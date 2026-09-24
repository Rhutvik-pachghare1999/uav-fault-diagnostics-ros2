#!/usr/bin/env python3
"""
ROS2 node for UAV fault inference.

Modes:
  1. Live inference: subscribes to IMU (and optionally rotor-RPM telemetry),
     publishes fault detections on a sliding window.
  2. Log replay: replays a CSV flight log through the model (offline).

Safety rules (added after the deployment audit):
  * If the model consumes rpm channels in live mode, a rotor RPM topic
    (--rpm-topic, std_msgs/Float32MultiArray, 4 values) is REQUIRED. Without
    it the node refuses to start — zero-filling RPMs would feed the model
    far-out-of-domain inputs and silently produce garbage. For IMU-only
    deployments use an IMU-only model (e.g. models/cnn_imu9.pth).
  * Replay mode REQUIRES every model channel to exist in the CSV. Missing
    columns abort the run (--allow-missing-columns zero-fills them and marks
    predictions as compromised).
  * Unknown model channels (not produced by IMU/RPM topics) abort startup.

Run after sourcing ROS2:
  source /opt/ros/jazzy/setup.sh

  # Live inference, model with RPM channels:
  python3 scripts/ros2_inference_node.py --model models/cnn_sealed.pth --mode live \
      --rpm-topic /rotor_rpms

  # Live inference, IMU-only model (no RPM telemetry needed):
  python3 scripts/ros2_inference_node.py --model models/cnn_imu9.pth --mode live

  # Offline replay with CSV (strict column check):
  python3 scripts/ros2_inference_node.py --model models/cnn_sealed.pth --mode replay \
      --input data/flight_log.csv
"""

import argparse
import collections
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

RPM_CHANNELS = ("rpm1", "rpm2", "rpm3", "rpm4")
IMU_CHANNELS = ("acc_x", "acc_y", "acc_z",
                "gyro_x", "gyro_y", "gyro_z",
                "roll", "pitch", "yaw")


def main():
    p = argparse.ArgumentParser(description="UAV Fault Inference ROS2 Node")
    p.add_argument("--model", required=True, help="Path to trained model (.pth)")
    p.add_argument("--mode", choices=["live", "replay"], default="live",
                   help="Operating mode: live (ROS2 subscription) or replay (offline CSV)")
    p.add_argument("--input", help="Input CSV file for replay mode")
    p.add_argument("--window", type=int, default=100, help="Sliding window size")
    p.add_argument("--step", type=int, default=10, help="Step size for replay mode")
    p.add_argument("--topic", default="/imu/data", help="IMU topic (live mode)")
    p.add_argument("--rpm-topic", default=None,
                   help="rotor RPM telemetry topic (std_msgs/Float32MultiArray, "
                        "4 rotor RPMs). REQUIRED in live mode if the model "
                        "consumes rpm channels.")
    p.add_argument("--rpm-max-age", type=float, default=1.0,
                   help="max age (s) of cached RPM values before inference is "
                        "suspended in live mode")
    p.add_argument("--publish_topic", default="/fault_detection", help="Output topic (live mode)")
    p.add_argument("--publish-severity", action="store_true", help="Publish severity alongside fault_id")
    p.add_argument("--severity-map", type=str, default="", help="JSON file mapping fault_id -> severity")
    p.add_argument("--allow-missing-columns", action="store_true",
                   help="replay escape hatch: zero-fill missing CSV columns "
                        "(predictions are marked compromised)")
    p.add_argument("--output", help="Output CSV for replay mode predictions")
    args = p.parse_args()

    # Load model
    import torch
    ck = torch.load(args.model, map_location="cpu")

    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))
    from cnn_classifier import PaperCNN

    meta = ck.get("meta", {})
    n_faults = meta.get("n_faults", 16)
    base_filters = int(meta.get("base_filters", 32))
    model = PaperCNN(in_channels=1, base_filters=base_filters, num_classes=n_faults)
    sd = ck.get("state_dict", ck)
    if all(k.startswith("module.") for k in sd.keys()):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    mean = meta.get("mean", None)
    std = meta.get("std", None)
    mean = np.array(mean, dtype="float32") if mean is not None else None
    std = np.array(std, dtype="float32") if std is not None else None

    fault_map = meta.get('fault_label_map', {})
    rev_map = {v: k for k, v in fault_map.items()}

    # Channel layout must match training exactly (stored in model meta).
    # NOTE: if the checkpoint has no vars list we CANNOT guess the layout —
    # refuse rather than silently mis-channel the data.
    var_list = meta.get("vars") or []
    if not var_list:
        raise SystemExit("checkpoint meta has no 'vars' channel list — cannot "
                         "safely map sensor data to model inputs. Retrain with "
                         "scripts/train_cnn.py so the model is self-describing.")
    n_ch = len(var_list)
    unknown = [v for v in var_list if v not in RPM_CHANNELS + IMU_CHANNELS]
    if unknown:
        raise SystemExit(f"model consumes channels this node cannot produce: {unknown}. "
                         f"Supported: {list(RPM_CHANNELS + IMU_CHANNELS)}")
    needs_rpm = any(v.startswith("rpm") for v in var_list)
    print(f"model channels ({n_ch}): {var_list}")

    if args.mode == "live" and needs_rpm and not args.rpm_topic:
        raise SystemExit(
            "MODEL REQUIRES RPM CHANNELS but no --rpm-topic was given.\n"
            "  Training RPMs are ~2757-3919; zero-filling rpm1-4 would feed the "
            "model far-out-of-domain inputs and silently invalidate predictions.\n"
            "  Fix one of:\n"
            "    1) pass --rpm-topic <Float32MultiArray topic with 4 rotor RPMs>, or\n"
            "    2) deploy the IMU-only model: --model models/cnn_imu9.pth")

    # Severity map
    sev_map = {}
    if args.publish_severity and args.severity_map:
        try:
            with open(args.severity_map, 'r') as f:
                m = json.load(f)
            for k, v in m.items():
                try:
                    sev_map[int(k)] = int(v)
                except Exception:
                    sev_map[k] = int(v)
        except Exception as e:
            print(f"Warning: Failed to load severity map: {e}")

    def run_inference(X_win):
        """Run inference on a single window (C, W) or (W, C)."""
        X = np.array(X_win)
        if X.shape[0] == args.window and X.shape[1] == n_ch:
            X = X.T
        elif X.shape[0] == n_ch and X.shape[1] == args.window:
            pass
        else:
            raise ValueError(f"Unexpected input shape: {X.shape} "
                             f"(expected ({n_ch}, {args.window}))")
        X = X[None, None, :, :].astype("float32")
        if mean is not None and std is not None:
            X = (X - mean) / (std + 1e-9)
        inp = torch.from_numpy(X)
        with torch.no_grad():
            pf = model(inp)
            fid = int(pf.argmax(dim=1).item())
        out = {"fault_id": fid, "label": rev_map.get(fid, f"class_{fid}")}
        if args.publish_severity:
            sev = sev_map.get(fid)
            if sev is not None:
                out['severity'] = int(sev)
        return out

    if args.mode == "replay":
        if not args.input:
            raise SystemExit("--input required for replay mode")
        print(f"Running offline replay on {args.input}...")

        data = {}
        with open(args.input, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, val in row.items():
                    data.setdefault(key, []).append(float(val))

        # STRICT column check: every model channel must exist in the CSV
        missing = [f for f in var_list if f not in data]
        if missing:
            if not args.allow_missing_columns:
                raise SystemExit(
                    f"Replay aborted: CSV {args.input} is missing model channels "
                    f"{missing}.\nZero-filling them would feed the model "
                    f"out-of-domain inputs and silently invalidate predictions.\n"
                    f"Fix the log to contain all of {var_list}, or pass "
                    f"--allow-missing-columns to accept compromised predictions.")
            print(f"WARNING: --allow-missing-columns set; zero-filling {missing}. "
                  f"Predictions involving these windows are COMPROMISED.")

        n_samples = len(data[next(iter(data))])
        features = np.zeros((n_ch, n_samples))
        for i, field in enumerate(var_list):
            if field in data:
                features[i] = data[field]

        predictions = []
        for i in range(0, n_samples - args.window + 1, args.step):
            win = features[:, i:i + args.window]
            pred = run_inference(win)
            pred['sample_index'] = i
            predictions.append(pred)
            if len(predictions) % 50 == 0:
                print(f"  Processed {len(predictions)} windows... latest: {pred['label']}")

        print(f"Completed {len(predictions)} predictions")

        if args.output:
            import pandas as pd
            pd.DataFrame(predictions).to_csv(args.output, index=False)
            print(f"Saved predictions to {args.output}")

        counts = collections.Counter([p['label'] for p in predictions])
        print("\nPrediction summary:")
        for label, count in sorted(counts.items()):
            print(f"  {label}: {count}")
        return

    # ---------------- Live ROS2 mode ----------------
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Imu
        from std_msgs.msg import Float32MultiArray, String
    except Exception:
        print("ROS2 python packages not available. Install rclpy and run in a ROS2 environment.")
        print("Try: source /opt/ros/jazzy/setup.sh")
        return

    class InferenceNode(Node):
        def __init__(self):
            super().__init__("fault_inference_node")
            self.window_size = args.window
            self.win = collections.deque(maxlen=self.window_size)
            self.rpm_cache = None
            self.rpm_time = None
            self.rpm_stale_warned = False
            self.sub = self.create_subscription(Imu, args.topic, self.cb_imu, 10)
            if needs_rpm:
                self.rpm_sub = self.create_subscription(
                    Float32MultiArray, args.rpm_topic, self.cb_rpm, 10)
            self.pub = self.create_publisher(String, args.publish_topic, 10)
            self.get_logger().info(f"Fault inference node started. Subscribing to {args.topic}")
            if needs_rpm:
                self.get_logger().info(f"Subscribing to RPM telemetry on {args.rpm_topic}")
            self.get_logger().info(f"Publishing to {args.publish_topic}")
            self.get_logger().info(f"Window size: {self.window_size}")
            self.sample_count = 0

        def cb_rpm(self, msg: Float32MultiArray):
            if len(msg.data) != 4:
                self.get_logger().warn(
                    f"expected 4 rotor RPMs on {args.rpm_topic}, got {len(msg.data)}")
                return
            self.rpm_cache = [float(v) for v in msg.data]
            self.rpm_time = time.monotonic()

        def cb_imu(self, msg: Imu):
            q = msg.orientation
            roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z),
                              1.0 - 2.0 * (q.x * q.x + q.y * q.y))
            sinp = 2.0 * (q.w * q.y - q.z * q.x)
            pitch = (math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0
                     else math.asin(sinp))
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            imu_fields = {
                "acc_x": msg.linear_acceleration.x,
                "acc_y": msg.linear_acceleration.y,
                "acc_z": msg.linear_acceleration.z,
                "gyro_x": msg.angular_velocity.x,
                "gyro_y": msg.angular_velocity.y,
                "gyro_z": msg.angular_velocity.z,
                "roll": roll, "pitch": pitch, "yaw": yaw,
            }
            self.sample_count += 1

            if needs_rpm:
                stale = (self.rpm_cache is None
                         or time.monotonic() - self.rpm_time > args.rpm_max_age)
                if stale:
                    if not self.rpm_stale_warned or self.sample_count % 500 == 0:
                        self.get_logger().error(
                            f"no fresh RPM telemetry on {args.rpm_topic} — "
                            f"INFERENCE SUSPENDED (would be out-of-domain)")
                        self.rpm_stale_warned = True
                    return
                rpm_fields = dict(zip(RPM_CHANNELS, self.rpm_cache))
            else:
                rpm_fields = {}

            row = []
            for v in var_list:
                if v in rpm_fields:
                    row.append(rpm_fields[v])
                else:
                    row.append(imu_fields[v])   # KeyError impossible: validated at startup
            self.win.append(row)

            if len(self.win) == self.window_size:
                X_win = np.array(self.win)
                pred = run_inference(X_win)
                out_str = str(pred)
                self.pub.publish(String(data=out_str))
                if self.sample_count % 100 == 0:
                    self.get_logger().info(f"Published: {out_str}")

    rclpy.init()
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
