#!/usr/bin/env python3
"""
ROS2 node for UAV fault inference.

Modes:
  1. Live inference: Subscribes to IMU topic, publishes fault detection
  2. Log replay: Replays CSV/ULog file through the model (offline)

Run after sourcing ROS2:
  source /opt/ros/jazzy/setup.sh
  
  # Live inference
  python3 scripts/ros2_inference_node.py --model models/cnn_multi.pth --mode live
  
  # Offline replay with CSV
  python3 scripts/ros2_inference_node.py --model models/cnn_multi.pth --mode replay --input data/flight_log.csv
"""

import argparse
import collections
import math
import sys
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description="UAV Fault Inference ROS2 Node")
    p.add_argument("--model", required=True, help="Path to trained model (.pth)")
    p.add_argument("--mode", choices=["live", "replay"], default="live",
                   help="Operating mode: live (ROS2 subscription) or replay (offline CSV)")
    p.add_argument("--input", help="Input CSV file for replay mode")
    p.add_argument("--window", type=int, default=100, help="Sliding window size")
    p.add_argument("--step", type=int, default=10, help="Step size for replay mode")
    p.add_argument("--topic", default="/imu/data", help="IMU topic (live mode)")
    p.add_argument("--publish_topic", default="/fault_detection", help="Output topic (live mode)")
    p.add_argument("--publish-severity", action="store_true", help="Publish severity alongside fault_id")
    p.add_argument("--severity-map", type=str, default="", help="JSON file mapping fault_id -> severity")
    p.add_argument("--output", help="Output CSV for replay mode predictions")
    args = p.parse_args()

    # Load model
    import torch
    ck = torch.load(args.model, map_location="cpu")
    
    # Add scripts to path for imports
    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))
    from cnn_classifier import PaperCNN
    
    meta = ck.get("meta", {})
    n_faults = meta.get("n_faults", 16)
    model = PaperCNN(in_channels=1, base_filters=32, num_classes=n_faults)
    sd = ck.get("state_dict", ck)
    if all(k.startswith("module.") for k in sd.keys()):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    
    # Normalization stats
    mean = meta.get("mean", None)
    std = meta.get("std", None)
    if mean is not None:
        mean = np.array(mean, dtype="float32")
    if std is not None:
        std = np.array(std, dtype="float32")
    
    # Class labels
    fault_map = meta.get('fault_label_map', {})
    rev_map = {v: k for k, v in fault_map.items()}

    # Channel layout must match training exactly (stored in model meta)
    var_list = meta.get("vars") or ["rpm1", "rpm2", "rpm3", "rpm4",
                                     "roll", "pitch", "yaw",
                                     "gyro_x", "gyro_y", "gyro_z"]
    n_ch = len(var_list)
    
    # Severity map
    sev_map = {}
    if args.publish_severity and args.severity_map:
        import json
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
        """Run inference on a single window.
        X_win can be (C, W) or (W, C) - we handle both.
        """
        X = np.array(X_win)
        # Ensure shape is (C, W) - channels first
        if X.shape[0] == args.window and X.shape[1] == n_ch:
            # (W, C) -> transpose to (C, W)
            X = X.T
        elif X.shape[0] == n_ch and X.shape[1] == args.window:
            # Already (C, W) - correct
            pass
        else:
            raise ValueError(f"Unexpected input shape: {X.shape} "
                             f"(expected ({n_ch}, {args.window}))")
        X = X[None, None, :, :].astype("float32")  # (1,1,C,W)
        if mean is not None and std is not None:
            try:
                X = (X - mean) / (std + 1e-9)
            except Exception:
                X = (X - mean.reshape(1, 1, mean.shape[-2], 1)) / (std.reshape(1, 1, std.shape[-2], 1) + 1e-9)
        inp = torch.from_numpy(X)
        with torch.no_grad():
            pf = model(inp)
            fid = int(pf.argmax(dim=1).item())
        label = rev_map.get(fid, f"class_{fid}")
        out = {"fault_id": fid, "label": label}
        if args.publish_severity:
            sev = sev_map.get(fid)
            if sev is not None:
                out['severity'] = int(sev)
        return out

    if args.mode == "replay":
        if not args.input:
            print("--input required for replay mode")
            return
        print(f"Running offline replay on {args.input}...")
        
        # Load CSV data
        import csv
        data = {}
        with open(args.input, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, val in row.items():
                    if key not in data:
                        data[key] = []
                    data[key].append(float(val))
        
        # Check required fields (channel list from model meta)
        required = list(var_list)
        available = set(data.keys())
        missing = [f for f in required if f not in available]
        if missing:
            print(f"Missing fields {missing} in {args.input}; "
                  "they will be zero-filled.")

        # Build feature matrix in the exact training channel order
        n_samples = len(data[next(iter(data))])
        features = np.zeros((n_ch, n_samples))
        for i, field in enumerate(var_list):
            if field in data:
                features[i] = data[field]
        
        # Extract windows and run inference
        predictions = []
        for i in range(0, n_samples - args.window + 1, args.step):
            win = features[:, i:i+args.window]
            pred = run_inference(win)
            pred['sample_index'] = i
            predictions.append(pred)
            if len(predictions) % 50 == 0:
                print(f"  Processed {len(predictions)} windows... latest: {pred['label']}")
        
        print(f"Completed {len(predictions)} predictions")
        
        # Save output if requested
        if args.output:
            import pandas as pd
            df = pd.DataFrame(predictions)
            df.to_csv(args.output, index=False)
            print(f"Saved predictions to {args.output}")
        
        # Print summary
        from collections import Counter
        counts = Counter([p['label'] for p in predictions])
        print("\nPrediction summary:")
        for label, count in sorted(counts.items()):
            print(f"  {label}: {count}")
        
        return

    # Live ROS2 mode
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Imu
        from std_msgs.msg import String
    except Exception:
        print("ROS2 python packages not available. Install rclpy and run in a ROS2 environment.")
        print("Try: source /opt/ros/jazzy/setup.sh")
        return

    class InferenceNode(Node):
        def __init__(self):
            super().__init__("fault_inference_node")
            self.window_size = args.window
            self.win = collections.deque(maxlen=self.window_size)
            self.sub = self.create_subscription(Imu, args.topic, self.cb_imu, 10)
            self.pub = self.create_publisher(String, args.publish_topic, 10)
            self.get_logger().info(f"Fault inference node started. Subscribing to {args.topic}")
            self.get_logger().info(f"Publishing to {args.publish_topic}")
            self.get_logger().info(f"Window size: {self.window_size}")
            self.sample_count = 0

        def cb_imu(self, msg: Imu):
            # Build the feature row in the model's channel order (meta-driven,
            # 13-channel capable). RPY comes from the orientation quaternion;
            # RPM channels are zero-filled unless provided by the platform.
            q = msg.orientation
            roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z),
                              1.0 - 2.0 * (q.x * q.x + q.y * q.y))
            sinp = 2.0 * (q.w * q.y - q.z * q.x)
            pitch = (math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0
                     else math.asin(sinp))
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            fields = {
                "acc_x": msg.linear_acceleration.x,
                "acc_y": msg.linear_acceleration.y,
                "acc_z": msg.linear_acceleration.z,
                "gyro_x": msg.angular_velocity.x,
                "gyro_y": msg.angular_velocity.y,
                "gyro_z": msg.angular_velocity.z,
                "roll": roll, "pitch": pitch, "yaw": yaw,
                "rpm1": 0.0, "rpm2": 0.0, "rpm3": 0.0, "rpm4": 0.0,
            }
            arr = [float(fields.get(v, 0.0)) for v in var_list]
            self.win.append(arr)
            self.sample_count += 1
            
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