#!/usr/bin/env python3
"""PX4 log replay for ROS 2, and contract-honoring offline inference.

Modes:
  ros2     Publish sensor_msgs/Imu messages from a CSV or PX4 ULog file at a
           given rate. Orientation is published as a real quaternion when the
           source has attitude data (CSV roll/pitch/yaw columns or ULog
           vehicle_attitude); otherwise the identity quaternion is sent with
           orientation_covariance[0] = -1 (the REP 1203 "orientation invalid"
           marker) so downstream consumers can refuse to trust it.
  offline  Run a sealed CNN checkpoint on sliding windows from a CSV, honoring
           the checkpoint's channel contract exactly like
           scripts/ros2_inference_node.py --mode replay.

Contract rules (mirroring ros2_inference_node.py):
  * Features are built in the checkpoint's meta['vars'] order, never guessed.
  * Every model channel must exist in the CSV. Missing columns abort the run;
    --allow-missing-columns zero-fills them and marks the predictions as
    compromised.
  * n_faults and base_filters are read from the checkpoint meta, not assumed.
  * No silent zero-filling anywhere.

Usage:
  python3 scripts/px4_log_replay.py flight_log.csv --mode ros2 --rate 500
  python3 scripts/px4_log_replay.py flight_log.ulg --mode ros2 --rate 200
  python3 scripts/px4_log_replay.py imu.csv --mode offline \
      --model models/cnn_imu9.pth --output results/px4_replay.csv

Note on real PX4 logs: PX4 logs use NED/FRD frame conventions while the
sealed models were trained on Isaac Sim telemetry (FLU-style body axes,
gravity along -Z). Raw PX4 accel/gyro values are therefore not directly
in-domain for these models; predictions on unconverted PX4 logs are
indicative only.
"""

import argparse
import collections
import csv
import math
import time
from pathlib import Path

import numpy as np

RPM_CHANNELS = ("rpm1", "rpm2", "rpm3", "rpm4")
EULER_CHANNELS = ("roll", "pitch", "yaw")


def load_csv_imu(csv_path):
    """Load a CSV into {column_name: np.ndarray} (all numeric columns)."""
    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key is None:
                    continue
                data.setdefault(key, []).append(float(val))
    return {k: np.array(v) for k, v in data.items()}


def _ulog_field(msg, base, i):
    """PX4 ORB multi-field accessor: try both 'base[i]' and 'base[i]/rad' styles."""
    for name in (f"{base}[{i}]", f"{base}[{i}]/rad", base):
        if name in msg.data:
            return np.asarray(msg.data[name], dtype="float64")
    return None


def load_ulog_imu(ulog_path):
    """Load IMU samples from a PX4 ULog.

    Returns dict with keys 't' (N,), 'accel' (N,3), 'gyro' (N,3) and optionally
    'quat' (N,4, w-first PX4 convention) aligned to the IMU timestamps via
    nearest-neighbor lookup against vehicle_attitude. Returns None when no
    recognized IMU message is present.
    """
    try:
        from pyulog import ULog
    except ImportError:
        print("pyulog not installed. Install with: pip install pyulog")
        return None

    ulog = ULog(ulog_path)

    imu_msg = None
    for name in ("sensor_combined", "vehicle_imu", "vehicle_gyro"):
        for msg in ulog.data_list:
            if msg.name == name:
                imu_msg = msg
                break
        if imu_msg is not None:
            break
    if imu_msg is None:
        avail = sorted({m.name for m in ulog.data_list})
        print(f"No IMU message found in {ulog_path}.")
        print(f"Messages present: {avail}")
        return None

    t = np.asarray(imu_msg.data["timestamp"], dtype="float64")
    accel = np.column_stack(
        [
            _ulog_field(imu_msg, base, i)
            for base, i in (("accelerometer_m_s2", 0), ("accelerometer_m_s2", 1), ("accelerometer_m_s2", 2))
        ]
    )
    gyro = np.column_stack(
        [
            _ulog_field(imu_msg, base, i)
            for base, i in (("gyroscope_rad", 0), ("gyroscope_rad", 1), ("gyroscope_rad", 2))
        ]
    )
    if accel is None or gyro is None or np.shape(accel)[1] != 3 or np.shape(gyro)[1] != 3:
        print(f"IMU message '{imu_msg.name}' lacks the expected accelerometer_m_s2/gyroscope_rad fields.")
        return None

    out = {"t": t, "accel": accel, "gyro": gyro, "quat": None}

    # Attitude (optional): nearest-neighbor align vehicle_attitude quaternions
    for msg in ulog.data_list:
        if msg.name == "vehicle_attitude" and "q[0]" in msg.data:
            tq = np.asarray(msg.data["timestamp"], dtype="float64")
            q = np.column_stack([np.asarray(msg.data[f"q[{i}]"], dtype="float64") for i in range(4)])
            idx = np.searchsorted(tq, t, side="right") - 1
            idx = np.clip(idx, 0, len(tq) - 1)
            out["quat"] = q[idx]
            break
    return out


def euler_to_quat(roll, pitch, yaw):
    """ZYX yaw-pitch-roll euler (rad) -> quaternion (x, y, z, w)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def create_ros2_imu_msg(accel, gyro, quat=None, timestamp=None):
    """Create a sensor_msgs/Imu message.

    quat: (x, y, z, w). When None, the identity quaternion is set with
    orientation_covariance[0] = -1 (invalid marker per REP 1203).
    """
    from sensor_msgs.msg import Imu
    from builtin_interfaces.msg import Time

    msg = Imu()
    if timestamp is not None:
        sec = int(timestamp)
        msg.header.stamp = Time(sec=sec, nanosec=int((timestamp - sec) * 1e9))
    msg.header.frame_id = "imu_link"

    msg.linear_acceleration.x = float(accel[0])
    msg.linear_acceleration.y = float(accel[1])
    msg.linear_acceleration.z = float(accel[2])
    msg.angular_velocity.x = float(gyro[0])
    msg.angular_velocity.y = float(gyro[1])
    msg.angular_velocity.z = float(gyro[2])

    if quat is not None:
        msg.orientation.x, msg.orientation.y = float(quat[0]), float(quat[1])
        msg.orientation.z, msg.orientation.w = float(quat[2]), float(quat[3])
        msg.orientation_covariance = [0.0] * 9
    else:
        msg.orientation.w = 1.0
        msg.orientation_covariance = [-1.0] + [0.0] * 8
    msg.angular_velocity_covariance = [0.0] * 9
    msg.linear_acceleration_covariance = [0.0] * 9
    return msg


def _ros2_publish(accel_gyro_iter, topic, rate):
    """Publish (accel, gyro, quat, timestamp) tuples to a ROS 2 Imu topic."""
    try:
        import rclpy
        from sensor_msgs.msg import Imu
    except Exception:
        print(
            "ROS2 python packages not available. Install rclpy and run in a "
            "ROS2 environment (source /opt/ros/jazzy/setup.sh)."
        )
        return

    rclpy.init()
    node = rclpy.create_node("px4_log_replay")
    pub = node.create_publisher(Imu, topic, 10)
    loop_rate = node.create_rate(rate)
    print(f"Publishing to {topic} at {rate} Hz...")

    n = 0
    try:
        for accel, gyro, quat, timestamp in accel_gyro_iter:
            pub.publish(create_ros2_imu_msg(accel, gyro, quat, timestamp))
            n += 1
            if n % 1000 == 0:
                print(f"Published {n} samples")
            try:
                loop_rate.sleep()
            except Exception:
                time.sleep(1.0 / rate)
    except KeyboardInterrupt:
        print("\nReplay interrupted")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def replay_csv_to_ros2(csv_path, topic, rate):
    """Replay a CSV to ROS 2. Orientation is real when roll/pitch/yaw exist."""
    data = load_csv_imu(csv_path)
    required = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
    missing = [f for f in required if f not in data]
    if missing:
        print(f"CSV {csv_path} is missing required IMU columns: {missing}")
        return

    has_euler = all(f in data for f in EULER_CHANNELS)
    if not has_euler:
        print(
            "WARNING: no roll/pitch/yaw columns — publishing orientation as "
            "INVALID (covariance[0] = -1). Models that need roll/pitch/yaw "
            "will suspend inference on this stream."
        )

    n = len(data["acc_x"])
    print(f"Loaded {n} samples from {csv_path}")

    def gen():
        for i in range(n):
            accel = [data["acc_x"][i], data["acc_y"][i], data["acc_z"][i]]
            gyro = [data["gyro_x"][i], data["gyro_y"][i], data["gyro_z"][i]]
            quat = None
            if has_euler:
                quat = euler_to_quat(data["roll"][i], data["pitch"][i], data["yaw"][i])
            yield accel, gyro, quat, i / rate

    _ros2_publish(gen(), topic, rate)


def replay_ulog_to_ros2(ulog_path, topic, rate):
    """Replay a PX4 ULog to ROS 2 with real orientation when available."""
    imu = load_ulog_imu(ulog_path)
    if imu is None:
        return
    if imu["quat"] is None:
        print(
            "WARNING: ULog has no vehicle_attitude message — publishing "
            "orientation as INVALID (covariance[0] = -1). Models that need "
            "roll/pitch/yaw will suspend inference on this stream."
        )

    n = len(imu["t"])
    print(f"Loaded {n} IMU samples from {ulog_path}")
    t0 = imu["t"][0]

    def gen():
        for i in range(n):
            quat = None
            if imu["quat"] is not None:
                qx, qy, qz, qw = imu["quat"][i]
                quat = (qx, qy, qz, qw)
            yield imu["accel"][i], imu["gyro"][i], quat, (imu["t"][i] - t0) * 1e-6

    _ros2_publish(gen(), topic, rate)


def run_offline_inference(csv_path, model_path, window=100, step=10, allow_missing_columns=False, output=None):
    """Offline inference honoring the checkpoint contract (see module docstring)."""
    import sys
    import torch

    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))
    from cnn_classifier import PaperCNN

    ck = torch.load(model_path, map_location="cpu")
    meta = ck.get("meta", {})

    var_list = meta.get("vars") or []
    if not var_list:
        raise SystemExit(
            f"checkpoint {model_path} meta has no 'vars' channel list — cannot "
            f"safely map CSV columns to model inputs. Retrain with "
            f"scripts/train_cnn.py so the model is self-describing."
        )

    n_faults = int(meta.get("n_faults", 16))
    base_filters = int(meta.get("base_filters", 32))
    model = PaperCNN(in_channels=1, base_filters=base_filters, num_classes=n_faults)
    sd = ck.get("state_dict", ck)
    if all(k.startswith("module.") for k in sd.keys()):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    rev_map = {v: k for k, v in meta.get("fault_label_map", {}).items()}
    print(f"model channels ({len(var_list)}): {list(var_list)}")

    data = load_csv_imu(csv_path)
    missing = [v for v in var_list if v not in data]
    compromised = False
    if missing:
        if not allow_missing_columns:
            raise SystemExit(
                f"Replay aborted: CSV {csv_path} is missing model channels "
                f"{missing}.\nZero-filling them would feed the model "
                f"out-of-domain inputs and silently invalidate predictions.\n"
                f"Fix the log to contain all of {list(var_list)}, or pass "
                f"--allow-missing-columns to accept compromised predictions."
            )
        compromised = True
        print(
            f"WARNING: --allow-missing-columns set; zero-filling {missing}. "
            f"All predictions from this run are COMPROMISED."
        )

    n_samples = len(data[next(iter(data))])
    features = np.zeros((len(var_list), n_samples), dtype="float32")
    for i, field in enumerate(var_list):
        if field in data:
            features[i] = data[field]

    mean = meta.get("mean")
    std = meta.get("std")
    mean = np.array(mean, dtype="float32") if mean is not None else None
    std = np.array(std, dtype="float32") if std is not None else None

    predictions = []
    for i in range(0, n_samples - window + 1, step):
        X = features[:, i : i + window][None, None, :, :]
        if mean is not None and std is not None:
            X = (X - mean) / (std + 1e-9)
        with torch.no_grad():
            logits = model(torch.from_numpy(X))
            fid = int(logits.argmax(dim=1).item())
        row = {"sample_index": i, "fault_id": fid, "label": rev_map.get(fid, f"class_{fid}")}
        if compromised:
            row["compromised"] = "MISSING_COLUMNS"
        predictions.append(row)
        if len(predictions) % 50 == 0:
            print(f"  processed {len(predictions)} windows... latest: {row['label']}")

    print(f"Completed {len(predictions)} predictions")
    counts = collections.Counter(p["label"] for p in predictions)
    print("\nPrediction summary:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

    if output:
        import pandas as pd

        pd.DataFrame(predictions).to_csv(output, index=False)
        print(f"Saved predictions to {output}")


def main():
    parser = argparse.ArgumentParser(description="PX4 Log Replay for ROS2")
    parser.add_argument("input", help="Input CSV or PX4 ULog (.ulg) file")
    parser.add_argument(
        "--mode",
        choices=["ros2", "offline"],
        default="ros2",
        help="Replay mode: ros2 (publish to ROS2) or offline (run inference)",
    )
    parser.add_argument("--model", help="Model checkpoint for offline inference")
    parser.add_argument("--topic", default="/imu/data", help="ROS2 topic to publish to")
    parser.add_argument(
        "--rate", type=float, default=500, help="Publish rate (Hz); the sealed dataset is sampled at 500 Hz"
    )
    parser.add_argument("--window", type=int, default=100, help="Window size for offline inference")
    parser.add_argument("--step", type=int, default=10, help="Step size for offline inference")
    parser.add_argument(
        "--allow-missing-columns",
        action="store_true",
        help="offline escape hatch: zero-fill missing CSV columns (predictions are marked compromised)",
    )
    parser.add_argument("--output", help="Output CSV for offline predictions")
    args = parser.parse_args()

    is_ulog = str(args.input).lower().endswith((".ulg", ".ulog"))

    if args.mode == "offline":
        if is_ulog:
            raise SystemExit(
                "offline inference from ULog is not supported; export the ULog "
                "to CSV first (columns must include the model's vars), or use "
                "ros2 mode to publish it into the inference node."
            )
        if not args.model:
            raise SystemExit("--model is required for offline mode")
        run_offline_inference(
            args.input,
            args.model,
            window=args.window,
            step=args.step,
            allow_missing_columns=args.allow_missing_columns,
            output=args.output,
        )
        return

    # ros2 mode
    if is_ulog:
        print(
            "Note: PX4 logs use FRD/NED conventions; the sealed models were "
            "trained on Isaac Sim telemetry (FLU-style axes). Predictions on "
            "raw PX4 streams are indicative only."
        )
        replay_ulog_to_ros2(args.input, args.topic, args.rate)
    else:
        replay_csv_to_ros2(args.input, args.topic, args.rate)


if __name__ == "__main__":
    main()
