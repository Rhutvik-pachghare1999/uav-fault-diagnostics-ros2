#!/usr/bin/env python3
"""
PX4 ULog replay utility for ROS2.
Converts PX4 ULog files to IMU messages for replay.
Can also replay from CSV format (e.g., from our synthetic dataset).
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_csv_imu(csv_path):
    """Load IMU data from CSV file."""
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                data[key].append(float(val))
    return {k: np.array(v) for k, v in data.items()}


def load_ulog_imu(ulog_path):
    """Load IMU data from PX4 ULog file."""
    try:
        from pyulog import ULog
    except ImportError:
        print("pyulog not installed. Install with: pip install pyulog")
        return None
    
    ulog = ULog(ulog_path)
    
    # Find sensor_combined or vehicle_imu messages
    imu_data = {}
    for msg in ulog.data_list:
        if msg.name in ['sensor_combined', 'vehicle_imu', 'vehicle_imu_status']:
            for field in msg.field_names:
                if field not in ['timestamp', 'timestamp_sample']:
                    imu_data[field] = msg.data[field]
    
    if not imu_data:
        print(f"No IMU data found in {ulog_path}")
        return None
    
    return imu_data


def create_ros2_imu_msg(accel, gyro, timestamp=None):
    """Create a ROS2 IMU message dictionary."""
    from sensor_msgs.msg import Imu
    from builtin_interfaces.msg import Time
    from std_msgs.msg import Header
    
    msg = Imu()
    if timestamp is not None:
        sec = int(timestamp)
        nanosec = int((timestamp - sec) * 1e9)
        msg.header.stamp = Time(sec=sec, nanosec=nanosec)
    msg.header.frame_id = "imu_link"
    
    # Linear acceleration
    msg.linear_acceleration.x = float(accel[0])
    msg.linear_acceleration.y = float(accel[1])
    msg.linear_acceleration.z = float(accel[2])
    
    # Angular velocity
    msg.angular_velocity.x = float(gyro[0])
    msg.angular_velocity.y = float(gyro[1])
    msg.angular_velocity.z = float(gyro[2])
    
    # Orientation (identity quaternion - not used by our model)
    msg.orientation.w = 1.0
    msg.orientation.x = 0.0
    msg.orientation.y = 0.0
    msg.orientation.z = 0.0
    
    # Covariances (unknown)
    msg.orientation_covariance = [-1.0] + [0.0] * 8
    msg.angular_velocity_covariance = [-1.0] + [0.0] * 8
    msg.linear_acceleration_covariance = [-1.0] + [0.0] * 8
    
    return msg


def replay_csv_to_ros2(csv_path, topic="/imu/data", rate=100, window=100):
    """
    Replay CSV IMU data to ROS2 topic.
    This is a standalone script that publishes IMU messages.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Imu
    except Exception:
        print("ROS2 python packages not available. Install rclpy and run in a ROS2 environment.")
        return
    
    data = load_csv_imu(csv_path)
    if not data:
        print(f"Failed to load data from {csv_path}")
        return
    
    # Check required fields
    required = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    for field in required:
        if field not in data:
            print(f"Missing required field: {field}")
            return
    
    n_samples = len(data['acc_x'])
    print(f"Loaded {n_samples} samples from {csv_path}")
    
    rclpy.init()
    node = Node('px4_log_replay')
    pub = node.create_publisher(Imu, topic, 10)
    
    # Create rate object
    loop_rate = node.create_rate(rate)
    
    print(f"Publishing to {topic} at {rate} Hz...")
    
    try:
        for i in range(n_samples):
            accel = [data['acc_x'][i], data['acc_y'][i], data['acc_z'][i]]
            gyro = [data['gyro_x'][i], data['gyro_y'][i], data['gyro_z'][i]]
            timestamp = i / rate
            
            msg = create_ros2_imu_msg(accel, gyro, timestamp)
            pub.publish(msg)
            
            if i % 1000 == 0:
                print(f"Published {i}/{n_samples} samples")
            
            try:
                loop_rate.sleep()
            except Exception:
                time.sleep(1.0 / rate)
                
    except KeyboardInterrupt:
        print("\nReplay interrupted")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def extract_imu_windows(csv_path, window=100, step=10):
    """Extract sliding windows from CSV for offline inference."""
    data = load_csv_imu(csv_path)
    if not data:
        return None, None
    
    required = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    for field in required:
        if field not in data:
            print(f"Missing required field: {field}")
            return None, None
    
    # Build feature matrix (C, T)
    C = len(required)
    T = len(data[required[0]])
    features = np.zeros((C, T))
    for i, field in enumerate(required):
        features[i] = data[field]
    
    # Add RPM columns if available (pad with zeros)
    rpm_fields = ['rpm1', 'rpm2', 'rpm3', 'rpm4']
    for rpm in rpm_fields:
        if rpm in data:
            features = np.vstack([features, data[rpm].reshape(1, -1)])
        else:
            features = np.vstack([features, np.zeros((1, T))])
    
    # Extract windows
    windows = []
    indices = []
    for i in range(0, T - window + 1, step):
        win = features[:, i:i+window]
        windows.append(win)
        indices.append(i)
    
    return np.array(windows), np.array(indices)


def run_offline_inference(csv_path, model_path, window=100, step=10):
    """Run offline inference on CSV data using trained model."""
    import torch
    import sys
    
    # Add scripts to path
    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))
    from cnn_classifier import PaperCNN
    
    # Load model
    ck = torch.load(model_path, map_location="cpu")
    meta = ck.get("meta", {})
    n_faults = meta.get("n_faults", 16)
    mean = meta.get("mean", None)
    std = meta.get("std", None)
    
    model = PaperCNN(in_channels=1, base_filters=32, num_classes=n_faults)
    sd = ck.get("state_dict", ck)
    if all(k.startswith("module.") for k in sd.keys()):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    
    if mean is not None:
        mean = np.array(mean, dtype="float32")
    if std is not None:
        std = np.array(std, dtype="float32")
    
    # Extract windows
    windows, indices = extract_imu_windows(csv_path, window, step)
    if windows is None:
        return
    
    print(f"Running inference on {len(windows)} windows...")
    
    # Load class labels
    fault_map = meta.get('fault_label_map', {})
    rev_map = {v: k for k, v in fault_map.items()}
    
    predictions = []
    for i, win in enumerate(windows):
        X = win[None, None, :, :].astype("float32")  # (1,1,C,W)
        if mean is not None and std is not None:
            X = (X - mean) / (std + 1e-9)
        
        inp = torch.from_numpy(X)
        with torch.no_grad():
            out = model(inp)
            pred = int(out.argmax(dim=1).item())
        
        label = rev_map.get(pred, f"class_{pred}")
        predictions.append((indices[i], pred, label))
        
        if i % 100 == 0:
            print(f"  Window {i}/{len(windows)}: {label}")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="PX4 Log Replay for ROS2")
    parser.add_argument("input", help="Input CSV or ULog file")
    parser.add_argument("--mode", choices=["ros2", "offline"], default="ros2",
                        help="Replay mode: ros2 (publish to ROS2) or offline (run inference)")
    parser.add_argument("--model", help="Model path for offline inference")
    parser.add_argument("--topic", default="/imu/data", help="ROS2 topic to publish to")
    parser.add_argument("--rate", type=int, default=100, help="Replay rate (Hz)")
    parser.add_argument("--window", type=int, default=100, help="Window size for offline inference")
    parser.add_argument("--step", type=int, default=10, help="Step size for offline inference")
    args = parser.parse_args()
    
    if args.mode == "ros2":
        replay_csv_to_ros2(args.input, args.topic, args.rate)
    elif args.mode == "offline":
        if not args.model:
            print("--model required for offline mode")
            return
        preds = run_offline_inference(args.input, args.model, args.window, args.step)
        if preds:
            # Print summary
            from collections import Counter
            counts = Counter([p[2] for p in preds])
            print("\nPrediction summary:")
            for label, count in sorted(counts.items()):
                print(f"  {label}: {count}")
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()