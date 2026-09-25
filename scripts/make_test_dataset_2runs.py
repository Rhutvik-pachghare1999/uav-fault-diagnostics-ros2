#!/usr/bin/env python3
"""Generate synthetic dataset with only 2 runs to test leaky fallback."""

import os
import json
import numpy as np
import h5py
import pandas as pd


def make_synthetic_imu(run_id, fault_mask, n_samples=500, window=100, step=10):
    """Generate one synthetic run with IMU data."""
    T = n_samples + window
    t = np.linspace(0, 10, T)

    rpm_base = 3000 + 50 * np.sin(0.5 * t)
    rpm = np.tile(rpm_base, (4, 1)).T

    for prop_idx in range(4):
        if (fault_mask >> prop_idx) & 1:
            rpm[:, prop_idx] *= np.random.uniform(0.6, 0.9)

    rpm += np.random.normal(0, 20, rpm.shape)

    acc = np.random.normal(0, 0.5, (T, 3))
    gyro = np.random.normal(0, 0.05, (T, 3))
    rpm_freq = np.mean(rpm) / 60.0
    acc[:, 0] += 0.5 * np.sin(2 * np.pi * rpm_freq * t)

    roll = np.random.normal(0, 0.1, T)
    pitch = np.random.normal(0, 0.1, T)
    yaw = np.random.normal(0, 0.1, T)

    df = pd.DataFrame(
        {
            "time": t,
            "acc_x": acc[:, 0],
            "acc_y": acc[:, 1],
            "acc_z": acc[:, 2],
            "gyro_x": gyro[:, 0],
            "gyro_y": gyro[:, 1],
            "gyro_z": gyro[:, 2],
            "rpm1": rpm[:, 0],
            "rpm2": rpm[:, 1],
            "rpm3": rpm[:, 2],
            "rpm4": rpm[:, 3],
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }
    )
    return df


def build_synthetic_dataset_2runs(n_windows_per_run=40, window=100, step=10, out_h5="ml_dataset_2runs.h5"):
    """Build synthetic dataset with only 2 runs."""
    DEFAULT_VARS = ["rpm1", "rpm2", "rpm3", "rpm4", "roll", "pitch", "yaw", "gyro_x", "gyro_y", "gyro_z"]

    runs = []
    fault_label_map = {}
    next_label = 0

    # Only 2 runs - less than 3
    for run_id in range(2):
        fault_mask = run_id % 4  # 4 different fault classes
        df = make_synthetic_imu(run_id, fault_mask, n_samples=500)

        fl = f"label_{fault_mask}"
        if fl not in fault_label_map:
            fault_label_map[fl] = next_label
            next_label += 1
        label = fault_label_map[fl]

        data = df[DEFAULT_VARS].values
        T = data.shape[0]
        for i in range(0, T - window + 1, step):
            win = data[i : i + window].T
            runs.append((win, label, run_id))

    if not runs:
        raise ValueError("No windows generated")

    X = np.stack([r[0] for r in runs], axis=0).astype("float32")
    X = X[:, None, :, :]
    y_fault = np.array([r[1] for r in runs], dtype="int64")
    run_ids = np.array([r[2] for r in runs], dtype="int64")

    print(f"Generated: {len(X)} windows, {len(np.unique(y_fault))} classes, {len(np.unique(run_ids))} runs")
    print(f"Run distribution: {np.bincount(run_ids)}")
    print(f"Class distribution: {np.bincount(y_fault)}")

    os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("X", data=X, compression="gzip")
        f.create_dataset("y_fault", data=y_fault)
        f.create_dataset("run_id", data=run_ids)
        f.attrs["meta"] = json.dumps({"fault_label_map": fault_label_map, "window": window, "vars": DEFAULT_VARS})

    print(f"Wrote {out_h5}")
    return out_h5


if __name__ == "__main__":
    build_synthetic_dataset_2runs(out_h5="ml_dataset_2runs.h5")
