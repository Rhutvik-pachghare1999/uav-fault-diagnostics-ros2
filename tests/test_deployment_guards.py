"""Deployment-safety tests for the ROS2 inference node (guards fire BEFORE
any rclpy import, so they run in plain CI without a ROS install).

Pins the live-deployment audit fixes:
  * RPM-consuming model in live mode without --rpm-topic refuses to start
  * checkpoint without a 'vars' channel list refuses to start
  * unknown model channels refuse to start
  * replay with missing CSV columns aborts unless --allow-missing-columns
"""

import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import VARS9, VARS13, make_checkpoint  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NODE = REPO / "scripts" / "ros2_inference_node.py"


def run_node(model, *args):
    return subprocess.run(
        [sys.executable, str(NODE), "--model", str(model), *args],
        capture_output=True, text=True)


def imu_csv(path, n=120):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(VARS9)
        for i in range(n):
            w.writerow([round(0.01 * i, 4)] * 9)
    return path


def test_live_rpm_model_requires_rpm_topic(tmp_path, rpm_ckpt):
    cp = run_node(rpm_ckpt, "--mode", "live")
    assert cp.returncode != 0
    assert "--rpm-topic" in cp.stdout + cp.stderr
    assert "IMU-only" in cp.stdout + cp.stderr


def test_checkpoint_without_vars_refuses(tmp_path):
    import torch
    bad = make_checkpoint(tmp_path / "no_vars.pth", VARS13)
    d = torch.load(bad, map_location="cpu")
    del d["meta"]["vars"]
    torch.save(d, bad)
    cp = run_node(bad, "--mode", "replay", "--input", "whatever.csv")
    assert cp.returncode != 0
    assert "'vars'" in cp.stdout + cp.stderr


def test_unknown_channels_refuse(tmp_path):
    ck = make_checkpoint(tmp_path / "alien.pth", ["rpm1", "warp_drive"])
    cp = run_node(ck, "--mode", "live")
    assert cp.returncode != 0
    assert "cannot produce" in cp.stdout + cp.stderr


def test_replay_missing_columns_aborts(tmp_path, rpm_ckpt):
    log = imu_csv(tmp_path / "log.csv")
    cp = run_node(rpm_ckpt, "--mode", "replay", "--input", str(log))
    assert cp.returncode != 0
    assert "missing model channels" in cp.stdout + cp.stderr


def test_replay_allow_missing_columns_compromised(tmp_path, rpm_ckpt):
    log = imu_csv(tmp_path / "log.csv")
    cp = run_node(rpm_ckpt, "--mode", "replay", "--input", str(log),
                  "--allow-missing-columns")
    assert cp.returncode == 0
    assert "COMPROMISED" in cp.stdout
    assert "Completed" in cp.stdout


def test_replay_imu9_model_happy_path(tmp_path, imu9_ckpt):
    log = imu_csv(tmp_path / "log.csv")
    cp = run_node(imu9_ckpt, "--mode", "replay", "--input", str(log))
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "Completed" in cp.stdout
