"""Deployment-safety tests for the ROS2 inference node (guards fire BEFORE
any rclpy import, so they run in plain CI without a ROS install).

Pins the live-deployment audit fixes:
  * RPM-consuming model in live mode without --rpm-topic refuses to start
  * checkpoint without a 'vars' channel list refuses to start
  * unknown model channels refuse to start
  * replay with missing CSV columns aborts unless --allow-missing-columns
"""

import ast
import csv
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import VARS9, VARS13, make_checkpoint  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
NODE = REPO / "scripts" / "ros2_inference_node.py"


def run_node(model, *args):
    return subprocess.run([sys.executable, str(NODE), "--model", str(model), *args], capture_output=True, text=True)


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
    cp = run_node(rpm_ckpt, "--mode", "replay", "--input", str(log), "--allow-missing-columns")
    assert cp.returncode == 0
    assert "COMPROMISED" in cp.stdout
    assert "Completed" in cp.stdout


def test_replay_imu9_model_happy_path(tmp_path, imu9_ckpt):
    log = imu_csv(tmp_path / "log.csv")
    cp = run_node(imu9_ckpt, "--mode", "replay", "--input", str(log))
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "Completed" in cp.stdout


PX4 = REPO / "scripts" / "px4_log_replay.py"


def run_px4(*args):
    return subprocess.run([sys.executable, str(PX4), *args], capture_output=True, text=True)


def test_px4_offline_happy_path(tmp_path, imu9_ckpt):
    log = imu_csv(tmp_path / "log.csv")
    out = tmp_path / "pred.csv"
    cp = run_px4(str(log), "--mode", "offline", "--model", str(imu9_ckpt), "--step", "10", "--output", str(out))
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "Completed" in cp.stdout
    assert out.exists()
    with open(out) as f:
        header = f.readline().strip().split(",")
    assert header == ["sample_index", "fault_id", "label"]


def test_px4_offline_missing_columns_abort(tmp_path, rpm_ckpt):
    log = imu_csv(tmp_path / "log.csv")  # 9 IMU channels, no rpm1-4
    cp = run_px4(str(log), "--mode", "offline", "--model", str(rpm_ckpt))
    assert cp.returncode != 0
    assert "missing model channels" in cp.stdout + cp.stderr
    assert "--allow-missing-columns" in cp.stdout + cp.stderr


def test_px4_offline_allow_missing_marks_compromised(tmp_path, rpm_ckpt):
    log = imu_csv(tmp_path / "log.csv")
    out = tmp_path / "pred.csv"
    cp = run_px4(
        str(log), "--mode", "offline", "--model", str(rpm_ckpt), "--allow-missing-columns", "--output", str(out)
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "COMPROMISED" in cp.stdout
    with open(out) as f:
        header = f.readline().strip().split(",")
    assert "compromised" in header


def test_px4_offline_without_vars_refuses(tmp_path):
    import torch

    bad = make_checkpoint(tmp_path / "no_vars.pth", VARS9)
    d = torch.load(bad, map_location="cpu")
    del d["meta"]["vars"]
    torch.save(d, bad)
    cp = run_px4(str(imu_csv(tmp_path / "log.csv")), "--mode", "offline", "--model", str(bad))
    assert cp.returncode != 0
    assert "'vars'" in cp.stdout + cp.stderr


def test_ros_package_vendored_sources_fresh():
    """The ROS package entry points must run the audited scripts/ copies.

    setup.py vendors scripts/{ros2_inference_node,cnn_classifier,
    px4_log_replay}.py into uav_aegis/vendor/ at build time; if a vendored
    source disappears from scripts/, the package build breaks. The stale
    pre-audit package modules were removed, so entry points must reference
    the vendor package.
    """
    setup_py = REPO / "ros2_ws/src/uav_aegis/setup.py"
    text = setup_py.read_text()
    assert 'VENDOR_FILES = ("ros2_inference_node.py", "cnn_classifier.py", "px4_log_replay.py")' in text
    for name in ("ros2_inference_node.py", "cnn_classifier.py", "px4_log_replay.py"):
        assert (REPO / "scripts" / name).is_file()
    for module in ("ros2_inference_node", "px4_log_replay"):
        assert f"uav_aegis.vendor.{module}:main" in text, f"entry point for {module} must target uav_aegis.vendor"
    for gone in ("uav_aegis/fault_inference_node.py", "uav_aegis/px4_log_replay.py"):
        assert not (REPO / "ros2_ws/src/uav_aegis" / gone).exists(), f"stale pre-audit module must stay deleted: {gone}"


def test_dashboard_reports_only_real_artifacts():
    """The dashboard must not regress into fabricated metrics.

    The pre-rewrite dashboard displayed invented numbers (Engine Reliability
    99.8, Mission Capability 92%, RUL 48 Hours, Health Score 94/100, Cloud
    Sync Certificate 0xAEG-8822) that had no basis in any artifact. The
    rewritten inspector computes everything from ml_sealed.h5, models/*.pth
    and the results/ JSONs. This test pins the absence of fabrication
    markers and the presence of the real artifact wiring.
    """
    path = REPO / "scripts/dashboard.py"
    src = path.read_text()
    tree = ast.parse(src)  # also proves the file compiles
    strings = " | ".join(
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ).lower()
    for marker in (
        "engine reliability",
        "sensor fidelity",
        "model confidence",
        "safety buffer",
        "mission capability",
        "health score",
        "cloud sync",
        "0xaeg",
        "certification",
        "ignite",
        "find_latest_model",
        "simulate",
        "fleet",
    ):
        assert marker not in strings, f"fabrication marker '{marker}' reappeared in the dashboard"
    assert not re.search(r"\brul\b", strings), (
        "RUL prognostics marker reappeared (the models only classify known "
        "fault classes; they do not estimate remaining useful life)"
    )
    for real in (
        "eval_sealed",
        "DATASET_FS_HZ = 500.0",
        "per_run_accuracy.json",
        "predictions.npz",
        "ML_DATASET_PATH",
        "cross_speed_loso.json",
        "uncertainty_sealed.json",
    ):
        assert real in src, f"dashboard lost its real-artifact wiring: {real}"
