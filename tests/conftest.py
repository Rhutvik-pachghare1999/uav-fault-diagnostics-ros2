"""Shared fixtures: a synthetic 96-run project for pipeline integrity tests.

Everything here is tiny (4 windows per run, random data) so the tests run in
CI seconds without the real Isaac Sim dataset or trained weights.
"""

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
VARS13 = [
    "rpm1",
    "rpm2",
    "rpm3",
    "rpm4",
    "roll",
    "pitch",
    "yaw",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_x",
    "acc_y",
    "acc_z",
]
VARS9 = ["roll", "pitch", "yaw", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]
FAULT_CLASSES = ["healthy"] + [f"label_{i}" for i in range(1, 16)]
PAYLOADS = [0.0, 0.005, 0.01]


def make_checkpoint(path, vars_list, n_faults=16, manifest_sha=None, base_filters=32, extra_meta=None):
    """Write a structurally valid PaperCNN checkpoint with the given meta."""
    import torch

    sys.path.insert(0, str(SCRIPTS))
    from cnn_classifier import PaperCNN

    model = PaperCNN(in_channels=1, base_filters=base_filters, num_classes=n_faults)
    meta = {
        "n_faults": n_faults,
        "base_filters": base_filters,
        "vars": list(vars_list),
        "mean": [[0.0]] * len(vars_list),
        "std": [[1.0]] * len(vars_list),
        "fault_label_map": {c: i for i, c in enumerate(FAULT_CLASSES)},
    }
    if manifest_sha is not None:
        meta["manifest_sha256"] = manifest_sha
    if extra_meta:
        meta.update(extra_meta)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    return path


def run_script(script, *args, expect_fail=False):
    """Run a pipeline script; return CompletedProcess (checked)."""
    cp = subprocess.run([sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True)
    if expect_fail:
        assert cp.returncode != 0, f"expected failure, got success:\n{cp.stdout}"
    else:
        assert cp.returncode == 0, f"script failed:\n{cp.stdout}\n{cp.stderr}"
    return cp


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory):
    """Tiny 96-run project (16 classes x 3 payloads x 2 seeds), meta.json only."""
    root = tmp_path_factory.mktemp("proj")
    ds = root / "isaac_dataset"
    for ci, cls in enumerate(FAULT_CLASSES):
        base = "healthy" if cls == "healthy" else f"mask{ci:02x}"
        for pi, payload in enumerate(PAYLOADS):
            for seed in (0, 1):
                name = f"run_cf2x_{base}_p{int(payload * 1000):02d}_s{seed}"
                d = ds / name
                d.mkdir(parents=True)
                (d / "meta.json").write_text(
                    json.dumps(
                        {
                            "fault_type": cls,
                            "payload_kg": payload,
                            "seed": seed,
                            "hover_rpm_mean": 3000.0 + 100.0 * ((ci + pi + seed) % 4),
                        }
                    )
                )
    return root


@pytest.fixture(scope="session")
def manifest(synth_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("manifest") / "split_manifest.json"
    run_script("make_split_manifest.py", "--project-root", str(synth_root), "--out", str(out))
    return out


@pytest.fixture(scope="session")
def manifest_data(manifest):
    return json.loads(manifest.read_text())


@pytest.fixture(scope="session")
def manifest_sha(manifest_data):
    return manifest_data["sha256"]


@pytest.fixture(scope="session")
def source_h5(synth_root, tmp_path_factory):
    """Unsealed source dataset: 96 runs x 4 random windows, run_id = glob order."""
    out = tmp_path_factory.mktemp("h5") / "ml_tiny.h5"
    run_dirs = sorted(glob.glob(os.path.join(str(synth_root), "isaac_dataset", "run_*")))
    assert len(run_dirs) == 96
    rng = np.random.default_rng(0)
    label_of = {c: i for i, c in enumerate(FAULT_CLASSES)}
    Xs, ys, rid = [], [], []
    for i, d in enumerate(run_dirs):
        meta = json.loads((Path(d) / "meta.json").read_text())
        for _ in range(4):
            Xs.append(rng.normal(size=(1, len(VARS13), 100)).astype("float32"))
            ys.append(label_of[meta["fault_type"]])
            rid.append(i)
    with h5py.File(out, "w") as f:
        f.create_dataset("X", data=np.stack(Xs), compression="gzip")
        f.create_dataset("y_fault", data=np.array(ys, dtype="int64"))
        f.create_dataset("y_sev", data=np.zeros(len(ys), dtype="int64"))
        f.create_dataset("ur", data=np.zeros(len(ys), dtype="float32"))
        f.create_dataset("run_id", data=np.array(rid, dtype="int64"))
        f.attrs["meta"] = json.dumps(
            {
                "vars": VARS13,
                "window": 100,
                "step": 25,
                "fault_label_map": {c: i for i, c in enumerate(FAULT_CLASSES)},
            }
        )
    return out


@pytest.fixture(scope="session")
def sealed_h5(synth_root, source_h5, manifest, tmp_path_factory):
    out = tmp_path_factory.mktemp("sealed") / "ml_sealed_tiny.h5"
    run_script(
        "build_sealed_dataset.py",
        "--h5",
        str(source_h5),
        "--manifest",
        str(manifest),
        "--out",
        str(out),
        "--project-root",
        str(synth_root),
        "--aug-copies",
        "2",
        "--aug-seed",
        "42",
    )
    return out


@pytest.fixture(scope="session")
def sealed_ckpt(sealed_h5, manifest, tmp_path_factory):
    """Real 1-epoch sealed training on the tiny dataset (train smoke test)."""
    out = tmp_path_factory.mktemp("ckpt") / "cnn_tiny.pth"
    run_script(
        "train_cnn.py",
        "--h5",
        str(sealed_h5),
        "--manifest",
        str(manifest),
        "--out",
        str(out),
        "--epochs",
        "1",
        "--history",
        str(out.with_suffix(".history.json")),
    )
    return out


@pytest.fixture(scope="session")
def rpm_ckpt(tmp_path_factory):
    """Structurally-valid random-weight checkpoint consuming 13 ch (incl. RPM)."""
    return make_checkpoint(tmp_path_factory.mktemp("rpm") / "cnn_rpm.pth", VARS13)


@pytest.fixture(scope="session")
def imu9_ckpt(tmp_path_factory):
    """Structurally-valid random-weight checkpoint consuming IMU-only 9 ch."""
    return make_checkpoint(tmp_path_factory.mktemp("imu9") / "cnn_imu9.pth", VARS9)
