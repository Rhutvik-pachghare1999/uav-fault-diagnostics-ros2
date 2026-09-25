"""Integrity tests for build_ml_dataset_v2 (the raw Isaac -> h5 builder).

Pins the audit fix: missing data must FAIL LOUDLY, never silently
zero-fill / skip and quietly shrink the dataset.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import build_ml_dataset_v2 as builder  # noqa: E402

FULL_VARS = [
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


def make_run(root, name, columns, n_rows=120, with_imu=True):
    d = Path(root) / name
    d.mkdir(parents=True)
    if with_imu:
        df = pd.DataFrame({c: np.random.randn(n_rows) for c in columns})
        df.to_csv(d / "imu.csv", index=False)
    (d / "meta.json").write_text('{"fault_type": "healthy", "payload_kg": 0.0}')
    return d


def test_read_run_missing_vars_raises(tmp_path):
    make_run(tmp_path, "run_a", ["rpm1", "rpm2", "roll"])
    with pytest.raises(ValueError) as ei:
        builder.read_run(str(tmp_path / "run_a"), vars=FULL_VARS)
    assert "missing from imu data" in str(ei.value)
    assert "rpm3" in str(ei.value)


def test_read_run_missing_vars_escape_hatch_zero_fills(tmp_path, capsys):
    make_run(tmp_path, "run_b", ["rpm1", "roll", "acc_x"])
    Xs, metas = builder.read_run(str(tmp_path / "run_b"), vars=FULL_VARS, allow_missing_vars=True)
    assert len(Xs) > 0
    # rpm2 (index 1) was missing -> must be exactly zeros
    assert (Xs[0][1] == 0).all()
    out = capsys.readouterr().out
    assert "WARNING" in out and "zero-filling" in out


def test_read_run_missing_imu_raises(tmp_path):
    make_run(tmp_path, "run_c", ["rpm1"], with_imu=False)
    with pytest.raises(FileNotFoundError) as ei:
        builder.read_run(str(tmp_path / "run_c"), vars=FULL_VARS)
    assert "allow-missing-runs" in str(ei.value)


def test_read_run_missing_imu_escape_hatch_skips(tmp_path):
    make_run(tmp_path, "run_d", ["rpm1"], with_imu=False)
    Xs, metas = builder.read_run(str(tmp_path / "run_d"), vars=FULL_VARS, allow_missing_runs=True)
    assert Xs == [] and metas == []


def test_build_dataset_empty_project_aborts(tmp_path):
    (tmp_path / "isaac_dataset").mkdir()
    out = tmp_path / "out.h5"
    with pytest.raises(SystemExit) as ei:
        builder.build_dataset(str(tmp_path), str(out), vars=FULL_VARS)
    assert "run" in str(ei.value).lower()


def test_build_dataset_no_windows_aborts_not_noop(tmp_path):
    """Run dirs exist but contain no IMU data: must abort loudly, never
    'succeed' by writing an empty dataset."""
    root = tmp_path / "isaac_dataset"
    root.mkdir()
    d = root / "run_empty"
    d.mkdir()
    (d / "meta.json").write_text('{"fault_type": "healthy"}')
    out = tmp_path / "out.h5"
    with pytest.raises(SystemExit) as ei:
        builder.build_dataset(str(tmp_path), str(out), vars=FULL_VARS, allow_missing_runs=True)
    assert "no windows" in str(ei.value).lower()
    assert not out.exists()


def test_build_dataset_happy_path(tmp_path):
    root = tmp_path / "isaac_dataset"
    root.mkdir()
    make_run(root, "run_x", FULL_VARS)
    make_run(root, "run_y", FULL_VARS)
    out = tmp_path / "out.h5"
    # project_root is the repo root (expects <root>/isaac_dataset/run_*)
    builder.build_dataset(str(tmp_path), str(out), vars=FULL_VARS)
    import h5py

    with h5py.File(out, "r") as f:
        assert f["X"].shape[0] > 0
        assert set(np.unique(f["run_id"][:])) == {0, 1}
