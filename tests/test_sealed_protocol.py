"""Integrity tests for the sealed run-grouped split protocol.

Pins the guarantees the audit demanded:
  * the manifest is a valid, disjoint, class-stratified, payload-covering split
  * the sealed dataset has augmented copies of TRAIN runs only
  * training verifies the h5 against the manifest
  * evaluation refuses anything but untouched test-run windows
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import VARS13, make_checkpoint, run_script


# ── Manifest ──────────────────────────────────────────────────────────────────
def test_manifest_structure(manifest_data):
    m = manifest_data
    tr, va, te = set(m["train"]), set(m["val"]), set(m["test"])
    all_runs = set(m["runs"].keys())
    assert len(tr) == 48 and len(va) == 16 and len(te) == 32
    assert len(all_runs) == 96
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == all_runs
    assert m["sha256"] and isinstance(m["seed"], int)


def test_manifest_class_stratification(manifest_data):
    m = manifest_data
    for run, info in m["runs"].items():
        part = "train" if run in m["train"] else "val" if run in m["val"] else "test"
        m["runs"][run]["part"] = part
    by_class = {}
    for run, info in m["runs"].items():
        by_class.setdefault(info["class"], {"train": 0, "val": 0, "test": 0})
        by_class[info["class"]][info["part"]] += 1
    assert len(by_class) == 16
    for cls, counts in by_class.items():
        assert counts == {"train": 3, "val": 1, "test": 2}, cls


def test_manifest_train_payload_coverage(manifest_data):
    m = manifest_data
    for cls in {info["class"] for info in m["runs"].values()}:
        payloads = {m["runs"][r]["payload_kg"] for r in m["train"] if m["runs"][r]["class"] == cls}
        assert len(payloads) == 3, f"class {cls} train runs miss payloads"


def test_manifest_deterministic(synth_root, manifest, tmp_path):
    second = tmp_path / "split_second.json"
    run_script("make_split_manifest.py", "--project-root", str(synth_root), "--out", str(second))
    a = json.loads(manifest.read_text())
    b = json.loads(second.read_text())
    assert a["train"] == b["train"]
    assert a["val"] == b["val"]
    assert a["test"] == b["test"]
    assert a["sha256"] == b["sha256"]


# ── Sealed dataset ────────────────────────────────────────────────────────────
def test_sealed_dataset_integrity(sealed_h5, manifest_data, manifest_sha):
    m = manifest_data
    with h5py.File(sealed_h5, "r") as f:
        meta = json.loads(f.attrs["meta"])
        assert meta["manifest_sha256"] == manifest_sha
        split = f["split"][:]
        is_aug = f["is_aug"][:]
        run_id = f["run_id"][:]
        names = meta["run_names"]

        part_of = {}
        for part in ("train", "val", "test"):
            for r in m[part]:
                part_of[r] = {"train": 0, "val": 1, "test": 2}[part]
        expected = np.array([part_of[names[int(r)]] for r in run_id])

        assert np.array_equal(split, expected), "h5 split column != manifest"
        # THE headline guarantee: augmented windows exist only in train
        assert (split[is_aug == 1] == 0).all(), "aug windows outside train split"
        assert set(np.unique(split[is_aug == 0])) == {0, 1, 2}


def test_sealed_dataset_preserves_originals(source_h5, sealed_h5):
    with h5py.File(source_h5, "r") as f:
        n_src = f["X"].shape[0]
    with h5py.File(sealed_h5, "r") as f:
        n_orig = int((f["is_aug"][:] == 0).sum())
        n_aug = int((f["is_aug"][:] == 1).sum())
    assert n_orig == n_src == 96 * 4
    assert n_aug == 48 * 4 * 2  # aug-copies=2, train runs only


# ── Training smoke + self-describing checkpoint ─────────────────────────────
def test_train_smoke_checkpoint_meta(sealed_ckpt, manifest_sha):
    import torch

    ck = torch.load(sealed_ckpt, map_location="cpu")
    meta = ck["meta"]
    assert meta["protocol"] == "sealed-run-grouped-v2"
    assert meta["manifest_sha256"] == manifest_sha
    assert meta["vars"] == VARS13
    assert meta["base_filters"] == 32
    assert meta["seed"] == 42
    assert "state_dict" in ck and "mean" in meta and "std" in meta


# ── Evaluation guards ─────────────────────────────────────────────────────────
def test_eval_refuses_unsealed_h5(source_h5, sealed_ckpt, tmp_path):
    cp = run_script(
        "eval_classifier.py",
        "--h5",
        str(source_h5),
        "--model",
        str(sealed_ckpt),
        "--out",
        str(tmp_path / "e1"),
        expect_fail=True,
    )
    assert "sealed" in (cp.stdout + cp.stderr).lower()


def test_eval_refuses_train_split(sealed_h5, sealed_ckpt, tmp_path):
    cp = run_script(
        "eval_classifier.py",
        "--h5",
        str(sealed_h5),
        "--model",
        str(sealed_ckpt),
        "--out",
        str(tmp_path / "e2"),
        "--split",
        "train",
        expect_fail=True,
    )
    assert "train" in (cp.stdout + cp.stderr).lower()


def test_eval_refuses_manifest_sha_mismatch(sealed_h5, manifest_sha, tmp_path):
    bad = make_checkpoint(tmp_path / "bad_sha.pth", VARS13, manifest_sha="deadbeef")
    cp = run_script(
        "eval_classifier.py",
        "--h5",
        str(sealed_h5),
        "--model",
        str(bad),
        "--out",
        str(tmp_path / "e3"),
        expect_fail=True,
    )
    assert "manifest" in (cp.stdout + cp.stderr).lower()


def test_eval_test_split_success(sealed_h5, sealed_ckpt, tmp_path):
    out = tmp_path / "eval_out"
    run_script("eval_classifier.py", "--h5", str(sealed_h5), "--model", str(sealed_ckpt), "--out", str(out))
    meta = json.loads((out / "eval_meta.json").read_text())
    assert "window_accuracy" in meta
    # tiny random data, 1 epoch: only assert the plumbing, not the accuracy
    assert (out / "confusion_matrix.png").exists()
    assert (out / "per_run_accuracy.json").exists()
