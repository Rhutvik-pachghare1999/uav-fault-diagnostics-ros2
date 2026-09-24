"""Integrity tests for uncertainty/OOD scoring.

Pins the two audit fixes:
  * the sealed-split scope guards (train split / unsealed h5 refused)
  * the AUROC orientation convention: OOD = label 1, measures oriented so
    HIGHER = more OOD-like (the historical bug scored energy backwards)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import VARS13, run_script  # noqa: E402
from uncertainty_quantification import ood_aurocs  # noqa: E402


def test_ood_aurocs_orientation_separable():
    rng = np.random.default_rng(0)
    id_energy = rng.normal(-20.0, 1.0, 200)
    ood_energy = rng.normal(0.0, 1.0, 200)
    a = ood_aurocs({"energy": (id_energy, ood_energy)})
    assert a["energy"] > 0.99


def test_ood_aurocs_flags_wrong_orientation():
    """If someone feeds raw max-prob (higher = ID) without negating,
    AUROC crashes far below 0.5 — the signature of the historical bug."""
    rng = np.random.default_rng(1)
    id_maxp = rng.normal(0.99, 0.005, 200)   # confident on ID
    ood_maxp = rng.normal(0.40, 0.05, 200)   # unconfident on OOD
    a = ood_aurocs({"raw_max_prob_NOT_negated": (id_maxp, ood_maxp)})
    assert a["raw_max_prob_NOT_negated"] < 0.1


def test_ood_aurocs_chance_scores():
    rng = np.random.default_rng(2)
    s = rng.normal(0, 1, 500)
    a = ood_aurocs({"noise": (s[:250], s[250:])})
    assert abs(a["noise"] - 0.5) < 0.1


def test_uncertainty_refuses_train_split(sealed_h5, sealed_ckpt, tmp_path):
    cp = run_script("uncertainty_quantification.py",
                    "--model", str(sealed_ckpt), "--dataset", str(sealed_h5),
                    "--output", str(tmp_path / "u.json"),
                    "--split", "train", "--n-samples", "16", "--n-ood", "8",
                    expect_fail=True)
    assert "train" in (cp.stdout + cp.stderr).lower()


def test_uncertainty_refuses_unsealed_h5(source_h5, sealed_ckpt, tmp_path):
    cp = run_script("uncertainty_quantification.py",
                    "--model", str(sealed_ckpt), "--dataset", str(source_h5),
                    "--output", str(tmp_path / "u.json"),
                    "--n-samples", "16", "--n-ood", "8",
                    expect_fail=True)
    assert "sealed" in (cp.stdout + cp.stderr).lower()


def test_uncertainty_success_on_sealed_test(sealed_h5, sealed_ckpt,
                                            manifest_sha, tmp_path):
    out = tmp_path / "u.json"
    cp = run_script("uncertainty_quantification.py",
                    "--model", str(sealed_ckpt), "--dataset", str(sealed_h5),
                    "--output", str(out),
                    "--n-samples", "64", "--n-ood", "32",
                    "--mc-samples", "3", "--batch-size", "32")
    res = json.loads(out.read_text())
    assert res["protocol"] == "sealed-test-originals"
    assert res["split"] == "test"
    assert res["manifest_sha256"] == manifest_sha
    assert res["n_eval_windows"] > 0
    # orientation must be the FIXED one: OOD scores higher => AUROC >= 0.5
    assert res["auroc_ood"]["Energy (higher=OOD)"] >= 0.5
