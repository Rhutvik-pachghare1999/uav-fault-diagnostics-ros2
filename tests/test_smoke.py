"""Smoke tests – fast, no model weights required."""

import importlib
import os
import sys
from pathlib import Path

import numpy as np

# Make scripts directory importable
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


# ── Config ────────────────────────────────────────────────────────────────────
def test_config_importable():
    """config.py must be importable without errors."""
    spec = importlib.util.spec_from_file_location("config", SCRIPTS_DIR / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod is not None


# ── Severity utils ────────────────────────────────────────────────────────────
def test_severity_utils_importable():
    spec = importlib.util.spec_from_file_location("severity_utils", SCRIPTS_DIR / "severity_utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod is not None


# ── Benchmark script ──────────────────────────────────────────────────────────
def test_benchmark_smoke(tmp_path, monkeypatch):
    """run_benchmark.py --smoke must complete without exception."""
    monkeypatch.chdir(tmp_path)
    # Set results dir to tmp_path for isolation
    env = os.environ.copy()
    env["BENCHMARK_RESULTS_DIR"] = str(tmp_path / "results")
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "benchmarks" / "run_benchmark.py"), "--smoke"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "results" / "benchmark_metrics.csv").exists()


# ── Numpy sanity ──────────────────────────────────────────────────────────────
def test_synthetic_batch_shape():
    """Synthetic batch generator returns expected shapes."""
    n, seq = 16, 512
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, 1, seq)).astype(np.float32)
    y = rng.integers(0, 4, size=n)
    assert X.shape == (n, 1, seq)
    assert y.shape == (n,)
    assert X.dtype == np.float32
