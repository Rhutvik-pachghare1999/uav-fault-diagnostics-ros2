#!/usr/bin/env python3
"""Reproducible benchmark for UAV-Aegis models.

Usage
-----
  # Full benchmark with real model and data (saves logs + plots to results/)
  python scripts/benchmarks/run_benchmark.py

  # CI smoke test – tiny synthetic dataset, no model weights required
  python scripts/benchmarks/run_benchmark.py --smoke

Outputs (all checked into results/)
--------------------------------------
  results/benchmark_metrics.csv   - per-class accuracy, F1, latency (ms)
  results/latency_histogram.png   - inference-latency distribution
  results/benchmark.log           - full console log
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add scripts to path
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from cnn_classifier import PaperCNN
from config import PROJECT_ROOT, MODELS_DIR, ML_DATASET_PATH

# Use environment variable or PROJECT_ROOT for results directory
RESULTS_DIR = Path(os.getenv("BENCHMARK_RESULTS_DIR", PROJECT_ROOT / "results"))
RESULTS_DIR.mkdir(exist_ok=True)

# ── Logging setup ─────────────────────────────────────────────────────────────
log_path = RESULTS_DIR / "benchmark.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SEED = 42


def synthetic_batch(n_samples: int, seq_len: int = 100, n_channels: int = 10):
    """Generate a reproducible synthetic vibration batch."""
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n_samples, n_channels, seq_len)).astype(np.float32)
    y = rng.integers(0, 16, size=n_samples)
    return X, y


def load_real_model():
    """Load the trained CNN model."""
    model_path = MODELS_DIR / "cnn_multi.pth"
    if not model_path.exists():
        log.warning(f"Model not found at {model_path}, using mock inference")
        return None, None
    
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
    
    return model, (mean, std)


def load_real_data(max_samples=512):
    """Load real test data from HDF5 dataset."""
    # Try to find the dataset
    candidates = [
        PROJECT_ROOT / "ml_dataset_v2_aug.h5",
        PROJECT_ROOT / "ml_dataset_v2.h5",
        ML_DATASET_PATH,
    ]
    dataset_path = next((p for p in candidates if p.exists()), None)
    
    if dataset_path is None:
        log.warning("No dataset found, using synthetic data")
        return None, None
    
    import h5py
    with h5py.File(dataset_path, "r") as f:
        X = f["X"][:]
        y = f["y_fault"][:]
        
        # Get class names from metadata
        import json, ast
        meta_raw = f.attrs.get("meta", "{}")
        if isinstance(meta_raw, (bytes, bytearray)):
            meta_raw = meta_raw.decode("utf-8", errors="ignore")
        try:
            meta = json.loads(meta_raw)
        except Exception:
            try:
                meta = ast.literal_eval(meta_raw)
            except Exception:
                meta = {}
        fault_map = meta.get("fault_label_map", {})
        rev_map = {v: k for k, v in fault_map.items()}
        classes = [rev_map.get(i, f"class_{i}") for i in range(len(fault_map))]
    
    # Subsample
    N = len(X)
    if N > max_samples:
        rng = np.random.default_rng(SEED)
        idxs = rng.choice(N, max_samples, replace=False)
        X = X[idxs]
        y = y[idxs]
    
    return (X, y, classes), dataset_path


def mock_infer(X: np.ndarray, n_classes=4) -> np.ndarray:
    """Stand-in inference that returns random logits reproducibly."""
    rng = np.random.default_rng(SEED + 1)
    logits = rng.standard_normal((len(X), n_classes)).astype(np.float32)
    return np.argmax(logits, axis=1)


def real_infer(model, X_batch, mean=None, std=None):
    """Run inference with real model."""
    X = X_batch.astype("float32")
    if mean is not None and std is not None:
        try:
            X = (X - mean) / (std + 1e-9)
        except Exception:
            pass
    inp = torch.from_numpy(X)
    if inp.dim() == 5 and inp.size(2) == 1:
        inp = inp.squeeze(2)
    if inp.dim() == 3:
        inp = inp.unsqueeze(1)
    with torch.no_grad():
        out = model(inp)
        pred = out.argmax(dim=1).numpy()
    return pred


def run(smoke: bool = False):
    n_samples = 32 if smoke else 512
    log.info("=" * 60)
    log.info("UAV-Aegis Benchmark  |  %s", datetime.now().isoformat(timespec="seconds"))
    log.info("Mode: %s  |  Samples: %d", "smoke" if smoke else "full", n_samples)
    log.info("=" * 60)

    # Try to load real model and data
    model, norm_stats = load_real_model()
    data_result = load_real_data(max_samples=n_samples)
    
    if data_result is not None:
        (X, y_true, classes), dataset_path = data_result
        log.info(f"Loaded real data from {dataset_path}")
        log.info(f"Classes: {classes}")
    else:
        X, y_true = synthetic_batch(n_samples)
        classes = ["Healthy", "Cracked", "Imbalanced", "Eroded"]
        log.info("Using synthetic data")

    # ── Latency measurement ───────────────────────────────────────────────────
    latencies_ms = []
    preds = []
    batch_size = 8
    
    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size]
        t0 = time.perf_counter()
        
        if model is not None:
            p = real_infer(model, batch, *norm_stats)
        else:
            p = mock_infer(batch, len(classes))
        
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1e3 / len(batch))
        preds.extend(p.tolist())

    y_pred = np.array(preds[:len(X)])
    latencies_ms = np.array(latencies_ms)

    # ── Per-class metrics ─────────────────────────────────────────────────────
    rows = []
    for cls_idx, cls_name in enumerate(classes):
        mask = y_true == cls_idx
        if mask.sum() == 0:
            continue
        acc = (y_pred[mask] == y_true[mask]).mean()
        tp = ((y_pred == cls_idx) & (y_true == cls_idx)).sum()
        fp = ((y_pred == cls_idx) & (y_true != cls_idx)).sum()
        fn = ((y_pred != cls_idx) & (y_true == cls_idx)).sum()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        rows.append(
            {
                "class": cls_name,
                "accuracy": round(float(acc), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "mean_latency_ms": round(float(latencies_ms.mean()), 3),
                "p99_latency_ms": round(float(np.percentile(latencies_ms, 99)), 3),
            }
        )
        log.info(
            "  %-12s  acc=%.4f  f1=%.4f  lat_mean=%.2f ms",
            cls_name,
            acc,
            f1,
            latencies_ms.mean(),
        )

    # ── Save CSV metrics ──────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "benchmark_metrics.csv"
    fieldnames = ["class", "accuracy", "precision", "recall", "f1",
                  "mean_latency_ms", "p99_latency_ms"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Metrics saved → %s", csv_path)

    # ── Save latency histogram ────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(latencies_ms, bins=20, color="#2196F3", edgecolor="white")
        ax.set_xlabel("Latency per sample (ms)")
        ax.set_ylabel("Count")
        ax.set_title("UAV-Aegis – Inference Latency Distribution")
        plt.tight_layout()
        plot_path = RESULTS_DIR / "latency_histogram.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        log.info("Latency histogram saved → %s", plot_path)
    except ImportError:
        log.warning("matplotlib not available – skipping histogram plot")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(X),
        "mean_latency_ms": round(float(latencies_ms.mean()), 3),
        "p99_latency_ms": round(float(np.percentile(latencies_ms, 99)), 3),
        "overall_accuracy": round(float((y_pred == y_true).mean()), 4),
        "per_class": rows,
    }
    with open(RESULTS_DIR / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("-" * 60)
    log.info("Overall accuracy : %.4f", summary["overall_accuracy"])
    log.info("Mean latency     : %.2f ms", summary["mean_latency_ms"])
    log.info("P99  latency     : %.2f ms", summary["p99_latency_ms"])
    log.info("Benchmark complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAV-Aegis benchmark")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke-test (32 samples, no model weights)")
    args = parser.parse_args()
    run(smoke=args.smoke)