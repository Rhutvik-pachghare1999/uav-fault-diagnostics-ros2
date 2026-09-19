# 🛡️ UAV Aegis — UAV Propeller Fault Diagnostics (ROS2 + Deep Learning)

<div align="center">

[![CI](https://github.com/Rhutvik-pachghare1999/uav-fault-diagnostics-ros2/actions/workflows/ci.yml/badge.svg)](https://github.com/Rhutvik-pachghare1999/uav-fault-diagnostics-ros2/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![CNN Accuracy](https://img.shields.io/badge/CNN_Accuracy-reported_98.4%25-yellow.svg)](#-model-performance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**UAV Aegis** is an industrial-grade diagnostic and prognostic suite designed to detect, classify, and predict mechanical failures in UAV propellers and motors using high-frequency vibration signatures and Deep Learning.

</div>

---

## 1. 🚨 The Problem: Hidden In-Flight Failures

Mechanical faults like micro-cracks, blade chips, or motor imbalances are often invisible to standard telemetry (RPM/Current) until it's too late.

| # | Critical Issue | Impact |
|---|----------------|--------|
| 1 | **Sensor Degradation** | IMU noise increases, affecting flight stability |
| 2 | **Structural Fatigue** | Undetected vibrations lead to airframe failure |
| 3 | **Mid-Air Failure** | Catastrophic loss of asset and mission capability |

---

## 2. 🛡️ The Solution: UAV Aegis v3

This suite provides a professional end-to-end pipeline:

| # | Solution Layer | Technical Implementation | Benefit |
|---|----------------|--------------------------|---------|
| 1 | **Deep Intelligence** | 1D/2D CNN architectures | reported 98.4% fault classification accuracy |
| 2 | **Spectral Analysis** | FFT-based feature extraction | Pinpoint resonance anomalies |
| 3 | **Predictive Maintenance** | Stacked LSTM for RUL estimation | RMSE of 4.2 cycles for mission planning |
| 4 | **Real-time Dashboard** | Hardened Streamlit Command Center | Fleet health monitoring at a glance |
| 5 | **ROS2 Integration** | Live inference node for edge deployment | Onboard real-time diagnostics |

---

## 3. 🔧 What I Built

This is a solo project spanning the full pipeline — data engineering, the ML
models, ROS2 integration, and the tooling/CI around it.

| # | Domain | Details | Specifications |
|---|--------|---------|----------------|
| 1 | **CNN Fault Classifier** | 1D/2D CNN architectures for mechanical fault signature identification | reported 98.4% accuracy, 0.97 F1 (see reproducibility note) |
| 2 | **RUL Prediction Engine** | Stacked LSTM for Remaining Useful Life estimation | RMSE ~4.2 cycles, temporal sequence modeling |
| 3 | **Spectral Engineering** | FFT-based feature extraction pipeline for resonance anomaly detection | Dominant frequency tracking, spectral entropy |
| 4 | **Data Pipeline** | Sliding-window segmentation (`build_ml_dataset_v2.py`) + augmentation (noise/shift/warp) | Multi-channel sensor sync, 5x expansion |
| 5 | **ROS2 Integration** | `ros2_inference_node.py` for live diagnostics + topic management | ROS2 Humble, real-time messaging |
| 6 | **Command Center Dashboard** | Streamlit monitoring dashboard for fleet visualization | Real-time telemetry, fault logging |
| 7 | **Synthetic Data Generation** | NVIDIA Isaac Sim for physics-accurate fault data in simulated flight | Fault-pattern injection |
| 8 | **Validation & CI** | Smoke tests (config/module import, benchmark exec, batch-shape) + CI | pytest, CI-enforced (no failure masking) |

---

## 4. 🏗️ System Architecture & Data Flow

```mermaid
graph TD
    A[UAV Propeller/Motor] -->|High-Freq Vibration| B[Raw Sensor Data: Acc/Gyro]
    B --> C[Pipeline Initialization]
    subgraph "Data Engineering"
        C --> D[build_ml_dataset.py]
        D -->|Sliding Window Segmentation| E[ML-Ready Windows]
        E --> F[augment_dataset.py]
        F -->|Noise + Shift + Warp| G[Augmented Dataset]
    end
    subgraph "Intelligence Layer"
        G --> H[cnn_classifier.py - 1D CNN]
        G --> I[propeller_lstm.py - Stacked LSTM]
        H -->|Fault Classification| J[Healthy / Cracked / Imbalanced / Eroded]
        I -->|RUL Prediction| K[Remaining Useful Life - Cycles]
    end
    subgraph "Deployment"
        H --> L[ros2_inference_node.py]
        L -->|ROS2 Topic| M[dashboard.py - Streamlit]
    end
```

---

## 5. 📊 Model Performance

*Reported per-class results (see the reproducibility note below — not yet independently reproducible from a clean clone).*

| # | Class | Description | CNN Accuracy |
|---|-------|-------------|--------------|
| 1 | ✅ **Healthy** | Nominal operation | 99.2% |
| 2 | ⚠️ **Cracked** | Structural integrity compromise | 97.5% |
| 3 | ⚠️ **Imbalanced** | Dynamic weight distribution fault | 98.1% |
| 4 | ❌ **Eroded** | Leading edge degradation | 98.8% |

### 5.1 Global Metrics

| # | Metric | CNN v3 (Classification) | LSTM (RUL Prediction) |
|---|--------|------------------------|-----------------------|
| 1 | **Accuracy** | **98.4%** | — |
| 2 | **F1-Score** | **0.97** | — |
| 3 | **RMSE** | — | **4.2 Cycles** |

> **Reproducibility note.** The accuracy/F1/RMSE figures above are the values
> reported from my training runs. The trained model weights and the
> corresponding evaluation artifacts (`results/`) are **not committed** to this
> public repo, so these headline numbers are **not independently reproducible
> from a fresh clone yet**. An episode-level, sealed held-out split and the
> published evaluation artifacts are pending. Treat the numbers as *reported*,
> not *independently verified here*.

---

## 6. 📈 Reproducible Benchmarks

The CI smoke benchmark (`--smoke`) runs on every push to validate the pipeline
end-to-end (I/O, metric computation, latency measurement) using a synthetic
batch — it exercises the plumbing, not the trained model. Full evaluation
artifacts from real training runs are not committed to this public repo.

### 6.1 Run Benchmarks Locally

```bash
# Full benchmark — saves CSV metrics, JSON summary, latency histogram
python benchmarks/run_benchmark.py

# Fast smoke test (32 samples, no model weights required)
python benchmarks/run_benchmark.py --smoke
```

### 6.2 Benchmark Outputs

Running the benchmark locally generates the following in `results/` (not
committed — regenerate with the commands above):

| File | Description |
|------|-------------|
| `results/benchmark_metrics.csv` | Per-class accuracy, precision, recall, F1, latency (ms) |
| `results/benchmark_summary.json` | Run metadata, overall accuracy, p99 latency |
| `results/latency_histogram.png` | Inference latency distribution plot |
| `results/benchmark.log` | Full console log from last run |

---

## 7. ⚙️ CI / Continuous Integration

This repo uses **GitHub Actions** for automated quality gates on every push and pull request.

```
Push / PR  →  Lint (flake8)  →  Unit tests (pytest)  →  Benchmark smoke test  →  Upload artifacts
```

- **Linter**: `flake8` — fails on syntax errors and undefined names; warns on style violations (max line length 120)
- **Tests**: `pytest tests/ -v` — smoke tests, shape checks, benchmark script execution
- **Benchmark**: `python benchmarks/run_benchmark.py --smoke` — verifies the metrics pipeline end-to-end
- **Artifacts**: benchmark results uploaded for 30-day retention on every passing run

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the full workflow definition.

---

## 8. 🚀 Quick Start

### 8.1 Setup Environment

```bash
git clone https://github.com/Rhutvik-pachghare1999/uav-fault-diagnostics-ros2.git
cd uav-fault-diagnostics-ros2
pip install -r requirements.txt
```

### 8.2 Full Pipeline Execution

| # | Step | Command | Output |
|---|------|---------|--------|
| 1 | **Segment Data** | `python scripts/build_ml_dataset.py` | ML-ready windows |
| 2 | **Augment Data** | `python scripts/augment_dataset.py` | Robust dataset |
| 3 | **Train CNN** | `python scripts/cnn_classifier.py` | `models/cnn_v3.pth` |
| 4 | **Train LSTM** | `python scripts/propeller_lstm.py` | `models/lstm_rul.pth` |
| 5 | **Launch C2** | `streamlit run scripts/dashboard.py` | Web UI |
| 6 | **Deploy ROS2** | `ros2 run uav_aegis ros2_inference_node.py` | Live topics |

---

## 9. 🔧 Key Algorithms

### 9.1 1D-CNN Fault Classifier

```
Input: Raw vibration window [N x 1]
  │
  ▼ Conv1D(64, kernel=5) + BatchNorm + ReLU + MaxPool
  │
  ▼ Conv1D(128, kernel=3) + BatchNorm + ReLU + MaxPool
  │
  ▼ GlobalAvgPool
  │
  ▼ Linear(128 → 64) + Dropout(0.3)
  │
  ▼ Linear(64 → 4)
  │
  ▼ Softmax → [Healthy, Cracked, Imbalanced, Eroded]
```

---

## 10. 🧹 Testing

```bash
pytest tests/ -v
```

CI-enforced (no failure masking). Current suite is fast smoke tests that keep
the pipeline importable and the benchmark runnable on every push:

| # | Test | Description | Status |
|---|------|-------------|--------|
| 1 | `test_config_importable` | `config.py` imports without error | ✅ |
| 2 | `test_severity_utils_importable` | `severity_utils.py` imports without error | ✅ |
| 3 | `test_benchmark_smoke` | `run_benchmark.py --smoke` completes + writes metrics CSV | ✅ |
| 4 | `test_synthetic_batch_shape` | Synthetic batch generator returns expected shapes | ✅ |

> Deeper model-level tests (trained-weight inference, held-out evaluation) are
> pending alongside the committed evaluation artifacts.

---

## 11. 📐 Design Decisions

| # | Decision | Rationale | Benefit |
|---|----------|-----------|---------|
| 1 | **1D-CNN over 2D** | Direct vibration processing | Lower latency for edge deployment |
| 2 | **Stacked LSTM** | Temporal dependency capture | Higher accuracy in degradation prediction |
| 3 | **ROS2 Humble** | Industry standard middleware | Robust real-time message passing |
| 4 | **Synthetic Training** | NVIDIA Isaac Sim data | Rare fault scenarios captured |

---

## 12. 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.

---

## 13. 👤 Author

**Rhutvik Pachghare** | Master's in Robotics & Automation | Arizona State University

- [GitHub](https://github.com/Rhutvik-pachghare1999)
- [LinkedIn](https://www.linkedin.com/in/rhutvik-pachghare/)

---

## 14. 📧 Contact & Support

For questions, issues, or collaboration:
- **GitHub Issues**: [UAV-Aegis/issues](https://github.com/Rhutvik-pachghare1999/uav-fault-diagnostics-ros2/issues)
- **Project Repository**: [github.com/Rhutvik-pachghare1999/uav-fault-diagnostics-ros2](https://github.com/Rhutvik-pachghare1999/uav-fault-diagnostics-ros2)
