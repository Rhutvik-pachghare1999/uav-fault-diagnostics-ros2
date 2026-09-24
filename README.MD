# 🛡️ UAV Aegis — UAV Propeller Fault Diagnostics (Isaac Sim Physics + Deep Learning + ROS2)

<div align="center">

[![CI](https://github.com/Rhutvik-pachghare1999/uav-fault-diagnostics-ros2/actions/workflows/ci.yml/badge.svg)](https://github.com/Rhutvik-pachghare1999/uav-fault-diagnostics-ros2/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Isaac Sim](https://img.shields.io/badge/Isaac_Sim-5.1.0-7F00FF.svg)](https://developer.nvidia.com/isaac/sim)
[![ROS2](https://img.shields.io/badge/ROS2-Jazzy_(prototype_node)-22314E?logo=ros&logoColor=white)](https://docs.ros.org/en/jazzy/)
[![Evaluation](https://img.shields.io/badge/evaluation-run--grouped,_leakage--audited-brightgreen.svg)](#-evaluation-methodology-what-makes-these-numbers-credible)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

**UAV Aegis** is an end-to-end propeller-fault diagnostics pipeline for a quadrotor
(Bitcraze Crazyflie 2.1 X-frame): **physics-based fault injection in NVIDIA Isaac Sim →
windowed IMU+RPM dataset → 1D-CNN classifier → leakage-audited evaluation → ROS2
deployment node.**

Every number below comes from **real physics-simulation flights** (96 logged missions,
500 Hz IMU), not hand-crafted synthetic signals. Faults are injected as *physical
parameter changes* — mass imbalance, aerodynamic asymmetry, efficiency and motor-gain
loss — and the resulting vibration signatures are validated against rotor dynamics
before any model is trained.

---

## 📊 Headline Results (real-simulation data, run-grouped splits)

| Metric | Value | Protocol |
|---|---|---|
| **Test accuracy** | **99.0%** (macro F1 0.99) | 16 classes, held-out *flight runs* (no flight crosses train/test) |
| **Healthy-class F1** | **1.00** (precision & recall) | The safety-critical "no fault" class — zero missed detections |
| **Worst fault class F1** | 0.95 (all-4-rotors-faulted) | All 15 fault classes ≥ 0.95 |
| **Cross-RPM generalization** | 98.8–99.5% | Leave-out hover-RPM bins (2757–3919 RPM) |
| **Predictive entropy** | 0.0010 nats (mean max-prob 0.9997) | MC-dropout uncertainty quantification |
| **Inference latency** | 0.20 ms mean / 1.53 ms p99 | Single window, GPU, production-shaped input |

Full reports: `results/eval_final/classification_report.txt`, `results/cross_speed_eval.json`,
`results/uncertainty_eval.json`, `results/benchmark_summary.json`. All figures: `results/figures/`.

---

## 🚨 The Problem

Mechanical faults — blade chips, motor imbalance, bearing wear — are usually invisible
to standard telemetry until they cascade into in-flight failure. Rotor imbalance creates
a **1P vibration line** (once-per-revolution) exactly at the faulted rotor's speed:
detectable in gyroscope/accelerometer spectra, but buried in flight data if you don't
know where to look. UAV Aegis learns to read that signature directly from raw
0.2-second IMU+RPM windows.

---

## 🛠️ What I Built

```
Isaac Sim (Crazyflie 2.1 USD asset, PhysX 500 Hz)
   │  96 flights · 9 s missions · 3 payloads (0/5/10 g) × 2 seeds × 16 classes
   │  physical fault injection: COM offset + aero asymmetry + efficiency/motor loss
   ▼
isaac_dataset/run_cf2x_*/  (imu.csv @500 Hz, state.csv @100 Hz, meta.json)
   │  build_ml_dataset_v2.py  → 13-ch windows (100 samples = 0.2 s, step 4)
   ▼
ml_dataset_v2.h5  (105,696 windows)  ──augment×3──▶  ml_dataset_v2_aug.h5  (317,088)
   │  GroupShuffleSplit by run_id → train/val/test contain DISJOINT flights
   ▼
PaperCNN (1D-CNN) → models/cnn_multi.pth   (+ optional SimCLR-style self-supervised encoder)
   │
   ├─ eval_classifier.py      → 99% acc, per-class report, predictions.npz
   ├─ eval_cross_speed.py     → leave-out hover-RPM bins
   ├─ uncertainty_quantification.py → MC-dropout entropy / mutual information
   ├─ benchmarks/run_benchmark.py   → latency + accuracy benchmark
   ├─ make_report_plots.py    → 8 publication figures
   ▼
ros2_inference_node.py  (ROS2 Jazzy node; live /imu/data or CSV replay; publishes fault_id+severity)
```

### Fault taxonomy (16 classes)

15 non-empty subsets of the 4 rotors (bitmask `0x01`…`0x0f`) + healthy.
Per faulted rotor: **0.6 mm COM offset**, **15% 1P aerodynamic asymmetry**,
**η = 0.85 thrust efficiency**, **0.75 motor gain**. Severity: single-rotor = 2
(Moderate), multi-rotor = 3 (Severe); utilization ratio `ur = 0.3·n_faulted` (cap 0.9).

---

## 🔬 Why the Data is Real Physics (not synthetic curves)

An earlier version of this project used hand-generated signals and paid for it with a
leakage audit that invalidated its own headline number (98.4% → see
[methodology](#-evaluation-methodology-what-makes-these-numbers-credible)). The dataset
was rebuilt from scratch in Isaac Sim, and the physics was **verified before training**:

- **1P line at the right frequency.** Rotor-1 fault (`mask01`): measured IMU spectral
  peak at **50.2 Hz** vs the faulted rotor's actual speed **3109 RPM = 51.8 Hz**. The
  healthy aircraft shows an empty 20–80 Hz band (peak 20.5 Hz, amplitude 0.2); the
  faulted one shows a 62.5-amplitude line — **~100× more gyro energy** (and **63× more
  accelerometer energy**) in that band. *Figure: `results/figures/fig3_vibration_signatures.png`*
- **Closed-loop RPM redistribution.** The flight controller compensates the loss:
  faulted rotor spins at 3109 RPM while its counter-rotor partner drops to 2881 RPM —
  the classic quadrotor rebalancing signature, emergent from the physics, not scripted.
- **Hover operating points track payload.** 2908 / 3165 / 3412 RPM at 0 / 5 / 10 g —
  consistent with $T = k\omega^2$ for the true vehicle mass, and a realistic ~1.5×
  throttle headroom.
- **No-crash audit.** All 96 missions fly: min altitude ≥ 0.034 m (0.55 m reference),
  mean altitude 0.37–0.60 m. This required an altitude **I-term** in the controller —
  without it, the 4-fault aircraft (η=0.85) is thrust-starved and can never leave the
  ground (a real diagnosis of a real simulator limitation, fixed in
  `isaac/cf2x_fault_sim.py`).

*Figures: `fig2_flight_mission.png` (mission + closed-loop behaviour), `fig1_dataset_overview.png` (96-run composition).*

---

## 🧪 Evaluation Methodology (what makes these numbers credible)

1. **Run-grouped splits.** Splits are made by `run_id` (GroupShuffleSplit): a flight
   contributes windows to train *or* val *or* test — never both. Near-duplicate windows
   cannot leak. During a 51-run pilot this audit caught a split that placed **all three
   `mask09` runs in test** (class absent from training → 0 recall). The fix wasn't a
   code tweak — it was generating 45 more seed-1 flights so every class has 6 runs
   (3 payloads × 2 seeds), which raised test accuracy from 89% → 99%.
2. **Cross-payload / cross-RPM protocol.** `eval_cross_speed.py` retrains and tests
   with whole hover-RPM bins (which map to payload operating points) held out.
3. **Uncertainty, not just accuracy.** MC-dropout entropy + mutual information +
   energy score per class (`results/uncertainty_eval.json`).
4. **Severity is reported separately** from fault identity
   (`results/eval_final/severity_distribution.txt`).
5. **Everything is regenerated from one script** (`reproduce.sh`) — data, augment,
   training, evals, figures. CI additionally smoke-tests the benchmark and unit tests
   on every push.

Known limitation (honest): the evaluation is **simulation-only**. Isaac Sim + PhysX
with a realistic Crazyflie asset is a far stronger basis than hand-written signal
generators, but **sim-to-real transfer is NOT established** by these numbers. The
grouped-split, cross-RPM and uncertainty protocols are the transfer-resistant parts;
the absolute accuracy is an upper bound for the current data source.

---

## 📈 The 8 Report Figures

| Figure | Content |
|---|---|
| `fig1_dataset_overview.png` | 96-run dataset composition, severity & hover-RPM distribution, vibration RMS vs #faulted rotors |
| `fig2_flight_mission.png` | 3D mission, altitude tracking, RPM command vs measured, IMU time series |
| `fig3_vibration_signatures.png` | Welch PSD: healthy vs single vs double fault, 1P lines annotated at rotor speeds |
| `fig4_training_curves.png` | Loss/accuracy curves (early stop epoch 22, best val acc 0.9931) |
| `fig5_confusion_matrix.png` | 16×16 confusion + per-class F1 (macro F1 0.989) |
| `fig6_cross_speed.png` | Accuracy & per-class recall across hover-RPM bins |
| `fig7_benchmark.png` | Per-class accuracy & latency (0.20 ms mean) |
| `fig8_uncertainty.png` | Predictive entropy & confidence by class |

---

## 🚀 Quick Start

### 1. Environment

```bash
# Python deps (PyTorch, h5py, scikit-learn, matplotlib, ...)
pip install -r requirements.txt
# ROS2 Jazzy (only for the live inference node)
# Isaac Sim 5.1.0 (only for data generation) -> set PYTHON_SH below
```

### 2. One-command reproduction (data → figures)

```bash
PYTHON_SH=~/.local/share/ov/pkg/isaac_sim-5.1.0/python.sh ./reproduce.sh          # full
PYTHON_SH=... ./reproduce.sh --quick                                     # quick pass
```

Generates the 96 flights (~13 min on a laptop GPU), builds/augments the HDF5
datasets, trains the CNN, runs all evals and writes the 8 figures.

### 3. Use the trained model

```bash
# Classification report on held-out flights
python3 scripts/eval_classifier.py --h5 ml_dataset_v2_aug.h5 \
    --model models/cnn_multi.pth --out results/eval_final --max-samples 10000

# ROS2 node (live IMU topic or CSV replay of a real flight log)
python3 scripts/ros2_inference_node.py --model models/cnn_multi.pth --mode live
python3 scripts/ros2_inference_node.py --model models/cnn_multi.pth \
    --mode replay --input isaac_dataset/run_cf2x_mask01_p00_s0/imu.csv --step 500
# → label_1  (9/9 windows) — channel layout is read from the model checkpoint meta
```

### 4. Tests & CI

```bash
python3 -m pytest tests/ -q          # 4 smoke tests: config, severity utils, benchmark, batch shapes
# CI (.github/workflows/ci.yml): lint + tests + benchmark --smoke (synthetic fallback,
# no GPU / no dataset required)
```

---

## 🗂️ Repository Layout

```
isaac/cf2x_fault_sim.py       # Isaac Sim mission generator (fault injection, controller, logging)
isaac/force_probe.py           # USD asset verification utility (masses, joints, arm offsets)
isaac/assets/Bitcraze/...     # Crazyflie 2.1 USD asset
isaac_dataset/run_cf2x_*/     # 96 flight logs (CSVs gitignored; meta.json manifests tracked)
scripts/build_ml_dataset_v2.py # flight logs → windowed 13-channel HDF5 (run_id-aware)
scripts/augment_dataset.py     # ×3 augmentation (noise/RPM-scale/drift/shift), run_id preserved
scripts/train_cnn.py           # PaperCNN training, grouped splits, history dump
scripts/eval_classifier.py     # test-set report + predictions.npz
scripts/eval_cross_speed.py    # leave-out RPM-bin generalization
scripts/uncertainty_quantification.py  # MC-dropout entropy/MI/energy
scripts/self_supervised_pretrain.py   # optional SimCLR-style encoder
scripts/benchmarks/run_benchmark.py  # latency + accuracy benchmark (CI-safe --smoke)
scripts/make_report_plots.py   # the 8 report figures (consumes only real artifacts)
scripts/ros2_inference_node.py# ROS2 deployment node (live + replay)
reproduce.sh                  # end-to-end pipeline
results/                       # all evaluation artifacts + figures
```

---

## 📐 Design Decisions (short)

- **13 channels** (`rpm1-4, roll, pitch, yaw, gyro_xyz, acc_xyz`): the accelerometer
  carries the strongest 1P signature (63× energy amplification vs 20× on gyro) —
  dropping it (the 10-channel default) discards signal. Channel order is stored in the
  model checkpoint and enforced end-to-end (builder → training → ROS2 node).
- **0.2 s windows @ 500 Hz**: long enough to resolve the 48–65 Hz 1P lines at
  hovering speeds, short enough for a 0.2 ms reactive onboard loop.
- **Altitude I-term in the simulator controller** so *faulted* vehicles still fly
  realistic missions (PD-only commands healthy-model RPMs and starves multi-fault
  vehicles).
- **Grouped splits everywhere**: the pilot's 89% result was an artifact of a class with
  zero training runs — fixed by more flights, not by metric surgery.

---

## 🧹 Known Limitations

- **Sim-to-real gap**: all 96 flights are Isaac Sim. Real-vehicle transfer requires
  domain randomization + flight data (the evaluation protocols are designed to make
  that step honest).
- **One fault severity profile** (fixed COM/aero/efficiency magnitudes); severity
  labels encode *how many* rotors faulted, not graded degradation levels.
- **16 classes on one vehicle model** (Crazyflie 2.1); per-vehicle calibration is
  out of scope.
- **ROS2 node is a prototype**: live mode assumes a synchronized IMU+RPM interface.

---

## 📜 License

MIT — see repository root.

## 👤 Author

**Rhutvik Pachghare** — [GitHub](https://github.com/Rhutvik-pachghare1999)

Built as a solo end-to-end robotics ML project: simulation, data engineering,
modeling, evaluation methodology, and deployment.
