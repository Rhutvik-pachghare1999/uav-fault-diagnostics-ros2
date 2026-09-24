# 🛡️ UAV Aegis — UAV Propeller Fault Diagnostics (Isaac Sim Physics + Deep Learning + ROS2)

<div align="center">

[![CI](https://github.com/Rhutvik-pachghare1999/uav-fault-diagnostics-ros2/actions/workflows/ci.yml/badge.svg)](https://github.com/Rhutvik-pachghare1999/uav-fault-diagnostics-ros2/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Isaac Sim](https://img.shields.io/badge/Isaac_Sim-5.1.0-7F00FF.svg)](https://developer.nvidia.com/isaac/sim)
[![ROS2](https://img.shields.io/badge/ROS2-Jazzy-22314E?logo=ros&logoColor=white)](https://docs.ros.org/en/jazzy/)
[![Evaluation](https://img.shields.io/badge/evaluation-sealed_split,_retrained_LOSO,_34_CI_tests-brightgreen.svg)](#-evaluation-methodology-what-makes-these-numbers-credible)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**End-to-end propeller fault diagnostics for a quadrotor** (Bitcraze Crazyflie 2.1 X-frame):
**physics-based fault injection in NVIDIA Isaac Sim → windowed 13-channel IMU dataset →
1D-CNN classifier → sealed-split evaluation with retrained cross-speed LOSO → ROS2 deployment node.**

Every number below comes from **real physics-simulation flights** (96 logged missions, 500 Hz IMU) —
not hand-crafted synthetic signals. Faults are injected as *physical parameter changes* — mass
imbalance, aerodynamic asymmetry, efficiency and motor-gain loss — and the resulting vibration
signatures are validated against rotor dynamics before any model is trained.

</div>

<br/>

<p align="center">
  <img src="results/figures/fig0_system_architecture.png" alt="UAV Aegis end-to-end pipeline: Isaac Sim flights, sealed dataset builder, PaperCNN training, evaluation, ROS2 deployment" width="96%">
</p>
<p align="center"><sub><b>Figure 1 — End-to-end system architecture.</b> Five stages, fully reproducible via <code>reproduce.sh</code>.</sub></p>

---

## 📊 Headline Results (sealed test protocol)

<div align="center">

| Metric | Value | Protocol |
|:---|:---:|:---|
| **Test accuracy (13-ch model)** | **81.7%** (macro F1 0.80) | 35,232 windows from **32 fully untouched test flights** |
| Test accuracy (IMU-only 9-ch model) | 79.9% | same flights, no RPM channels |
| **Healthy class F1** | **1.00** (precision & recall) | zero missed faults *and* zero false alarms on the safety-critical class |
| Worst fault class | `label_12` (mask0c, rotors R3+R4) recall **0.13** | the honest weak spot — see [failure analysis](#-where-it-fails) |
| Per-run spread | 81.7% ± 24.8%, **min 0%** (4/32 runs < 50%) | accuracy varies dramatically across flights |
| **Cross-RPM generalization (LOSO)** | **90.3 / 91.3 / 99.0 / 92.3%** (mean 93.2% ± 3.4) | whole hover-RPM bins held out, model **retrained from scratch per fold** |
| OOD detection | Energy score AUROC **0.996** | confidence-based scores fail (0.40–0.60) — see [uncertainty](#-uncertainty--what-the-model-does-not-know) |
| Inference latency | 0.20 ms mean / 0.37 ms p99 | single window, CPU, production-shaped input |

</div>

Full reports: `results/eval_sealed/` · `results/eval_imu9/` · `results/cross_speed_loso.json` ·
`results/uncertainty_sealed.json` · `results/benchmark_summary.json` — all figures in `results/figures/`.

> **Why not 99%?** An earlier version of this project reported 99.0% test accuracy. A full leakage
> audit found that number was produced by a non-frozen split and mixed evaluation, and invalidated
> it. The dataset was rebuilt, the split frozen into a manifest, every evaluator hardened, and the
> models retrained. **81.7% is the real number** for the sealed protocol below — reported with its
> failure modes, not around them.

---

## 🚨 The Problem

Mechanical faults — blade chips, motor imbalance, bearing wear — are usually invisible to standard
telemetry until they cascade into in-flight failure. Rotor imbalance creates a **1P vibration line**
(once-per-revolution) exactly at the faulted rotor's speed: detectable in gyroscope/accelerometer
spectra, but buried in flight data if you don't know where to look. UAV Aegis learns to read that
signature directly from raw 0.2-second IMU+RPM windows.

---

## 🔬 Fault Injection — Physics, Not Curves

<p align="center">
  <img src="results/figures/fig_fault_injection.png" alt="Quadrotor rotor bitmask schematic, per-rotor fault profile, and measured vibration signatures" width="92%">
</p>
<p align="center"><sub><b>Figure 2 — Fault model.</b> Left: X-quad rotor layout with the 4-bit fault mask (15 non-empty combinations + healthy). Right: the per-rotor physical modifications and the signatures they produce in the flight data.</sub></p>

### Verified physics evidence

- **1P line at the right frequency.** Rotor-1 fault (`mask01`): measured IMU spectral peak at
  **50.2 Hz** vs the faulted rotor's actual speed **3109 RPM = 51.8 Hz**. The healthy aircraft shows
  an empty 20–80 Hz band; the faulted one shows a dominant spectral line — **~100× more gyro
  energy** (and **63× more accelerometer energy**) in that band.
- **Closed-loop RPM redistribution.** The flight controller compensates the loss: faulted rotor
  spins at 3109 RPM while its counter-rotor partner drops to 2881 RPM — the classic quadrotor
  rebalancing signature, emergent from the physics, not scripted.
- **Hover operating points track payload.** 2908 / 3165 / 3412 RPM at 0 / 5 / 10 g — consistent
  with $T = k\omega^2$ for the true vehicle mass, with a realistic ~1.5× throttle headroom.
- **No-crash audit.** All 96 missions fly: min altitude ≥ 0.034 m (0.55 m reference), mean altitude
  0.37–0.60 m. This required an altitude **I-term** in the controller — without it, the 4-fault
  aircraft (η=0.85) is thrust-starved and can never leave the ground (a real diagnosis of a real
  simulator limitation, fixed in `isaac/cf2x_fault_sim.py`).

<p align="center">
  <img src="results/figures/fig3_vibration_signatures.png" alt="Welch PSD of healthy vs single-fault vs double-fault gyroscope with 1P lines annotated" width="88%">
</p>
<p align="center"><sub><b>Figure 3 — Vibration signatures.</b> Welch PSD of the gyro signal for healthy, single-rotor and double-rotor faults; the 1P lines appear exactly at the faulted rotors' speeds.</sub></p>

### Fault taxonomy (16 classes)

| Bitmask | Meaning | Class | Severity |
|:--:|:--|:--|:--:|
| `0x00` | no fault | `healthy` | 0 |
| `0x01`–`0x0f` | each non-empty subset of rotors R1–R4 | `label_1` … `label_15` | 2 (single) / 3 (multi) |

Per faulted rotor: **0.6 mm COM offset**, **15% 1P aerodynamic asymmetry**, **η = 0.85 thrust
efficiency**, **0.75 motor gain**; utilization ratio `ur = 0.3·n_faulted` (cap 0.9).

---

## 📦 Dataset — 96 Real Physics Flights, Sealed Split

<p align="center">
  <img src="results/figures/fig1_dataset_overview.png" alt="Dataset composition: 96 runs, 16 classes, severity and hover RPM distributions, vibration RMS vs number of faulted rotors" width="92%">
</p>
<p align="center"><sub><b>Figure 4 — Dataset overview.</b> 96 flights = 16 classes × 6 runs (3 payloads × 2 seeds); severity and hover-RPM distributions; vibration RMS grows with the number of faulted rotors.</sub></p>

<p align="center">
  <img src="results/figures/fig2_flight_mission.png" alt="3D mission trajectory, altitude tracking, RPM command vs measured, and IMU time series" width="92%">
</p>
<p align="center"><sub><b>Figure 5 — Flight mission.</b> 9-second waypoint mission at 0.55 m: 3D trajectory, altitude tracking, closed-loop RPM and raw 500 Hz IMU.</sub></p>

| Property | Value |
|:--|:--|
| Flights | 96 (16 classes × 6 runs: payloads 0/5/10 g × seeds 0/1) |
| Sensors | IMU @ 500 Hz (rpm1-4, roll, pitch, yaw, gyro xyz, acc xyz = 13 ch) |
| Window | 100 samples = 0.2 s, step 4 → 105,696 windows (1,101 per flight) |
| Split | **frozen manifest** (`splits/split_manifest.json`, sha256-pinned): 48 train / 16 val / 32 test runs — per class 3/1/2, every payload level in every class's train set |
| Augmentation | ×2 (noise, RPM-scale, drift, shift) → **train runs only**, applied after the split → 105,696 extra train windows |

The sealed dataset (`ml_sealed.h5`, 211,392 windows) stores a `split` and `is_aug` column per
window. **Augmented windows are mathematically guaranteed to appear only in the train split** —
a property enforced by the builder and pinned by a CI test.

---

## 🧠 Model — PaperCNN

<p align="center">
  <img src="results/figures/fig_model_architecture.png" alt="PaperCNN block diagram: three conv blocks 32-64-128, global average pooling, FC 128, 16-class head" width="94%">
</p>
<p align="center"><sub><b>Figure 6 — PaperCNN.</b> Three 3×3 conv blocks with doubling feature maps (32→64→128), global average pooling, and a 16-class head — 112K parameters, ≈450 KB checkpoint.</sub></p>

The window is treated as a 1 × 13 × 100 "image". **The channel layout is stored in the checkpoint
meta (`vars`) and enforced identically** in the dataset builder, trainer and ROS2 node — so the
deployment node assembles live IMU features in exactly the order the model was trained on.
Checkpoints are self-describing (vars, mean/std, manifest sha, base_filters) and the ROS2 node
**refuses to start** on one that isn't.

<p align="center">
  <img src="results/figures/fig4_training_curves.png" alt="Training and validation loss and accuracy curves for the 13-channel and IMU-only models" width="86%">
</p>
<p align="center"><sub><b>Figure 7 — Training.</b> 50 epochs max, early stop at epoch 27; best validation accuracy 90.0% (13-ch) / 87.6% (IMU-only 9-ch) on the 16 sealed validation runs.</sub></p>

---

## 📈 Results — and Where It Fails

<p align="center">
  <img src="results/figures/fig5_confusion_matrix.png" alt="16x16 confusion matrix and per-class F1 on the 32 untouched sealed test flights" width="88%">
</p>
<p align="center"><sub><b>Figure 8 — Confusion matrix (sealed test flights).</b> 81.7% accuracy, macro F1 0.80; healthy precision/recall 1.00; errors concentrate between fault combinations that share rotor-overlap and operating point.</sub></p>

### 🎯 Where it works

- **`healthy` is never misclassified in either direction** — F1 1.00. For a deployment story
  (fault detection before classification), this is the class that matters most.
- **High-severity, multi-rotor faults** (bigger vibration signatures) classify near-perfectly.
- **Cross-operating-point generalization is real**: with three RPM bins in training, a model
  retrained from scratch reads a never-seen fourth bin at 90–99% (Figure 9).

### 💥 Where it fails

- **`label_12` (mask0c — rotors R3+R4) recall is 0.13** (precision 1.00: when the model says
  `label_12` it is right, but it rarely says it — true-12 windows get called other two-rotor
  faults). This is the single biggest accuracy sink.
- **Failures concentrate at the lowest hover-RPM operating point** (0 g payload ≈ 2900–3100 RPM),
  where 1P excitation is weakest: the four test runs below 50% accuracy (`mask0c_p05_s1` → **0%**,
  `mask0c_p00_s1` → 27%, `mask02_p00_s0` → 29%, `mask0e_p00_s0` → 45%) are all low-payload runs.
- **Per-run spread is huge** (±24.8%): mean window accuracy hides flight-level failures — which is
  exactly why per-run metrics are reported and plotted, not just the aggregate.

The LOSO results add an instructive contrast: folds see *more* operating-point diversity (3 bins)
than the sealed train split, and score higher (93.2% mean). **More diverse training operating
points would likely lift the weak low-RPM corner** — the clearest next data-collection target.

<p align="center">
  <img src="results/figures/fig6_cross_speed.png" alt="Accuracy per held-out RPM bin with retrained models, and per-run accuracy spread" width="88%">
</p>
<p align="center"><sub><b>Figure 9 — Cross-RPM generalization (LOSO, retrained).</b> One hover-RPM bin (24 flights, 26,424 windows) held out per fold; the model is retrained from scratch on the other three bins. 90.3 / 91.3 / 99.0 / 92.3%. Right: per-run spread — each dot is one held-out flight.</sub></p>

<p align="center">
  <img src="results/figures/fig7_benchmark.png" alt="Per-class accuracy and inference latency benchmark on sealed test windows" width="88%">
</p>
<p align="center"><sub><b>Figure 10 — Latency benchmark (sealed test windows, CPU).</b> 0.20 ms mean / 0.37 ms p99 per window — three orders of magnitude faster than the 0.2 s window itself, leaving ample budget for onboard use.</sub></p>

### 🎲 Uncertainty — what the model does not know

<p align="center">
  <img src="results/figures/fig8_uncertainty.png" alt="Per-class predictive entropy and OOD-detection AUROC by score" width="88%">
</p>
<p align="center"><sub><b>Figure 11 — Uncertainty & OOD detection.</b> Left: predictive entropy by class on sealed test windows. Right: AUROC for detecting gaussian-noise OOD windows — the energy score separates them almost perfectly; softmax confidence does not.</sub></p>

Measured on sealed test windows vs gaussian-noise OOD (`results/uncertainty_sealed.json`):

| Score | AUROC | Verdict |
|:--|:--:|:--|
| **Energy** | **0.996** | excellent OOD detector |
| Entropy | 0.60 | weak |
| Mutual information (MC-dropout) | 0.51 | **no-op — PaperCNN has no dropout layers**, so MC sampling returns identical logits (documented, not hidden) |
| Neg. max probability | 0.40 | **worse than chance — the model is confidently wrong on OOD**, i.e. softmax confidence must not be used as an OOD gate in deployment |

Lesson pinned by CI tests: the AUROC convention here is *higher = more OOD-like*, and the
orientation of every score is asserted in `tests/test_uncertainty_guards.py` — an earlier version
of this script scored energy backwards and silently reported 0.996 for the wrong reason.

### Severity is reported separately

Fault *identity* (which rotors) and *severity* (from the number of faulted rotors, plus `ur`) are
kept as distinct outputs; per-window true/predicted severity is stored in
`results/eval_sealed/predictions.npz` (`sev` key).

---

## 🚀 Deployment — ROS2 Jazzy Node

<p align="center">
  <img src="results/figures/fig_ros2_deployment.png" alt="ROS2 node block diagram: live IMU subscription or CSV replay, sliding window, meta-driven normalization, PaperCNN, fault_detection publisher" width="92%">
</p>
<p align="center"><sub><b>Figure 12 — ROS2 deployment.</b> The node runs live on <code>/imu/data</code> (+ <code>--rpm-topic</code> for RPM-telemetry models) or replays CSV flight logs offline.</sub></p>

The node is deployment-audited: it **refuses to start** rather than silently produce garbage.

| Guard | Behavior |
|:--|:--|
| RPM-consuming model in live mode without `--rpm-topic` | refuses to start (zero-filled RPMs would be far out-of-domain) — deploy the IMU-only model instead, or provide telemetry |
| Replay CSV missing any model channel | aborts; `--allow-missing-columns` zero-fills but marks predictions **COMPROMISED** |
| Checkpoint without a `vars` channel list | refuses to start (channel layout cannot be guessed) |
| Unknown model channels (not IMU/RPM) | refuses to start |

```bash
# Live inference, model with RPM channels (ROS2 Jazzy, after colcon build + source):
ros2 launch uav_aegis fault_inference.launch.py model:=models/cnn_sealed.pth rpm_topic:=/rotor_rpms
#   ...or directly:
python3 scripts/ros2_inference_node.py --model models/cnn_sealed.pth --mode live --rpm-topic /rotor_rpms

# Live inference, IMU-only model (no RPM telemetry needed):
python3 scripts/ros2_inference_node.py --model models/cnn_imu9.pth --mode live

# Offline replay of a real flight log (no ROS2 required):
python3 scripts/ros2_inference_node.py --model models/cnn_imu9.pth \
    --mode replay --input isaac_dataset/run_cf2x_mask01_p00_s0/imu.csv --step 500
# → label_1  (9/9 windows — verified on this held-out flight)
```

The `ros2_ws/src/uav_aegis` package vendors the node scripts at build time, so `ros2 launch` /
`ros2 run` work from the install tree without the repo on `PYTHONPATH`.

---

## 🧪 Evaluation Methodology (what makes these numbers credible)

1. **Sealed run-grouped split, frozen in a manifest.** `splits/split_manifest.json` (seed 42,
   sha256-pinned) fixes 48/16/32 flights as train/val/test — per class 3/1/2, every payload level
   in every class's train set. Windows of a flight never cross partitions. The manifest's sha256
   hashes the split *content*, so regenerating with the same seed reproduces the identical
   fingerprint (pinned by CI).
2. **Augmentation after the split, train-only.** ×2 augmented copies are attached only to train
   runs, inside the sealed h5 — val/test windows are untouched originals, and a CI test asserts
   `is_aug == 1 ⇒ split == 0`.
3. **Guard-rail evaluators.** `eval_classifier.py`, `uncertainty_quantification.py` and the
   trainer all refuse to run on: unsealed datasets, the train split (without an explicit
   `--allow-train-eval`), or a model/dataset manifest-sha mismatch. The h5 builder
   (`build_ml_dataset_v2.py`) aborts loudly on missing runs/channels instead of silently
   zero-filling.
4. **Cross-RPM LOSO with true retraining.** Each fold holds out one hover-RPM bin (24 flights,
   26,424 windows), validates on 11 flights and **retrains from scratch** on the remaining 61
   (+×2 augmentation). Zero-shot numbers on bins the model trained near are meaningless for
   deployment claims, so they are not reported.
5. **34 CI integrity tests** (`pytest tests/`, self-contained synthetic fixtures — no dataset or
   weights needed): manifest determinism/stratification/payload coverage, sealed aug-isolation,
   evaluator guard rails, builder loud-failure behavior, OOD-AUROC orientation, and the deployment
   node's startup guards.
6. **Everything is regenerated from one script** (`reproduce.sh`) — flights, sealed dataset,
   both models, all evals, figures. CI additionally lints and runs the tests on every push.

Known limitation (honest): the evaluation is **simulation-only**. Isaac Sim + PhysX with a
realistic Crazyflie asset is a far stronger basis than hand-written signal generators, but
**sim-to-real transfer is NOT established** by these numbers. The sealed-split, retrained-LOSO and
uncertainty protocols are the transfer-resistant parts; the absolute accuracy is an upper bound for
the current data source.

---

## 🛠️ Quick Start

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
PYTHON_SH=... ./reproduce.sh --quick                                             # quick pass
```

Generates the 96 flights (~13 min on a laptop GPU), builds the sealed HDF5 dataset (manifest →
`ml_sealed.h5`), trains both models (13-ch + IMU-only), runs all evals (sealed test, LOSO
retrain, uncertainty, benchmark) and writes the 12 report figures.

### 3. Use the trained model

```bash
# Classification report on the sealed test split (untouched flights only)
python3 scripts/eval_classifier.py --h5 ml_sealed.h5 \
    --model models/cnn_sealed.pth --out results/eval_sealed

# Cross-RPM generalization, retrained per fold:
python3 scripts/eval_cross_speed.py --dataset ml_sealed.h5 --retrain \
    --output results/cross_speed_loso.json
```

### 4. Tests & CI

```bash
python3 -m pytest tests/ -q          # 34 integrity tests (self-contained fixtures)
# CI (.github/workflows/ci.yml): flake8 + pytest + benchmark --smoke (synthetic fallback,
# no GPU / no dataset required)
```

---

## 🗂️ Repository Layout

```
isaac/cf2x_fault_sim.py           # Isaac Sim mission generator (fault injection, controller, logging)
isaac/force_probe.py              # USD asset verification utility (masses, joints, arm offsets)
isaac/assets/Bitcraze/…           # Crazyflie 2.1 USD asset
isaac_dataset/run_cf2x_*/         # 96 flight logs (CSVs gitignored; meta.json manifests tracked)
splits/split_manifest.json        # FROZEN train/val/test flight split (sha256-pinned)
scripts/make_split_manifest.py     # builds the frozen split manifest
scripts/build_ml_dataset_v2.py     # flight logs → windowed 13-channel HDF5 (loud-failure guards)
scripts/build_sealed_dataset.py    # manifest + source h5 → ml_sealed.h5 (train-only augmentation)
scripts/augment_dataset.py         # noise/RPM-scale/drift/shift augmentation (used by sealed builder)
scripts/train_cnn.py               # PaperCNN training on the sealed split (manifest-verified)
scripts/eval_classifier.py         # sealed test report + predictions.npz (guard-railed)
scripts/eval_cross_speed.py        # leave-one-RPM-bin-out, RETRAINED per fold
scripts/uncertainty_quantification.py    # entropy/MI/energy + OOD AUROC (sealed scope)
scripts/self_supervised_pretrain.py      # optional SimCLR-style encoder (not in main pipeline)
scripts/benchmarks/run_benchmark.py      # latency + accuracy benchmark (sealed test, CI-safe --smoke)
scripts/make_report_plots.py       # report figures (consumes only real artifacts)
scripts/make_architecture_diagrams.py    # block diagrams (pipeline, model, faults, ROS2 node)
scripts/ros2_inference_node.py    # ROS2 deployment node (live + replay, deployment guards)
ros2_ws/src/uav_aegis/            # ROS2 package (vendored scripts, launch file)
tests/                            # 34 CI integrity tests (synthetic fixtures)
reproduce.sh                      # end-to-end pipeline
results/                          # all evaluation artifacts + figures
```

---

## 📐 Design Decisions (short)

- **13 channels** (`rpm1-4, roll, pitch, yaw, gyro_xyz, acc_xyz`): the accelerometer carries the
  strongest 1P signature (63× energy amplification vs 20× on gyro) — dropping it (the 10-channel
  default) discards signal. Channel order is stored in the model checkpoint and enforced
  end-to-end (builder → training → ROS2 node).
- **0.2 s windows @ 500 Hz**: long enough to resolve the 48–65 Hz 1P lines at hovering speeds,
  short enough for a 0.2 ms reactive onboard loop.
- **Sealed split over ad-hoc splits**: a frozen manifest makes "which flight was this window
  trained on" a property of the repo, not of the last script run — every evaluator and the model
  checkpoint cross-check the same sha256.
- **Retrained LOSO over zero-shot cross-eval**: deployment claims need "never saw this operating
  point", which means retraining without it, not predicting from a model that trained next door.
- **Refuse-don't-degrade deployment**: the ROS2 node aborts on missing RPM telemetry, missing
  replay columns, or non-self-describing checkpoints, instead of zero-filling inputs and
  publishing confident garbage.
- **Altitude I-term in the simulator controller** so *faulted* vehicles still fly realistic missions
  (PD-only commands healthy-model RPMs and starves multi-fault vehicles).

---

## 🧹 Known Limitations

- **Sim-to-real gap**: all 96 flights are Isaac Sim. Real-vehicle transfer requires domain
  randomization + flight data (the evaluation protocols are designed to make that step honest).
- **`mask0c` (R3+R4) is nearly inseparable at low RPM** (recall 0.13 on the sealed test split).
  More low-RPM training flights — or features that expose rotor *identity* rather than just
  cardinality — are needed; the current data cannot solve this by tuning alone.
- **MC-dropout is a no-op** on PaperCNN (no dropout layers): mutual information is reported for
  completeness but is ≈ chance. Dropout would need to be added to the architecture for meaningful
  epistemic uncertainty.
- **Softmax confidence is not an OOD gate here** (AUROC 0.40 on gaussian noise — confidently
  wrong). The energy score is; deployment should use it.
- **One fault severity profile** (fixed COM/aero/efficiency magnitudes); severity labels encode
  *how many* rotors faulted, not graded degradation levels.
- **16 classes on one vehicle model** (Crazyflie 2.1); per-vehicle calibration is out of scope.

---

## 📜 License

MIT — see repository root.

## 👤 Author

**Rhutvik Pachghare** — [GitHub](https://github.com/Rhutvik-pachghare1999)

Built as a solo end-to-end robotics ML project: simulation, data engineering, modeling,
evaluation methodology, and deployment.
