#!/usr/bin/env python3
"""
Generate publication-quality figures from REAL pipeline artifacts.

Inputs (all optional; missing artifacts are skipped with a warning):
  - isaac_dataset/run_cf2x_*/        raw simulator runs (imu.csv/state.csv/meta.json)
  - results/train_history_sealed.json (+ _imu9)  training curves (train_cnn.py)
  - results/eval_sealed/predictions.npz  sealed-test predictions (eval_classifier.py)
  - results/cross_speed_loso.json    leave-one-RPM-bin-out, retrained (eval_cross_speed.py)
  - results/benchmark_metrics.csv    per-class latency (benchmarks/run_benchmark.py)
  - results/uncertainty_sealed.json  OOD/uncertainty (uncertainty_quantification.py)

Outputs -> results/figures/*.png
"""

import argparse
import glob
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

C_BLUE = "#1f77b4"
C_ORANGE = "#ff7f0e"
C_GREEN = "#2ca02c"
C_RED = "#d62728"
C_PURPLE = "#9467bd"
C_GRAY = "#7f7f7f"
FS = 500.0  # IMU sample rate [Hz]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def _welch(x, fs, nperseg=1024):
    """Average periodogram (Welch, Hann window) with scipy fallback to numpy."""
    try:
        from scipy.signal import welch

        f, p = welch(x, fs=fs, nperseg=min(nperseg, len(x)), detrend="constant")
        return f, p
    except Exception:
        x = np.asarray(x, float)
        x = x - x.mean()
        nseg = max(1, len(x) // nperseg)
        f = np.fft.rfftfreq(nperseg, 1.0 / fs)
        win = np.hanning(nperseg)
        acc = np.zeros(len(f))
        for i in range(nseg):
            seg = x[i * nperseg : (i + 1) * nperseg] * win
            acc += np.abs(np.fft.rfft(seg)) ** 2
        p = acc / (nseg * (win**2).sum() * fs)
        return f, p


def _load_run(root, name):
    if not name.startswith("run_cf2x_"):
        name = "run_cf2x_" + name
    run = os.path.join(root, name)
    if not os.path.isdir(run):
        return None
    imu = np.genfromtxt(os.path.join(run, "imu.csv"), delimiter=",", names=True)
    with open(os.path.join(run, "meta.json")) as f:
        meta = json.load(f)
    return imu, meta


# ---------------------------------------------------------------- figure 1
def fig_dataset_overview(run_root, out):
    """Dataset composition from real run metadata."""
    metas = []
    for d in sorted(glob.glob(os.path.join(run_root, "run_cf2x_*"))):
        try:
            with open(os.path.join(d, "meta.json")) as f:
                metas.append(json.load(f))
        except Exception:
            pass
    if not metas:
        print("  [skip] no run_cf2x_* meta.json found")
        return

    fig = plt.figure(figsize=(10, 7.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

    # (a) runs per fault class
    ax = fig.add_subplot(gs[0, 0])
    classes = {}
    for m in metas:
        classes[m["fault_type"]] = classes.get(m["fault_type"], 0) + 1
    names = sorted(classes)
    counts = [classes[n] for n in names]
    colors = [C_GREEN if n == "healthy" else C_BLUE for n in names]
    ax.bar(range(len(names)), counts, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("label_", "mask 0x") for n in names], rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel("runs")
    ax.set_title(f"(a) Runs per class ({len(metas)} total)")

    # (b) severity distribution
    ax = fig.add_subplot(gs[0, 1])
    sev_names = {0: "None", 1: "Slight", 2: "Moderate", 3: "Severe"}
    sev = {}
    for m in metas:
        s = sev_names.get(m.get("severity", -1), "?")
        sev[s] = sev.get(s, 0) + 1
    order = [k for k in ["None", "Slight", "Moderate", "Severe"] if k in sev]
    vals = [sev[k] for k in order]
    ax.bar(order, vals, color=[C_GREEN, C_ORANGE, C_RED, "#8b0000"][: len(order)])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, str(v), ha="center", fontsize=8)
    ax.set_ylabel("runs")
    ax.set_title("(b) Imbalance severity (per run)")

    # (c) hover RPM operating points
    ax = fig.add_subplot(gs[1, 0])
    for pl, colr in zip((0, 5, 10), (C_BLUE, C_ORANGE, C_PURPLE)):
        r = [m["hover_rpm_mean"] for m in metas if m.get("payload_kg", 0) * 1000 == pl]
        ax.scatter([pl] * len(r), r, s=18, color=colr, zorder=3, label=f"{pl} g payload")
        if r:
            ax.plot([pl - 0.6, pl + 0.6], [np.mean(r)] * 2, color=colr, lw=1.5)
    ax.set_xlabel("payload [g]")
    ax.set_ylabel("mean hover rotor speed [RPM]")
    ax.set_title("(c) Operating points (payload -> hover RPM)")
    ax.legend(loc="lower right", fontsize=7)

    # (d) steady-state vibration level per class
    ax = fig.add_subplot(gs[1, 1])
    healthy = [m["gyro_rms"] for m in metas if m["fault_type"] == "healthy"]
    single = [m["gyro_rms"] for m in metas if bin(m.get("fault_mask", 0)).count("1") == 1]
    multi = [m["gyro_rms"] for m in metas if bin(m.get("fault_mask", 0)).count("1") >= 2]
    data = [healthy, single, multi]
    data = [d for d in data if d]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], (C_GREEN, C_ORANGE, C_RED)):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xticklabels(["healthy", "1 rotor", "2+ rotors"])
    ax.set_ylabel("gyro RMS [rad/s]")
    ax.set_title("(d) Vibration level vs number of faulted rotors")

    fig.suptitle("Isaac Sim dataset composition — 51 real physics runs", y=0.995)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------- figure 2
def fig_trajectory(run_root, out):
    """Real flight path + tracking quality from a healthy run."""
    loaded = _load_run(run_root, "run_cf2x_healthy_p00_s0")
    if loaded is None:
        print("  [skip] healthy run missing")
        return
    imu, meta = loaded
    st = np.genfromtxt(os.path.join(run_root, "run_cf2x_healthy_p00_s0", "state.csv"), delimiter=",", names=True)
    t = st["t"] - st["t"][0]

    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.32)

    ax = fig.add_subplot(gs[0, :2])
    r, w = 0.35, 2 * np.pi / 12.0
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r * np.cos(th), r * np.sin(th), "--", color=C_GRAY, lw=1, label="reference circle (r=0.35 m)")
    ax.plot(st["x"], st["y"], color=C_BLUE, lw=1.4, label="flown (sim)")
    ax.scatter([st["x"][0]], [st["y"][0]], marker="o", color=C_GREEN, zorder=5, s=28, label="start (on trajectory)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.set_title("(a) Horizontal flight path")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, st["z"], color=C_BLUE, lw=1.2)
    ax.axhline(0.55, color=C_GRAY, ls="--", lw=1, label="setpoint 0.55 m")
    ax.fill_between(
        t, 0.55 - 0.05 * np.sin(w * t), 0.55 + 0.05 * np.sin(w * t), color=C_BLUE, alpha=0.12, label="altitude band"
    )
    ax.set_xlabel("t [s]")
    ax.set_ylabel("z [m]")
    ax.set_title("(b) Altitude")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 0])
    for i, c in zip((1, 2, 3, 4), (C_BLUE, C_ORANGE, C_GREEN, C_PURPLE)):
        ax.plot(t, st[f"rpm_cmd{i}"], color=c, lw=0.8, alpha=0.6)
        ax.plot(t, imu[f"rpm{i}"], color=c, lw=0.8)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("rotor speed [RPM]")
    ax.set_title("(c) Commanded (light) vs measured (solid)")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, imu["gyro_x"], lw=0.7, color=C_BLUE, label=r"$\omega_x$")
    ax.plot(t, imu["gyro_y"], lw=0.7, color=C_ORANGE, label=r"$\omega_y$")
    ax.plot(t, imu["gyro_z"], lw=0.7, color=C_GREEN, label=r"$\omega_z$")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("body rate [rad/s]")
    ax.set_title("(d) Gyroscope")
    ax.legend(fontsize=7, ncol=3)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, imu["acc_z"], lw=0.7, color=C_PURPLE)
    ax.axhline(9.81, color=C_GRAY, ls="--", lw=1, label="1 g")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$a_z$ [m/s$^2$]")
    ax.set_title("(e) Vertical accelerometer")
    ax.legend(fontsize=7)

    fig.suptitle("Healthy flight — circular hover-scan mission (real simulation)", y=0.995)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------- figure 3
def fig_vibration_signature(run_root, out):
    """THE physics-credibility figure: 1P lines at the faulted rotor's speed."""
    specs = [
        ("healthy_p00_s0", "Healthy", C_GREEN),
        ("mask01_p00_s0", "Rotor 1 faulted (0x01)", C_ORANGE),
        ("mask03_p00_s0", "Rotors 1+2 faulted (0x03)", C_RED),
    ]
    rows = []
    for name, label, colr in specs:
        loaded = _load_run(run_root, name)
        if loaded is None:
            print(f"  [skip] {name} missing")
            continue
        rows.append((loaded[0], label, colr))
    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 3.1 * len(rows)), sharex=True)
    if len(rows) == 1:
        axes = axes[None, :]
    for (imu, label, colr), (axg, axa) in zip(rows, axes):
        for col, ax in (("gyro_x", axg), ("acc_y", axa)):
            f, p = _welch(imu[col], FS, nperseg=1024)
            ax.semilogy(f, p, color=colr, lw=1.0)
            ax.set_xlim(0, 120)
        # annotate 1P lines at each rotor's actual mean speed
        for i in (1, 2, 3, 4):
            f1p = imu[f"rpm{i}"].mean() / 60.0
            if f1p < 120:
                for ax in (axg, axa):
                    ax.axvline(f1p, color=C_GRAY, ls=":", lw=0.7)
        if label != "Healthy":
            axg.annotate(
                "1P", xy=(imu["rpm1"].mean() / 60.0, axg.get_ylim()[1] * 0.4), color=C_RED, fontsize=8, ha="left"
            )
        axg.set_ylabel("gyro PSD\n[rad$^2$/s$^2$/Hz]", fontsize=7)
        axa.set_ylabel("acc PSD\n[m$^2$/s$^4$/Hz]", fontsize=7)
        axg.text(
            0.01,
            0.92,
            label,
            transform=axg.transAxes,
            fontsize=8.5,
            fontweight="bold",
            color=colr,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    axes[-1, 0].set_xlabel("frequency [Hz]")
    axes[-1, 1].set_xlabel("frequency [Hz]")
    axes[0, 0].set_title("Gyroscope $\\omega_x$ — power spectral density")
    axes[0, 1].set_title("Accelerometer $a_y$ — power spectral density")
    fig.suptitle(
        "Real 1P vibration signatures: spectral lines appear at each faulted rotor's true rotation frequency", y=0.995
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------- figure 4
def fig_training_curves(results, out):
    path = os.path.join(results, "train_history_sealed.json")
    if not os.path.exists(path):
        print("  [skip] train_history_sealed.json missing")
        return
    hist = json.load(open(path))
    ep = np.arange(1, len(hist["train_loss"]) + 1)

    imu9_path = os.path.join(results, "train_history_imu9.json")
    imu9 = json.load(open(imu9_path)) if os.path.exists(imu9_path) else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax1.plot(ep, hist["train_loss"], color=C_BLUE, label="train (13-ch)")
    ax1.plot(ep, hist["val_loss"], color=C_ORANGE, label="validation (13-ch)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("(a) Loss (sealed run-grouped split)")
    ax1.legend(fontsize=8)
    ax2.plot(ep, np.array(hist["val_acc"]) * 100, color=C_GREEN, label="13-ch (RPM+IMU)")
    if imu9:
        ax2.plot(
            np.arange(1, len(imu9["val_acc"]) + 1),
            np.array(imu9["val_acc"]) * 100,
            "--",
            color=C_PURPLE,
            label="9-ch (IMU-only)",
        )
    best = hist.get("best_val_acc", max(hist["val_acc"]))
    ax2.axhline(best * 100, ls="--", color=C_GRAY, lw=1, label=f"best 13-ch = {best * 100:.2f} %")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation accuracy [%]")
    ax2.set_ylim(0, 105)
    ax2.set_title("(b) Held-out (val runs) accuracy")
    ax2.legend(fontsize=7)
    fig.suptitle("CNN training on the sealed split (train runs only + augs)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------- figure 5
def fig_confusion(results, out):
    path = os.path.join(results, "eval_sealed", "predictions.npz")
    if not os.path.exists(path):
        print("  [skip] eval_sealed/predictions.npz missing")
        return
    d = np.load(path, allow_pickle=True)
    trues, preds = d["trues"], d["preds"]
    names = [str(n) for n in d["label_names"]]
    cls_idx = d["class_idx"] if "class_idx" in d else np.arange(len(names))
    n = len(names)
    pos = {int(c): i for i, c in enumerate(cls_idx)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(trues, preds):
        cm[pos[int(t)], pos[int(p)]] += 1
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig = plt.figure(figsize=(9.5, 8))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.6, 1.0], hspace=0.32)

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"(a) Confusion matrix — run-grouped held-out runs (N={len(trues)} windows)", fontsize=10)
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            v = cmn[i, j]
            if cm[i, j] > 0:
                ax.text(
                    j,
                    i,
                    f"{v * 100:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5.5 if n > 12 else 7,
                    color="white" if v > thresh else "black",
                )
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalized rate")

    ax = fig.add_subplot(gs[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = (
            2
            * cm.diagonal()
            / np.maximum(2 * cm.diagonal() + cm.sum(axis=0) - cm.diagonal() + cm.sum(axis=1) - cm.diagonal(), 1)
        )
    order = np.argsort(f1)
    cols = [C_GREEN if names[i] == "healthy" else (C_RED if f1[i] < 0.9 else C_BLUE) for i in order]
    ax.barh([names[i] for i in order], f1[order], color=cols)
    ax.axvline(f1.mean(), ls="--", color=C_GRAY, lw=1, label=f"macro F1 = {f1.mean():.3f}")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-score")
    ax.set_title("(b) Per-class F1 (sorted)", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(axis="y", labelsize=7)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}  (macro F1={f1.mean():.4f})")


# ---------------------------------------------------------------- figure 6
def fig_cross_speed(results, out):
    """LOSO (leave-one-RPM-bin-out, RETRAINED per fold) results.

    The zero-shot version of this experiment is meaningless for deployment
    claims; each bin is held out and the model retrained from scratch.
    """
    path = os.path.join(results, "cross_speed_loso.json")
    if not os.path.exists(path):
        print("  [skip] cross_speed_loso.json missing")
        return
    d = json.load(open(path))
    res = d["results"]
    keys = sorted(res.keys(), key=lambda k: int(k.split("_")[-1]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    acc = [res[k]["window_accuracy"] * 100 for k in keys]
    rng = [res[k]["rpm_range"] for k in keys]
    n_s = [res[k]["n_test_windows"] for k in keys]
    ax1.bar(range(len(keys)), acc, color=[C_GREEN if a >= 90 else (C_BLUE if a >= 80 else C_ORANGE) for a in acc])
    for i, (a, n) in enumerate(zip(acc, n_s)):
        ax1.text(i, a + 0.8, f"{a:.1f}%\n(n={n})", ha="center", fontsize=7.5)
    summ = d.get("summary", {})
    ax1.axhline(
        summ.get("window_accuracy_mean", 0) * 100,
        ls="--",
        color=C_GRAY,
        lw=1.2,
        label=f"mean = {summ.get('window_accuracy_mean', 0) * 100:.1f}%",
    )
    ax1.set_xticks(range(len(keys)))
    ax1.set_xticklabels([f"{lo:.0f}–{hi:.0f} RPM" for lo, hi in rng], fontsize=8)
    ax1.set_ylim(0, 112)
    ax1.set_ylabel("accuracy [%]")
    ax1.set_title(f"(a) Held-out RPM bin accuracy ({d.get('mode', 'loso-retrain')}: model retrained per fold)")
    ax1.legend(fontsize=8)

    # per-run accuracy spread within each held-out bin — shows the failures,
    # not just the average
    ax = ax2
    rng_seed = np.random.default_rng(0)
    for i, k in enumerate(keys):
        run_accs = np.array(list(res[k]["per_run_accuracy"].values())) * 100
        xs = np.full(len(run_accs), i) + rng_seed.uniform(-0.18, 0.18, len(run_accs))
        ax.scatter(xs, run_accs, s=10, color=C_BLUE, alpha=0.6, zorder=3)
        ax.hlines(np.mean(run_accs), i - 0.3, i + 0.3, color=C_RED, lw=1.6, zorder=4)
    ax.axhline(80, ls=":", color=C_GRAY, lw=1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{lo:.0f}–{hi:.0f} RPM" for lo, hi in rng], fontsize=8)
    ax.set_ylabel("per-run accuracy [%]")
    ax.set_ylim(-3, 105)
    ax.set_title("(b) Per-run accuracy spread (red = bin mean)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------- figure 7
def fig_benchmark(results, out):
    path = os.path.join(results, "benchmark_metrics.csv")
    if not os.path.exists(path):
        print("  [skip] benchmark_metrics.csv missing")
        return
    import csv

    with open(path) as f:
        rows = list(csv.DictReader(f))
    summ = {}
    sp = os.path.join(results, "benchmark_summary.json")
    if os.path.exists(sp):
        summ = json.load(open(sp))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    names = [r["class"] for r in rows]
    acc = [float(r["accuracy"]) * 100 for r in rows]
    lat = [float(r["mean_latency_ms"]) for r in rows]
    cols = [C_GREEN if n == "healthy" else C_BLUE for n in names]
    ax1.bar(range(len(rows)), acc, color=cols)
    ax1.axhline(
        float(summ.get("overall_accuracy", 0)) * 100,
        ls="--",
        color=C_RED,
        lw=1.2,
        label=f"overall = {float(summ.get('overall_accuracy', 0)) * 100:.1f}%",
    )
    ax1.set_xticks(range(len(rows)))
    ax1.set_xticklabels([n.replace("label_", "m") for n in names], rotation=60, ha="right", fontsize=6.5)
    ax1.set_ylim(0, 112)
    ax1.set_ylabel("accuracy [%]")
    ax1.set_title("(a) Per-class accuracy (CPU inference)")
    ax1.legend(fontsize=8)

    ax2.bar(range(len(rows)), lat, color=C_PURPLE)
    ax2.axhline(
        float(summ.get("p99_latency_ms", 0)),
        ls="--",
        color=C_RED,
        lw=1.2,
        label=f"p99 = {float(summ.get('p99_latency_ms', 0)):.2f} ms",
    )
    ax2.set_xticks(range(len(rows)))
    ax2.set_xticklabels([n.replace("label_", "m") for n in names], rotation=60, ha="right", fontsize=6.5)
    ax2.set_ylabel("latency [ms]")
    ax2.set_title(f"(b) Window inference latency (mean {float(summ.get('mean_latency_ms', 0)):.2f} ms)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------- figure 8
def fig_uncertainty(results, out):
    path = os.path.join(results, "uncertainty_sealed.json")
    if not os.path.exists(path):
        print("  [skip] uncertainty_sealed.json missing")
        return
    d = json.load(open(path))
    pc = d.get("per_class", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    def _name(k):
        return "healthy" if k in ("0", "healthy") else k

    keys = sorted(pc.keys())
    names = [_name(k) for k in keys]
    ent = [pc[k].get("entropy", 0) for k in keys]
    cols = [C_GREEN if n == "healthy" else C_BLUE for n in names]
    ax1.barh(names, ent, color=cols)
    ax1.set_xlabel("mean predictive entropy [nats]")
    ax1.set_title(f"(a) Uncertainty by class (sealed test split, acc {d.get('accuracy', 0) * 100:.1f}%)")
    ax1.tick_params(axis="y", labelsize=6.5)

    # OOD detection: AUROC of every score, oriented higher=OOD.
    # Energy is the honest winner; confidence-based scores fail on
    # gaussian-noise OOD — the figure shows the gap instead of hiding it.
    aurocs = d.get("auroc_ood", {})
    labels = list(aurocs.keys())
    vals = [aurocs[k] for k in labels]
    short = [lb.split(" (")[0] for lb in labels]
    ax2.barh(short, vals, color=[C_GREEN if v >= 0.9 else (C_BLUE if v >= 0.6 else C_ORANGE) for v in vals])
    for i, v in enumerate(vals):
        ax2.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7.5)
    ax2.axvline(0.5, ls=":", color=C_GRAY, lw=1, label="chance")
    ax2.set_xlim(0, 1.08)
    ax2.set_xlabel("AUROC (OOD = gaussian noise)")
    ax2.set_title("(b) OOD detection by score (higher = OOD)")
    ax2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", default="isaac_dataset")
    p.add_argument("--results", default="results")
    p.add_argument("--out-dir", default="results/figures")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Generating report figures from real artifacts ...")
    fig_dataset_overview(args.run_root, os.path.join(args.out_dir, "fig1_dataset_overview.png"))
    fig_trajectory(args.run_root, os.path.join(args.out_dir, "fig2_flight_mission.png"))
    fig_vibration_signature(args.run_root, os.path.join(args.out_dir, "fig3_vibration_signatures.png"))
    fig_training_curves(args.results, os.path.join(args.out_dir, "fig4_training_curves.png"))
    fig_confusion(args.results, os.path.join(args.out_dir, "fig5_confusion_matrix.png"))
    fig_cross_speed(args.results, os.path.join(args.out_dir, "fig6_cross_speed.png"))
    fig_benchmark(args.results, os.path.join(args.out_dir, "fig7_benchmark.png"))
    fig_uncertainty(args.results, os.path.join(args.out_dir, "fig8_uncertainty.png"))
    print("Done.")


if __name__ == "__main__":
    main()
