#!/usr/bin/env python3
"""
Generate architecture / block diagrams for the README (no data dependencies).

Outputs -> results/figures/
  fig0_system_architecture.png  end-to-end pipeline block diagram
  fig_model_architecture.png    PaperCNN block diagram (shapes + params)
  fig_fault_injection.png       quad schematic + fault profile + class encoding
  fig_ros2_deployment.png       ROS2 node I/O diagram (live + replay)

All figures are verified programmatically (size, non-blank, ink coverage).
"""
import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

# ---------------------------------------------------------------- palette ----
NAVY = "#0F2D52"
INK = "#1A1A2E"
GRAY = "#5A6472"
WHITE = "#FFFFFF"

STAGES = [  # (fill, edge)
    ("#EAF2FB", "#1F77B4"),  # simulation  blue
    ("#E3F4F7", "#0E8A9E"),  # dataset     teal
    ("#FDEEE1", "#E8722A"),  # training    orange
    ("#E8F6EC", "#2CA02C"),  # evaluation  green
    ("#F0ECF8", "#7B5EA7"),  # deployment  purple
]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.size": 9, "axes.titlesize": 11,
    "axes.grid": False, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False,
    "axes.spines.bottom": False,
})


def _box(ax, x, y, w, h, title, lines, fill, edge, fs_title=9.5, fs_body=8,
         title_color=NAVY, lw=1.6, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.6,rounding_size={r}",
                                fc=fill, ec=edge, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top",
            fontsize=fs_title, fontweight="bold", color=title_color, zorder=3)
    ty = y + h - 4.4
    for ln in lines:
        ax.text(x + 1.3, ty, ln, ha="left", va="top", fontsize=fs_body,
                color=INK, zorder=3)
        ty -= 2.5
    return ty


def _arrow(ax, x0, x1, y, color=GRAY, lw=2.2, style="-|>", ms=22):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y),
                                 arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, zorder=1,
                                 shrinkA=0, shrinkB=0))


def _chip(ax, x, y, w, text, edge="#8896A6", fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, 3.2,
                                boxstyle="round,pad=0.35,rounding_size=0.5",
                                fc=WHITE, ec=edge, lw=1.1, zorder=2))
    ax.text(x + w / 2, y + 1.6, text, ha="center", va="center",
            fontsize=fs, family="monospace", color=NAVY, zorder=3)


def _title(ax, x, y, text, sub=None, fs=13, fs_sub=9.5):
    ax.text(x, y, text, ha="center", va="top", fontsize=fs,
            fontweight="bold", color=NAVY)
    if sub:
        ax.text(x, y - 2.6, sub, ha="center", va="top", fontsize=fs_sub,
                color=GRAY)


# ------------------------------------------------------ fig 0: pipeline ------
def fig_system_architecture(out):
    fig, ax = plt.subplots(figsize=(13.2, 5.4))
    ax.set_xlim(0, 120); ax.set_ylim(0, 52); ax.axis("off")

    _title(ax, 60, 52, "UAV Aegis — End-to-End Fault Diagnostics Pipeline",
           "Isaac Sim physics flights → windowed dataset → CNN → leakage-audited evaluation → ROS2 deployment")

    W, H, Y = 21.5, 24.5, 19
    xs = [2.0, 25.5, 49.0, 72.5, 96.0]

    stages = [
        ("1 · SIMULATION",
         ["Isaac Sim 5.1.0 + PhysX", "Crazyflie 2.1 X-quad (USD)",
          "IMU @ 500 Hz · 9 s missions", "physical fault injection",
          "96 flights · 16 classes"]),
        ("2 · DATASET",
         ["13-channel windows", "100 samples = 0.2 s · step 4",
          "105,696 windows", "augment ×3 → 317,088",
          "run-grouped splits"]),
        ("3 · TRAINING",
         ["PaperCNN · 112K params", "train / val / test = disjoint",
          "flights (GroupShuffleSplit)", "50 epochs · early stop @ 22",
          "best val acc 0.9931"]),
        ("4 · EVALUATION",
         ["test acc 99.0% · macro F1 0.99", "healthy F1 1.00",
          "cross-RPM bins 98.8–99.5%", "MC-dropout entropy 0.0010",
          "latency 0.20 ms mean"]),
        ("5 · DEPLOYMENT",
         ["ROS2 Jazzy node", "live /imu/data @ 500 Hz",
          "or CSV flight-log replay", "publishes /fault_detection",
          "meta-driven 13-ch input"]),
    ]
    chips = ["isaac_dataset/", "ml_dataset_v2_aug.h5", "models/cnn_multi.pth",
             "results/ · evals + figures", "/fault_detection topic"]

    for i, ((x, (title, lines)), (fill, edge)) in enumerate(zip(zip(xs, stages), STAGES)):
        _box(ax, x, Y, W, H, title, lines, fill, edge)
        _chip(ax, x + 0.8, 12.2, W - 1.6, chips[i], edge=edge)
        _arrow(ax, x + W / 2, x + W / 2, Y - 0.2, color=edge, lw=1.4, ms=14)
        if i < 4:
            _arrow(ax, x + W + 0.4, xs[i + 1] - 0.4, Y + H / 2)

    ax.text(60, 6.5,
            "96 flights  ·  16 classes  ·  317,088 training windows  ·  "
            "99.0% run-grouped test accuracy  ·  0.20 ms inference",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color=NAVY,
            bbox=dict(boxstyle="round,pad=0.5", fc="#F5F8FC", ec="#8896A6", lw=1.0))
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------ fig: model -----------
def fig_model_architecture(out):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cnn_classifier import PaperCNN

    model = PaperCNN(in_channels=1, base_filters=32, num_classes=16)
    n_params = sum(p.numel() for p in model.parameters())

    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    ax.set_xlim(0, 120); ax.set_ylim(0, 40); ax.axis("off")

    _title(ax, 60, 40, f"PaperCNN — Fault Classifier ({n_params:,} parameters)",
           "input: 0.2 s window as a 1 × 13 × 100 \u201cimage\u201d  ·  output: 16-class fault diagnosis  ·  "
           "99.0% test accuracy  ·  0.20 ms mean latency")

    W, H, Y = 16.5, 17, 12
    xs = [3.0, 22.0, 41.0, 60.0, 79.0, 100.0]
    blocks = [
        ("Input", "1 × 13 × 100", ["rpm1-4 · roll · pitch · yaw", "gyro xyz · acc xyz",
                                   "normalized (checkpoint", "mean / std)"], STAGES[1]),
        ("Conv Block 1", "32 × 6 × 50", ["Conv2d 3×3 → 32", "BatchNorm · ReLU",
                                        "MaxPool 2 × 2", "— 384 params"], STAGES[2]),
        ("Conv Block 2", "64 × 3 × 25", ["Conv2d 3×3 → 64", "BatchNorm · ReLU",
                                        "MaxPool 2 × 2", "— 18,624 params"], STAGES[2]),
        ("Conv Block 3", "128 × 1 × 1", ["Conv2d 3×3 → 128", "BatchNorm · ReLU",
                                         "Global Avg Pool", "— 74,112 params"], STAGES[2]),
        ("FC", "128", ["Flatten 128", "Linear → 128", "ReLU", "— 16,512 params"], STAGES[3]),
        ("Head", "16 logits", ["Linear → 16", "healthy + 15 rotor", "fault combinations",
                               "— 2,064 params"], STAGES[3]),
    ]
    for (title, shape, lines, (fill, edge)), x in zip(blocks, xs):
        ax.add_patch(FancyBboxPatch((x, Y), W, H,
                                    boxstyle="round,pad=0.6,rounding_size=0.02",
                                    fc=fill, ec=edge, lw=1.6, zorder=2))
        ax.text(x + W / 2, Y + H - 1.3, title, ha="center", va="top",
                fontsize=9.5, fontweight="bold", color=NAVY, zorder=3)
        ax.text(x + W / 2, Y + H - 5.0, shape, ha="center", va="top",
                fontsize=9, fontweight="bold", color=edge, zorder=3)
        ty = Y + H - 7.8
        for ln in lines:
            ax.text(x + W / 2, ty, ln, ha="center", va="top", fontsize=7.3,
                    color=INK, zorder=3)
            ty -= 2.3

    for i in range(5):
        _arrow(ax, xs[i] + W + 0.3, xs[i + 1] - 0.3, Y + H / 2, lw=2.4)

    ax.text(60, 7.5,
            "3 convolutional blocks double the feature maps (32 → 64 → 128); global average pooling makes the "
            "classifier robust to window length; total checkpoint ≈ 450 KB.",
            ha="center", va="center", fontsize=8.5, color=GRAY)
    ax.text(60, 3.4,
            "Channel order is stored in the model checkpoint meta (\u201cvars\u201d) and enforced identically in the "
            "dataset builder, trainer and ROS2 node.",
            ha="center", va="center", fontsize=8.5, color=GRAY)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------ fig: fault injection -------
def fig_fault_injection(out):
    fig = plt.figure(figsize=(13.2, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], height_ratios=[1, 1],
                         left=0.03, right=0.98, top=0.90, bottom=0.05, wspace=0.12, hspace=0.42)

    fig.suptitle("Physical Fault Injection — Crazyflie 2.1 X-Quad (Isaac Sim + PhysX)",
                 fontsize=13, fontweight="bold", color=NAVY, x=0.5, y=0.965)

    # ---- panel A: quad schematic -------------------------------------------
    axA = fig.add_subplot(gs[:, 0]); axA.axis("off")
    axA.set_xlim(-5.4, 5.4); axA.set_ylim(-4.9, 4.9)
    axA.set_title("Rotor bitmask → 15 fault combinations + healthy",
                  fontsize=9.5, color=NAVY, pad=2)

    cx, cy = 0, 0
    axA.add_patch(FancyBboxPatch((cx - 0.9, cy - 0.65), 1.8, 1.3,
                                 boxstyle="round,pad=0.08,rounding_size=0.15",
                                 fc="#22314A", ec=NAVY, lw=1.5, zorder=3))
    axA.text(cx, cy, "CF 2.1", ha="center", va="center", fontsize=8,
             color="white", fontweight="bold", zorder=4)

    rotors = [  # (x, y, name, hex, dir +1=CCW)
        (1.95, 1.95, "R1", "0x01", +1, "front-right"),
        (1.95, -1.95, "R2", "0x02", -1, "rear-right"),
        (-1.95, -1.95, "R3", "0x04", +1, "rear-left"),
        (-1.95, 1.95, "R4", "0x08", -1, "front-left"),
    ]
    for (rx, ry, name, hx, d, pos) in rotors:
        axA.plot([cx, rx], [cy, ry], color="#4A5568", lw=3.2, zorder=1, solid_capstyle="round")
        faulted = name == "R1"
        col = "#D64545" if faulted else "#0E8A9E"
        axA.add_patch(Circle((rx, ry), 0.72, fc="#FFFFFF" if not faulted else "#FDECEC",
                             ec=col, lw=2.4 if faulted else 1.8, zorder=2))
        axA.text(rx, ry + 0.14, name, ha="center", va="center", fontsize=9.5,
                 fontweight="bold", color=col, zorder=4)
        axA.text(rx, ry - 0.33, hx, ha="center", va="center", fontsize=7,
                 family="monospace", color=GRAY, zorder=4)
        # rotation-direction arc
        th = np.linspace(0.35, 2.6, 24) * d
        arc_r = 1.05
        axA.plot(rx + arc_r * np.cos(th), ry + arc_r * np.sin(th),
                 color=col, lw=1.2, alpha=0.85, zorder=1)
        arr_th = 2.6 * d
        axA.annotate("", xy=(rx + arc_r * np.cos(arr_th + 0.14 * d), ry + arc_r * np.sin(arr_th + 0.14 * d)),
                     xytext=(rx + arc_r * np.cos(arr_th), ry + arc_r * np.sin(arr_th)),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, alpha=0.85))
        axA.text(rx * 1.32, ry * 1.32, pos, ha="center", va="center",
                 fontsize=7, color=GRAY, zorder=4,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    axA.annotate("↑ front", xy=(0, 3.6), ha="center", fontsize=8.5, color=NAVY,
                 fontweight="bold")
    axA.annotate("example: mask 0x03 = R1 + R2 faulted → class label_3 (severity: Severe)",
                 xy=(0, -3.9), ha="center", fontsize=8, color=GRAY)
    axA.annotate("R1 faulted (example)", xy=(1.95, 0.98), xytext=(2.6, 0.2),
                 fontsize=7.5, color="#D64545", fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color="#D64545", lw=1.2))

    # ---- panel B: fault profile --------------------------------------------
    axB = fig.add_subplot(gs[0, 1]); axB.axis("off")
    axB.set_xlim(0, 10); axB.set_ylim(0, 10)
    _box(axB, 0.2, 0.2, 9.6, 9.4, "Fault profile — applied to each faulted rotor",
         [], "#FDEEE1", "#E8722A", fs_title=10)
    profile = [
        ("COM offset", "+0.6 mm propeller eccentricity → true 1P centrifugal force"),
        ("1P aero asymmetry", "15% once-per-rev thrust harmonic (chipped blade)"),
        ("Efficiency", "η = 0.85 thrust coefficient loss (eroded blade)"),
        ("Motor gain", "0.75 drive gain → RPM sag under load"),
        ("Reported ur", "0.30 × n_faulted (capped 0.9)"),
    ]
    y = 7.0
    for k, v in profile:
        axB.text(0.75, y, "•", fontsize=11, color="#E8722A", va="top")
        axB.text(1.15, y, k, fontsize=8.8, fontweight="bold", color=NAVY, va="top")
        axB.text(3.55, y, v, fontsize=8, color=INK, va="top")
        y -= 1.55

    # ---- panel C: measured effects ------------------------------------------
    axC = fig.add_subplot(gs[1, 1]); axC.axis("off")
    axC.set_xlim(0, 10); axC.set_ylim(0, 10)
    _box(axC, 0.2, 0.2, 9.6, 9.4, "Signatures measured in the flight data",
         [], "#E8F6EC", "#2CA02C", fs_title=10)
    effects = [
        ("1P vibration line", "gyro PSD peak 50.2 Hz vs faulted rotor 51.8 Hz (3109 RPM)"),
        ("Energy amplification", "~100× gyro · ~63× accel energy in 20–80 Hz band"),
        ("RPM redistribution", "faulted 3109 RPM vs counter-rotor 2881 RPM (controller)"),
        ("Severity labels", "single rotor → Moderate (2) · multi → Severe (3)"),
    ]
    y = 7.1
    for k, v in effects:
        axC.text(0.75, y, "•", fontsize=11, color="#2CA02C", va="top")
        axC.text(1.15, y, k, fontsize=8.8, fontweight="bold", color=NAVY, va="top")
        axC.text(3.55, y, v, fontsize=7.8, color=INK, va="top")
        y -= 1.72

    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------ fig: ROS2 deployment --------
def fig_ros2_deployment(out):
    fig, ax = plt.subplots(figsize=(12.6, 5.2))
    ax.set_xlim(0, 120); ax.set_ylim(0, 48); ax.axis("off")

    _title(ax, 60, 48, "ROS2 Jazzy Deployment — uav_aegis · fault_inference_node",
           "single-window inference: 0.20 ms mean · channel layout read from model checkpoint (meta-driven)")

    # inputs (left)
    _box(ax, 2, 30, 26, 11, "LIVE  ·  sensor_msgs/Imu",
         ["/imu/data @ 500 Hz", "acc xyz · gyro xyz", "rpy from orientation",
          "rpm channels zero-filled"], STAGES[0][0], STAGES[0][1], fs_title=9)
    _box(ax, 2, 12, 26, 11, "REPLAY  ·  CSV flight log",
         ["isaac_dataset/run_*/imu.csv", "13 columns @ 500 Hz",
          "offline smoke-testing", "--mode replay --step N"], STAGES[1][0], STAGES[1][1], fs_title=9)

    # node (center)
    nx, nw = 38, 44
    ax.add_patch(FancyBboxPatch((nx, 6), nw, 34,
                                boxstyle="round,pad=0.8,rounding_size=0.03",
                                fc="#F0ECF8", ec="#7B5EA7", lw=2.2, zorder=2))
    ax.text(nx + nw / 2, 37.6, "fault_inference_node", ha="center", va="top",
            fontsize=11, fontweight="bold", color=NAVY, zorder=3)
    steps = [
        ("①  sliding window", "deque of 100 samples = 0.2 s"),
        ("②  normalization", "13 ch · mean/std from checkpoint meta"),
        ("③  PaperCNN", "1 × 13 × 100 → 16 logits"),
        ("④  diagnosis", "argmax → label + severity map"),
    ]
    y = 32.4
    for k, v in steps:
        ax.add_patch(FancyBboxPatch((nx + 3, y - 5.6), nw - 6, 5.0,
                                    boxstyle="round,pad=0.3,rounding_size=0.4",
                                    fc="white", ec="#B9A7DC", lw=1.0, zorder=3))
        ax.text(nx + 5, y - 3.1, k, ha="left", va="center", fontsize=8.6,
                fontweight="bold", color=NAVY, zorder=4)
        ax.text(nx + 21, y - 3.1, v, ha="left", va="center", fontsize=7.8,
                color=GRAY, zorder=4)
        y -= 6.4

    # output (right)
    _box(ax, 92, 26, 26, 13, "PUBLISH",
         ["/fault_detection", "std_msgs/String", "predicted fault label", "e.g.  “label_1”"],
         STAGES[4][0], STAGES[4][1], fs_title=9)
    ax.text(105, 21.5, "severity with --publish-severity", ha="center",
            fontsize=7.2, color=GRAY, style="italic")

    # arrows
    _arrow(ax, 28.4, 37.6, 35.5)
    _arrow(ax, 28.4, 17.5, 35.5)
    _arrow(ax, nx + nw + 0.4, 32.5, 91.6)

    ax.text(60, 2.8,
            "replay mode verified on held-out flights: healthy → healthy · mask01 → label_1 · mask0f → label_15 (9/9 windows each)",
            ha="center", va="center", fontsize=8.5, color=GRAY)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------ verify ---------
def verify(path):
    from PIL import Image
    im = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    ink = float((im < 245).mean())  # fraction of non-white pixels
    ok = im.size > 0 and im.std() > 10 and ink > 0.02
    status = "PASS" if ok else "FAIL"
    print(f"  {path.name:<32} {path.stat().st_size/1024:7.1f} KB  "
          f"std={im.std():6.1f}  ink={ink*100:5.1f}%  [{status}]")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    figs = [
        ("fig0_system_architecture.png", fig_system_architecture),
        ("fig_model_architecture.png", fig_model_architecture),
        ("fig_fault_injection.png", fig_fault_injection),
        ("fig_ros2_deployment.png", fig_ros2_deployment),
    ]
    print("Generating architecture diagrams ->", out_dir)
    all_ok = True
    for name, fn in figs:
        out = out_dir / name
        fn(out)
        all_ok &= verify(out)
    if not all_ok:
        raise SystemExit("diagram verification FAILED")
    print("All diagrams verified.")


if __name__ == "__main__":
    main()
