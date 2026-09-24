#!/usr/bin/env python3
"""
make_model_slide_fig.py

Creates three figures for your PPT slide:
  1) architecture_schematic.png    -- schematic: Input -> LSTM (regression) and CNN (classification)
  2) window_heatmap.png            -- heatmap of one example input window (10 x 100 expected)
  3) model_slide_composite.png     -- single composite image combining schematic + heatmap + small training thumbnails

Usage:
  python3 make_model_slide_fig.py \
    --h5 ml_dataset_v2.h5 \
    --train_loss results_plots/train_loss.png \
    --val_acc results_plots/val_acc.png \
    --outdir figures

All args optional (defaults shown). The script is defensive if files are missing.
"""
from __future__ import annotations
import os
import argparse
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Optional h5py import (defensive)
try:
    import h5py  # type: ignore
    H5PY_AVAILABLE = True
except Exception:
    H5PY_AVAILABLE = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def draw_schematic(outpath: str):
    """Draw a simple block schematic for the model architecture."""
    plt.figure(figsize=(10,5))
    ax = plt.gca(); ax.axis('off')

    def box(x, y, w, h, txt, fc="#f7f7f7"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                              linewidth=1.2, facecolor=fc, edgecolor="#333")
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=10, wrap=True)

    box(0.05, 0.35, 0.22, 0.30, "Input\n(10 × 100)\nRPMs, roll/pitch/yaw, gyro")
    ax.annotate("", xy=(0.3, 0.5), xytext=(0.27, 0.5), arrowprops=dict(arrowstyle="->", lw=1.8))
    ax.annotate("", xy=(0.47, 0.75), xytext=(0.33, 0.75), arrowprops=dict(arrowstyle="->", lw=1.8))
    ax.annotate("", xy=(0.47, 0.25), xytext=(0.33, 0.25), arrowprops=dict(arrowstyle="->", lw=1.8))

    box(0.33, 0.60, 0.33, 0.28, "Stacked LSTM\n→ FC\nRegression\n(thrust, torque)\nLoss = MSE")
    box(0.33, 0.07, 0.33, 0.28, "1D-CNN Encoder\n→ Dense → Softmax\nClassification (fault loc.)\nOptional severity head")

    ax.text(0.7, 0.92, "Normalization: models/norm_params.json", fontsize=9, ha='left')
    ax.text(0.7, 0.87, "Reg head: MSE | Class head: Cross-Entropy", fontsize=9, ha='left')

    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved schematic: {outpath}")

def make_window_heatmap(h5path: str, outpath: str, prefer_key_list=None) -> bool:
    """Create a heatmap from the first window in the HDF5 file. Returns True on success."""
    if not H5PY_AVAILABLE:
        print("[SKIP] h5py not installed; cannot create heatmap.")
        return False
    if not h5path or not os.path.exists(h5path):
        print(f"[SKIP] HDF5 file not found at: {h5path}")
        return False
    prefer_key_list = prefer_key_list or ['X_train','X','data','windows','X_val','X_test']
    try:
        with h5py.File(h5path, 'r') as f:
            key = None
            for k in prefer_key_list:
                if k in f:
                    key = k; break
            if key is None:
                # pick the first 3D dataset
                for k in f:
                    try:
                        if isinstance(f[k], h5py.Dataset) and len(f[k].shape) == 3:
                            key = k; break
                    except Exception:
                        continue
            if key is None:
                print("[SKIP] No suitable 3D dataset found in HDF5. Keys:", list(f.keys()))
                return False

            sample = f[key][0]  # first sample
            arr = np.array(sample)
            # Normalize to (channels, window)
            if arr.ndim == 3:
                arr = arr.squeeze()
            if arr.ndim != 2:
                print("[SKIP] Unexpected sample shape:", arr.shape)
                return False

            a, b = arr.shape
            # heuristic: if a <= 50 assume channels; else assume window-length on axis 1
            if a <= 50 and b > 50:
                mat = arr
                C, W = a, b
            elif b <= 50 and a > 50:
                mat = arr.T
                C, W = b, a
            else:
                # default: treat first dim as channels
                mat = arr
                C, W = a, b

            # Normalize for visualization
            mat_norm = (mat - np.nanmean(mat)) / (np.nanstd(mat) + 1e-8)

            plt.figure(figsize=(6,3))
            plt.imshow(mat_norm, aspect='auto', cmap='viridis', origin='lower')
            plt.xlabel('Time (steps)')
            plt.ylabel('Channels')
            plt.title(f'Example window (C={C}, W={W})')
            plt.colorbar(label='normalized value', fraction=0.045)
            plt.tight_layout()
            plt.savefig(outpath, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"[OK] Saved window heatmap: {outpath}  (dataset key used: {key})")
            return True
    except Exception as e:
        print("[ERROR] Failed to create heatmap from HDF5:", e)
        return False

def make_composite(schematic_path: str, heatmap_path: str, train_loss_path: str, val_acc_path: str, outpath: str):
    """Compose schematic + heatmap + training thumbnails into one image."""
    import matplotlib.pyplot as plt
    import os
    fig = plt.figure(figsize=(12,6))

    # left: schematic (60% width)
    ax1 = fig.add_axes([0.03, 0.08, 0.58, 0.84])
    ax1.axis('off')
    if schematic_path and os.path.exists(schematic_path):
        ax1.imshow(plt.imread(schematic_path))
    else:
        ax1.text(0.5, 0.5, "Schematic missing", ha='center', va='center')
    ax1.set_title("Model architecture", fontsize=12)

    # right top: heatmap
    ax2 = fig.add_axes([0.65, 0.55, 0.33, 0.4])
    ax2.axis('off')
    if heatmap_path and isinstance(heatmap_path, str) and os.path.exists(heatmap_path):
        ax2.imshow(plt.imread(heatmap_path))
    else:
        ax2.text(0.5, 0.5, "Window heatmap missing", ha='center', va='center')
    ax2.set_title("Example input window (10 × 100)", fontsize=10)

    # right bottom left: train loss
    ax3 = fig.add_axes([0.65, 0.28, 0.16, 0.2])
    ax3.axis('off')
    if train_loss_path and os.path.exists(train_loss_path):
        ax3.imshow(plt.imread(train_loss_path))
    else:
        ax3.text(0.5, 0.5, "train_loss.png missing", ha='center', va='center')
    ax3.set_title("Propeller LSTM loss", fontsize=9)

    # right bottom right: val acc
    ax4 = fig.add_axes([0.82, 0.28, 0.16, 0.2])
    ax4.axis('off')
    if val_acc_path and os.path.exists(val_acc_path):
        ax4.imshow(plt.imread(val_acc_path))
    else:
        ax4.text(0.5, 0.5, "val_acc.png missing", ha='center', va='center')
    ax4.set_title("Classifier val accuracy", fontsize=9)

    # footer caption
    fig.text(0.03, 0.02,
             "Input shape: channels × window_length = 10 × 100 | LSTM → regression (MSE) | CNN → classification (CE)",
             fontsize=9)
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved composite: {outpath}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', default='ml_dataset_v2.h5', help='HDF5 dataset path')
    p.add_argument('--train_loss', default='results_plots/train_loss.png', help='train loss plot path')
    p.add_argument('--val_acc', default='results_plots/val_acc.png', help='val acc plot path')
    p.add_argument('--outdir', default='figures', help='output directory')
    args = p.parse_args()

    ensure_dir(args.outdir)
    schematic_p = os.path.join(args.outdir, 'architecture_schematic.png')
    heatmap_p = os.path.join(args.outdir, 'window_heatmap.png')
    composite_p = os.path.join(args.outdir, 'model_slide_composite.png')

    # 1) schematic
    draw_schematic(schematic_p)

    # 2) heatmap (if possible)
    heat_ok = make_window_heatmap(args.h5, heatmap_p)

    # 3) composite (uses available files). pass empty string if heatmap not created
    make_composite(schematic_p, heatmap_p if heat_ok else "", args.train_loss, args.val_acc, composite_p)

    print("[DONE] All generated files are in:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
