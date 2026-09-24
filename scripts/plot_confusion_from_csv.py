#!/usr/bin/env python3
"""
plot_confusion_fixed.py

Usage:
  python plot_confusion_fixed.py --csv /path/to/confusion.csv --outdir results

This script robustly reads common CSV layouts and plots an annotated confusion matrix PNG.
"""
import argparse
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_matrix(csv_path: Path):
    # Try reading with index_col=0 first, then without
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except Exception:
        df = pd.read_csv(csv_path)

    # If first column looks like labels, ensure it's the index
    if df.shape[1] >= 2:
        first_col = df.columns[0]
        # if first column non-numeric while others numeric, set as index
        first_col_is_non_numeric = df[first_col].apply(lambda v: pd.to_numeric(v, errors="coerce")).isna().any()
        other_cols_numeric = df.iloc[:, 1:].apply(lambda col: pd.to_numeric(col, errors="coerce").notna().all()).sum() >= 1
        if first_col_is_non_numeric and other_cols_numeric:
            df = df.set_index(first_col)

    # Coerce whole frame to numeric (column-wise) with pd.to_numeric (no shadowing)
    coerced = df.apply(pd.to_numeric, errors="coerce")
    # Replace NaN (non-numeric) with 0. Warn if any
    n_non_numeric = int(coerced.isna().sum().sum())
    if n_non_numeric:
        print(f"Warning: found {n_non_numeric} non-numeric cells — coercing them to 0.")
        coerced = coerced.fillna(0.0)

    mat = coerced.values.astype(float)

    # Determine labels: prefer index, then columns, otherwise integer labels
    if coerced.index is not None and len(coerced.index) == mat.shape[0]:
        labels = [str(x) for x in coerced.index.tolist()]
    elif len(coerced.columns) == mat.shape[1]:
        labels = [str(x) for x in coerced.columns.tolist()]
    else:
        labels = [str(i) for i in range(mat.shape[0])]

    return labels, mat

def ensure_square_and_fix(labels, mat):
    r, c = mat.shape
    if r == c:
        return labels, mat
    # Try trimming extras (common when totals row/col included)
    if r > c:
        print(f"Warning: matrix {r}x{c} -> trimming rows to {c}")
        mat = mat[:c, :]
        labels = labels[:c]
        return labels, mat
    if c > r:
        print(f"Warning: matrix {r}x{c} -> trimming cols to {r}")
        mat = mat[:, :r]
        labels = labels[:mat.shape[0]]
        return labels, mat
    raise RuntimeError("Cannot coerce to square matrix.")

def plot_and_save(labels, mat, outpath: Path, title=None):
    int_mat = np.rint(mat).astype(int)
    figsize = (max(6, len(labels)*0.5), max(6, len(labels)*0.5))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(int_mat, cmap="Blues", interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)

    vmax = int_mat.max() if int_mat.size else 0
    for i in range(int_mat.shape[0]):
        for j in range(int_mat.shape[1]):
            val = int_mat[i, j]
            if val == 0:
                continue
            color = "white" if val > vmax/2 else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(str(outpath), dpi=200)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV path")
    parser.add_argument("--outdir", default="results", help="Output dir")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        labels, mat = read_matrix(csv_path)
    except Exception as e:
        print("ERROR reading CSV:", e)
        sys.exit(1)

    try:
        labels, mat = ensure_square_and_fix(labels, mat)
    except Exception as e:
        print("ERROR ensuring square matrix:", e)
        sys.exit(1)

    out_png = outdir / "confusion_matrix_corrected.png"
    plot_and_save(labels, mat, out_png, title=f"Confusion matrix — {csv_path.name}")
    print("Saved", out_png)

if __name__ == "__main__":
    main()
