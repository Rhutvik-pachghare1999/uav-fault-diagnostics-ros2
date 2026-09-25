"""Generate synthetic run folders with labeled faults by perturbing RPMs and attitudes.

Creates multiple `isaac_dataset/run_synth_<label>_<i>/` directories by copying a base run
and modifying `imu.csv` values to simulate faults. Adds `meta.json` with `fault_type`
so the dataset builder will create multiple classes.

Usage:
  python3 scripts/generate_synthetic_runs.py --base isaac_dataset/run_auto_1764651130 --n-per-label 2
"""

import argparse
import os
import shutil
import json
import numpy as np
import pandas as pd


def perturb_run(base_run, out_dir, label_mask):
    # base_run: path to existing run dir; out_dir: target new run dir; label_mask: 0..15 bitmask
    os.makedirs(out_dir, exist_ok=True)
    # copy state.csv and other files if exist
    for name in ("state.csv", "thrust_tau.csv"):
        src = os.path.join(base_run, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, name))
    # copy imu csv (use imu_full.csv if present)
    imu_src = None
    for name in ("imu.csv", "imu_full.csv"):
        p = os.path.join(base_run, name)
        if os.path.exists(p):
            imu_src = p
            break
    if imu_src is None:
        print(f"Warning: Base run has no imu file: {base_run}")
        return False
    df = pd.read_csv(imu_src)
    # perturb RPM columns: rpm1..rpm4
    for i in range(1, 5):
        col = f"rpm{i}"
        if col not in df.columns:
            df[col] = 0.0
    # apply per-prop perturbation depending on label mask
    for prop_idx in range(4):
        col = f"rpm{prop_idx + 1}"
        if (label_mask >> prop_idx) & 1:
            # faulty prop: reduce rpm by a factor between 0.6 and 0.95, and add noise
            factor = np.random.uniform(0.6, 0.95)
            df[col] = df[col] * factor * (1 + np.random.normal(0, 0.01, size=len(df)))
        else:
            # healthy prop: small variation
            df[col] = df[col] * (1 + np.random.normal(0, 0.005, size=len(df)))
    # small attitude drift to simulate CG offset
    for a in ("roll", "pitch", "yaw"):
        if a in df.columns:
            df[a] = df[a] + np.random.normal(0, 0.001, size=len(df))
    # write imu.csv in out_dir
    out_imu = os.path.join(out_dir, "imu.csv")
    df.to_csv(out_imu, index=False)
    # write meta.json with fault_type and fault_params
    fault_label = f"label_{label_mask}"
    meta = {"fault_type": fault_label, "fault_params": {"unbalance": float(np.random.uniform(0.05, 0.6))}}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Path to existing run to copy")
    p.add_argument("--n-per-label", type=int, default=2)
    p.add_argument("--out-prefix", default="run_synth")
    args = p.parse_args()

    base = args.base
    if not os.path.isdir(base):
        print("Base run not found:", base)
        return
    # write into the run directory sibling of the base run
    isaac_root = os.path.join(os.path.dirname(base))
    # generate runs
    created = []
    for mask in range(1, 16):  # Skip mask=0 (same as healthy)
        for i in range(args.n_per_label):
            run_name = f"{args.out_prefix}_{mask:02d}_{i}"
            run_dir = os.path.join(isaac_root, run_name)
            print("Creating", run_dir)
            ok = perturb_run(base, run_dir, mask)
            if ok is False:
                print("Skipping creation of", run_dir, "due to missing base imu")
                continue
            created.append(run_dir)
    print("Created", len(created), "synthetic runs")


if __name__ == "__main__":
    main()
