# scripts/build_ml_dataset_v2.py

import os
import glob
import h5py
import json
import argparse
import numpy as np
import pandas as pd
from severity_utils import severity_from_ur

# Per paper: 10 variables: 4 RPMs, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate
DEFAULT_VARS = ["rpm1", "rpm2", "rpm3", "rpm4", "roll", "pitch", "yaw", "gyro_x", "gyro_y", "gyro_z"]


def read_run(run_dir, window=100, step=1, vars=DEFAULT_VARS, allow_missing_vars=False, allow_missing_runs=False):
    # accept either imu.csv or imu_full.csv (some recorders write imu_full.csv)
    imu_csv = None
    for name in ("imu.csv", "imu_full.csv"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            imu_csv = p
            break
    meta_json = os.path.join(run_dir, "meta.json")
    if imu_csv is None:
        if allow_missing_runs:
            print(f"WARNING: no imu.csv/imu_full.csv in {run_dir}; skipping run")
            return [], []
        raise FileNotFoundError(
            f"no imu.csv/imu_full.csv in {run_dir}. Every run directory must "
            f"contain IMU data; a silently skipped run silently shrinks the "
            f"dataset and corrupts splits. Pass --allow-missing-runs to skip."
        )
    df = pd.read_csv(imu_csv)
    # If state.csv exists, merge columns (e.g., roll,pitch,yaw) from it to ensure rates/angles present
    state_csv = os.path.join(run_dir, "state.csv")
    if os.path.exists(state_csv):
        try:
            sdf = pd.read_csv(state_csv)
            # prefer state columns when present (e.g., roll,pitch,yaw)
            for c in ["roll", "pitch", "yaw", "roll_rate", "pitch_rate", "yaw_rate"]:
                if c in sdf.columns:
                    df[c] = sdf[c]
        except Exception:
            pass
    missing = [v for v in vars if v not in df.columns]
    if missing:
        if allow_missing_vars:
            print(f"WARNING: {run_dir}: zero-filling missing vars {missing} (model will see dead channels)")
            for v in missing:
                df[v] = 0.0
        else:
            raise ValueError(
                f"{run_dir}: requested vars missing from imu data: {missing}. "
                f"The old behaviour silently zero-filled them, training the "
                f"model on dead channels. Record the channels or pass "
                f"--allow-missing-vars."
            )
    data = df[vars].values
    Xs, metas = [], []
    T = data.shape[0]
    for i in range(0, T - window + 1, step):
        win = data[i : i + window].T  # (C, W)
        Xs.append(win)
        metas.append(i)
    if os.path.exists(meta_json):
        meta = json.load(open(meta_json, "r"))
    else:
        meta = {}
    ur = None
    if "fault_params" in meta:
        ur = meta["fault_params"].get("unbalance", meta["fault_params"].get("ur", None))
    sev = severity_from_ur(float(ur) if ur is not None else 0.0)
    fault_label = meta.get("fault_type", "unknown")
    return Xs, [{"fault_label": fault_label, "severity": sev, "ur": ur, "meta": meta} for _ in Xs]


def build_dataset(
    project_root: str,
    out_h5: str,
    window: int = 100,
    step: int = 1,
    vars=DEFAULT_VARS,
    allow_missing_vars=False,
    allow_missing_runs=False,
):
    runs = sorted(glob.glob(os.path.join(project_root, "isaac_dataset", "run_*")))
    print("Found runs:", len(runs))
    if not runs:
        raise SystemExit(
            f"no run_* directories under {project_root}/isaac_dataset — nothing "
            f"to build. Generate runs first (see README / reproduce.sh)."
        )
    all_X, all_fault_labels, all_sev, all_ur, all_run = [], [], [], [], []
    fault_label_map = {}
    next_label = 0
    skipped = []
    for run_id, r in enumerate(runs):
        Xs, metas = read_run(
            r,
            window=window,
            step=step,
            vars=vars,
            allow_missing_vars=allow_missing_vars,
            allow_missing_runs=allow_missing_runs,
        )
        if not Xs:
            skipped.append(os.path.basename(r))
            continue
        for x, m in zip(Xs, metas):
            fl = m.get("fault_label") or "unknown"
            if fl not in fault_label_map:
                fault_label_map[fl] = next_label
                next_label += 1
            all_X.append(x)
            all_fault_labels.append(fault_label_map[fl])
            all_sev.append(int(m.get("severity", 0)))
            all_ur.append(float(m.get("ur") or 0.0))
            all_run.append(run_id)  # which physical run this window came from
    if skipped:
        print(f"WARNING: skipped {len(skipped)} runs: {skipped}")
    if not all_X:
        # No windows found despite run directories existing: data-integrity
        # error. Fail loudly — a graceful no-op lets orchestration scripts
        # "succeed" while producing no dataset at all.
        raise SystemExit("No windows found in any run. Check ISAAC dataset directory and imu.csv placement.")
    X = np.stack(all_X, axis=0).astype("float32")  # (N, C, W)
    X = X[:, None, :, :]  # (N,1,C,W)
    y_fault = np.array(all_fault_labels, dtype="int64")
    y_sev = np.array(all_sev, dtype="int64")
    ur_arr = np.array(all_ur, dtype="float32")
    run_ids = np.array(all_run, dtype="int64")

    # NOTE: augmentation is intentionally NOT done here. Duplicating/perturbing
    # windows before the train/test split would place near-duplicates of the same
    # window on both sides of the split (leakage). Augmentation must happen AFTER
    # a run-grouped split, on the training partition only — see train_cnn.py.
    os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("X", data=X, compression="gzip")
        f.create_dataset("y_fault", data=y_fault)
        f.create_dataset("y_sev", data=y_sev)
        f.create_dataset("ur", data=ur_arr)
        f.create_dataset("run_id", data=run_ids)  # for run-grouped splitting
        f.attrs["meta"] = json.dumps({"fault_label_map": fault_label_map, "window": window, "vars": vars})
    print("Wrote", out_h5, "samples=", len(X), "runs=", int(run_ids.max()) + 1 if len(run_ids) else 0)
    return out_h5


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--out", type=str, default="ml_dataset_v2.h5")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--step", type=int, default=1)
    p.add_argument(
        "--vars", type=str, default=None, help="comma-separated variable list to include (overrides defaults)"
    )
    p.add_argument(
        "--allow-missing-vars",
        action="store_true",
        help="zero-fill requested vars missing from a run's imu data (loud WARNING per run)",
    )
    p.add_argument(
        "--allow-missing-runs",
        action="store_true",
        help="skip run directories without imu.csv/imu_full.csv instead of failing",
    )
    args = p.parse_args()
    if args.vars:
        var_list = [v.strip() for v in args.vars.split(",") if v.strip()]
    else:
        var_list = DEFAULT_VARS
    build_dataset(
        args.project_root,
        out_h5=args.out,
        window=args.window,
        step=args.step,
        vars=var_list,
        allow_missing_vars=args.allow_missing_vars,
        allow_missing_runs=args.allow_missing_runs,
    )
