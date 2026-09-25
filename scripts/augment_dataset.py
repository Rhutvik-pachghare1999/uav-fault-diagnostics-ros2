import argparse
import h5py
import numpy as np
import json


def augment_sample(x, vars_meta):
    # x shape: (C,W)
    C, W = x.shape
    y = x.copy()
    # vars_meta is list of variable names; determine which indices are RPMs and attitudes
    rpm_idx = [i for i, v in enumerate(vars_meta) if v.startswith("rpm")]
    att_idx = []
    gyro_idx = []
    for i, v in enumerate(vars_meta):
        if v in ("roll", "pitch", "yaw"):
            att_idx.append(i)
        if v.startswith("gyro") or v.endswith("_rate"):
            gyro_idx.append(i)

    # 1) Gaussian noise: small std relative to variable scale
    chan_std = y.std(axis=1, keepdims=True)
    noise = np.random.randn(C, W) * (0.02 * (chan_std + 1e-6))
    y = y + noise

    # 2) RPM scaling: multiply rpm channels by 0.96-1.04
    if rpm_idx:
        scale = np.random.uniform(0.96, 1.04)
        for i in rpm_idx:
            y[i] = y[i] * scale

    # 3) low-frequency drift on attitude channels (simulate bias/CG offset)
    for i in att_idx:
        drift = np.linspace(0, np.random.uniform(-0.02, 0.02), W)
        y[i] = y[i] + drift

    # 4) small circular time-shift (jitter) up to +/-3 samples
    shift = np.random.randint(-3, 4)
    if shift != 0:
        y = np.roll(y, shift, axis=1)

    return y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_h5", required=True)
    p.add_argument("--out", dest="out_h5", required=True)
    p.add_argument("--times", type=int, default=5, help="Augmentation multiplier")
    args = p.parse_args()

    with h5py.File(args.in_h5, "r") as f:
        X = f["X"][:]  # (N,1,C,W)
        y = f["y_fault"][:]
        ur = f["ur"][:] if "ur" in f else np.zeros(len(X))
        run_id = f["run_id"][:] if "run_id" in f else None
        y_sev = f["y_sev"][:] if "y_sev" in f else None
        meta = json.loads(f.attrs.get("meta", "{}"))
        vars_meta = meta.get("vars", [])

    N = X.shape[0]
    print(f"Loaded {args.in_h5} samples={N}, vars={vars_meta}")

    new_N = N * args.times
    out_X = np.zeros((new_N, 1, X.shape[2], X.shape[3]), dtype="float32")
    out_y = np.zeros((new_N,), dtype="int64")
    out_ur = np.zeros((new_N,), dtype="float32")
    out_run = np.zeros((new_N,), dtype="int64") if run_id is not None else None
    out_sev = np.zeros((new_N,), dtype="int64") if y_sev is not None else None

    idx = 0
    for i in range(N):
        orig = X[i, 0]
        label = int(y[i])
        urv = float(ur[i])
        # include original and augmented versions
        for t in range(args.times):
            if t == 0:
                aug = orig
            else:
                aug = augment_sample(orig.copy(), vars_meta or [])
            out_X[idx, 0] = aug
            out_y[idx] = label
            out_ur[idx] = urv
            if out_run is not None:
                out_run[idx] = run_id[i]
            if out_sev is not None:
                out_sev[idx] = y_sev[i]
            idx += 1

    print("Writing", args.out_h5, "samples=", out_X.shape[0])
    with h5py.File(args.out_h5, "w") as f:
        f.create_dataset("X", data=out_X, compression="gzip")
        f.create_dataset("y_fault", data=out_y)
        f.create_dataset("ur", data=out_ur)
        if out_run is not None:
            # propagate run ids so augmented copies stay grouped with their
            # source run in run-grouped train/test splits (prevents leakage)
            f.create_dataset("run_id", data=out_run)
        if out_sev is not None:
            f.create_dataset("y_sev", data=out_sev)
        meta["augmented_from"] = args.in_h5
        meta["aug_times"] = args.times
        f.attrs["meta"] = json.dumps(meta)

    print("Done")


if __name__ == "__main__":
    main()
