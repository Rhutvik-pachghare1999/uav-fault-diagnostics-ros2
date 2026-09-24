#!/usr/bin/env python3
"""
Build the SEALED dataset (protocol v2) from the frozen run-level split manifest.

Reads the original-window dataset (ml_dataset_v2.h5 — every window keeps its
run_id) and writes ml_sealed.h5:

  * ALL 96 runs' original windows  -> is_aug = 0, split = train/val/test
  * seeded augmented copies (x2)   -> TRAIN runs ONLY, is_aug = 1, split = train

val/test windows are never augmented and exist in the file exactly once, so a
sealed evaluation can never touch a training-augmented copy.

Extra columns (vs the legacy builder):
  is_aug  - 0 original / 1 augmented copy
  split   - 0 train / 1 val / 2 test (from the manifest, never recomputed)

Usage:
  python3 scripts/build_sealed_dataset.py --h5 ml_dataset_v2.h5 \
      --manifest splits/split_manifest.json --out ml_sealed.h5
"""
import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment_dataset import augment_sample  # noqa: E402

SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", required=True, help="original-window dataset (with run_id)")
    ap.add_argument("--manifest", required=True, help="frozen split manifest JSON")
    ap.add_argument("--out", default="ml_sealed.h5")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--aug-seed", type=int, default=42)
    ap.add_argument("--aug-copies", type=int, default=2,
                    help="augmented copies per TRAIN window (total x3 with original)")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    # run_id in the source h5 was assigned by sorted(glob(...)) order in
    # build_ml_dataset_v2.py; replicate it exactly to map ids -> run names.
    run_names = [os.path.basename(p)
                 for p in sorted(glob.glob(os.path.join(args.project_root,
                                                        "isaac_dataset", "run_*")))]
    if len(run_names) != 96:
        raise SystemExit(f"expected 96 run dirs, found {len(run_names)}")
    run_split = {}
    for part, names in (("train", manifest["train"]), ("val", manifest["val"]),
                        ("test", manifest["test"])):
        for n in names:
            run_split[n] = SPLIT_IDS[part]
    missing = set(run_names) - set(run_split)
    if missing:
        raise SystemExit(f"manifest does not cover runs: {sorted(missing)}")

    with h5py.File(args.h5, "r") as f:
        meta_raw = f.attrs.get("meta", "{}")
        if isinstance(meta_raw, (bytes, bytearray)):
            meta_raw = meta_raw.decode("utf-8", errors="ignore")
        src_meta = json.loads(meta_raw)
        X = f["X"][:]                      # (N,1,C,W)
        y_fault = f["y_fault"][:]
        y_sev = f["y_sev"][:] if "y_sev" in f else np.zeros(len(X), dtype="int64")
        ur = f["ur"][:] if "ur" in f else np.zeros(len(X), dtype="float32")
        run_id = f["run_id"][:]
        fault_label_map = src_meta.get("fault_label_map", {})

    n_orig = len(X)
    id_to_name = {i: run_names[i] for i in range(len(run_names))}
    uniq = np.unique(run_id)
    if len(uniq) != len(run_names) or max(uniq) >= len(run_names):
        raise SystemExit("run_id layout does not match sorted run dirs")

    split_arr = np.array([run_split[id_to_name[int(r)]] for r in run_id], dtype="int8")

    # cross-check labels per run against the manifest class (fault_label_map)
    for rid in list(uniq[:5]) + list(uniq[-5:]):
        name = id_to_name[int(rid)]
        cls = manifest["runs"][name]["class"]
        labels = set(y_fault[run_id == rid].tolist())
        if labels != {fault_label_map[cls]}:
            raise SystemExit(f"run {name}: h5 labels {labels} != manifest class {cls}")

    n_train_win = int((split_arr == 0).sum())
    n_out = n_orig + args.aug_copies * n_train_win
    print(f"source: {n_orig} windows over {len(run_names)} runs "
          f"({n_train_win} train-run windows)")
    print(f"output: {n_out} windows "
          f"({n_train_win} original + {args.aug_copies * n_train_win} aug of TRAIN runs only)")

    # augment_sample draws from the GLOBAL numpy RNG -> seed it for determinism
    np.random.seed(args.aug_seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # assemble everything in RAM, then write each dataset in one shot
    # (per-window h5 writes with gzip compression are orders of magnitude slower)
    X_out = np.empty((n_out,) + X.shape[1:], dtype="float32")
    yf_out = np.empty(n_out, dtype="int64")
    ys_out = np.empty(n_out, dtype="int64")
    ur_out = np.empty(n_out, dtype="float32")
    rid_out = np.empty(n_out, dtype="int64")
    aug_out = np.zeros(n_out, dtype="int8")
    split_out = np.empty(n_out, dtype="int8")

    vars_meta = src_meta.get("vars", [])
    pos = 0
    for rid in uniq:
        idxs = np.where(run_id == rid)[0]
        sp = run_split[id_to_name[int(rid)]]
        n_run = len(idxs)
        n_write = n_run * (args.aug_copies + 1 if sp == SPLIT_IDS["train"] else 1)
        sl = slice(pos, pos + n_write)
        block = np.repeat(X[idxs], 1, axis=0)          # originals first
        if sp == SPLIT_IDS["train"]:
            aug_blocks = [np.stack([augment_sample(X[i][0].astype("float64"),
                                                   vars_meta).astype("float32")
                                     for i in idxs])[:, None]   # (n,1,C,W)
                          for _ in range(args.aug_copies)]
            block = np.concatenate([block] + aug_blocks, axis=0)
            aug_out[pos + n_run:pos + n_write] = 1
        X_out[sl] = block
        yf_out[sl] = np.tile(y_fault[idxs], 1 if sp != SPLIT_IDS["train"]
                             else args.aug_copies + 1)
        ys_out[sl] = np.tile(y_sev[idxs], 1 if sp != SPLIT_IDS["train"]
                             else args.aug_copies + 1)
        ur_out[sl] = np.tile(ur[idxs], 1 if sp != SPLIT_IDS["train"]
                             else args.aug_copies + 1)
        rid_out[sl] = rid
        split_out[sl] = sp
        pos += n_write
    assert pos == n_out

    with h5py.File(args.out, "w") as f:
        f.create_dataset("X", data=X_out, compression="gzip")
        f.create_dataset("y_fault", data=yf_out)
        f.create_dataset("y_sev", data=ys_out)
        f.create_dataset("ur", data=ur_out)
        f.create_dataset("run_id", data=rid_out)
        f.create_dataset("is_aug", data=aug_out)
        f.create_dataset("split", data=split_out)
        out_meta = dict(src_meta)
        out_meta.update({
            "protocol": "sealed-run-grouped-v2",
            "manifest": os.path.basename(args.manifest),
            "manifest_sha256": manifest.get("sha256"),
            "aug_seed": args.aug_seed,
            "aug_copies": args.aug_copies,
            "split_encoding": SPLIT_IDS,
            "run_names": run_names,          # run_id -> run name mapping
            "augmentation_scope": "train runs only",
        })
        f.attrs["meta"] = json.dumps(out_meta)
    del X_out, block

    # verification pass
    with h5py.File(args.out, "r") as f:
        aug = f["is_aug"][:]
        sp = f["split"][:]
        n_bad = int(((aug == 1) & (sp != 0)).sum())
        per_split = {k: int((sp == v).sum()) for k, v in SPLIT_IDS.items()}
        n_runs = len(np.unique(f["run_id"][:]))
    if n_bad:
        raise SystemExit(f"LEAKAGE: {n_bad} augmented windows outside train split")
    print(f"verified: {n_out} windows, {n_runs} runs, per-split {per_split}, "
          f"aug-outside-train = {n_bad}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
