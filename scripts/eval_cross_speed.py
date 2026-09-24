#!/usr/bin/env python3
"""
Cross-speed (leave-one-RPM-bin-out) robustness evaluation — sealed protocol.

TRUE unseen-speed evaluation: runs are grouped into RPM bins by their RUN-LEVEL
median RPM (window-level percentiles would let a run straddle bins and leak).
For each bin, a fresh model is trained FROM SCRATCH on all runs OUTSIDE the
bin (originals + seeded augmented copies, train-only) and evaluated on the
bin's ORIGINAL windows. Nothing from the held-out bin is used for training or
early stopping.

Zero-shot mode (no --retrain) evaluates an existing model per bin WITHOUT
retraining — informative, but must never be reported as unseen-speed TRAINING.

Usage:
  LOSO retraining (the honest number; ~4 full retrains):
    python3 scripts/eval_cross_speed.py --dataset ml_sealed.h5 --retrain \
        --output results/cross_speed_loso.json
  Zero-shot per-bin (clearly labeled as such):
    python3 scripts/eval_cross_speed.py --dataset ml_sealed.h5 \
        --model models/cnn_sealed.pth --output results/cross_speed_zeroshot.json
"""
import argparse
import ast
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))
from cnn_classifier import PaperCNN  # noqa: E402
from augment_dataset import augment_sample  # noqa: E402

SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


class ArrayDataset(Dataset):
    def __init__(self, X, y, mean, std):
        self.X = X.astype("float32", copy=False)
        self.y = y.astype("int64")
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.mean is not None:
            x = (x - self.mean) / (self.std + 1e-9)
        return x, int(self.y[i])


def parse_h5_meta(f):
    raw = f.attrs.get("meta", "{}")
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return {}


def load_originals(path):
    """Load ONLY original windows (is_aug == 0) from the sealed h5.

    Reads X sequentially (fast for gzip datasets; fancy-indexed h5py reads
    decompress one chunk per row and are orders of magnitude slower), then
    masks in memory.
    """
    with h5py.File(path, "r") as f:
        if "is_aug" not in f or "run_id" not in f:
            raise SystemExit("dataset must be ml_sealed.h5 (needs run_id + is_aug); "
                             "build it with scripts/build_sealed_dataset.py")
        meta = parse_h5_meta(f)
        mask = f["is_aug"][:] == 0
        X_all = f["X"][:]
        X = X_all[mask]
        del X_all
        y = f["y_fault"][:][mask]
        run_id = f["run_id"][:][mask]
    return X, y, run_id, meta


def run_level_rpm_bins(X, run_id, n_bins):
    """Assign every RUN to an RPM bin by its median RPM (rpm1-4 channels)."""
    rpm = X[:, 0, :4, :].mean(axis=(1, 2))            # per-window mean RPM
    run_ids = np.unique(run_id)
    run_med = {int(r): float(np.median(rpm[run_id == r])) for r in run_ids}
    medians = np.array([run_med[int(r)] for r in run_ids])
    edges = np.quantile(medians, np.linspace(0, 1, n_bins + 1))
    # make edges strictly increasing (identical hover speeds across runs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    run_bin = {int(r): int(np.searchsorted(edges, run_med[int(r)], side="right") - 1)
               for r in run_ids}
    for r in run_bin:                                # clamp
        run_bin[r] = min(max(run_bin[r], 0), n_bins - 1)
    bin_of_window = np.array([run_bin[int(r)] for r in run_id], dtype="int64")
    bin_ranges = [(float(edges[i]), float(edges[i + 1])) for i in range(n_bins)]
    return run_bin, bin_of_window, bin_ranges


def make_fold_data(X, y, run_id, fold_bin, bin_of_window, aug_copies, fold_seed, vars_meta):
    """Train data = non-bin runs (originals + seeded augs); val = grouped run holdout."""
    rng = np.random.RandomState(fold_seed)
    np.random.seed(fold_seed)          # augment_sample uses the global RNG
    torch.manual_seed(fold_seed)

    train_mask = bin_of_window != fold_bin
    tr_orig = np.where(train_mask)[0]

    # grouped run-level val holdout from the fold's training runs
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=fold_seed)
    tr_rel, val_rel = next(gss.split(tr_orig, y[tr_orig], groups=run_id[tr_orig]))
    tr_idx, val_idx = tr_orig[tr_rel], tr_orig[val_rel]
    val_runs = set(np.unique(run_id[val_idx]).tolist())

    # augmented copies of TRAINING runs only (post-split, seeded, fold-local).
    # Preallocate the output array and fill it in chunks: building copies via
    # np.stack/np.concatenate peaks at several extra GB and OOMs laptops.
    aug_idx = [i for i in tr_idx if int(run_id[i]) not in val_runs]
    n_aug_rows = len(aug_idx) * aug_copies
    X_tr = np.empty((len(tr_idx) + n_aug_rows,) + X.shape[1:], dtype="float32")
    X_tr[:len(tr_idx)] = X[tr_idx]
    out_row = len(tr_idx)
    chunk = 4096
    for copy_i in range(aug_copies):
        for s in range(0, len(aug_idx), chunk):
            block = aug_idx[s:s + chunk]
            for j, i in enumerate(block):
                X_tr[out_row + j, 0] = augment_sample(
                    X[i][0].astype("float64"), vars_meta).astype("float32")
            out_row += len(block)
    y_tr = np.empty(len(tr_idx) + n_aug_rows, dtype=y.dtype)
    y_tr[:len(tr_idx)] = y[tr_idx]
    for copy_i in range(aug_copies):
        y_tr[len(tr_idx) + copy_i * len(aug_idx):
             len(tr_idx) + (copy_i + 1) * len(aug_idx)] = y[aug_idx]
    return X_tr, y_tr, X[val_idx], y[val_idx], len(tr_idx), len(val_idx)


def train_fold(X_tr, y_tr, X_val, y_val, n_faults, device, epochs, batch_size, patience):
    mean = X_tr.mean(axis=(0, 1, 3), keepdims=True).astype("float32")   # (1,1,C,1)
    std = X_tr.std(axis=(0, 1, 3), keepdims=True).astype("float32")
    tr_loader = DataLoader(ArrayDataset(X_tr, y_tr, mean, std), batch_size=batch_size,
                           shuffle=True, num_workers=2)
    val_loader = DataLoader(ArrayDataset(X_val, y_val, mean, std), batch_size=batch_size,
                            shuffle=False, num_workers=2)
    model = PaperCNN(in_channels=1, base_filters=32, num_classes=n_faults).to(device)
    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    best_val, no_improve, best_state = 0.0, 0, None
    hist = {"train_loss": [], "val_acc": []}
    for ep in range(epochs):
        model.train()
        tot, cnt = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            if xb.dim() == 5 and xb.size(2) == 1:
                xb = xb.squeeze(2)
            if xb.dim() == 3:
                xb = xb.unsqueeze(1)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); cnt += 1
        model.eval()
        correct, total, vloss, vcnt = 0, 0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                if xb.dim() == 5 and xb.size(2) == 1:
                    xb = xb.squeeze(2)
                if xb.dim() == 3:
                    xb = xb.unsqueeze(1)
                pf = model(xb)
                correct += int((pf.argmax(1) == yb).sum().item()); total += len(yb)
                vloss += float(loss_fn(pf, yb).item()); vcnt += 1
        val_acc = correct / total if total else 0.0
        hist["train_loss"].append(tot / cnt if cnt else 0.0)
        hist["val_acc"].append(val_acc)
        sched.step(vloss / vcnt if vcnt else 0.0)
        if val_acc > best_val:
            best_val, no_improve = val_acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std, best_val, hist


def predict(model, X, mean, std, device, batch_size=1024):
    preds = np.empty(len(X), dtype="int64")
    model.eval()
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = X[s:s + batch_size].astype("float32")
            xb = (xb - mean) / (std + 1e-9)
            inp = torch.from_numpy(xb)
            if inp.dim() == 3:
                inp = inp.unsqueeze(1)
            preds[s:s + batch_size] = model(inp.to(device)).argmax(1).cpu().numpy()
    return preds


def eval_bin(y_true, y_pred, run_id_bin):
    acc = float(accuracy_score(y_true, y_pred))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_run = {}
    for r in np.unique(run_id_bin):
        m = run_id_bin == r
        per_run[int(r)] = float((y_pred[m] == y_true[m]).mean())
    return acc, bal, f1m, per_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ml_sealed.h5")
    p.add_argument("--model", default=None,
                   help="pretrained model for ZERO-SHOT mode (ignored with --retrain)")
    p.add_argument("--retrain", action="store_true",
                   help="TRUE leave-one-RPM-bin-out: fresh training per bin")
    p.add_argument("--n-speed-bins", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30,
                   help="epoch cap per LOSO retrain (compute budget)")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--aug-copies", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models-out", default="models/loso")
    p.add_argument("--output", default="results/cross_speed_loso.json")
    args = p.parse_args()

    if not args.retrain and not args.model:
        raise SystemExit("need --model (zero-shot) or --retrain (true LOSO)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading originals from {args.dataset} …", flush=True)
    X, y, run_id, h5_meta = load_originals(args.dataset)
    n_faults = len(h5_meta.get("fault_label_map", {})) or int(y.max()) + 1
    n_runs = len(np.unique(run_id))
    print(f"{len(X)} original windows over {n_runs} runs, {n_faults} classes", flush=True)

    run_bin, bin_of_window, bin_ranges = run_level_rpm_bins(X, run_id, args.n_speed_bins)
    bins_summary = {}
    for b, (lo, hi) in enumerate(bin_ranges):
        runs_in_b = sorted(int(r) for r, bb in run_bin.items() if bb == b)
        bins_summary[b] = {"rpm_range": [lo, hi], "n_runs": len(runs_in_b),
                           "runs": runs_in_b,
                           "n_windows": int((bin_of_window == b).sum())}
        print(f"  bin {b}: {lo:.0f}-{hi:.0f} RPM, {len(runs_in_b)} runs, "
              f"{bins_summary[b]['n_windows']} windows")

    mode = "loso-retrain" if args.retrain else "zero-shot"
    results = {}
    if args.retrain:
        os.makedirs(args.models_out, exist_ok=True)

    zero_model, z_mean, z_std = None, None, None
    if not args.retrain:
        ck = torch.load(args.model, map_location="cpu")
        m = ck.get("meta", {}) if isinstance(ck, dict) else {}
        sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
        zero_model = PaperCNN(in_channels=1, base_filters=int(m.get("base_filters", 32)),
                              num_classes=int(m.get("n_faults", n_faults)))
        zero_model.load_state_dict(sd)
        zero_model.to(device).eval()
        z_mean = np.array(m.get("mean"), dtype="float32") if m.get("mean") is not None else None
        z_std = np.array(m.get("std"), dtype="float32") if m.get("std") is not None else None

    for b in range(args.n_speed_bins):
        lo, hi = bin_ranges[b]
        te_mask = bin_of_window == b
        te_idx = np.where(te_mask)[0]
        print(f"\n=== bin {b}: {lo:.0f}-{hi:.0f} RPM | "
              f"{len(bins_summary[b]['runs'])} test runs, {len(te_idx)} test windows ===",
              flush=True)
        if args.retrain:
            X_tr, y_tr, X_val, y_val, n_tr, n_va = make_fold_data(
                X, y, run_id, b, bin_of_window, args.aug_copies,
                fold_seed=args.seed + 100 * b,
                vars_meta=h5_meta.get("vars", []))
            print(f"  fold train: {len(X_tr)} windows ({n_tr} orig + augs), "
                  f"fold val: {n_va} windows — bin {b} fully held out", flush=True)
            model, mean, std, best_val, hist = train_fold(
                X_tr, y_tr, X_val, y_val, n_faults, device,
                epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)
            mpath = os.path.join(args.models_out, f"bin{b}.pth")
            torch.save({"state_dict": model.state_dict(),
                        "meta": {"n_faults": n_faults, "mean": mean.tolist(),
                                 "std": std.tolist(), "protocol": "loso-retrain",
                                 "held_out_bin": b,
                                 "rpm_range": [lo, hi], "seed": args.seed + 100 * b,
                                 "fault_label_map": h5_meta.get("fault_label_map", {}),
                                 "vars": h5_meta.get("vars", [])}}, mpath)
            print(f"  retrained (best fold-val acc {best_val:.4f}) -> {mpath}", flush=True)
            preds = predict(model, X[te_idx], mean, std, device)
            del X_tr, model
            torch.cuda.empty_cache() if device == "cuda" else None
        else:
            preds = predict(zero_model, X[te_idx], z_mean, z_std, device)

        acc, bal, f1m, per_run = eval_bin(y[te_idx], preds, run_id[te_idx])
        run_accs = list(per_run.values())
        results[f"bin_{b}"] = {
            "mode": mode,
            "rpm_range": [lo, hi],
            "n_test_runs": len(bins_summary[b]["runs"]),
            "n_test_windows": int(len(te_idx)),
            "window_accuracy": acc,
            "balanced_accuracy": bal,
            "macro_f1": f1m,
            "per_run_accuracy": per_run,
            "per_run_accuracy_mean": float(np.mean(run_accs)),
            "per_run_accuracy_min": float(np.min(run_accs)),
        }
        print(f"  window_acc={acc:.4f} balanced_acc={bal:.4f} macro_f1={f1m:.4f} "
              f"per-run mean={np.mean(run_accs):.4f}", flush=True)

    accs = [v["window_accuracy"] for v in results.values()]
    bals = [v["balanced_accuracy"] for v in results.values()]
    summary = {"mode": mode,
               "n_bins": args.n_speed_bins,
               "window_accuracy_mean": float(np.mean(accs)),
               "window_accuracy_std": float(np.std(accs)),
               "window_accuracy_min": float(np.min(accs)),
               "balanced_accuracy_mean": float(np.mean(bals)),
               "balanced_accuracy_min": float(np.min(bals))}
    print("\n" + "=" * 60)
    print(f"{mode.upper()} SUMMARY over {args.n_speed_bins} RPM bins:")
    print(f"  window acc: mean={summary['window_accuracy_mean']:.4f} "
          f"std={summary['window_accuracy_std']:.4f} min={summary['window_accuracy_min']:.4f}")
    print(f"  balanced acc: mean={summary['balanced_accuracy_mean']:.4f} "
          f"min={summary['balanced_accuracy_min']:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"mode": mode, "dataset": args.dataset, "seed": args.seed,
                   "bins": {str(b): {k: v for k, v in bins_summary[b].items()}
                            for b in bins_summary},
                   "config": {"epochs_cap": args.epochs, "patience": args.patience,
                              "batch_size": args.batch_size,
                              "aug_copies": args.aug_copies,
                              "rpm_bins": args.n_speed_bins},
                   "summary": summary, "results": results}, f, indent=2)
    print(f"Results saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
