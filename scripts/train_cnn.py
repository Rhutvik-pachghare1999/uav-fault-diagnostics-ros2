# scripts/train_cnn.py
"""
Train the 2D-CNN multi-head classifier.

SEALED protocol (recommended):
  python3 scripts/train_cnn.py --h5 ml_sealed.h5 --manifest splits/split_manifest.json \
      --out models/cnn_sealed.pth --epochs 50 --history results/train_history_sealed.json

  Train/val/test come from the FROZEN run-level manifest carried in the h5's
  `split` column (0/1/2). The trainer trains on split==train (augmented copies
  included), early-stops on split==val ORIGINAL windows only, and NEVER loads
  split==test windows. The h5 split column is cross-checked against the
  manifest (run_id -> run name -> manifest part) before training starts.

Legacy mode (no --manifest): ephemeral in-process GroupShuffleSplit — metrics
from this mode are NOT sealed and must not be reported as held-out results.
"""

import argparse
import h5py
import numpy as np
import os
import json
import ast

try:
    from tqdm import tqdm
except ImportError:

    class tqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, *args, **kwargs):
            pass

        def set_description(self, *args, **kwargs):
            pass

        def close(self):
            pass


import logging
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def setup_logging(log_file="logs/train.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    return logging.getLogger()


class H5Dataset(Dataset):
    def __init__(self, X, yf, idxs, mean=None, std=None):
        self.X = X[idxs].astype("float32")
        self.yf = yf[idxs]
        # mean/std shape: (C,1) or (1,C,1) broadcasting OK
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-9)
        return x, int(self.yf[idx])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", "--data", required=True, help="Path to HDF5 dataset")
    p.add_argument("--out", default="models/cnn_sealed.pth", help="Output model path")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--base-filters", type=int, default=32)
    p.add_argument(
        "--manifest",
        default=None,
        help="frozen split manifest JSON; --h5 must carry split/is_aug columns (ml_sealed.h5). Sealed protocol v2.",
    )
    p.add_argument(
        "--no-manifest-check",
        action="store_true",
        help="use the h5 split column as-is without cross-checking the "
        "manifest (for LOSO fold datasets with fold-specific splits)",
    )
    p.add_argument(
        "--vars-subset",
        default=None,
        help="comma-separated channel subset to train on (e.g. IMU-only: "
        "'roll,pitch,yaw,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z')",
    )
    p.add_argument("--history", default="results/train_history_sealed.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="legacy", help="protocol tag recorded in the model meta")
    args = p.parse_args()

    logger = setup_logging()

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        from cnn_classifier import PaperCNN
    except Exception:
        print("Torch not installed in this environment. Install torch and retry.")
        raise

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with h5py.File(args.h5, "r") as f:
        X_all = f["X"][:]  # (N,1,C,W)
        run_id = f["run_id"][:] if "run_id" in f else None
        # use safe dataset read API
        y_fault = f["y_fault"][:]
        split_col = f["split"][:] if "split" in f else None
        is_aug = f["is_aug"][:] if "is_aug" in f else None
        # metadata may be bytes or str; parse safely (never eval untrusted data)
        meta_raw = f.attrs.get("meta", "{}")
        if isinstance(meta_raw, (bytes, bytearray)):
            meta_raw = meta_raw.decode("utf-8", errors="ignore")
        try:
            meta = json.loads(meta_raw)
        except Exception:
            try:
                meta = ast.literal_eval(meta_raw)  # safe: literals only
            except Exception:
                meta = {}
        n_faults = len(meta.get("fault_label_map", {})) or int(y_fault.max() + 1)

    # optional channel subset (e.g. IMU-only live-deployment model)
    effective_vars = list(meta.get("vars", []))
    if args.vars_subset:
        if not effective_vars:
            raise SystemExit("--vars-subset requires meta['vars'] in the h5")
        want = [v.strip() for v in args.vars_subset.split(",") if v.strip()]
        missing = [v for v in want if v not in effective_vars]
        if missing:
            raise SystemExit(f"--vars-subset unknown channels: {missing} (h5 has {effective_vars})")
        ch = [effective_vars.index(v) for v in want]
        X_all = np.ascontiguousarray(X_all[:, :, ch, :])
        effective_vars = want
        print(f"vars subset ({len(want)} ch): {want}")

    manifest_sha = None
    if args.manifest:
        # ---- SEALED PROTOCOL v2: split comes from the FROZEN manifest ----
        if split_col is None or is_aug is None:
            raise SystemExit(
                "--manifest requires an h5 with split/is_aug columns (build it with scripts/build_sealed_dataset.py)"
            )
        with open(args.manifest) as f:
            manifest = json.load(f)
        manifest_sha = manifest.get("sha256")
        if not args.no_manifest_check:
            # integrity guard: every ORIGINAL window's split must equal the
            # manifest partition of its run — catches stale/mutated datasets
            run_names = meta.get("run_names")
            if run_id is None or not run_names:
                raise SystemExit("manifest check requires run_id and meta['run_names']")
            id_to_part = {}
            for part, sid in (("train", 0), ("val", 1), ("test", 2)):
                for n in manifest[part]:
                    id_to_part[n] = sid
            orig_mask = is_aug == 0
            bad = []
            for r in np.unique(run_id[orig_mask]):
                name = run_names[int(r)]
                if name not in id_to_part:
                    raise SystemExit(f"run {name} missing from manifest")
                if not np.all(split_col[orig_mask][run_id[orig_mask] == r] == id_to_part[name]):
                    bad.append(name)
            if bad:
                raise SystemExit(f"LEAKAGE GUARD: h5 split disagrees with manifest for runs {bad}")
        tr_idx = np.where(split_col == 0)[0]  # originals + train-only augs
        val_idx = np.where((split_col == 1) & (is_aug == 0))[0]  # val originals, never augmented
        n_test_guard = int(((split_col == 2) & (is_aug == 0)).sum())
        if len(val_idx) == 0 or n_test_guard == 0:
            raise SystemExit("sealed h5 must contain non-empty val and test partitions")
        print(
            f"SEALED split from {os.path.basename(args.manifest)} "
            f"(sha256 {str(manifest_sha)[:12]}…): {len(tr_idx)} train / {len(val_idx)} val / "
            f"{n_test_guard} test windows (test NEVER loaded for training or early stop)"
        )
        args.tag = "sealed-run-grouped-v2"
    else:
        # ---- LEGACY MODE: ephemeral split, NOT sealed ----
        print(
            "WARNING: no --manifest — using ephemeral in-process splits. Metrics from "
            "this run are NOT sealed held-out results; do not report them as such."
        )
        idx = np.arange(len(X_all))
        if run_id is not None and len(np.unique(run_id)) >= 3:
            from sklearn.model_selection import GroupShuffleSplit

            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            tr_full, te_idx = next(gss.split(idx, y_fault, groups=run_id))
            tr_idx = idx[tr_full]
            te_idx = idx[te_idx]
            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
            tr_rel, val_rel = next(gss2.split(tr_idx, y_fault[tr_idx], groups=run_id[tr_idx]))
            tr_idx, val_idx = tr_idx[tr_rel], tr_idx[val_rel]
            print(
                f"Run-grouped split: {len(tr_idx)} train / {len(val_idx)} val / {len(te_idx)} test "
                f"windows over {len(np.unique(run_id))} runs (no run crosses partitions)."
            )
        else:
            print(
                "WARNING: run_id unavailable or <3 runs — falling back to stratified random "
                "split. Windows from the same run may leak across partitions; treat metrics as optimistic."
            )
            try:
                tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_fault)
                tr_idx, val_idx = train_test_split(tr_idx, test_size=0.125, random_state=42, stratify=y_fault[tr_idx])
            except Exception:
                tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=None)
                tr_idx, val_idx = train_test_split(tr_idx, test_size=0.125, random_state=42, stratify=None)

    # compute per-channel mean/std on training set for normalization
    X_tr = X_all[tr_idx].astype("float32")
    # X_tr shape: (N,1,C,W) -> compute mean/std per channel over samples and time
    # collapse sample and time dims to compute per-channel stats
    C = X_tr.shape[2]
    vals = X_tr.reshape(X_tr.shape[0], C, -1).transpose(1, 0, 2).reshape(C, -1)
    mean = vals.mean(axis=1).reshape(1, 1, C, 1)
    std = vals.std(axis=1).reshape(1, 1, C, 1)

    tr_ds = H5Dataset(X_all, y_fault, tr_idx, mean=mean, std=std)
    val_ds = H5Dataset(X_all, y_fault, val_idx, mean=mean, std=std)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = PaperCNN(in_channels=1, base_filters=args.base_filters, num_classes=n_faults).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    model_meta = {
        "n_faults": n_faults,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "fault_label_map": meta.get("fault_label_map", {}),
        "vars": effective_vars,  # channels the model ACTUALLY consumes
        "base_filters": args.base_filters,
        "protocol": args.tag,
        "manifest": os.path.basename(args.manifest) if args.manifest else None,
        "manifest_sha256": manifest_sha,
        "n_train_windows": int(len(tr_idx)),
        "n_val_windows": int(len(val_idx)),
        "seed": args.seed,
        "h5": os.path.basename(args.h5),
    }

    def save_ckpt(path, epoch=None):
        ckpt = {"state_dict": model.state_dict(), "meta": model_meta}
        if epoch is not None:
            ckpt["epoch"] = epoch
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(ckpt, path)

    best_val = 0.0
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        model.train()
        tot, cnt = 0.0, 0
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for xb, yf in pbar:
            xb = xb.to(device)
            yf = yf.to(device)
            # handle unexpected extra singleton dimension from collate
            if xb.dim() == 5 and xb.size(2) == 1:
                xb = xb.squeeze(2)
            if xb.dim() == 3:
                xb = xb.unsqueeze(1)
            pred = model(xb)
            loss = loss_fn(pred, yf)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            cnt += 1
            pbar.set_postfix(loss=f"{tot / cnt:.4f}")

        train_loss = tot / cnt if cnt > 0 else 0.0

        # val acc
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        val_cnt = 0
        with torch.no_grad():
            for xb, yf in val_loader:
                xb = xb.to(device)
                yf = yf.to(device)
                if xb.dim() == 5 and xb.size(2) == 1:
                    xb = xb.squeeze(2)
                if xb.dim() == 3:
                    xb = xb.unsqueeze(1)
                pf = model(xb)
                pred = pf.argmax(dim=1)
                correct += int((pred == yf).sum().item())
                total += len(yf)
                loss = loss_fn(pf, yf)
                val_loss += float(loss.item())
                val_cnt += 1

        val_acc = correct / total if total > 0 else 0.0
        val_loss_avg = (val_loss / val_cnt) if val_cnt else 0.0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss_avg)
        history["val_acc"].append(val_acc)
        msg = (
            f"Epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss_avg:.4f} val_acc={val_acc:.4f}"
        )
        logger.info(msg)

        scheduler.step(val_loss_avg)

        # checkpoint best
        if val_acc > best_val:
            best_val = val_acc
            no_improve = 0
            # Save best model to a separate path or fixed name
            best_model_path = args.out.replace(".pth", "_best.pth")
            save_ckpt(best_model_path, epoch=epoch)
            print(f"  --> Saved new best model (acc={val_acc:.4f}) to {best_model_path}")
        else:
            no_improve += 1

        if no_improve >= 15:  # Increased patience slightly
            print(f"Early stopping triggered after {no_improve} epochs of no improvement.")
            break

    # final save
    save_ckpt(args.out)
    print(f"Saved final model to {args.out}. Best validation accuracy: {best_val:.4f}")

    # dump training history for reproducible plots/reports
    history["best_val_acc"] = best_val
    history["epochs_run"] = len(history["train_loss"])
    history["protocol"] = args.tag
    history["manifest_sha256"] = manifest_sha
    os.makedirs(os.path.dirname(args.history) or ".", exist_ok=True)
    with open(args.history, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {args.history}")


if __name__ == "__main__":
    main()
