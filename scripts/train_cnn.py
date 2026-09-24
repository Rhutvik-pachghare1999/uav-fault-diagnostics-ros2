# scripts/train_cnn.py
"""
Train the 2D-CNN multi-head classifier:
  python3 scripts/train_cnn.py --h5 ml_dataset_v2.h5 --out models/cnn_multi.pth --epochs 50
"""
import argparse, h5py, numpy as np, os
from sklearn.model_selection import train_test_split

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

import torch, logging
from torch.utils.data import Dataset

def setup_logging(log_file="logs/train.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

class H5Dataset(Dataset):
    def __init__(self, X, yf, idxs, mean=None, std=None):
        self.X = X[idxs].astype('float32')
        self.yf = yf[idxs]
        # mean/std shape: (C,1) or (1,C,1) broadcasting OK
        self.mean = mean
        self.std = std
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-9)
        return x, int(self.yf[idx])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", "--data", required=True, help="Path to HDF5 dataset")
    p.add_argument("--out", default="models/cnn_multi.pth", help="Output model path")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--base-filters", type=int, default=32)
    args = p.parse_args()
    
    logger = setup_logging()

    try:
        import torch, torch.nn as nn, torch.optim as optim
        from torch.utils.data import DataLoader
        from cnn_classifier import PaperCNN
    except Exception:
        print("Torch not installed in this environment. Install torch and retry.")
        raise

    with h5py.File(args.h5, "r") as f:
        X_all = f["X"][:]  # (N,1,C,W)
        run_id = f["run_id"][:] if "run_id" in f else None
        # use safe dataset read API
        y_fault = f["y_fault"][:]
        # metadata may be bytes or str; parse safely (never eval untrusted data)
        import json, ast
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
        n_faults = len(meta.get("fault_label_map", {})) or int(y_fault.max()+1)
    # Split so that windows from the SAME physical run never cross the
    # train/val/test boundary. Overlapping/adjacent windows within one run are
    # near-duplicates; a plain random split would leak them across partitions and
    # inflate accuracy. When run_id is available we use GroupShuffleSplit (grouped
    # by run); otherwise we fall back to a stratified random split and warn.
    idx = np.arange(len(X_all))
    if run_id is not None and len(np.unique(run_id)) >= 3:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_full, te_idx = next(gss.split(idx, y_fault, groups=run_id))
        tr_idx = idx[tr_full]; te_idx = idx[te_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        tr_rel, val_rel = next(gss2.split(tr_idx, y_fault[tr_idx], groups=run_id[tr_idx]))
        tr_idx, val_idx = tr_idx[tr_rel], tr_idx[val_rel]
        print(f"Run-grouped split: {len(tr_idx)} train / {len(val_idx)} val / {len(te_idx)} test "
              f"windows over {len(np.unique(run_id))} runs (no run crosses partitions).")
    else:
        print("WARNING: run_id unavailable or <3 runs — falling back to stratified random "
              "split. Windows from the same run may leak across partitions; treat metrics as optimistic.")
        try:
            tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_fault)
            tr_idx, val_idx = train_test_split(tr_idx, test_size=0.125, random_state=42, stratify=y_fault[tr_idx])
        except Exception:
            tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=None)
            tr_idx, val_idx = train_test_split(tr_idx, test_size=0.125, random_state=42, stratify=None)

    # compute per-channel mean/std on training set for normalization
    X_tr = X_all[tr_idx].astype('float32')
    # X_tr shape: (N,1,C,W) -> compute mean/std per channel over samples and time
    # collapse sample and time dims to compute per-channel stats
    C = X_tr.shape[2]
    vals = X_tr.reshape(X_tr.shape[0], C, -1).transpose(1,0,2).reshape(C, -1)
    mean = vals.mean(axis=1).reshape(1,1,C,1)
    std = vals.std(axis=1).reshape(1,1,C,1)

    tr_ds = H5Dataset(X_all, y_fault, tr_idx, mean=mean, std=std)
    val_ds = H5Dataset(X_all, y_fault, val_idx, mean=mean, std=std)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = PaperCNN(in_channels=1, base_filters=args.base_filters, num_classes=n_faults).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    best_val = 0.0
    no_improve = 0
    best_path = args.out
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        model.train()
        tot, cnt = 0.0, 0
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for xb, yf in pbar:
            xb = xb.to(device); yf = yf.to(device)
            # handle unexpected extra singleton dimension from collate
            if xb.dim() == 5 and xb.size(2) == 1:
                xb = xb.squeeze(2)
            if xb.dim() == 3:
                xb = xb.unsqueeze(1)
            pred = model(xb)
            loss = loss_fn(pred, yf)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); cnt += 1
            pbar.set_postfix(loss=f"{tot/cnt:.4f}")
        
        train_loss = tot/cnt if cnt > 0 else 0.0
        
        # val acc
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0; val_cnt = 0
        with torch.no_grad():
            for xb, yf in val_loader:
                xb = xb.to(device); yf = yf.to(device)
                if xb.dim() == 5 and xb.size(2) == 1:
                    xb = xb.squeeze(2)
                if xb.dim() == 3:
                    xb = xb.unsqueeze(1)
                pf = model(xb)
                pred = pf.argmax(dim=1)
                correct += int((pred == yf).sum().item()); total += len(yf)
                l = loss_fn(pf, yf)
                val_loss += float(l.item()); val_cnt += 1
        
        val_acc = correct/total if total>0 else 0.0
        val_loss_avg = (val_loss/val_cnt) if val_cnt else 0.0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss_avg)
        history["val_acc"].append(val_acc)
        msg = f"Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss_avg:.4f} val_acc={val_acc:.4f}"
        logger.info(msg)
        
        scheduler.step(val_loss_avg)
        
        # checkpoint best
        if val_acc > best_val:
            best_val = val_acc
            no_improve = 0
            # Save best model to a separate path or fixed name
            best_model_path = args.out.replace(".pth", "_best.pth")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(), 
                "meta": {
                    "n_faults": n_faults, 
                    "mean": mean.tolist(), 
                    "std": std.tolist(), 
                    "fault_label_map": meta.get('fault_label_map', {}),
                    "vars": meta.get('vars', [])
                }
            }, best_model_path)
            print(f"  --> Saved new best model (acc={val_acc:.4f}) to {best_model_path}")
        else:
            no_improve += 1
            
        if no_improve >= 15: # Increased patience slightly
            print(f"Early stopping triggered after {no_improve} epochs of no improvement.")
            break

    # final save
    torch.save({
        "state_dict": model.state_dict(), 
        "meta": {
            "n_faults": n_faults, 
            "mean": mean.tolist(), 
            "std": std.tolist(), 
            "fault_label_map": meta.get('fault_label_map', {}),
            "vars": meta.get('vars', [])
        }
    }, args.out)
    print(f"Saved final model to {args.out}. Best validation accuracy: {best_val:.4f}")

    # dump training history for reproducible plots/reports
    import json
    history["best_val_acc"] = best_val
    history["epochs_run"] = len(history["train_loss"])
    os.makedirs("results", exist_ok=True)
    with open("results/train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Saved training history to results/train_history.json")

if __name__ == "__main__":
    main()
