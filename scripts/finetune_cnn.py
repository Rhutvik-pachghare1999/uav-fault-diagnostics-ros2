#!/usr/bin/env python3
"""Fine-tune a pretrained CNN on a given HDF5 dataset with class balancing.

Usage:
  python3 scripts/finetune_cnn.py --h5 scripts/data_out/ml_dataset_small_real.h5 \
       --pretrained scripts/models/cnn_multi_retrain.pth --out scripts/models/cnn_multi_finetune.pth \
       --epochs 40 --batch-size 64 --lr 1e-4
"""

import ast
import argparse
import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--pretrained", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.model_selection import train_test_split
    from cnn_classifier import PaperCNN

    # load dataset
    with h5py.File(args.h5, "r") as f:
        X_all = f["X"][:]  # (N,1,C,W)
        y_fault = f["y_fault"][:]
        try:
            meta = ast.literal_eval(f.attrs.get("meta", "{}"))
        except Exception:
            meta = {}
        # determine dataset classes (use unique labels found)
        unique_labels = np.unique(y_fault)
        n_faults = len(unique_labels)

    # split train/val
    idx = np.arange(len(X_all))
    try:
        tr_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_fault)
    except Exception:
        tr_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    # compute mean/std on training
    X_tr = X_all[tr_idx].astype("float32")
    C = X_tr.shape[2]
    vals = X_tr.reshape(X_tr.shape[0], C, -1).transpose(1, 0, 2).reshape(C, -1)
    mean = vals.mean(axis=1).reshape(1, 1, C, 1)
    std = vals.std(axis=1).reshape(1, 1, C, 1)

    class TrainDataset(Dataset):
        def __init__(self, X, y, idxs, mean, std):
            self.X = X[idxs].astype("float32")
            self.y = y[idxs]
            self.mean = mean
            self.std = std

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            x = self.X[i]
            x = (x - self.mean) / (self.std + 1e-9)
            return x, int(self.y[i])

    tr_ds = TrainDataset(X_all, y_fault, tr_idx, mean, std)
    val_ds = TrainDataset(X_all, y_fault, val_idx, mean, std)

    # weighted sampler to address class imbalance
    unique, counts = np.unique(y_fault[tr_idx], return_counts=True)
    weights = {u: 1.0 / float(c) for u, c in zip(unique, counts)}
    sample_weights = np.array([weights[int(y)] for y in y_fault[tr_idx]], dtype="float32")
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load pretrained
    ck = torch.load(args.pretrained, map_location="cpu")
    # set model classes to dataset classes to avoid label-out-of-range errors
    n_classes = n_faults
    model = PaperCNN(in_channels=1, base_filters=32, num_classes=n_classes).to(device)
    # extract state_dict and perform partial load where shapes match
    state_dict = None
    if isinstance(ck, dict):
        if "state_dict" in ck:
            state_dict = ck["state_dict"]
        elif "model_state_dict" in ck:
            state_dict = ck["model_state_dict"]
        else:
            state_dict = ck
    else:
        state_dict = ck
    # normalize keys
    sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model_sd = model.state_dict()
    load_sd = {}
    skipped = []
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            load_sd[k] = v
        else:
            skipped.append(k)
    if load_sd:
        model_sd.update(load_sd)
        model.load_state_dict(model_sd)
    if skipped:
        print("Skipped loading keys due to shape mismatch (likely classifier head):", skipped[:10])

    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_val = 0.0
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        cnt = 0
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if xb.dim() == 5 and xb.size(2) == 1:
                xb = xb.squeeze(2)
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
            cnt += 1
        train_loss = train_loss / cnt if cnt else 0.0

        # validate
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        vcnt = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                if xb.dim() == 5 and xb.size(2) == 1:
                    xb = xb.squeeze(2)
                out = model(xb)
                pred = out.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += len(yb)
                val_loss += float(loss_fn(out, yb).item())
                vcnt += 1
        val_acc = correct / total if total else 0.0
        print(
            f"Epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.4f} "
            f"val_acc={val_acc:.4f} val_loss={val_loss / (vcnt or 1):.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            no_improve = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "meta": {
                        "n_faults": n_classes,
                        "mean": mean.tolist(),
                        "std": std.tolist(),
                        "fault_label_map": meta.get("fault_label_map", {}),
                    },
                },
                args.out,
            )
            print("Saved best model to", args.out)
        else:
            no_improve += 1
        if no_improve >= 8:
            print("Early stopping")
            break

    # final save
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {
                "n_faults": n_classes,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "fault_label_map": meta.get("fault_label_map", {}),
            },
        },
        args.out,
    )
    print("Saved final model to", args.out)


if __name__ == "__main__":
    main()
