"""scripts/smoke_test.py
Quick smoke test: load a windowed dataset and a trained checkpoint, run one
forward pass on CPU, and print the prediction against the window's true
label. Uses the checkpoint's own meta (vars, base_filters, n_faults,
mean/std) so the smoke test exercises the same contract as production
inference.

Usage (defaults point at the sealed artifacts):
  python3 scripts/smoke_test.py
  python3 scripts/smoke_test.py --h5 ml_sealed.h5 --model models/cnn_sealed.pth
"""

import argparse
import json
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)


def main():
    p = argparse.ArgumentParser()
    repo = os.path.dirname(SCRIPTS_DIR)
    p.add_argument("--h5", default=os.path.join(repo, "ml_sealed.h5"))
    p.add_argument("--model", default=os.path.join(repo, "models", "cnn_sealed.pth"))
    p.add_argument("--idx", type=int, default=0, help="window index into the dataset")
    args = p.parse_args()

    if not os.path.exists(args.h5):
        print("HDF5 dataset not found:", args.h5)
        return
    if not os.path.exists(args.model):
        print("Model checkpoint not found:", args.model)
        return

    import h5py
    import numpy as np
    import torch
    from cnn_classifier import PaperCNN

    with h5py.File(args.h5, "r") as f:
        raw = f.attrs.get("meta", "{}")
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="ignore")
        try:
            meta = json.loads(raw)
        except Exception:
            meta = {}
        fault_map = meta.get("fault_label_map", {})
        ds_vars = meta.get("vars", [])
        X = f["X"][args.idx]  # (1, C, W) or (C, W)
        y_true = int(f["y_fault"][args.idx]) if "y_fault" in f else None
    print("Loaded h5:", args.h5, "window shape =", X.shape)
    X = np.asarray(X, dtype="float32")
    if X.ndim == 2:
        X = X[None, ...]
    X = X[None, ...]  # -> (1, 1, C, W)

    ck = torch.load(args.model, map_location="cpu")
    m = ck.get("meta", {})
    n_faults = int(m.get("n_faults", 16))
    base_filters = int(m.get("base_filters", 32))
    model_vars = m.get("vars", ds_vars)
    model = PaperCNN(in_channels=1, base_filters=base_filters, num_classes=n_faults)
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    model.eval()

    if ds_vars and model_vars:
        missing = [v for v in model_vars if v not in ds_vars]
        if missing:
            print("ERROR: model channels absent from dataset:", missing)
            sys.exit(2)
        idx = [ds_vars.index(v) for v in model_vars]
        X = X[:, :, idx, :]

    mean, std = m.get("mean"), m.get("std")
    if mean is not None and std is not None:
        mean, std = np.asarray(mean, "float32"), np.asarray(std, "float32")
        X = (X - mean) / (std + 1e-9)

    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        probs = torch.nn.functional.softmax(logits, dim=1)[0].numpy()
    fid = int(probs.argmax())
    rev = {v: k for k, v in fault_map.items()}
    out = {
        "fault_id": fid,
        "label": rev.get(fid, str(fid)),
        "confidence": round(float(probs[fid]), 4),
        "true_fault_id": y_true,
        "true_label": rev.get(y_true, str(y_true)) if y_true is not None else None,
        "correct": (fid == y_true) if y_true is not None else None,
    }
    print("Smoke test result:", out)


if __name__ == "__main__":
    main()
