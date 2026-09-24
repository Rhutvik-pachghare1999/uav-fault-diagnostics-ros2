#!/usr/bin/env python3
"""
Uncertainty Quantification and Open-Set Recognition for UAV Fault Diagnostics.

Implements:
1. Monte Carlo Dropout for epistemic uncertainty
2. Energy-based OOD detection
3. Prediction confidence calibration

Usage:
  python3 scripts/uncertainty_quantification.py --model models/cnn_sealed.pth --dataset ml_sealed.h5
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import entropy

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))
from cnn_classifier import PaperCNN

SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


class MCDropoutWrapper(nn.Module):
    """Wrapper to enable dropout at inference time for MC Dropout."""
    
    def __init__(self, model, dropout_rate=0.3):
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        
        # Replace dropout layers to be active during eval
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep dropout active
    
    def forward(self, x):
        return self.model(x)


def mc_dropout_predict(model, x, n_samples=20):
    """
    Monte Carlo Dropout prediction.
    Returns mean prediction and uncertainty estimates.
    """
    model.eval()
    # Enable dropout
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x)
            probs = F.softmax(out, dim=1)
            predictions.append(probs.cpu().numpy())
    
    predictions = np.stack(predictions, axis=0)  # (n_samples, batch, n_classes)
    
    # Mean prediction
    mean_probs = predictions.mean(axis=0)
    mean_pred = mean_probs.argmax(axis=1)
    
    # Uncertainty measures
    # 1. Predictive entropy (total uncertainty)
    predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
    
    # 2. Mutual information (epistemic uncertainty)
    expected_entropy = -np.mean(np.sum(predictions * np.log(predictions + 1e-10), axis=2), axis=0)
    mutual_info = predictive_entropy - expected_entropy
    
    # 3. Variance of predictions
    pred_variance = np.var(predictions, axis=0).mean(axis=1)
    
    return {
        'mean_probs': mean_probs,
        'mean_pred': mean_pred,
        'predictive_entropy': predictive_entropy,
        'mutual_info': mutual_info,
        'pred_variance': pred_variance,
        'all_probs': predictions
    }


def energy_score(logits):
    """
    Energy-based OOD score (lower energy = more in-distribution).
    Energy = -T * log(sum(exp(logits/T)))
    """
    T = 1.0  # Temperature
    energy = -T * torch.logsumexp(logits / T, dim=1)
    return energy


def ood_aurocs(measures):
    """AUROC of OOD detection, labels 0 = in-distribution, 1 = OOD.

    Every measure passed in must already be oriented so that HIGHER = more
    OOD-like: energy / entropy / mutual information as-is, max probability
    NEGATED. Passing a wrongly-oriented pair silently yields AUROC < 0.5 —
    the historical bug this function exists to prevent.
    """
    out = {}
    for name, (id_scores, ood_scores) in measures.items():
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        scores = np.concatenate([id_scores, ood_scores])
        out[name] = float(roc_auc_score(labels, scores))
    return out


def evaluate_uncertainty(args):
    """Main evaluation function."""

    if args.split == "train" and not args.allow_train_eval:
        raise SystemExit("refusing to evaluate on the TRAIN split (training "
                         "windows give inflated confidence metrics); pass "
                         "--allow-train-eval to override.")

    # Load dataset (sealed h5: evaluate ONE split's ORIGINAL windows only —
    # legacy mixed datasets sample training windows and inflate every metric)
    print(f"Loading dataset from {args.dataset}...")
    with h5py.File(args.dataset, 'r') as f:
        meta_raw = f.attrs.get('meta', '{}')
        if isinstance(meta_raw, (bytes, bytearray)):
            meta_raw = meta_raw.decode('utf-8', errors='ignore')
        try:
            meta = json.loads(meta_raw)
        except Exception:
            meta = ast.literal_eval(meta_raw)
        fault_map = meta.get('fault_label_map', {})
        rev_map = {v: k for k, v in fault_map.items()}
        h5_vars = meta.get('vars', None)

        if 'split' in f and 'is_aug' in f:
            X_all = f['X'][:]
            y_all = f['y_fault'][:]
            sel = (f['split'][:] == SPLIT_IDS[args.split]) & (f['is_aug'][:] == 0)
            X, y = X_all[sel], y_all[sel]
            del X_all, y_all
            protocol = f"sealed-{args.split}-originals"
            print(f"  split={args.split} originals only: {len(X)} windows")
        else:
            if not args.allow_unsealed:
                raise SystemExit(
                    f"{args.dataset} lacks split/is_aug columns. Refusing: "
                    f"legacy mixed datasets contain TRAINING windows and "
                    f"inflate confidence/OOD metrics. Use ml_sealed.h5 "
                    f"(scripts/build_sealed_dataset.py) or pass --allow-unsealed.")
            X, y = f['X'][:], f['y_fault'][:]
            protocol = "unsealed-legacy (WARNING: may include training windows)"

    # Load model
    print(f"Loading model from {args.model}...")
    ck = torch.load(args.model, map_location='cpu')
    meta_ck = ck.get('meta', {}) if isinstance(ck, dict) else {}
    n_faults = int(meta_ck.get('n_faults', 16))
    base_filters = int(meta_ck.get('base_filters', 32))

    # Align channels by name (supports IMU-only models against 13-channel h5)
    model_vars = meta_ck.get('vars', None)

    # Sealed protocol provenance: model and dataset must come from the same
    # frozen manifest, else confidence/OOD metrics mix incompatible splits.
    h5_sha = meta.get('manifest_sha256', None)
    model_sha = meta_ck.get('manifest_sha256', None)
    if h5_sha and model_sha and h5_sha != model_sha:
        raise SystemExit(
            f"MANIFEST MISMATCH: model trained on manifest {model_sha[:12]}... "
            f"but dataset built from {h5_sha[:12]}... — refusing to mix splits.")

    if model_vars and h5_vars and list(model_vars) != list(h5_vars):
        try:
            col_idx = [list(h5_vars).index(v) for v in model_vars]
        except ValueError as e:
            raise SystemExit(f"model channels missing in dataset: {e}")
        X = X[:, :, col_idx, :]
        print(f"  channel alignment: using {len(col_idx)} of {len(h5_vars)} "
              f"channels ({','.join(model_vars)})")

    n_ch = X.shape[2] if X.ndim == 4 else X.shape[1]
    mean = meta_ck.get('mean', None)
    std = meta_ck.get('std', None)

    if mean is not None:
        mean = np.array(mean, dtype='float32')
        if mean.size == n_ch:
            mean = mean.reshape(1, 1, n_ch, 1)
    if std is not None:
        std = np.array(std, dtype='float32')
        if std.size == n_ch:
            std = std.reshape(1, 1, n_ch, 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PaperCNN(in_channels=1, base_filters=base_filters,
                     num_classes=n_faults).to(device)
    sd = ck.get('state_dict', ck)
    if all(k.startswith('module.') for k in sd.keys()):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    
    # Create MC Dropout wrapper
    mc_model = MCDropoutWrapper(model, dropout_rate=0.3)
    
    # Prepare test data (use a subset for speed)
    n_test = min(args.n_samples, len(X))
    rng = np.random.default_rng(42)
    test_idx = rng.choice(len(X), n_test, replace=False)
    
    X_test = X[test_idx].astype('float32')
    y_test = y[test_idx]
    
    # Normalize
    if mean is not None and std is not None:
        X_test = (X_test - mean) / (std + 1e-9)
    
    # Create dataloader
    class TestDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y
        def __len__(self): return len(self.X)
        def __getitem__(self, idx):
            x = self.X[idx]
            if x.ndim == 5: x = x.squeeze(1)
            if x.ndim == 4 and x.shape[0] == 1: x = x.squeeze(0)
            return x, int(self.y[idx])
    
    test_ds = TestDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Run MC Dropout predictions
    print(f"Running MC Dropout with {args.mc_samples} samples...")
    all_mean_probs = []
    all_preds = []
    all_true = []
    all_entropy = []
    all_mutual_info = []
    all_variance = []
    all_energy = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            if xb.dim() == 5 and xb.size(2) == 1:
                xb = xb.squeeze(2)
            if xb.dim() == 3:
                xb = xb.unsqueeze(1)
            
            # Standard prediction
            logits = model(xb)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            
            # Energy score
            energy = energy_score(logits).cpu().numpy()
            
            # MC Dropout
            mc_result = mc_dropout_predict(mc_model, xb, n_samples=args.mc_samples)
            
            all_mean_probs.append(mc_result['mean_probs'])
            all_preds.append(mc_result['mean_pred'])
            all_true.append(yb.numpy())
            all_entropy.append(mc_result['predictive_entropy'])
            all_mutual_info.append(mc_result['mutual_info'])
            all_variance.append(mc_result['pred_variance'])
            all_energy.append(energy)
    
    # Concatenate results
    mean_probs = np.vstack(all_mean_probs)
    preds = np.concatenate(all_preds)
    true = np.concatenate(all_true)
    entropy_vals = np.concatenate(all_entropy)
    mutual_info = np.concatenate(all_mutual_info)
    variance = np.concatenate(all_variance)
    energy_vals = np.concatenate(all_energy)
    
    # Metrics
    acc = accuracy_score(true, preds)
    print(f"\nOverall Accuracy: {acc:.4f}")
    
    # Confidence-based metrics
    max_probs = mean_probs.max(axis=1)
    print(f"Mean Max Probability: {max_probs.mean():.4f}")
    print(f"Mean Predictive Entropy: {entropy_vals.mean():.4f}")
    print(f"Mean Mutual Info (Epistemic): {mutual_info.mean():.4f}")
    print(f"Mean Energy: {energy_vals.mean():.4f}")
    
    # OOD Detection: Create synthetic OOD data
    print("\n--- OOD Detection Evaluation ---")
    
    # Generate OOD samples (random noise)
    ood_X = rng.standard_normal((args.n_ood, *X_test.shape[1:])).astype('float32')
    ood_ds = TestDataset(ood_X, np.zeros(args.n_ood, dtype=int))
    ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    ood_energy = []
    ood_entropy = []
    ood_mutual_info = []
    ood_max_prob = []
    
    with torch.no_grad():
        for xb, _ in ood_loader:
            xb = xb.to(device)
            if xb.dim() == 5 and xb.size(2) == 1:
                xb = xb.squeeze(2)
            if xb.dim() == 3:
                xb = xb.unsqueeze(1)
            
            logits = model(xb)
            probs = F.softmax(logits, dim=1)
            
            energy = energy_score(logits).cpu().numpy()
            mc_result = mc_dropout_predict(mc_model, xb, n_samples=args.mc_samples)
            
            ood_energy.append(energy)
            ood_entropy.append(mc_result['predictive_entropy'])
            ood_mutual_info.append(mc_result['mutual_info'])
            ood_max_prob.append(probs.max(dim=1).values.cpu().numpy())
    
    ood_energy = np.concatenate(ood_energy)
    ood_entropy = np.concatenate(ood_entropy)
    ood_mutual_info = np.concatenate(ood_mutual_info)
    ood_max_prob = np.concatenate(ood_max_prob)
    
    # Binary labels: 0 = in-distribution, 1 = OOD
    # roc_auc_score expects the positive class (OOD) to have HIGHER scores:
    #   energy / entropy / MI are naturally higher for OOD -> use as-is
    #   max prob is lower for OOD -> negate
    aurocs = ood_aurocs({
        "Energy (higher=OOD)": (energy_vals, ood_energy),
        "Entropy (higher=OOD)": (entropy_vals, ood_entropy),
        "Mutual Info (higher=OOD)": (mutual_info, ood_mutual_info),
        "Neg Max Prob (lower conf=OOD)": (-max_probs, -ood_max_prob),
    })
    for name, auroc in aurocs.items():
        print(f"  {name} AUROC: {auroc:.4f}")
    
    # Per-class uncertainty
    print("\n--- Per-Class Uncertainty ---")
    for class_id in sorted(np.unique(true)):
        mask = true == class_id
        if mask.sum() > 0:
            class_name = rev_map.get(class_id, f"class_{class_id}")
            print(f"  {class_name}: Acc={accuracy_score(true[mask], preds[mask]):.4f}, "
                  f"Entropy={entropy_vals[mask].mean():.4f}, "
                  f"MI={mutual_info[mask].mean():.4f}, "
                  f"Energy={energy_vals[mask].mean():.4f}")
    
    # Save results
    results = {
        'protocol': protocol,
        'split': args.split if protocol.startswith('sealed-') else None,
        'manifest_sha256': h5_sha,
        'model': args.model,
        'dataset': args.dataset,
        'n_eval_windows': int(len(y)),
        'n_ood': int(args.n_ood),
        'ood_kind': 'synthetic gaussian noise',
        'accuracy': float(acc),
        'mean_max_prob': float(max_probs.mean()),
        'mean_entropy': float(entropy_vals.mean()),
        'mean_mutual_info': float(mutual_info.mean()),
        'mean_energy': float(energy_vals.mean()),
        'auroc_ood': aurocs,
        'per_class': {}
    }
    
    for class_id in sorted(np.unique(true)):
        mask = true == class_id
        class_name = rev_map.get(class_id, f"class_{class_id}")
        results['per_class'][class_name] = {
            'accuracy': float(accuracy_score(true[mask], preds[mask])),
            'entropy': float(entropy_vals[mask].mean()),
            'mutual_info': float(mutual_info[mask].mean()),
            'energy': float(energy_vals[mask].mean()),
            'support': int(mask.sum())
        }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Uncertainty Quantification and OOD Detection")
    parser.add_argument("--model", required=True, help="Path to trained model (.pth)")
    parser.add_argument("--dataset", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="sealed split to evaluate (originals only; default test)")
    parser.add_argument("--allow-train-eval", action="store_true",
                        help="explicitly allow evaluating the train split")
    parser.add_argument("--allow-unsealed", action="store_true",
                        help="allow legacy datasets without split/is_aug columns")
    parser.add_argument("--output", default="results/uncertainty_sealed.json", help="Output JSON file")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--n-ood", type=int, default=500, help="Number of OOD samples")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--mc-samples", type=int, default=20, help="MC Dropout samples")
    args = parser.parse_args()
    
    evaluate_uncertainty(args)


if __name__ == "__main__":
    main()