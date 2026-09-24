#!/usr/bin/env python3
"""
Cross-speed robustness evaluation.

Evaluates model performance across different motor speed conditions.
This simulates the "unseen speed" evaluation ladder rung.

Usage:
  python3 scripts/eval_cross_speed.py --model models/cnn_multi.pth --dataset ml_dataset_v2_aug.h5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))
from cnn_classifier import PaperCNN


def estimate_rpm_from_data(X):
    """Estimate average RPM from first 4 channels (rpm1-rpm4)."""
    # X shape: (N, 1, C, W)
    rpm_channels = X[:, 0, :4, :]  # (N, 4, W)
    avg_rpm = rpm_channels.mean(axis=(1, 2))  # (N,)
    return avg_rpm


def create_speed_bins(rpm_values, n_bins=4):
    """Create speed bins based on RPM percentiles."""
    percentiles = np.percentile(rpm_values, np.linspace(0, 100, n_bins + 1))
    bins = []
    for i in range(n_bins):
        bins.append((percentiles[i], percentiles[i + 1]))
    return bins


def assign_speed_bin(rpm_values, bins):
    """Assign each sample to a speed bin."""
    bin_labels = np.zeros(len(rpm_values), dtype=int)
    for i, (low, high) in enumerate(bins):
        mask = (rpm_values >= low) & (rpm_values < high)
        bin_labels[mask] = i
    # Handle edge case for max value
    bin_labels[rpm_values == bins[-1][1]] = len(bins) - 1
    return bin_labels


def evaluate_speed_generalization(model, X, y, run_ids, speed_bin_labels, speed_bin_edges, mean, std, device):
    """Evaluate model on leave-one-speed-out splits."""
    results = {}
    n_bins = len(speed_bin_edges)
    
    for test_speed in range(n_bins):
        print(f"\n=== Leave-out Speed Bin {test_speed} ({speed_bin_edges[test_speed][0]:.0f}-{speed_bin_edges[test_speed][1]:.0f} RPM) ===")
        
        # Create train/test split based on speed
        train_mask = speed_bin_labels != test_speed
        test_mask = speed_bin_labels == test_speed
        
        # But we must respect run grouping - no run crosses train/test
        # So we check which runs fall into test speed
        test_runs = set(run_ids[test_mask])
        train_mask_final = np.array([rid not in test_runs for rid in run_ids])
        test_mask_final = np.array([rid in test_runs for rid in run_ids])
        
        if test_mask_final.sum() == 0:
            print(f"  No test runs for speed bin {test_speed}")
            continue
        
        tr_idx = np.where(train_mask_final)[0]
        te_idx = np.where(test_mask_final)[0]
        
        print(f"  Train runs: {len(np.unique(run_ids[tr_idx]))}, Test runs: {len(np.unique(run_ids[te_idx]))}")
        print(f"  Train samples: {len(tr_idx)}, Test samples: {len(te_idx)}")
        
        # Create datasets
        from torch.utils.data import DataLoader, Dataset
        
        class EvalDataset(Dataset):
            def __init__(self, X, y, idxs, mean, std):
                self.X = X[idxs].astype('float32')
                self.y = y[idxs]
                self.mean = mean
                self.std = std
            def __len__(self): return len(self.X)
            def __getitem__(self, idx):
                x = self.X[idx]
                if x.ndim == 5: x = x.squeeze(1)
                if x.ndim == 4 and x.shape[0] == 1: x = x.squeeze(0)
                if self.mean is not None and self.std is not None:
                    x = (x - self.mean) / (self.std + 1e-9)
                return x, int(self.y[idx])
        
        tr_ds = EvalDataset(X, y, tr_idx, mean, std)
        te_ds = EvalDataset(X, y, te_idx, mean, std)
        tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=False, num_workers=0)
        te_loader = DataLoader(te_ds, batch_size=64, shuffle=False, num_workers=0)
        
        # Evaluate
        model.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for xb, yb in te_loader:
                xb = xb.to(device)
                if xb.dim() == 5 and xb.size(2) == 1:
                    xb = xb.squeeze(2)
                if xb.dim() == 3:
                    xb = xb.unsqueeze(1)
                out = model(xb)
                pred = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(pred)
                all_true.extend(yb.numpy())
        
        acc = accuracy_score(all_true, all_preds)
        macro_f1 = f1_score(all_true, all_preds, average='macro', zero_division=0)
        
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  Test Macro F1: {macro_f1:.4f}")
        
        results[f'speed_bin_{test_speed}'] = {
            'speed_range': speed_bin_edges[test_speed],
            'test_runs': len(np.unique(run_ids[te_idx])),
            'test_samples': len(te_idx),
            'accuracy': float(acc),
            'macro_f1': float(macro_f1),
            'per_class': classification_report(all_true, all_preds, output_dict=True, zero_division=0)
        }
    
    return results


def evaluate_speed_generalization_retrain(X, y, run_ids, speed_bin_labels, speed_bin_edges, device, epochs=10):
    """Evaluate with retraining on each leave-one-speed-out split (more realistic)."""
    results = {}
    
    for test_speed in range(len(speed_bin_edges)):
        print(f"\n=== Retrain Leave-out Speed Bin {test_speed} ({speed_bin_edges[test_speed][0]:.0f}-{speed_bin_edges[test_speed][1]:.0f} RPM) ===")
        
        test_runs = set()
        for i, bin_id in enumerate(speed_bin_labels):
            if bin_id == test_speed:
                # Find runs in this speed bin
                pass
        # This is more complex - for now return placeholder
        results[f'speed_bin_{test_speed}_retrain'] = {'status': 'not_implemented'}
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-speed robustness evaluation")
    parser.add_argument("--model", required=True, help="Path to trained model (.pth)")
    parser.add_argument("--dataset", default="ml_dataset_v2_aug.h5", help="Path to HDF5 dataset")
    parser.add_argument("--n-speed-bins", type=int, default=4, help="Number of speed bins")
    parser.add_argument("--output", default="results/cross_speed_eval.json", help="Output JSON file")
    parser.add_argument("--retrain", action="store_true", help="Retrain for each split (slower but more realistic)")
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    with h5py.File(args.dataset, 'r') as f:
        X = f['X'][:]
        y = f['y_fault'][:]
        run_ids = f['run_id'][:] if 'run_id' in f else np.arange(len(X))
        
        import ast
        meta_raw = f.attrs.get('meta', '{}')
        if isinstance(meta_raw, (bytes, bytearray)):
            meta_raw = meta_raw.decode('utf-8', errors='ignore')
        try:
            meta = json.loads(meta_raw)
        except Exception:
            meta = ast.literal_eval(meta_raw)
    
    print(f"Dataset: {len(X)} samples, {len(np.unique(y))} classes, {len(np.unique(run_ids))} runs")
    
    # Estimate RPM for each sample
    print("Estimating RPM for speed binning...")
    rpm_values = estimate_rpm_from_data(X)
    print(f"RPM range: {rpm_values.min():.0f} - {rpm_values.max():.0f}")
    
    # Create speed bins
    speed_bins_edges = create_speed_bins(rpm_values, args.n_speed_bins)
    speed_bin_labels = assign_speed_bin(rpm_values, speed_bins_edges)
    
    print(f"\nSpeed bins:")
    for i, (low, high) in enumerate(speed_bins_edges):
        count = (speed_bin_labels == i).sum()
        print(f"  Bin {i}: {low:.0f}-{high:.0f} RPM ({count} samples)")
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    ck = torch.load(args.model, map_location='cpu')
    meta_ck = ck.get('meta', {})
    n_faults = meta_ck.get('n_faults', 16)
    mean = meta_ck.get('mean', None)
    std = meta_ck.get('std', None)
    
    if mean is not None:
        mean = np.array(mean, dtype='float32')
    if std is not None:
        std = np.array(std, dtype='float32')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PaperCNN(in_channels=1, base_filters=32, num_classes=n_faults).to(device)
    sd = ck.get('state_dict', ck)
    if all(k.startswith('module.') for k in sd.keys()):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    
    # Run evaluation
    print("\n" + "="*60)
    print("CROSS-SPEED ROBUSTNESS EVALUATION")
    print("="*60)
    
    if args.retrain:
        results = evaluate_speed_generalization_retrain(X, y, run_ids, speed_bin_labels, speed_bins_edges, device)
    else:
        results = evaluate_speed_generalization(model, X, y, run_ids, speed_bin_labels, speed_bins_edges, mean, std, device)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for key, val in results.items():
        if 'accuracy' in val:
            print(f"  {key} ({val['speed_range'][0]:.0f}-{val['speed_range'][1]:.0f} RPM): "
                  f"Acc={val['accuracy']:.4f}, Macro F1={val['macro_f1']:.4f}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            'model': args.model,
            'dataset': args.dataset,
            'n_speed_bins': args.n_speed_bins,
            'speed_bins': speed_bins_edges,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()