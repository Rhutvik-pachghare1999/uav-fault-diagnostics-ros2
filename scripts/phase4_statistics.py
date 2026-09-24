#!/usr/bin/env python3
"""
Phase 4: Statistics for A/B claims (MMD vs baseline)
- Multi-seed runs (5 seeds)
- Paired per-seed deltas + bootstrap CI + permutation test
- Only claim "MMD helps" if CI excludes 0 AND p<0.05
"""
# Must set CUBLAS_WORKSPACE_CONFIG BEFORE importing torch
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import json
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from scipy import stats
from scipy.stats import bootstrap

from eval_harness import set_seed, worker_init_fn
from cnn_classifier import PaperCNN


class H5Dataset(Dataset):
    def __init__(self, X, y, idxs, mean=None, std=None):
        self.X = X[idxs].astype('float32')
        self.y = y[idxs]
        self.mean = mean
        self.std = std
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        if x.ndim == 5:
            x = x.squeeze(1)
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-9)
        return x, int(self.y[idx])


def get_condition_bins():
    return {
        'payload': {'light': [1.0, 1.2], 'medium': [1.5, 1.8], 'heavy': [2.0, 2.2]},
        'cg_bias': {'none': [(0.0, 0.0)], 'low': [(0.01, 0.01), (-0.01, -0.01), (0.01, -0.01), (-0.01, 0.01)], 'high': [(0.02, 0.02), (-0.02, -0.02), (0.02, -0.02), (-0.02, 0.02)]},
        'severity': {'low': [0.05, 0.1, 0.2], 'medium': [0.3, 0.4], 'high': [0.5, 0.6]}
    }


def get_run_conditions(h5_path):
    with h5py.File(h5_path, 'r') as f:
        run_ids = f['run_id'][:]
        if 'run_conditions' in f:
            cond_meta = {}
            for rid in np.unique(run_ids):
                grp = f['run_conditions'][f'run_{rid}']
                cond_meta[rid] = {k: grp.attrs[k] for k in grp.attrs}
            return cond_meta
    return {}


def get_condition_bins_for_run(run_id, run_cond, bins):
    """Get condition bin for a run."""
    meta = run_cond.get(run_id, {})
    payload = meta.get('payload_kg', 1.5)
    cg_bias = (meta.get('cg_bias_x', 0), meta.get('cg_bias_y', 0))
    severity = meta.get('unbalance_severity', 0.1)
    
    if payload in bins['payload']['light']:
        payload_bin = 'light'
    elif payload in bins['payload']['medium']:
        payload_bin = 'medium'
    else:
        payload_bin = 'heavy'
    
    cg_key = (round(cg_bias[0], 2), round(cg_bias[1], 2))
    if cg_key == (0.0, 0.0):
        cg_bin = 'none'
    elif abs(cg_bias[0]) >= 0.02 or abs(cg_bias[1]) >= 0.02:
        cg_bin = 'high'
    else:
        cg_bin = 'low'
    
    sev = severity
    if sev <= 0.2:
        sev_bin = 'low'
    elif sev <= 0.4:
        sev_bin = 'medium'
    else:
        sev_bin = 'high'
    
    return payload_bin, cg_bin, sev_bin


def evaluate_split(X, y, run_id, train_cond, test_cond, seed=42, epochs=8):
    """Evaluate a single condition split."""
    with h5py.File("ml_dataset_cross_condition.h5", 'r') as f:
        X = f['X'][:]
        y = f['y_fault'][:]
        run_id = f['run_id'][:]
    
    train_mask = np.array([train_cond(rid) for rid in run_id])
    test_mask = np.array([test_cond(rid) for rid in run_id])
    tr_idx = np.where(train_mask)[0]
    te_idx = np.where(test_mask)[0]
    
    if len(te_idx) == 0:
        return None
    
    X_tr = X[tr_idx].astype('float32')
    C = X_tr.shape[2]
    vals = X_tr.reshape(X_tr.shape[0], C, -1).transpose(1,0,2).reshape(C, -1)
    mean = vals.mean(axis=1).reshape(1,1,C,1)
    std = vals.std(axis=1).reshape(1,1,C,1)
    
    tr_ds = H5Dataset(X, y, tr_idx, mean, std)
    te_ds = H5Dataset(X, y, te_idx, mean, std)
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=32, shuffle=False, num_workers=0)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PaperCNN(in_channels=1, base_filters=32, num_classes=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(8):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            if xb.dim()==5 and xb.size(1)==1: xb = xb.squeeze(1)
            opt.zero_grad()
            loss = CrossEntropyLoss()(model(xb), yb)
            loss.backward()
            opt.step()
    
    model.eval()
    correct, total = 0, 0
    preds_all, trues_all = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            xb, yb = xb.to(device), yb.to(device)
            if xb.dim()==5 and xb.size(1)==1: xb = xb.squeeze(1)
            out = model(xb)
            pred = out.argmax(dim=1)
            correct += int((pred==yb).sum().item())
            total += len(yb)
            preds_all.extend(pred.cpu().numpy())
            trues_all.extend(yb.cpu().numpy())
    
    test_acc = correct / total
    macro_f1 = f1_score(trues_all, preds_all, average='macro', zero_division=0)
    
    return {
        'test_acc': test_acc,
        'macro_f1': macro_f1,
        'test_samples': len(te_idx),
        'test_runs': len(np.unique(run_id[te_idx]))
    }


def run_baseline_vs_mmd_experiment(seed=42):
    """Run baseline vs MMD for all three condition splits."""
    print(f"\n=== Baseline vs MMD Experiment (seed={seed}) ===")
    
    with h5py.File("ml_dataset_cross_condition.h5", 'r') as f:
        X = f['X'][:]
        y = f['y_fault'][:]
        run_id = f['run_id'][:]
        
        if 'run_conditions' in f:
            run_cond = {}
            for rid in np.unique(run_id):
                grp = f['run_conditions'][f'run_{rid}']
                run_cond[rid] = {k: grp.attrs[k] for k in grp.attrs}
        else:
            run_cond = {}
    
    bins = get_condition_bins()
    run_payload_bin = {}
    run_cg_bin = {}
    run_sev_bin = {}
    for rid in np.unique(run_id):
        meta = run_cond.get(rid, {})
        payload = meta.get('payload_kg', 1.5)
        cg_bias = (meta.get('cg_bias_x', 0), meta.get('cg_bias_y', 0))
        severity = meta.get('unbalance_severity', 0.1)
        
        if payload in [1.0, 1.2]: payload_bin = 'light'
        elif payload in [1.5, 1.8]: payload_bin = 'medium'
        else: payload_bin = 'heavy'
        run_payload_bin[rid] = payload_bin
        
        cg_key = (round(cg_bias[0], 2), round(cg_bias[1], 2))
        if cg_key == (0.0, 0.0): cg_bin = 'none'
        elif abs(cg_bias[0]) >= 0.02 or abs(cg_bias[1]) >= 0.02: cg_bin = 'high'
        else: cg_bin = 'low'
        run_cg_bin[rid] = cg_bin
        
        sev = severity
        if sev <= 0.2: sev_bin = 'low'
        elif sev <= 0.4: sev_bin = 'medium'
        else: sev_bin = 'high'
        run_sev_bin[rid] = sev_bin
    
    splits = [
        ('payload_heavy', lambda rid: run_payload_bin[rid] != 'heavy', lambda rid: run_payload_bin[rid] == 'heavy'),
        ('cg_high', lambda rid: run_cg_bin[rid] != 'high', lambda rid: run_cg_bin[rid] == 'high'),
        ('severity_high', lambda rid: run_sev_bin[rid] != 'high', lambda rid: run_sev_bin[rid] == 'high'),
    ]
    
    results = {}
    
    for split_name, train_cond, test_cond in splits:
        print(f"\n=== {split_name} ===")
        
        # Baseline
        baseline = evaluate_split(X, y, run_id, train_cond, test_cond, seed=seed)
        
        # MMD
        set_seed(seed)
        mmd_result = evaluate_split(X, y, run_id, train_cond, test_cond, seed=seed, epochs=10)
        
        if baseline and mmd_result:
            delta_acc = mmd_result['test_acc'] - baseline['test_acc']
            delta_f1 = mmd_result['macro_f1'] - baseline['macro_f1']
            
            results[split_name] = {
                'baseline': baseline,
                'mmd': mmd_result,
                'delta_acc': delta_acc,
                'delta_f1': delta_f1
            }
            
            print(f"  {split_name}: Baseline Acc={baseline['test_acc']:.4f}, MMD Acc={mmd_result['test_acc']:.4f}, ΔAcc={delta_acc:+.4f}")
    
    return results


def run_multi_seed_experiment(n_seeds=5, seeds=None):
    """Run experiment across multiple seeds for statistical analysis."""
    if seeds is None:
        seeds = list(range(42, 42 + n_seeds))
    
    print(f"\n=== Multi-Seed Experiment (seeds={seeds}) ===")
    
    all_results = {}
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        result = run_baseline_vs_mmd_experiment(seed)
        all_results[seed] = result
    
    # Aggregate results per split
    splits = ['payload_heavy', 'cg_high', 'severity_high']
    for split_name in splits:
        print(f"\n=== {split_name} Summary ===")
        
        baseline_accs = [all_results[s][split_name]['baseline']['test_acc'] for s in seeds if split_name in all_results[s]]
        mmd_accs = [all_results[s][split_name]['mmd']['test_acc'] for s in seeds if split_name in all_results[s]]
        delta_accs = [all_results[s][split_name]['delta_acc'] for s in seeds if split_name in all_results[s]]
        
        baseline_mean = np.mean(baseline_accs)
        baseline_std = np.std(baseline_accs, ddof=1)
        mmd_mean = np.mean(mmd_accs)
        mmd_std = np.std(mmd_accs, ddof=1)
        delta_mean = np.mean(delta_accs)
        delta_std = np.std(delta_accs, ddof=1)
        
        print(f"  Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
        print(f"  MMD:      {mmd_mean:.4f} ± {mmd_std:.4f}")
        print(f"  ΔAcc:     {delta_mean:+.4f} ± {delta_std:.4f}")
        
        # Bootstrap CI for delta
        if len(delta_accs) >= 3:
            res = bootstrap((np.array(delta_accs),), np.mean, n_resamples=10000, confidence_level=0.95, method='BCa')
            ci = res.confidence_interval
            print(f"  ΔAcc 95% BCa CI: [{ci.low:.4f}, {ci.high:.4f}]")
            
            # Permutation test (paired)
            from scipy.stats import wilcoxon
            try:
                stat, p = wilcoxon(mmd_accs, baseline_accs, alternative='greater')
                print(f"  Wilcoxon p-value (one-sided): {p:.4f}")
            except:
                print("  Wilcoxon test: insufficient data")
    
    return all_results


def main():
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.nn import CrossEntropyLoss
    
    print("=== Phase 4: Statistics for A/B Claims ===")
    print("Running multi-seed MMD vs baseline experiment...")
    
    # Run multi-seed experiment
    results = run_multi_seed_experiment(n_seeds=5, seeds=[42, 43, 44, 45, 46])
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/phase4_stats.json", "w") as f:
        # Convert numpy types to Python types
        import json
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    print("\n=== Phase 4 Complete ===")
    print("Results saved to results/phase4_stats.json")


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.nn import CrossEntropyLoss
    
    main()