"""Evaluate a trained CNN classifier on an HDF5 dataset and produce realistic plots.

Produces:
 - `results/confusion_matrix.png` (heatmap)
 - `results/classification_report.txt`
 - `results/sample_signals/` plots showing example signals with predicted vs true labels

Usage:
  python3 scripts/eval_classifier.py --h5 ml_dataset_synth.h5 --model models/cnn_multi_synth.pth --out results
"""
import argparse, os, h5py, numpy as np, random, json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_confusion(cm, labels, outpath):
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_ylabel('True'); ax.set_xlabel('Pred')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i,j]), ha='center', va='center', color='black', fontsize=6)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_sample_signal(X, true_label, pred_label, label_map, outpath, channels=(0,1,2,3)):
    # X: (C,W)
    C, W = X.shape
    t = np.arange(W)
    fig, ax = plt.subplots(4,1, figsize=(8,6), sharex=True)
    for i,ch in enumerate(channels[:4]):
        if ch < C:
            ax[i].plot(t, X[ch], lw=0.7)
            ax[i].set_ylabel(f'ch{ch}')
    ax[-1].set_xlabel('t')
    fig.suptitle(f'True: {label_map.get(true_label,str(true_label))} Pred: {label_map.get(pred_label,str(pred_label))}')
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', default='results')
    p.add_argument('--max-samples', type=int, default=5000)
    args = p.parse_args()

    # Defer creating output directories until after we successfully load a compatible checkpoint

    import torch
    ck = torch.load(args.model, map_location='cpu')
    meta = ck.get('meta', {})
    mean = meta.get('mean', None)
    std = meta.get('std', None)
    if mean is not None and std is not None:
        mean = np.array(mean, dtype='float32')
        std = np.array(std, dtype='float32')

    # load dataset
    with h5py.File(args.h5, 'r') as f:
        X = f['X'][:]  # (N,1,C,W)
        y = f['y_fault'][:]
        y_sev = f['y_sev'][:] if 'y_sev' in f else None
        import json, ast
        _mr = f.attrs.get('meta', '{}')
        if isinstance(_mr, (bytes, bytearray)):
            _mr = _mr.decode('utf-8', errors='ignore')
        try:
            meta_attr = json.loads(_mr)
        except Exception:
            try:
                meta_attr = ast.literal_eval(_mr)  # safe: literals only
            except Exception:
                meta_attr = {}
        fault_map = meta_attr.get('fault_label_map', {})
        rev = {v:k for k,v in fault_map.items()} if fault_map else {i:str(i) for i in range(int(y.max())+1)}

    # subsample for speed
    N = len(X)
    idxs = np.arange(N)
    if args.max_samples and N > args.max_samples:
        idxs = np.random.choice(idxs, args.max_samples, replace=False)

    # prepare model
    from cnn_classifier import PaperCNN
    n_faults = meta.get('n_faults', int(y.max())+1)
    model = PaperCNN(in_channels=1, base_filters=32, num_classes=n_faults)
    # Support multiple checkpoint formats:
    # - checkpoint dict with 'state_dict'
    # - checkpoint dict with 'model_state_dict'
    # - raw state_dict saved directly
    state_dict = None
    if isinstance(ck, dict):
        if 'state_dict' in ck:
            state_dict = ck['state_dict']
        elif 'model_state_dict' in ck:
            state_dict = ck['model_state_dict']
        else:
            # Heuristic: if the dict values look like tensors (have .dim), treat as state_dict
            try:
                vals = list(ck.values())
                if vals and all(hasattr(v, 'dim') for v in vals):
                    state_dict = ck
            except Exception:
                state_dict = None
    else:
        # ck may be a raw state_dict (mapping)
        try:
            vals = list(ck.values())
            if vals and all(hasattr(v, 'dim') for v in vals):
                state_dict = ck
        except Exception:
            state_dict = None

    if state_dict is None:
        print('WARNING: checkpoint', args.model, 'does not contain a recognized state_dict format; skipping evaluation for this model')
        return

    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print('ERROR loading state_dict for', args.model, ':', e)
        print('Skipping evaluation for this model')
        return
    model.eval()

    # Now that model is successfully loaded, create output folders
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, 'sample_signals'), exist_ok=True)

    preds = []
    trues = []
    trues_sev = []
    for ii in idxs:
        x = X[ii:ii+1].astype('float32')
        if mean is not None and std is not None:
            try:
                x = (x - mean) / (std + 1e-9)
            except Exception:
                x = (x - mean.reshape(1,1,mean.shape[-2],1)) / (std.reshape(1,1,std.shape[-2],1) + 1e-9)
        with torch.no_grad():
            inp = torch.from_numpy(x)
            if inp.dim() == 5 and inp.size(2) == 1:
                inp = inp.squeeze(2)
            pf = model(inp)
            pred = int(pf.argmax(dim=1).item())
        preds.append(pred); trues.append(int(y[ii]))
        if y_sev is not None:
            trues_sev.append(int(y_sev[ii]))

    # metrics
    cm = confusion_matrix(trues, preds, labels=sorted(list(set(trues) | set(preds))))
    labels_sorted = [rev.get(i, str(i)) for i in sorted(list(set(trues) | set(preds)))]
    plot_confusion(cm, labels_sorted, os.path.join(args.out, 'confusion_matrix.png'))
    report = classification_report(trues, preds, target_names=labels_sorted, zero_division=0)
    with open(os.path.join(args.out, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # sample signal plots: pick some correct/incorrect examples
    os.makedirs(os.path.join(args.out,'sample_signals'), exist_ok=True)
    # pick up to 20 examples
    sample_idxs = list(idxs)
    random.shuffle(sample_idxs)
    count = 0
    for ii in sample_idxs:
        if count >= 20: break
        true = int(y[ii]); pred = preds[list(idxs).index(ii)] if ii in idxs else None
        Xsig = X[ii,0]  # (C,W)
        sev_tag = (f"_s{int(y_sev[ii])}" if y_sev is not None else "")
        outp = os.path.join(args.out,'sample_signals', f'sample_{ii}_t{true}_p{pred}{sev_tag}.png')
        plot_sample_signal(Xsig, true, pred, rev, outp)
        count += 1

    print('Saved results to', args.out)
    # save severity distribution if available
    if y_sev is not None:
        unique, counts = np.unique(y_sev, return_counts=True)
        sev_counts = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
        with open(os.path.join(args.out, 'severity_distribution.txt'), 'w') as f:
            f.write(json.dumps(sev_counts, indent=2))
        print('Saved severity distribution to', os.path.join(args.out, 'severity_distribution.txt'))

if __name__ == '__main__':
    main()
