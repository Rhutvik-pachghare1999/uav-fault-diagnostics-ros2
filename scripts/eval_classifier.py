"""Evaluate a trained CNN classifier on the SEALED held-out test split.

Protocol v2 (leakage-audited):
  * Evaluates ONLY windows of the requested split (default: test) that are
    ORIGINALS (is_aug == 0). Augmented copies and train/val windows are
    rejected, so the reported accuracy is a true held-out number.
  * By default EVERY test window is scored (no random subsampling of the
    whole dataset, which historically mixed train/val/test windows).
  * If both the checkpoint and the h5 carry a manifest sha256, they must
    MATCH or evaluation refuses to run (stale-model guard).
  * The model's channel list (meta['vars']) is aligned to the h5 channels by
    name, so IMU-only (9-ch) models evaluate correctly against the 13-ch h5.

Outputs (in --out):
  confusion_matrix.png, classification_report.txt, predictions.npz,
  eval_meta.json (protocol + provenance + metrics), per_run_accuracy.json,
  sample_signals/ (a few example windows)

Usage:
  python3 scripts/eval_classifier.py --h5 ml_sealed.h5 --model models/cnn_sealed.pth \
      --out results/eval_sealed
"""
import argparse, ast, json, os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix)

SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


def parse_h5_meta(f):
    raw = f.attrs.get("meta", "{}")
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(raw)  # safe: literals only
        except Exception:
            return {}


def plot_confusion(cm, labels, outpath):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_ylabel('True'); ax.set_xlabel('Pred')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center',
                    color='black', fontsize=6)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_sample_signal(X, true_label, pred_label, label_map, outpath):
    C, W = X.shape
    t = np.arange(W)
    fig, ax = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
    for i, ch in enumerate(range(min(4, C))):
        ax[i].plot(t, X[ch], lw=0.7)
        ax[i].set_ylabel(f'ch{ch}')
    ax[-1].set_xlabel('t')
    fig.suptitle(f'True: {label_map.get(true_label, str(true_label))} '
                 f'Pred: {label_map.get(pred_label, str(pred_label))}')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def select_split_windows(f, split_name):
    if "split" not in f or "is_aug" not in f:
        raise SystemExit(
            "This h5 has no split/is_aug columns. Sealed evaluation requires "
            "ml_sealed.h5 (build it with scripts/build_sealed_dataset.py). "
            "Evaluating against a non-sealed file would mix train/val/test "
            "windows and produce inflated, non-held-out numbers.")
    split_col = f["split"][:]
    is_aug = f["is_aug"][:]
    mask = (split_col == SPLIT_IDS[split_name]) & (is_aug == 0)
    return np.where(mask)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--h5', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', default='results/eval_sealed')
    p.add_argument('--split', default='test', choices=list(SPLIT_IDS),
                   help='which sealed partition to evaluate (default: held-out test)')
    p.add_argument('--max-samples', type=int, default=0,
                   help='optional: subsample WITHIN the selected split only '
                        '(0 = use every window of the split)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--allow-train-eval', action='store_true',
                   help='escape hatch for diagnostics; NEVER for reported metrics')
    args = p.parse_args()
    if args.split == 'train' and not args.allow_train_eval:
        raise SystemExit("Refusing to evaluate on the TRAIN split without "
                         "--allow-train-eval (training metrics are not "
                         "held-out results).")

    import torch
    ck = torch.load(args.model, map_location='cpu')
    if isinstance(ck, dict) and 'state_dict' in ck:
        state_dict = ck['state_dict']
    elif isinstance(ck, dict) and 'model_state_dict' in ck:
        state_dict = ck['model_state_dict']
    else:
        state_dict = ck  # raw state_dict
    meta = ck.get('meta', {}) if isinstance(ck, dict) else {}
    mean = np.array(meta.get('mean'), dtype='float32') if meta.get('mean') is not None else None
    std = np.array(meta.get('std'), dtype='float32') if meta.get('std') is not None else None
    n_faults = int(meta.get('n_faults', 0)) or None
    base_filters = int(meta.get('base_filters', 32))

    with h5py.File(args.h5, 'r') as f:
        h5_meta = parse_h5_meta(f)
        idxs = select_split_windows(f, args.split)
        if len(idxs) == 0:
            raise SystemExit(f"split '{args.split}' has no original windows")
        # manifest provenance guard
        model_sha = meta.get('manifest_sha256')
        h5_sha = h5_meta.get('manifest_sha256')
        if model_sha and h5_sha and model_sha != h5_sha:
            raise SystemExit(f"MANIFEST MISMATCH: model trained on manifest "
                             f"{str(model_sha)[:12]}… but h5 was built from "
                             f"{str(h5_sha)[:12]}… — refusing to evaluate")
        X = f['X'][sorted(idxs)]                 # h5py needs sorted unique idxs
        X = X[np.argsort(idxs, kind='stable')]    # restore requested order
        y = f['y_fault'][sorted(idxs)][np.argsort(idxs, kind='stable')]
        y_sev = (f['y_sev'][sorted(idxs)][np.argsort(idxs, kind='stable')]
                 if 'y_sev' in f else None)
        run_id = (f['run_id'][sorted(idxs)][np.argsort(idxs, kind='stable')]
                  if 'run_id' in f else None)
        fault_map = h5_meta.get('fault_label_map', {})
        h5_vars = list(h5_meta.get('vars', []))

    # channel alignment by name
    model_vars = list(meta.get('vars', []))
    if model_vars:
        if h5_vars:
            missing = [v for v in model_vars if v not in h5_vars]
            if missing:
                raise SystemExit(f"model needs channels {missing} not present in h5 ({h5_vars})")
            ch = [h5_vars.index(v) for v in model_vars]
            if ch != list(range(len(h5_vars))):   # not identity -> subset model
                X = np.ascontiguousarray(X[:, :, ch, :])
                print(f"channel subset from model meta: {model_vars}")
        else:
            print("WARNING: h5 has no channel metadata; assuming identical layout")

    if n_faults is None:
        n_faults = int(y.max()) + 1

    from cnn_classifier import PaperCNN
    model = PaperCNN(in_channels=1, base_filters=base_filters, num_classes=n_faults)
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        raise SystemExit(f"ERROR loading state_dict from {args.model}: {e}")
    model.eval()

    if args.max_samples and args.max_samples > 0 and len(X) > args.max_samples:
        rng = np.random.RandomState(args.seed)
        keep = rng.choice(len(X), args.max_samples, replace=False)
        X, y = X[keep], y[keep]
        if y_sev is not None: y_sev = y_sev[keep]
        if run_id is not None: run_id = run_id[keep]
        print(f"subsampled WITHIN {args.split} split: {args.max_samples}/{len(idxs)} windows (seeded)")

    # batched inference over ALL selected windows
    preds = np.empty(len(X), dtype='int64')
    with torch.no_grad():
        for s in range(0, len(X), args.batch_size):
            xb = X[s:s + args.batch_size].astype('float32')
            if mean is not None and std is not None:
                xb = (xb - mean) / (std + 1e-9)
            inp = torch.from_numpy(xb)
            if inp.dim() == 3:
                inp = inp.unsqueeze(1)
            preds[s:s + args.batch_size] = model(inp).argmax(dim=1).numpy()

    rev = {v: k for k, v in fault_map.items()} if fault_map else {}
    labels_used = sorted(set(y.tolist()) | set(preds.tolist()))
    label_names = [rev.get(i, str(i)) for i in labels_used]

    win_acc = float((preds == y).mean())
    bal_acc = float(balanced_accuracy_score(y, preds))
    per_run = {}
    if run_id is not None:
        accs = []
        for r in np.unique(run_id):
            m = run_id == r
            per_run[int(r)] = {"acc": float((preds[m] == y[m]).mean()), "n": int(m.sum())}
            accs.append(per_run[int(r)]["acc"])
        run_mean = float(np.mean(accs)); run_std = float(np.std(accs)); run_min = float(np.min(accs))
    else:
        run_mean = run_std = run_min = None

    os.makedirs(args.out, exist_ok=True)
    cm = confusion_matrix(y, preds, labels=labels_used)
    plot_confusion(cm, label_names, os.path.join(args.out, 'confusion_matrix.png'))
    report = classification_report(y, preds, target_names=label_names, zero_division=0)
    with open(os.path.join(args.out, 'classification_report.txt'), 'w') as f:
        f.write(report)
    np.savez(os.path.join(args.out, 'predictions.npz'),
             trues=y, preds=preds, sev=y_sev if y_sev is not None else np.array([], dtype='int64'),
             run_id=run_id if run_id is not None else np.array([], dtype='int64'),
             label_names=np.array(label_names), class_idx=np.array(labels_used, dtype='int64'))
    eval_meta = {
        "protocol": "sealed-run-grouped-v2",
        "split_evaluated": args.split,
        "originals_only": True,
        "model": args.model,
        "model_manifest_sha256": model_sha,
        "h5_manifest_sha256": h5_sha,
        "n_windows": int(len(X)),
        "n_windows_full_split": int(len(idxs)),
        "window_accuracy": win_acc,
        "balanced_accuracy": bal_acc,
        "per_run_accuracy_mean": run_mean,
        "per_run_accuracy_std": run_std,
        "per_run_accuracy_min": run_min,
        "seed": args.seed,
    }
    with open(os.path.join(args.out, 'eval_meta.json'), 'w') as f:
        json.dump(eval_meta, f, indent=2)
    if per_run:
        with open(os.path.join(args.out, 'per_run_accuracy.json'), 'w') as f:
            json.dump(per_run, f, indent=2)

    # a few example signal plots
    sig_dir = os.path.join(args.out, 'sample_signals')
    os.makedirs(sig_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    for ii in rng.choice(len(X), size=min(8, len(X)), replace=False):
        outp = os.path.join(sig_dir, f'sample_{ii}_t{int(y[ii])}_p{int(preds[ii])}.png')
        plot_sample_signal(X[ii, 0], int(y[ii]), int(preds[ii]), rev, outp)

    print(f"[{args.split}] windows={len(X)}  window_acc={win_acc:.4f}  "
          f"balanced_acc={bal_acc:.4f}")
    if run_mean is not None:
        print(f"per-run accuracy: mean={run_mean:.4f} std={run_std:.4f} "
              f"min={run_min:.4f} over {len(per_run)} runs")
    print('Saved results to', args.out)


if __name__ == '__main__':
    main()
