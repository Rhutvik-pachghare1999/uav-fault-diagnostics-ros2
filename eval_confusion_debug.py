#!/usr/bin/env python3
"""Evaluate model and produce numeric confusion matrix CSV plus top confusion pairs.

Usage:
  python3 scripts/eval_confusion_debug.py --h5 <h5> --model <pth> --out <outdir>
"""
import ast
import argparse, os, json
import h5py, numpy as np, pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', default='results')
    p.add_argument('--max-samples', type=int, default=20000)
    args = p.parse_args()

    import torch
    from sklearn.metrics import confusion_matrix, classification_report
    from cnn_classifier import PaperCNN

    os.makedirs(args.out, exist_ok=True)

    # load dataset
    with h5py.File(args.h5, 'r') as f:
        X = f['X'][:]
        y = f['y_fault'][:]
        try:
            meta = ast.literal_eval(f.attrs.get('meta','{}'))
        except Exception:
            meta = {}
        dataset_label_map = meta.get('fault_label_map', {})
    unique_labels = np.unique(y)

    # load model and meta
    ck = torch.load(args.model, map_location='cpu')
    meta_ck = ck.get('meta', {})
    model_map = meta_ck.get('fault_label_map', {})
    n_faults = len(unique_labels)

    # Build readable label names for dataset indices
    name_by_idx = {}
    if dataset_label_map:
        for name, idx in dataset_label_map.items():
            name_by_idx[int(idx)] = name
    else:
        for idx in unique_labels:
            name_by_idx[int(idx)] = str(int(idx))

    # instantiate model sized to dataset classes
    model = PaperCNN(in_channels=1, base_filters=32, num_classes=n_faults)
    # load state_dict robustly
    state_dict = None
    if isinstance(ck, dict):
        state_dict = ck.get('state_dict', ck)
    else:
        state_dict = ck
    # normalize keys
    sd = {k.replace('module.',''):v for k,v in state_dict.items()}
    # partial load
    msd = model.state_dict()
    to_load = {k:v for k,v in sd.items() if k in msd and msd[k].shape == v.shape}
    msd.update(to_load)
    model.load_state_dict(msd)
    model.eval()

    # subsample
    N = len(X)
    idxs = np.arange(N)
    if args.max_samples and N > args.max_samples:
        idxs = np.random.choice(idxs, args.max_samples, replace=False)

    preds = []
    trues = []
    with torch.no_grad():
        for ii in idxs:
            x = X[ii:ii+1].astype('float32')
            inp = torch.from_numpy(x)
            if inp.dim() == 5 and inp.size(2) == 1:
                inp = inp.squeeze(2)
            outp = model(inp)
            pred = int(outp.argmax(dim=1).item())
            preds.append(pred); trues.append(int(y[ii]))

    labels_union = sorted(list(set(trues) | set(preds)))
    cm = confusion_matrix(trues, preds, labels=labels_union)
    # map label indices to names
    names = [name_by_idx.get(int(l), str(int(l))) for l in labels_union]

    # save CSV
    df = pd.DataFrame(cm, index=names, columns=names)
    csv_path = os.path.join(args.out, 'confusion_matrix_numeric.csv')
    df.to_csv(csv_path)

    # print top confusion pairs (off-diagonal largest values)
    cm_off = cm.copy().astype(int)
    np.fill_diagonal(cm_off, 0)
    pairs = []
    for i in range(cm_off.shape[0]):
        for j in range(cm_off.shape[1]):
            if cm_off[i,j] > 0:
                pairs.append((cm_off[i,j], names[i], names[j]))
    pairs.sort(reverse=True)
    top_pairs = pairs[:20]

    out_txt = os.path.join(args.out, 'confusion_debug.txt')
    with open(out_txt, 'w') as f:
        f.write('labels_union:\n')
        f.write(json.dumps({int(k):v for k,v in zip(labels_union,names)}, indent=2))
        f.write('\n\nTop confusion pairs (count, true, pred):\n')
        for c,t,p in top_pairs:
            f.write(f"{c:5d}  {t} -> {p}\n")

    print('Wrote', csv_path)
    print('Wrote', out_txt)
    print('Top confusions:')
    for row in top_pairs[:10]:
        print(row)

if __name__ == '__main__':
    main()
