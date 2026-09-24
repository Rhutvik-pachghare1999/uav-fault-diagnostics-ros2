"""scripts/smoke_test.py
Quick smoke test: load `ml_dataset_v2.h5` and a `models/cnn_multi.pth` checkpoint
and run a single forward pass on CPU. Prints shapes and predicted labels.

Usage:
  python3 scripts/smoke_test.py --h5 ml_dataset_v2.h5 --model models/cnn_multi.pth
"""
import argparse, os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--model", required=True)
    args = p.parse_args()

    if not os.path.exists(args.h5):
        print("HDF5 dataset not found:", args.h5); return
    if not os.path.exists(args.model):
        print("Model checkpoint not found:", args.model); return

    import h5py, numpy as np, torch
    from cnn_classifier import PaperCNN

    with h5py.File(args.h5, 'r') as f:
        X_all = f['X'][:]
        y_sev = f['y_sev'][:] if 'y_sev' in f else None
        ur_arr = f['ur'][:] if 'ur' in f else None
        import json, ast
        _mr = f.attrs.get('meta', '{}')
        if isinstance(_mr, (bytes, bytearray)):
            _mr = _mr.decode('utf-8', errors='ignore')
        try:
            meta = json.loads(_mr)
        except Exception:
            try:
                meta = ast.literal_eval(_mr)  # safe: literals only
            except Exception:
                meta = {}
        fault_map = meta.get('fault_label_map', {})
    print('Loaded h5:', args.h5, 'X shape=', X_all.shape)

    ck = torch.load(args.model, map_location='cpu')
    n_faults = ck.get('meta', {}).get('n_faults', 16)
    model = PaperCNN(in_channels=1, base_filters=32, num_classes=n_faults)
    model.load_state_dict(ck['state_dict'])
    model.eval()

    inp = torch.from_numpy(X_all[:1].astype('float32'))
    if inp.dim() == 5 and inp.size(2) == 1:
        inp = inp.squeeze(2)
    with torch.no_grad():
        pf = model(inp)
    fid = int(pf.argmax(dim=1).item())
    rev = {v:k for k,v in fault_map.items()}
    out = {'fault_id': fid, 'label': rev.get(fid, str(fid))}
    # include sample true severity if available for the first sample
    if y_sev is not None and len(y_sev) > 0:
        out['true_severity'] = int(y_sev[0])
    if ur_arr is not None and len(ur_arr) > 0:
        out['true_ur'] = float(ur_arr[0])
    print('Smoke test result:', out)

if __name__ == '__main__':
    main()
