"""Generate more realistic synthetic runs by varying waypoints, payload, CG offsets, and motor behavior.

This produces many runs with `imu.csv` and `meta.json` that emulate different flight paths,
payloads, and faulty propellers. It builds on the simple perturbation approach but adds
per-run waypoint-driven RPM envelopes and payload scaling to make datasets more realistic.

Usage:
  python3 scripts/generate_realistic_runs.py --base isaac_dataset/run_auto_1764651130 --out-count 50
"""
import argparse, os, json, shutil
import numpy as np, pandas as pd

def generate_run(base_run, out_run, label_mask, payload_kg=1.5, cg_bias=(0.0,0.0)):
    os.makedirs(out_run, exist_ok=True)
    # read base imu
    src = None
    for nm in ('imu.csv','imu_full.csv'):
        p = os.path.join(base_run, nm)
        if os.path.exists(p): src = p; break
    if src is None:
        # Base imu not found for this base run. Log and signal failure to caller.
        print(f"Warning: base imu not found in base run: {base_run}")
        return False
    df = pd.read_csv(src)
    T = len(df)
    # create waypoint-driven RPM envelope: low-frequency sinusoid per motor
    t = np.linspace(0, 2*np.pi, T)
    base_rpms = np.zeros((4,T))
    for i in range(4):
        freq = np.random.uniform(0.5, 1.5)
        phase = np.random.uniform(0, 2*np.pi)
        amp = np.random.uniform(0.02, 0.08)  # relative variation
        # use median rpm of that column as baseline if present
        col = f'rpm{i+1}'
        if col in df.columns:
            baseline = np.nanmedian(df[col])
        else:
            baseline = 3000.0
        base_rpms[i] = baseline * (1 + amp * np.sin(freq * t + phase))

    # apply label mask: for each prop that is marked faulty reduce RPM capability
    for i in range(4):
        if (label_mask >> i) & 1:
            defect_factor = np.random.uniform(0.6, 0.9)
            base_rpms[i] *= defect_factor

    # apply payload scaling (heavier payload increases rpms to maintain lift)
    payload_factor = 1.0 + (payload_kg - 1.5) * 0.1
    base_rpms *= payload_factor

    # construct new dataframe: copy original and replace rpm columns if present
    new_df = df.copy()
    for i in range(4):
        col = f'rpm{i+1}'
        if col in new_df.columns:
            new_df[col] = base_rpms[i]
        else:
            new_df[col] = base_rpms[i]

    # add small attitude drift from CG bias
    for j,angle in enumerate(['roll','pitch']):
        if angle in new_df.columns:
            new_df[angle] = new_df[angle] + (cg_bias[j] * (1 + 0.005 * np.sin(0.2*t)))

    new_df.to_csv(os.path.join(out_run, 'imu.csv'), index=False)
    # copy state.csv etc if exist
    for nm in ('state.csv','thrust_tau.csv'):
        s = os.path.join(base_run, nm)
        if os.path.exists(s): shutil.copy(s, os.path.join(out_run, nm))

    # meta.json
    fault_label = f'label_{label_mask}'
    meta = {'fault_type': fault_label, 'fault_params': {'unbalance': float(np.random.uniform(0.01, 0.6))}, 'payload': payload_kg, 'cg_bias': list(cg_bias)}
    with open(os.path.join(out_run, 'meta.json'),'w') as f:
        json.dump(meta, f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--out-count', type=int, default=50)
    p.add_argument('--out-prefix', default='run_real')
    args = p.parse_args()

    base = args.base
    if not os.path.isdir(base):
        print('Base not found:', base); return
    # Quick check: ensure base run contains imu data
    has_imu = any(os.path.exists(os.path.join(base, nm)) for nm in ('imu.csv','imu_full.csv'))
    if not has_imu:
        print('Base run has no imu.csv or imu_full.csv; skipping realistic run generation for base:', base)
        return
    out_root = os.path.dirname(base)
    created = 0
    # cycle through labels and generate runs with varied payloads and cg biases
    masks = list(range(16))
    i = 0
    while created < args.out_count:
        mask = masks[i % len(masks)]
        payload = np.random.choice([1.2, 1.5, 1.8, 2.0])
        cg_bias = (np.random.uniform(-0.02, 0.02), np.random.uniform(-0.02, 0.02))
        run_name = f"{args.out_prefix}_{mask:02d}_{created}"
        run_dir = os.path.join(out_root, run_name)
        ok = generate_run(base, run_dir, mask, payload_kg=payload, cg_bias=cg_bias)
        if not ok:
            print('Skipping creation of', run_dir, 'because base imu missing')
            break
        created += 1; i += 1
    print('Generated', created, 'realistic runs in', out_root)

if __name__ == '__main__':
    main()
