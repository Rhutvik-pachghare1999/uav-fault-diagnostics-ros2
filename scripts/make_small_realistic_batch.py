
import argparse, os, shutil, json
import numpy as np, pandas as pd

def make_variant(base_run, out_run, unbalance=0.1, seed=None):
    np.random.seed(seed)
    os.makedirs(out_run, exist_ok=True)
    # copy state and thrust files if present
    for nm in ('state.csv','thrust_tau.csv'):
        s = os.path.join(base_run, nm)
        if os.path.exists(s): shutil.copy(s, os.path.join(out_run, nm))
    # pick imu source
    imu_src = None
    for nm in ('imu.csv','imu_full.csv'):
        p = os.path.join(base_run, nm)
        if os.path.exists(p): imu_src = p; break
    if imu_src is None:
        print('Base run has no imu file:', base_run); return False
    df = pd.read_csv(imu_src)
    T = len(df)
    # ensure rpm columns present
    for i in range(1,5):
        col = f'rpm{i}'
        if col not in df.columns:
            df[col] = 3000.0
    # time array from file if available
    if 'time' in df.columns:
        t = df['time'].values
    else:
        t = np.arange(T) * 0.01

    # white noise levels (conservative, realistic)
    gyro_sigma = 0.01  # rad/s
    accel_sigma = 0.05  # m/s^2

    # low-frequency bias random walk
    def random_walk(scale, size):
        steps = np.random.normal(loc=0.0, scale=scale, size=size)
        return np.cumsum(steps)

    # generate vibration sinusoid based on rpm1 median frequency
    rpm_median = np.median(df['rpm1'].values)
    f_r = (rpm_median/60.0) if rpm_median>0 else 50.0
    omega = 2*np.pi*f_r

    # create perturbations
    acc_vib = (unbalance * 0.5) * np.sin(2*np.pi*f_r*t)  # m/s^2-ish
    gyro_vib = (unbalance * 0.005) * np.sin(2*np.pi*f_r*t + 0.5)

    # apply to accel and gyro columns if present
    for acc_col in ('acc_x','acc_y','acc_z'):
        if acc_col in df.columns:
            df[acc_col] = df[acc_col] + acc_vib + np.random.normal(0, accel_sigma, size=T)
    for g_col in ('gyro_x','gyro_y','gyro_z'):
        if g_col in df.columns:
            df[g_col] = df[g_col] + gyro_vib + np.random.normal(0, gyro_sigma, size=T)

    # add small RPM jitter and occasional transients
    for i in range(1,5):
        col = f'rpm{i}'
        jitter = np.random.normal(0, 0.01, size=T)
        df[col] = df[col] * (1.0 + jitter)
        # if this prop is 'faulty' (simulate single-prop unbalance by index choose), reduce RPM
    # create a single faulty prop index based on unbalance magnitude
    bad_idx = int(min(3, max(0, int(unbalance*10) % 4)))
    df[f'rpm{bad_idx+1}'] = df[f'rpm{bad_idx+1}'] * (1.0 - 0.1*unbalance)

    # add slow bias drift to orientations
    for a in ('roll','pitch','yaw'):
        if a in df.columns:
            df[a] = df[a] + 0.001 * random_walk(0.0001, T)

    # write new imu.csv
    df.to_csv(os.path.join(out_run, 'imu.csv'), index=False)
    # write meta
    meta = {'fault_type': f'unb_{int(unbalance*100)}', 'fault_params': {'unbalance': float(unbalance)}}
    with open(os.path.join(out_run, 'meta.json'),'w') as f:
        json.dump(meta, f)
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--n', type=int, default=6)
    p.add_argument('--out-prefix', default='run_real_small')
    args = p.parse_args()
    base = args.base
    if not os.path.isdir(base):
        print('Base not found:', base); return
    out_root = os.path.dirname(base)
    created = 0
    for i in range(args.n):
        unb = np.random.uniform(0.02, 0.45)
        run_name = f"{args.out_prefix}_{i:02d}_0"
        run_dir = os.path.join(out_root, run_name)
        ok = make_variant(base, run_dir, unbalance=unb, seed=42+i)
        if ok:
            print('Created', run_dir, 'unbalance=', unb)
            created += 1
    print('Created', created, 'small realistic runs')

if __name__ == '__main__':
    main()
