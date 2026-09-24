
import argparse, os, json
import h5py, numpy as np, matplotlib.pyplot as plt

def inspect(h5path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(h5path, 'r') as f:
        X = f['X'][:]  # (N,1,C,W)
        y_fault = f['y_fault'][:]
        y_sev = f['y_sev'][:]
        ur = f['ur'][:]
        meta = json.loads(f.attrs.get('meta', b"{}"))

    N,_,C,W = X.shape
    stats = []
    stats.append(f"file: {h5path}")
    stats.append(f"samples: {N}")
    # class counts
    unique, counts = np.unique(y_fault, return_counts=True)
    stats.append("class_counts:")
    for u,c in zip(unique, counts):
        stats.append(f"  {u}: {c}")
    # severity counts
    unique2, counts2 = np.unique(y_sev, return_counts=True)
    stats.append("\nseverity_counts:")
    for u,c in zip(unique2, counts2):
        stats.append(f"  {u}: {c}")
    # ur summary
    stats.append(f"\nur summary:")
    stats.append(f"  min={ur.min():.6f}, max={ur.max():.6f}, mean={ur.mean():.6f}")

    stats.append("\nper-channel stats (mean,std,min,max):")
    # X shape: (N,1,C,W) -> collapse windows
    Xc = X.reshape(N, C, W)
    for ch in range(C):
        arr = Xc[:, ch, :].ravel()
        stats.append(f" ch{ch}: mean={arr.mean():.6f} std={arr.std():.6f} min={arr.min():.6f} max={arr.max():.6f}")

    txt_path = os.path.join(out_dir, 'data_inspect_small.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(stats))

    # Plot a few example windows (time-domain)
    n_plot = min(6, N)
    t = np.arange(W)
    plt.figure(figsize=(12, 8))
    for i in range(n_plot):
        plt.subplot(n_plot, 1, i+1)
        # plot rpm channel 0 and an IMU accel/gyro channel scaled
        rpm = X[i,0,0,:]
        acc0 = X[i,0,4,:] if C>4 else np.zeros_like(rpm)
        plt.plot(t, rpm, label='rpm1')
        plt.plot(t, acc0 * 1000.0, label='acc_x x1000', alpha=0.7)
        plt.ylabel(f'win{i}')
        if i==0:
            plt.legend(loc='upper right')
    plt.xlabel('sample')
    plt.tight_layout()
    out_ts = os.path.join(out_dir, 'sample_windows.png')
    plt.savefig(out_ts)
    plt.close()

    # Spectrogram of an IMU channel (acc_z if present index 6 else last)
    ch_idx = 6 if C>6 else (C-1)
    sig = X[0,0,ch_idx,:]
    plt.figure(figsize=(8,4))
    plt.specgram(sig, NFFT=256, Fs=100.0, noverlap=128, cmap='viridis')
    plt.title(f'Spectrogram ch{ch_idx}')
    plt.xlabel('Time (s)')
    plt.ylabel('Freq (Hz)')
    out_sp = os.path.join(out_dir, 'spectrogram_ch{}.png'.format(ch_idx))
    plt.tight_layout()
    plt.savefig(out_sp)
    plt.close()

    print('Wrote', txt_path)
    print('Wrote', out_ts)
    print('Wrote', out_sp)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--out', default='scripts/results_proper')
    args = p.parse_args()
    inspect(args.h5, args.out)

if __name__ == '__main__':
    main()
