
import argparse, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def parse_log(path):
    train_losses = []
    val_acc = []
    val_loss = []
    epochs = []
    if not os.path.exists(path):
        print(f"[plot_training_curves] Warning: log file not found: {path}")
        return np.array([]), np.array([]), np.array([]), np.array([])
    with open(path,'r') as f:
        for line in f:
            # Epoch 1/100 train_loss=0.2412
            m = re.search(r'Epoch\s+(\d+)/(\d+)\s+train_loss=([0-9.]+)', line)
            if m:
                e = int(m.group(1))
                tl = float(m.group(3))
                epochs.append(e)
                train_losses.append(tl)
            m2 = re.search(r'Val fault acc:\s*([0-9.]+)\s+val_loss=([0-9.]+)', line)
            if m2:
                va = float(m2.group(1)); vl = float(m2.group(2))
                val_acc.append(va); val_loss.append(vl)
    return np.array(epochs), np.array(train_losses), np.array(val_acc), np.array(val_loss)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log', default='logs/train_realistic.log')
    p.add_argument('--out', default='results_plots')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    e, tl, va, vl = parse_log(args.log)
    if len(e):
        plt.figure(); plt.plot(e, tl, label='train_loss'); plt.xlabel('epoch'); plt.ylabel('loss'); plt.grid(True); plt.legend(); plt.savefig(os.path.join(args.out,'train_loss.png'))
    if len(va):
        plt.figure(); plt.plot(np.arange(1,len(va)+1), va, label='val_acc'); plt.xlabel('epoch'); plt.ylabel('val_acc'); plt.grid(True); plt.legend(); plt.savefig(os.path.join(args.out,'val_acc.png'))
    if len(vl):
        plt.figure(); plt.plot(np.arange(1,len(vl)+1), vl, label='val_loss'); plt.xlabel('epoch'); plt.ylabel('val_loss'); plt.grid(True); plt.legend(); plt.savefig(os.path.join(args.out,'val_loss.png'))
    print('Saved plots to', args.out)

if __name__=='__main__':
    main()
