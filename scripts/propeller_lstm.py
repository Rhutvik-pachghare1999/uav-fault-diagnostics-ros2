
import argparse, os, numpy as np
from pathlib import Path

def synthesize_data(n_samples=3000, window=50):
    X = np.random.uniform(1000, 12000, size=(n_samples, window, 1)).astype("float32")
    thrust = 1e-6 * (X.squeeze(-1)**2)
    torque = 5e-8 * (X.squeeze(-1)**2)
    Y = np.stack([thrust.mean(axis=1), torque.mean(axis=1)], axis=1).astype("float32")
    return X, Y

def load_csv_windows(csv_path, window=50, step=1):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if 'rpm' not in df.columns:
        raise ValueError("CSV must include a 'rpm' column")
    rpms = df['rpm'].to_numpy()
    Xs, Ys = [], []
    for i in range(0, len(rpms)-window+1, step):
        Xw = rpms[i:i+window].reshape(window, 1)
        thrust = (1e-6 * (Xw.squeeze()**2)).mean()
        torque = (5e-8 * (Xw.squeeze()**2)).mean()
        Xs.append(Xw); Ys.append([thrust, torque])
    return np.array(Xs, dtype="float32"), np.array(Ys, dtype="float32")


import torch
import torch.nn as nn


class StackedLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, out_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.ReLU(), nn.Linear(hidden_size//2, out_dim))
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-csv", type=str, help="Load-cell CSV with columns 'rpm','thrust','torque' (optional)")
    p.add_argument("--out", type=str, default="models/prop_lstm.pth")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--synthesize", action='store_true')
    args = p.parse_args()

    if args.synthesize or not args.data_csv:
        X, Y = synthesize_data(n_samples=2000, window=args.window)
    else:
        X, Y = load_csv_windows(args.data_csv, window=args.window, step=1)
    try:
        import torch, torch.optim as optim
    except Exception as e:
        print("Torch not available. Create the venv and install torch. Exiting.")
        raise

    # Use module-level StackedLSTM defined above

    device = "cuda" if (hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available()) else "cpu"
    import torch
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = StackedLSTM().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    for epoch in range(args.epochs):
        model.train()
        tot, cnt = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); cnt += 1
        print(f"Epoch {epoch+1}/{args.epochs} loss={tot/cnt:.6f}")
    os.makedirs(Path(args.out).parent, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print("Saved LSTM model to", args.out)

if __name__ == "__main__":
    main()
