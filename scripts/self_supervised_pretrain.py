#!/usr/bin/env python3
"""
Self-supervised pre-training on healthy data using contrastive learning (SimCLR-style).

This pre-trains the encoder on healthy (normal) flight data to learn robust
representations before fine-tuning on fault classification.

Usage:
  python3 scripts/self_supervised_pretrain.py --h5 ml_dataset_v2_aug.h5 --out models/encoder_pretrained.pth --epochs 50
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))
from cnn_classifier import PaperCNN
from eval_harness import set_seed


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with two augmented views."""
    
    def __init__(self, X, healthy_indices, augment_fn, mean=None, std=None):
        self.X = X[healthy_indices].astype('float32')
        self.augment_fn = augment_fn
        self.mean = mean.astype('float32') if mean is not None else None
        self.std = std.astype('float32') if std is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]  # (1, C, W)
        
        # Create two augmented views
        x1 = self.augment_fn(x.copy())
        x2 = self.augment_fn(x.copy())
        
        # Normalize
        if self.mean is not None and self.std is not None:
            x1 = (x1 - self.mean) / (self.std + 1e-9)
            x2 = (x2 - self.mean) / (self.std + 1e-9)
        
        # Ensure correct shape (C, W)
        if x1.ndim == 4:  # (1, C, W)
            x1 = x1.squeeze(0)
            x2 = x2.squeeze(0)
        
        return x1.astype('float32'), x2.astype('float32')


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR)."""
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, z_i, z_j):
        """
        z_i, z_j: (batch_size, embedding_dim)
        """
        batch_size = z_i.shape[0]
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate for similarity computation
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, dim)
        
        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2*batch_size, 2*batch_size)
        
        # Create labels for positive pairs
        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels + batch_size, labels])  # (2*batch_size,)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        sim.masked_fill_(mask, -float('inf'))
        
        loss = self.criterion(sim, labels)
        return loss


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""
    
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


def augment_sample(x, noise_std=0.02, scale_range=(0.95, 1.05), shift_range=3):
    """
    Augment a single sample for contrastive learning.
    x shape: (1, C, W) or (C, W)
    """
    # Handle various input shapes
    if x.ndim == 4:  # (1, 1, C, W) or (1, C, W, 1)
        if x.shape[1] == 1:
            x = x.squeeze(1)  # (1, C, W)
        x = x.squeeze(0)  # (C, W)
    elif x.ndim == 3:  # (1, C, W)
        if x.shape[0] == 1:
            x = x.squeeze(0)  # (C, W)
    
    # Now x should be (C, W)
    if x.ndim != 2:
        raise ValueError(f"Unexpected shape after squeeze: {x.shape}")
    
    C, W = x.shape
    y = x.copy()
    
    # 1) Gaussian noise
    chan_std = y.std(axis=1, keepdims=True)
    noise = np.random.randn(C, W) * (noise_std * (chan_std + 1e-6))
    y = y + noise
    
    # 2) Channel-wise scaling (mainly for RPM channels)
    scale = np.random.uniform(scale_range[0], scale_range[1], size=(C, 1))
    y = y * scale
    
    # 3) Time shift (circular)
    shift = np.random.randint(-shift_range, shift_range + 1)
    if shift != 0:
        y = np.roll(y, shift, axis=1)
    
    # 4) Random channel dropout (simulate sensor dropout)
    if np.random.rand() < 0.1:
        drop_ch = np.random.randint(0, C)
        y[drop_ch] = 0
    
    return y[None, :, :]  # (1, C, W)


def pretrain_encoder(args):
    """Main pre-training loop."""
    
    # Load dataset
    print(f"Loading dataset from {args.h5}...")
    with h5py.File(args.h5, 'r') as f:
        X = f['X'][:]  # (N, 1, C, W)
        y = f['y_fault'][:]
        run_ids = f['run_id'][:] if 'run_id' in f else np.arange(len(X))
        
        import json, ast
        meta_raw = f.attrs.get('meta', '{}')
        if isinstance(meta_raw, (bytes, bytearray)):
            meta_raw = meta_raw.decode('utf-8', errors='ignore')
        try:
            meta = json.loads(meta_raw)
        except Exception:
            meta = ast.literal_eval(meta_raw)
        fault_map = meta.get('fault_label_map', {})
        rev_map = {v: k for k, v in fault_map.items()}
        
        # Find healthy class index
        healthy_idx = None
        for label, idx in fault_map.items():
            if 'healthy' in label.lower():
                healthy_idx = idx
                break
        
        if healthy_idx is None:
            # Assume class 0 is healthy
            healthy_idx = 0
            print(f"Warning: No 'healthy' class found, using index 0")
        else:
            print(f"Found healthy class: {healthy_idx}")
    
    # Get healthy samples
    healthy_mask = y == healthy_idx
    healthy_indices = np.where(healthy_mask)[0]
    print(f"Healthy samples: {len(healthy_indices)}")
    
    if len(healthy_indices) < 100:
        print("Not enough healthy samples for pre-training")
        return
    
    # Split healthy data (run-grouped)
    healthy_run_ids = run_ids[healthy_mask]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_rel, val_rel = next(gss.split(healthy_indices, groups=healthy_run_ids))
    tr_idx = healthy_indices[tr_rel]
    val_idx = healthy_indices[val_rel]
    
    print(f"Train: {len(tr_idx)} samples, Val: {len(val_idx)} samples")
    
    # Compute normalization stats on healthy training data
    X_tr = X[tr_idx].astype('float32')
    C = X_tr.shape[2]
    vals = X_tr.reshape(X_tr.shape[0], C, -1).transpose(1, 0, 2).reshape(C, -1)
    mean = vals.mean(axis=1).reshape(1, 1, C, 1)
    std = vals.std(axis=1).reshape(1, 1, C, 1)
    
    # Create datasets
    train_ds = ContrastiveDataset(X, tr_idx, augment_sample, mean, std)
    val_ds = ContrastiveDataset(X, val_idx, augment_sample, mean, std)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create encoder (without classification head)
    encoder = PaperCNN(in_channels=1, base_filters=32, num_classes=args.embed_dim)
    # Remove the final classification head, keep up to the FC layer
    encoder.head = nn.Identity()  # Output will be 128-dim embedding
    encoder = encoder.to(device)
    
    # Projection head
    projector = ProjectionHead(args.embed_dim, hidden_dim=512, output_dim=args.proj_dim).to(device)
    
    # Loss and optimizer
    criterion = NTXentLoss(temperature=args.temperature)
    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(args.epochs):
        # Train
        encoder.train()
        projector.train()
        train_loss = 0.0
        
        for x1, x2 in train_loader:
            x1 = x1.to(device)  # (B, C, W)
            x2 = x2.to(device)
            
            # Add channel dimension if needed
            if x1.dim() == 3:
                x1 = x1.unsqueeze(1)
                x2 = x2.unsqueeze(1)
            
            # Forward
            h1 = encoder(x1)  # (B, embed_dim)
            h2 = encoder(x2)
            z1 = projector(h1)
            z2 = projector(h2)
            
            loss = criterion(z1, z2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        encoder.eval()
        projector.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for x1, x2 in val_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                
                if x1.dim() == 3:
                    x1 = x1.unsqueeze(1)
                    x2 = x2.unsqueeze(1)
                
                h1 = encoder(x1)
                h2 = encoder(x2)
                z1 = projector(h1)
                z2 = projector(h2)
                
                loss = criterion(z1, z2)
                val_loss += loss.item()
        
        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'projector_state_dict': projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'meta': {
                    'embed_dim': args.embed_dim,
                    'proj_dim': args.proj_dim,
                    'temperature': args.temperature,
                    'mean': mean.tolist(),
                    'std': std.tolist(),
                    'fault_label_map': fault_map
                }
            }, args.out)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
    
    print(f"\nPre-training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {args.out}")
    
    return encoder, history


def main():
    parser = argparse.ArgumentParser(description="Self-supervised pre-training on healthy data")
    parser.add_argument("--h5", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--out", required=True, help="Output model path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--proj-dim", type=int, default=128, help="Projection dimension")
    parser.add_argument("--temperature", type=float, default=0.5, help="NT-Xent temperature")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    pretrain_encoder(args)


if __name__ == "__main__":
    main()