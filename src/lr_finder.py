#!/usr/bin/env python3
"""
LR Finder: quickly find a good learning rate by training for a few batches
while exponentially increasing the learning rate, then plotting loss vs LR.

Usage:
  python src/lr_finder.py --data-path data/prepared --output-fig lr_finder.png
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset import VideoFacesDataset
from models import DeepfakeModel
from train_improved import collate_batch, FocalLoss


class LRFinder:
    def __init__(self, model, device, criterion, optimizer):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lrs = []
        self.losses = []
        self.best_loss = None

    def find_lr(self, train_loader, init_lr=1e-4, final_lr=10, num_batches=100):
        """
        Exponentially increase LR over num_batches and record loss.
        """
        self.model.train()
        self.lrs = []
        self.losses = []
        self.best_loss = None

        # exponentially space learning rates
        lr_schedule = np.logspace(np.log10(init_lr), np.log10(final_lr), num_batches)

        pbar = tqdm(enumerate(train_loader), total=num_batches, desc="LR Finder")

        for batch_idx, batch in pbar:
            if batch_idx >= num_batches:
                break

            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                frames, A_norm, labels = batch
            else:
                continue

            frames = frames.to(self.device)
            A_norm = A_norm.to(self.device)
            labels = labels.to(self.device)

            # set LR
            lr = float(lr_schedule[batch_idx])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # forward pass
            self.optimizer.zero_grad()
            logits = self.model(frames, A_norm)
            loss = self.criterion(logits, labels)

            # check for NaN
            if torch.isnan(loss):
                print(f"\n⚠ NaN loss encountered at LR={lr:.2e}, stopping.")
                break

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # record
            loss_val = loss.item()
            self.lrs.append(lr)
            self.losses.append(loss_val)

            # track best loss
            if self.best_loss is None or loss_val < self.best_loss:
                self.best_loss = loss_val

            pbar.set_postfix({'loss': f'{loss_val:.4f}', 'lr': f'{lr:.2e}'})

        return self.lrs, self.losses

    def plot(self, output_fig='lr_finder.png', skip_first=10, skip_last=5):
        """Plot LR vs loss. Skip first and last few points for clarity."""
        if len(self.lrs) < skip_first + skip_last:
            print("Not enough data points to plot.")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Skipping plot.")
            return

        lrs = self.lrs[skip_first:-skip_last]
        losses = self.losses[skip_first:-skip_last]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(lrs, losses, 'b-', linewidth=2)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        ax.set_title('LR Finder: Loss vs Learning Rate')
        ax.grid(True, alpha=0.3)

        fig.savefig(output_fig, dpi=100, bbox_inches='tight')
        print(f"✓ Plot saved to {output_fig}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='LR Finder for Deepfake Detection')
    parser.add_argument('--data-path', type=str, default='data/prepared', help='Path to prepared data directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--init-lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--final-lr', type=float, default=10, help='Final learning rate')
    parser.add_argument('--num-batches', type=int, default=100, help='Number of batches to sample')
    parser.add_argument('--output-fig', type=str, default='lr_finder.png', help='Output figure path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"LR Finder for Deepfake Detection")
    print(f"{'='*80}")
    print(f"Data Path: {args.data_path}")
    print(f"Batch Size: {args.batch_size}")
    print(f"LR Range: [{args.init_lr:.2e}, {args.final_lr:.2e}]")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")

    # Setup device
    device = torch.device(args.device)

    # Load dataset
    print("Loading dataset...")
    dataset = VideoFacesDataset(args.data_path)
    print(f"Total videos: {len(dataset)}")

    # Use first 80% for LR finding
    train_size = int(len(dataset) * 0.8)
    train_dataset, _ = random_split(dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=False
    )

    print("Initializing model...")
    model = DeepfakeModel().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.init_lr,
        weight_decay=0.0001,
        betas=(0.9, 0.999)
    )

    # Loss function (simple cross-entropy for quick search)
    criterion = nn.CrossEntropyLoss()

    # LR Finder
    lr_finder = LRFinder(model, device, criterion, optimizer)
    print("\nFinding optimal LR range...")
    lrs, losses = lr_finder.find_lr(train_loader, init_lr=args.init_lr, final_lr=args.final_lr, num_batches=args.num_batches)

    # Report
    print(f"\n{'='*80}")
    print(f"LR Finder Results:")
    print(f"  Best LR: {lrs[losses.index(min(losses))]:.2e} (loss={min(losses):.4f})")
    print(f"  Range: [~{lrs[len(lrs)//4]:.2e}, ~{lrs[3*len(lrs)//4]:.2e}]")
    print(f"{'='*80}\n")

    # Plot
    lr_finder.plot(output_fig=args.output_fig)

    print(f"Recommended LR for training: {lrs[losses.index(min(losses))]:.2e}")


if __name__ == '__main__':
    main()
