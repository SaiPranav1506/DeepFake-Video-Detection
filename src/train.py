import argparse
import os
import random
from pathlib import Path
import sys
try:
    import torch
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    # Friendly message for missing PyTorch
    msg = (
        "PyTorch is not installed or failed to import.\n"
        "Please install PyTorch before running this script.\n\n"
        "Recommended (CPU-only) PowerShell commands:\n"
        "  python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1\n"
        "  pip install --upgrade pip\n"
        "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n\n"
        "If you have an NVIDIA GPU and want CUDA support, get the correct command from https://pytorch.org 'Get Started'\n"
        "Example (CUDA 11.8):\n"
        "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n\n"
        "After installing, re-run your training command.\n"
        "Full requirements file is available at requirements.txt and can be installed with:\n"
        "  pip install -r requirements.txt\n"
    )
    print(msg, file=sys.stderr)
    raise
import numpy as np

from dataset import VideoFacesDataset
from models import DeepfakeModel, CNNLSTMHybrid
from utils import normalize_adjacency
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import csv


def collate_batch_cnn_lstm(batch, max_frames=16, image_size=(224, 224)):
    B = len(batch)
    frames_list = []
    labels = []
    for item in batch:
        faces = item['faces']
        M = faces.shape[0]
        if M >= max_frames:
            idxs = np.linspace(0, M - 1, max_frames).astype(int)
            sel = faces[idxs]
        else:
            pad = max_frames - M
            if M == 0:
                sel = np.zeros((max_frames, image_size[0], image_size[1], 3), dtype=np.uint8)
            else:
                pads = np.repeat(faces[-1][None], pad, axis=0)
                sel = np.concatenate([faces, pads], axis=0)
        frames_list.append(sel)
        labels.append(item['label'] if item['label'] is not None else -1)

    frames = np.stack(frames_list)
    frames = torch.from_numpy(frames).permute(0, 1, 4, 2, 3).float() / 255.0
    labels = torch.tensor(labels, dtype=torch.long)
    return frames, labels


def collate_batch(batch, max_nodes=16, image_size=(224, 224)):
    # batch: list of {'faces': np.ndarray (M,H,W,C), 'label': int}
    B = len(batch)
    nodes = []
    labels = []
    for item in batch:
        faces = item['faces']
        M = faces.shape[0]
        if M >= max_nodes:
            # sample evenly
            idxs = np.linspace(0, M - 1, max_nodes).astype(int)
            sel = faces[idxs]
        else:
            # pad with last frame
            pad = max_nodes - M
            if M == 0:
                sel = np.zeros((max_nodes, image_size[0], image_size[1], 3), dtype=np.uint8)
            else:
                pads = np.repeat(faces[-1][None], pad, axis=0)
                sel = np.concatenate([faces, pads], axis=0)
        nodes.append(sel)
        labels.append(item['label'] if item['label'] is not None else -1)

    nodes = np.stack(nodes)  # (B, N, H, W, C)
    # convert to tensor and reorder to (B, N, C, H, W)
    nodes = torch.from_numpy(nodes).permute(0, 1, 4, 2, 3).float() / 255.0
    labels = torch.tensor(labels, dtype=torch.long)

    # Build adjacency: simple chain adjacency for each sample
    N = nodes.shape[1]
    A = np.zeros((B, N, N), dtype=np.float32)
    for b in range(B):
        for i in range(N - 1):
            A[b, i, i + 1] = 1.0
            A[b, i + 1, i] = 1.0
    A_norm = np.stack([normalize_adjacency(A[b]) for b in range(B)])
    A_norm = torch.from_numpy(A_norm).float()
    return nodes, A_norm, labels


def train_epoch(model, loader, device, optimizer, criterion, model_type='vit_gcn'):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for batch in loader:
        if model_type == 'cnn_lstm':
            images, labels = collate_batch_cnn_lstm(batch)
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(images)
        else:
            images, A_norm, labels = collate_batch(batch)
            images = images.to(device)
            A_norm = A_norm.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(images, A_norm)
        
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def collate_batch_cnn_lstm_with_files(batch, max_frames=16, image_size=(224, 224)):
    B = len(batch)
    frames_list = []
    labels = []
    files = []
    for item in batch:
        faces = item['faces']
        M = faces.shape[0]
        if M >= max_frames:
            idxs = np.linspace(0, M - 1, max_frames).astype(int)
            sel = faces[idxs]
        else:
            pad = max_frames - M
            if M == 0:
                sel = np.zeros((max_frames, image_size[0], image_size[1], 3), dtype=np.uint8)
            else:
                pads = np.repeat(faces[-1][None], pad, axis=0)
                sel = np.concatenate([faces, pads], axis=0)
        frames_list.append(sel)
        labels.append(item['label'] if item['label'] is not None else -1)
        files.append(item.get('file', ''))

    frames = np.stack(frames_list)
    frames = torch.from_numpy(frames).permute(0, 1, 4, 2, 3).float() / 255.0
    labels = torch.tensor(labels, dtype=torch.long)
    return frames, labels, files


def collate_batch_with_files(batch, max_nodes=16, image_size=(224, 224)):
    B = len(batch)
    nodes = []
    labels = []
    files = []
    for item in batch:
        faces = item['faces']
        M = faces.shape[0]
        if M >= max_nodes:
            idxs = np.linspace(0, M - 1, max_nodes).astype(int)
            sel = faces[idxs]
        else:
            pad = max_nodes - M
            if M == 0:
                sel = np.zeros((max_nodes, image_size[0], image_size[1], 3), dtype=np.uint8)
            else:
                pads = np.repeat(faces[-1][None], pad, axis=0)
                sel = np.concatenate([faces, pads], axis=0)
        nodes.append(sel)
        labels.append(item['label'] if item['label'] is not None else -1)
        files.append(item.get('file', ''))

    nodes = np.stack(nodes)
    nodes = torch.from_numpy(nodes).permute(0, 1, 4, 2, 3).float() / 255.0
    labels = torch.tensor(labels, dtype=torch.long)

    N = nodes.shape[1]
    A = np.zeros((B, N, N), dtype=np.float32)
    for b in range(B):
        for i in range(N - 1):
            A[b, i, i + 1] = 1.0
            A[b, i + 1, i] = 1.0
    A_norm = np.stack([normalize_adjacency(A[b]) for b in range(B)])
    A_norm = torch.from_numpy(A_norm).float()
    return nodes, A_norm, labels, files


def validate_epoch(model, loader, device, epoch, out_csv=None, model_type='vit_gcn'):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    all_files = []
    with torch.no_grad():
        for batch in loader:
            if model_type == 'cnn_lstm':
                images, labels, files = collate_batch_cnn_lstm_with_files(batch)
                images = images.to(device)
                out = model(images)
            else:
                images, A_norm, labels, files = collate_batch_with_files(batch)
                images = images.to(device)
                A_norm = A_norm.to(device)
                out = model(images, A_norm)
            
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy() if out.shape[1] > 1 else out.squeeze(1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()

            mask = labels >= 0
            if mask.sum() == 0:
                continue
            all_preds.extend(preds[mask].tolist())
            all_probs.extend(probs[mask].tolist())
            all_labels.extend(labels[mask].tolist())
            all_files.extend(np.array(files)[mask].tolist())

    if len(all_labels) == 0:
        print('No labeled validation samples found; skipping metrics')
        return None

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    print(f'Validation Epoch {epoch} - acc: {acc:.4f} prec: {prec:.4f} rec: {rec:.4f} f1: {f1:.4f}')
    print('Confusion matrix:\n', confusion_matrix(all_labels, all_preds))
    try:
        auc = roc_auc_score(all_labels, all_probs)
        print(f'ROC AUC: {auc:.4f}')
    except Exception:
        auc = None

    if out_csv:
        with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['file', 'label', 'pred', 'prob'])
            for f, l, p, pr in zip(all_files, all_labels, all_preds, all_probs):
                writer.writerow([f, l, p, pr])

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory with prepared .npz files')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--smoke', action='store_true', help='Run tiny smoke test')
    parser.add_argument('--balance', choices=['none', 'loss', 'sampler'], default='loss', help='How to handle class imbalance')
    parser.add_argument('--save_best', action='store_true', help='Save best model by validation F1')
    parser.add_argument('--model', choices=['vit_gcn', 'cnn_lstm'], default='vit_gcn', help='Model architecture to use')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint (requires --checkpoint)')
    args = parser.parse_args()

    ds = VideoFacesDataset(args.data_dir)
    if len(ds) == 0:
        print('No data found in', args.data_dir)
        return

    if args.smoke:
        # small subset
        indices = list(range(min(16, len(ds))))
        ds.files = [ds.files[i] for i in indices]

    # split into train/validation
    n = len(ds)
    n_val = max(1, int(0.2 * n))
    n_train = n - n_val
    if n_train <= 0:
        n_train = max(1, n - 1)
        n_val = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    # Compute class distribution for weighting/sampling
    all_labels = [ds.infer_label(p.name) for p in ds.files]
    labeled = [l for l in all_labels if l in (0, 1)]
    from collections import Counter
    counts = Counter(labeled)
    print('Dataset label counts:', counts)

    # Use a simple collate_fn that returns the raw list to handle variable-length node counts per sample
    if args.balance == 'sampler':
        # build per-sample weights for the original dataset files
        weights = []
        for l in all_labels:
            if l == 0:
                weights.append(1.0 / (counts.get(0, 1)))
            elif l == 1:
                weights.append(1.0 / (counts.get(1, 1)))
            else:
                weights.append(0.0)
        # for Subset, produce weights for each element in train_ds
        train_weights = [weights[i] for i in train_ds.indices]
        from torch.utils.data.sampler import WeightedRandomSampler
        sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=lambda x: x, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=lambda x: x)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=lambda x: x)

    device = torch.device(args.device)
    if args.model == 'cnn_lstm':
        model = CNNLSTMHybrid(input_channels=3, hidden_size=256, num_layers=2, num_classes=2, dropout=0.3)
    else:
        model = DeepfakeModel()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Class-weighted loss if requested
    if args.balance == 'loss' and len(counts) > 0:
        # compute inverse-frequency class weights
        total = sum(counts.values())
        num_classes = 2
        weights = [0.0] * num_classes
        for c in range(num_classes):
            cnt = counts.get(c, 0)
            if cnt > 0:
                weights[c] = total / (num_classes * cnt)
            else:
                weights[c] = 0.0
        class_weights = torch.tensor(weights, dtype=torch.float)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print('Using class-weighted loss, weights=', weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # simple LR scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_f1 = -1.0
    start_epoch = 1

    # Handle checkpoint resumption
    if args.resume:
        if not args.checkpoint or not os.path.exists(args.checkpoint):
            print('Error: --resume requires a valid --checkpoint path')
            return
        try:
            print(f'Loading checkpoint from {args.checkpoint}...')
            ckpt = torch.load(args.checkpoint, map_location=device)
            
            # Load model state
            if 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
                print('✓ Model state loaded')
            else:
                print('Warning: No model_state found in checkpoint')
            
            # Load optimizer state if available
            if 'optimizer_state' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state'])
                print('✓ Optimizer state loaded')
            
            # Load scheduler state if available
            if 'scheduler_state' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state'])
                print('✓ Scheduler state loaded')
            
            # Load training state
            if 'epoch' in ckpt:
                start_epoch = ckpt['epoch'] + 1
                print(f'✓ Resuming from epoch {start_epoch}')
            
            # Load best_f1 if available
            if 'best_f1' in ckpt:
                best_f1 = ckpt['best_f1']
                print(f'✓ Best F1 loaded: {best_f1:.4f}')
            
            print(f'Checkpoint loaded successfully. Training will continue from epoch {start_epoch}')
        except Exception as e:
            print(f'Error loading checkpoint: {e}')
            return

    for epoch in range(start_epoch, args.epochs + 1):
        loss, acc = train_epoch(model, train_loader, device, optimizer, criterion, model_type=args.model)
        print(f'Epoch {epoch}/{args.epochs} - loss: {loss:.4f} acc: {acc:.4f}')
        metrics = validate_epoch(model, val_loader, device, epoch, out_csv=f'preds_epoch_{epoch}.csv', model_type=args.model)
        try:
            scheduler.step()
        except Exception:
            pass

        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'metrics': metrics,
            'best_f1': best_f1
        }
        torch.save(ckpt, f'checkpoint_epoch_{epoch}.pt')
        if args.save_best and metrics and metrics.get('f1') is not None:
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                ckpt['best_f1'] = best_f1
                torch.save(ckpt, 'checkpoint_best.pt')
                print(f'New best model saved (f1={best_f1:.4f})')


if __name__ == '__main__':
    main()
