import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import csv
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
except Exception as e:
    print(f"Missing dependencies: {e}")
    sys.exit(1)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset import VideoFacesDataset
from models import DeepfakeModel


class FocalLoss(nn.Module):
    """Focal Loss for classification with optional label smoothing.

    Args:
        gamma: focusing parameter
        weight: optional class weights tensor
        label_smoothing: float in [0,1)
    """
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        if isinstance(weight, torch.Tensor) or weight is None:
            wt = weight
        else:
            wt = torch.tensor(weight, dtype=torch.float32)
        if isinstance(wt, torch.Tensor):
            wt = wt.to(dtype=torch.float32)
        self.register_buffer('weight', wt)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (B, C), targets: (B,) long
        C = logits.size(1)
        log_p = torch.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        # one-hot with label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.label_smoothing / (C - 1) if C > 1 else 0.0)
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.label_smoothing)

        # focal weight
        pt = (p * true_dist).sum(dim=1)
        focal_factor = (1 - pt) ** self.gamma

        loss = - (true_dist * log_p).sum(dim=1)
        loss = focal_factor * loss

        # apply class weight per sample if provided
        if self.weight is not None:
            wt = self.weight[targets].to(logits.device)
            loss = loss * wt

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class EarlyStoppingCallback:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_acc = None
        self.best_epoch = None
        
    def __call__(self, val_acc, epoch):
        if self.best_val_acc is None:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
        elif val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


class ImprovedTrainer:
    """Enhanced trainer with better monitoring and optimization"""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['max_epochs'],
            eta_min=config['learning_rate'] * 0.01
        )
        
        # Alternative: Reduce LR on plateau
        self.plateau_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Loss selection: support cross-entropy (with label smoothing) and focal loss
        loss_name = config.get('loss', 'cross_entropy')
        label_smoothing = float(config.get('label_smoothing', 0.0))
        focal_gamma = float(config.get('focal_gamma', 2.0))
        weight_tensor = torch.tensor(config['class_weights'], dtype=torch.float32, device=device)

        if loss_name == 'focal':
            self.criterion = FocalLoss(gamma=focal_gamma, weight=weight_tensor, label_smoothing=label_smoothing)
        else:
            # use built-in label_smoothing if available
            try:
                self.criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
            except TypeError:
                # older PyTorch: implement manual smoothing by adjusting targets in training loop (fallback to no smoothing)
                self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        
        # Early stopping
        self.early_stopping = EarlyStoppingCallback(
            patience=config['early_stopping_patience'],
            min_delta=0.001
        )
        
        # Metrics tracking
        self.train_metrics = {
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
        }
        self.val_metrics = {
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': [],
        }
        
        # Checkpoint tracking
        self.best_model_path = None
        self.best_val_acc = 0
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                frames, A_norm, labels = batch
            else:
                continue
            frames = frames.to(self.device)
            A_norm = A_norm.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(frames, A_norm)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy_score(all_labels, all_preds):.4f}'
            })
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            
            for batch in pbar:
                if isinstance(batch, (tuple, list)) and len(batch) == 3:
                    frames, A_norm, labels = batch
                else:
                    continue
                frames = frames.to(self.device)
                A_norm = A_norm.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(frames, A_norm)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(probs[:, 1])  # Prob of being fake
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy_score(all_labels, all_preds):.4f}'
                })
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # ROC-AUC (if binary classification)
        try:
            roc_auc = roc_auc_score(all_labels, all_scores)
        except:
            roc_auc = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
        }, (all_labels, all_preds, all_scores)
    
    def train(self, train_loader, val_loader, checkpoint_dir='checkpoints'):
        """Main training loop"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Starting training for {self.config['max_epochs']} epochs")
        print(f"Initial LR: {self.config['learning_rate']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Early stopping patience: {self.config['early_stopping_patience']}")
        print(f"{'='*80}\n")
        
        for epoch in range(self.config['max_epochs']):
            print(f"\n[Epoch {epoch+1}/{self.config['max_epochs']}]")
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            for key, val in train_metrics.items():
                self.train_metrics[key].append(val)
            
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train Acc:  {train_metrics['accuracy']:.4f}")
            print(f"  Train F1:   {train_metrics['f1']:.4f}")
            
            # Validation phase
            val_metrics, val_preds_data = self.validate(val_loader)
            for key, val in val_metrics.items():
                self.val_metrics[key].append(val)
            
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Val Acc:    {val_metrics['accuracy']:.4f}")
            print(f"  Val F1:     {val_metrics['f1']:.4f}")
            print(f"  Val ROC-AUC:{val_metrics['roc_auc']:.4f}")
            
            # Update learning rate
            self.scheduler.step()
            self.plateau_scheduler.step(val_metrics['accuracy'])
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_best.pt')
                torch.save(
                    {
                        'model_state': self.model.state_dict(),
                        'model_config': self.config.get('model_config', {}),
                    },
                    checkpoint_path,
                )
                print(f"  ✓ Saved best model (Acc: {val_metrics['accuracy']:.4f})")
                self.best_model_path = checkpoint_path
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save(
                    {
                        'model_state': self.model.state_dict(),
                        'model_config': self.config.get('model_config', {}),
                    },
                    checkpoint_path,
                )
                print(f"  ✓ Saved epoch checkpoint")
            
            # Early stopping check
            should_stop = self.early_stopping(val_metrics['accuracy'], epoch)
            if should_stop:
                print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                break
            
            # Log current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Current LR: {current_lr:.2e}")
        
        print(f"\n{'='*80}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Best model saved at: {self.best_model_path}")
        print(f"{'='*80}\n")
        
        return self.train_metrics, self.val_metrics


def collate_batch(batch):
    """Custom collate to extract frames and labels from dict items and stack them."""
    frames_list = []
    labels_list = []
    for item in batch:
        frames = item['faces']  # (N, H, W, C) numpy array
        label = item['label']
        frames_list.append(torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0)  # (N, C, H, W)
        labels_list.append(torch.tensor(label, dtype=torch.long))
    
    max_n = max(f.shape[0] for f in frames_list)
    frames_padded = []
    for f in frames_list:
        if f.shape[0] < max_n:
            pad = f[-1:].repeat(max_n - f.shape[0], 1, 1, 1)
            f = torch.cat([f, pad], dim=0)
        frames_padded.append(f)
    
    frames_batch = torch.stack(frames_padded)  # (B, N, C, H, W)
    labels_batch = torch.stack(labels_list)  # (B,)
    
    # Build simple adjacency matrix (chain: 0-1-2-...-N)
    B = frames_batch.shape[0]
    N = frames_batch.shape[1]
    A_norm = np.zeros((B, N, N), dtype=np.float32)
    for b in range(B):
        for i in range(N - 1):
            A_norm[b, i, i + 1] = 1.0
            A_norm[b, i + 1, i] = 1.0
    A_norm = torch.from_numpy(A_norm).float()
    
    return frames_batch, A_norm, labels_batch


def main():
    parser = argparse.ArgumentParser(description='Improved Deepfake Detection Training')
    parser.add_argument('--data-path', type=str, default='data/prepared',
                        help='Path to prepared data directory')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Weight decay (default: 0.0001)')
    parser.add_argument('--early-stopping-patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--train-val-split', type=float, default=0.8,
                        help='Train-validation split ratio (default: 0.8)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: cuda if available, else cpu)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--init-from', type=str, default=None,
                        help='Optional checkpoint (.pt) to initialize model weights from before training')
    parser.add_argument('--augment', action='store_true', help='Enable training-time augmentations')
    parser.add_argument('--loss', choices=['cross_entropy', 'focal'], default='cross_entropy', help='Loss to use')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing factor (0 = none)')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--backbone', choices=['timm_vit', 'clip_vit', 'dinov2_vit'], default='timm_vit', help='Vision backbone to use')
    parser.add_argument('--pretrained-vit', action='store_true', help='Use timm pretrained ViT backbone (only for timm_vit)')
    parser.add_argument('--vit-model-name', type=str, default='vit_base_patch16_224', help='timm ViT model name')
    parser.add_argument('--vit-pretrained-path', type=str, default=None, help='Optional path to ViT weights to load')
    parser.add_argument('--clip-model-name', type=str, default='openai/clip-vit-base-patch32', help='HuggingFace CLIP vision model id')
    parser.add_argument('--clip-pretrained', action=argparse.BooleanOptionalAction, default=True, help='Load pretrained CLIP weights')
    parser.add_argument('--dinov2-model-name', type=str, default='facebook/dinov2-base', help='HuggingFace DINOv2 model id')
    parser.add_argument('--dinov2-pretrained', action=argparse.BooleanOptionalAction, default=True, help='Load pretrained DINOv2 weights')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Deepfake Detection - Improved Training")
    print(f"{'='*80}")
    print(f"Data Path: {args.data_path}")
    print(f"Max Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Early Stopping Patience: {args.early_stopping_patience}")
    print(f"Train-Val Split: {args.train_val_split}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = VideoFacesDataset(args.data_path, augment=args.augment)
    print(f"Total videos: {len(dataset)}")
    
    # Calculate class weights for imbalanced data
    fake_count = sum(1 for item in dataset if item['label'] == 1)
    real_count = len(dataset) - fake_count
    total = len(dataset)
    
    class_weights = [
        total / (2 * real_count),   # Real
        total / (2 * fake_count),   # Fake
    ]
    print(f"Class distribution: Real={real_count}, Fake={fake_count}")
    print(f"Class weights: {class_weights}")
    
    # Split dataset
    train_size = int(len(dataset) * args.train_val_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Val set size: {len(val_dataset)}")
    
    # Create dataloaders
    # Use a weighted random sampler for class-balanced training when possible
    def _subset_labels(subset):
        labels = []
        try:
            ds = subset.dataset
            idxs = subset.indices
            for i in idxs:
                labels.append(ds.infer_label(ds.files[i].name))
        except Exception:
            # fallback: iterate (slower)
            for item in subset:
                labels.append(item['label'])
        return np.array(labels, dtype=int)

    train_labels = _subset_labels(train_dataset)
    # Only consider labeled samples for weighting
    valid_mask = train_labels >= 0
    if valid_mask.sum() > 0:
        counts = np.bincount(train_labels[valid_mask])
        if counts.size >= 2 and np.all(counts > 0):
            num_classes = counts.size
            class_weights = (valid_mask.sum() / (num_classes * counts)).astype(float)
            sample_weights = [float(class_weights[l]) if l >= 0 else 0.0 for l in train_labels]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                collate_fn=collate_batch,
                num_workers=0,
                pin_memory=False
            )
        else:
            # fallback to simple shuffle if class counts are degenerate
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_batch,
                num_workers=0,
                pin_memory=False
            )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            num_workers=0,
            pin_memory=False
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=False
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = DeepfakeModel(
        backbone=args.backbone,
        pretrained_vit=args.pretrained_vit,
        vit_model_name=args.vit_model_name,
        vit_pretrained_path=args.vit_pretrained_path,
        clip_model_name=args.clip_model_name,
        clip_pretrained=bool(args.clip_pretrained),
        dinov2_model_name=args.dinov2_model_name,
        dinov2_pretrained=bool(args.dinov2_pretrained),
    ).to(device)

    # Optionally initialize weights from a checkpoint (fine-tuning / continued training)
    if args.init_from:
        try:
            ckpt = torch.load(args.init_from, map_location='cpu')
            sd = ckpt.get('model_state', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(sd, strict=False)
            print(f"✓ Initialized model weights from: {args.init_from}")
        except Exception as e:
            print(f"⚠ Warning: failed to init from checkpoint {args.init_from}: {e}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training config
    config = {
        'max_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': args.early_stopping_patience,
        'class_weights': class_weights,
        'loss': args.loss,
        'label_smoothing': args.label_smoothing,
        'focal_gamma': args.focal_gamma,
        'model_config': {
            'backbone': args.backbone,
            'pretrained_vit': bool(args.pretrained_vit),
            'vit_model_name': args.vit_model_name,
            'vit_pretrained_path': args.vit_pretrained_path,
            'clip_model_name': args.clip_model_name,
            'clip_pretrained': bool(args.clip_pretrained),
            'dinov2_model_name': args.dinov2_model_name,
            'dinov2_pretrained': bool(args.dinov2_pretrained),
        },
    }
    
    # Create trainer
    trainer = ImprovedTrainer(model, device, config)
    
    # Train
    train_metrics, val_metrics = trainer.train(
        train_loader,
        val_loader,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Save metrics
    metrics_file = 'training_metrics_improved.csv'
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Epoch', 'Train_Loss', 'Train_Acc', 'Train_F1',
            'Val_Loss', 'Val_Acc', 'Val_F1', 'Val_ROC_AUC'
        ])
        for i in range(len(train_metrics['loss'])):
            writer.writerow([
                i + 1,
                f"{train_metrics['loss'][i]:.6f}",
                f"{train_metrics['accuracy'][i]:.6f}",
                f"{train_metrics['f1'][i]:.6f}",
                f"{val_metrics['loss'][i]:.6f}",
                f"{val_metrics['accuracy'][i]:.6f}",
                f"{val_metrics['f1'][i]:.6f}",
                f"{val_metrics['roc_auc'][i]:.6f}",
            ])
    
    print(f"\n✓ Metrics saved to {metrics_file}")
    print(f"\nFinal Results:")
    print(f"  Best Val Accuracy: {max(val_metrics['accuracy']):.4f}")
    print(f"  Best Val ROC-AUC:  {max(val_metrics['roc_auc']):.4f}")


if __name__ == '__main__':
    main()
