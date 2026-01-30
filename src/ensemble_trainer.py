"""
Enhanced Ensemble Training with Confidence Calibration and Uncertainty Estimation
Designed to boost accuracy from 50% to 75%+ through intelligent ensembling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import csv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """
    Temperature scaling for confidence calibration.
    Improves reliability of model predictions without retraining.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.temperature = nn.Parameter(torch.ones(1, device=device))
        self.optimizer = None
    
    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor, epochs: int = 50, lr: float = 0.01):
        """
        Calibrate temperature using validation set
        
        Args:
            logits: (N, num_classes) uncalibrated model outputs
            labels: (N,) ground truth labels
        """
        self.optimizer = optim.LBFGS([self.temperature], lr=lr)
        
        def loss_fn():
            self.optimizer.zero_grad()
            calibrated_logits = logits / self.temperature
            loss = F.cross_entropy(calibrated_logits, labels)
            loss.backward()
            return loss
        
        for _ in range(epochs):
            self.optimizer.step(loss_fn)
        
        logger.info(f"✓ Calibration complete. Temperature: {self.temperature.item():.3f}")
        return self.temperature.item()
    
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits"""
        return logits / self.temperature


class UncertaintyEstimator:
    """
    Estimates prediction uncertainty using ensemble disagreement and confidence margins.
    """
    
    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
    
    def estimate_from_ensemble(self, logits_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate uncertainty from ensemble predictions
        
        Args:
            logits_list: List of (B, num_classes) logits from different models
        
        Returns:
            uncertainty: (B,) uncertainty scores (higher = more uncertain)
            confidence: (B,) confidence scores (higher = more certain)
        """
        # Convert logits to probabilities
        probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
        probs_stack = torch.stack(probs_list, dim=0)  # (num_models, B, num_classes)
        
        # Mean and variance across ensemble
        mean_probs = probs_stack.mean(dim=0)  # (B, num_classes)
        var_probs = probs_stack.var(dim=0)  # (B, num_classes)
        
        # Uncertainty metrics
        ensemble_disagreement = var_probs.sum(dim=1)  # (B,)
        max_prob, _ = mean_probs.max(dim=1)  # (B,)
        confidence_margin = max_prob - mean_probs.topk(2, dim=1)[0][:, 1]  # Margin to 2nd best
        
        # Combined uncertainty (higher = more uncertain)
        uncertainty = ensemble_disagreement / (confidence_margin + 1e-6)
        
        # Confidence (inverse of uncertainty)
        confidence = 1.0 / (1.0 + uncertainty)
        
        return uncertainty, confidence


class EnsembleTrainer:
    """
    Advanced ensemble training with progressive fine-tuning and confidence calibration
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        checkpoint_dir: str = 'checkpoints/ensemble',
        ensemble_method: str = 'weighted'
    ):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.ensemble_method = ensemble_method
        
        # Calibration and uncertainty
        self.calibrator = ConfidenceCalibrator(device=device)
        self.uncertainty_estimator = UncertaintyEstimator()
        
        # Metrics tracking
        self.training_history = []
        self.best_metrics = {
            'epoch': 0,
            'val_accuracy': 0,
            'val_auc': 0,
            'val_f1': 0,
            # Which metric we are selecting checkpoints by (may differ from argmax accuracy)
            'select_metric': 'accuracy',
            'select_score': float('-inf'),
        }
    
    def prepare_optimizers(self, lr: float = 1e-4, weight_decay: float = 1e-5) -> Tuple:
        """Prepare optimizers for different model components"""
        if hasattr(self.model, 'models'):
            # Ensemble model
            params = list(self.model.parameters())
        else:
            params = list(self.model.parameters())
        
        # Main optimizer
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler with warmup and decay
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        return optimizer, scheduler
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, dict):
                images = batch.get('images', batch.get('frames'))
                labels = batch['label']
            else:
                images, labels = batch
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'models'):
                # Ensemble model - get ensemble output
                outputs, _ = self.model(images)
            else:
                outputs = self.model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Tracking
            total_loss += loss.item()
            
            with torch.no_grad():
                probs = F.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        # (uncertainty estimation can be added later; currently unused)
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                if isinstance(batch, dict):
                    images = batch.get('images', batch.get('frames'))
                    labels = batch['label']
                else:
                    images, labels = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'models'):
                    outputs, _ = self.model(images)
                else:
                    outputs = self.model(images)
                
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Compute metrics (default threshold = argmax)
        labels_unique = set(int(x) for x in all_labels)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        auc = roc_auc_score(all_labels, all_probs) if len(labels_unique) > 1 else 0.5
        cm = confusion_matrix(all_labels, all_preds)

        metrics: Dict[str, float | list] = {
            'loss': total_loss / len(val_loader),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'confusion_matrix': cm.tolist(),
        }

        # Threshold sweep on probabilities to maximize Accuracy/F1 on *this* val split.
        # NOTE: Keep this sweep bounded and reasonably coarse to avoid extreme thresholds
        # (e.g. ~0) that can artificially inflate validation metrics.
        try:
            import numpy as np

            probs_arr = np.array(all_probs, dtype=np.float32)
            labels_arr = np.array(all_labels, dtype=np.int64)
            if len(labels_unique) > 1 and len(probs_arr) == len(labels_arr) and len(labels_arr) > 0:
                # Bounded grid sweep
                cands = np.linspace(0.05, 0.95, 19, dtype=np.float32)
                best_acc = -1.0
                best_thr_acc = 0.5
                best_f1 = -1.0
                best_thr_f1 = 0.5

                for thr in cands.tolist():
                    thr_f = float(thr)
                    preds_thr = (probs_arr >= thr_f).astype(int)
                    acc_thr = float(accuracy_score(labels_arr, preds_thr))
                    f1_thr = float(
                        precision_recall_fscore_support(labels_arr, preds_thr, average='binary', zero_division=0)[2]
                    )
                    if acc_thr > best_acc:
                        best_acc = acc_thr
                        best_thr_acc = thr_f
                    if f1_thr > best_f1:
                        best_f1 = f1_thr
                        best_thr_f1 = thr_f

                metrics['best_thr_accuracy'] = float(best_thr_acc)
                metrics['accuracy_thr'] = float(best_acc)
                metrics['best_thr_f1'] = float(best_thr_f1)
                metrics['f1_thr'] = float(best_f1)
        except Exception:
            pass
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        start_epoch: int = 1,
        class_weights: torch.Tensor | None = None,
        select_metric: str = 'accuracy',
    ):
        """Full training loop"""

        if start_epoch < 1:
            raise ValueError('start_epoch must be >= 1')
        
        optimizer, scheduler = self.prepare_optimizers(lr=lr, weight_decay=weight_decay)
        
        # Loss function with class weighting.
        # Default to inverse-frequency weights computed from the training split to avoid
        # degenerate solutions on imbalanced datasets.
        if class_weights is None:
            class_weights = self._infer_class_weights(train_loader)
        else:
            class_weights = class_weights.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        logger.info(f"🚀 Starting ensemble training for {epochs} epochs")
        logger.info(f"   Ensemble method: {self.ensemble_method}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   LR: {lr}, Weight decay: {weight_decay}")
        try:
            cw = class_weights.detach().cpu().tolist()
            logger.info(f"   Class weights: real={cw[0]:.4f}, fake={cw[1]:.4f}")
        except Exception:
            pass
        
        current_epoch = start_epoch
        try:
            for epoch in range(start_epoch, epochs + 1):
                current_epoch = epoch
                # Train
                train_metrics = self.train_epoch(train_loader, optimizer, criterion, epoch)

                # Validate
                val_metrics = self.validate(val_loader, criterion, epoch)

                # Update learning rate
                scheduler.step()

                # Log metrics
                self.training_history.append({
                    'epoch': epoch,
                    'train': train_metrics,
                    'val': val_metrics
                })

                # Persist history every epoch for live monitoring
                self._save_history()

                # Save best model (allow selecting metric)
                key = (select_metric or 'accuracy').strip().lower()
                score = None
                if key in val_metrics:
                    score = float(val_metrics[key])
                else:
                    # friendly aliases
                    alias = {
                        'acc': 'accuracy',
                        'auc': 'auc',
                        'f1score': 'f1',
                        'acc_thr': 'accuracy_thr',
                        'f1_thr': 'f1_thr',
                    }.get(key)
                    if alias and alias in val_metrics:
                        score = float(val_metrics[alias])

                if score is None:
                    score = float(val_metrics.get('accuracy', 0.0))
                    key = 'accuracy'

                # Compare against the best score for the *selected* metric.
                # Previous code compared against argmax accuracy, which could cause
                # overwriting checkpoint_best.pt even when the selected metric got worse.
                best_score = float(self.best_metrics.get('select_score', float('-inf')))
                if str(self.best_metrics.get('select_metric', key)) != key:
                    best_score = float('-inf')

                if score > best_score:
                    self.best_metrics = {
                        'epoch': epoch,
                        # Keep existing keys for backward-compatible printing,
                        # but store the selection metric and score.
                        'val_accuracy': float(val_metrics.get('accuracy', 0.0)),
                        'val_auc': float(val_metrics.get('auc', 0.0)),
                        'val_f1': float(val_metrics.get('f1', 0.0)),
                        'select_metric': key,
                        'select_score': float(score),
                    }
                    self._save_checkpoint(epoch, is_best=True)
                    self._save_calibration(val_metrics)
                    logger.info(
                        f"✓ Epoch {epoch}: NEW BEST ({key}={score:.4f}) - "
                        f"Acc: {val_metrics.get('accuracy', 0.0):.4f}, AUC: {val_metrics.get('auc', 0.0):.4f}, F1: {val_metrics.get('f1', 0.0):.4f}"
                    )
                else:
                    extra = ''
                    if 'accuracy_thr' in val_metrics:
                        extra = f", Acc@thr={val_metrics['accuracy_thr']:.4f} (thr={val_metrics.get('best_thr_accuracy', 0.5):.2f})"
                    logger.info(
                        f"  Epoch {epoch}: Acc: {val_metrics.get('accuracy', 0.0):.4f}, "
                        f"AUC: {val_metrics.get('auc', 0.0):.4f}, F1: {val_metrics.get('f1', 0.0):.4f}{extra}"
                    )

                # Periodic checkpointing
                if epoch % 10 == 0:
                    self._save_checkpoint(epoch, is_best=False)
        except KeyboardInterrupt:
            logger.warning(
                "\n⚠️  Training interrupted by user (KeyboardInterrupt). Saving checkpoint and history (epoch %s)...",
                current_epoch,
            )
            self._save_checkpoint(current_epoch, is_best=False, tag='interrupt')
            self._save_history()
            logger.warning(
                "✓ Saved interrupt checkpoint. Resume with --resume --checkpoint %s",
                str((self.checkpoint_dir / f"checkpoint_epoch_{current_epoch}_interrupt.pt").as_posix()),
            )
            return

        # Final save (history already written each epoch, but keep for clarity)
        self._save_history()
        
        logger.info("✅ Training complete!")
        sel_key = str(self.best_metrics.get('select_metric', 'accuracy'))
        sel_score = float(self.best_metrics.get('select_score', self.best_metrics.get('val_accuracy', 0.0)))
        logger.info(f"   Best selection: {sel_key} = {sel_score:.4f}")
        logger.info(f"   Best accuracy (argmax): {float(self.best_metrics.get('val_accuracy', 0.0)):.4f}")
        logger.info(f"   Best AUC: {float(self.best_metrics.get('val_auc', 0.0)):.4f}")


    def _save_calibration(self, val_metrics: Dict):
        """Persist calibration hints (best thresholds) for downstream inference."""
        try:
            payload = {
                'epoch': int(self.best_metrics.get('epoch', 0) or 0),
                'best_thr_accuracy': float(val_metrics.get('best_thr_accuracy', 0.5)),
                'accuracy_thr': float(val_metrics.get('accuracy_thr', 0.0)),
                'best_thr_f1': float(val_metrics.get('best_thr_f1', 0.5)),
                'f1_thr': float(val_metrics.get('f1_thr', 0.0)),
            }
            p = self.checkpoint_dir / 'calibration_best.json'
            import json
            p.write_text(json.dumps(payload, indent=2))
        except Exception:
            pass


    def _infer_class_weights(self, train_loader: DataLoader) -> torch.Tensor:
        """Infer inverse-frequency class weights (binary: 0=real, 1=fake).

        Prefers using filename-based labels to avoid loading all arrays.
        Falls back to a single pass over `train_loader` if needed.
        """
        # Attempt fast path: Subset(VideoFacesDataset)
        ds = getattr(train_loader, 'dataset', None)
        base = getattr(ds, 'dataset', ds)
        indices = getattr(ds, 'indices', None)

        n_real = 0
        n_fake = 0

        if base is not None and hasattr(base, 'files') and hasattr(base, 'infer_label') and indices is not None:
            try:
                for i in indices:
                    name = getattr(base.files[i], 'name', str(base.files[i]))
                    y = int(base.infer_label(name))
                    if y == 0:
                        n_real += 1
                    elif y == 1:
                        n_fake += 1
            except Exception:
                n_real = 0
                n_fake = 0

        # Fallback: scan loader once
        if (n_real + n_fake) == 0:
            try:
                for batch in train_loader:
                    labels = None
                    if isinstance(batch, dict):
                        labels = batch.get('label', None)
                    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                        labels = batch[1]
                    if labels is None:
                        continue
                    labels = labels.detach().cpu().numpy().tolist()
                    for y in labels:
                        if int(y) == 0:
                            n_real += 1
                        elif int(y) == 1:
                            n_fake += 1
            except Exception:
                pass

        total = max(1, n_real + n_fake)
        # Avoid division by zero
        n_real = max(1, n_real)
        n_fake = max(1, n_fake)

        # Inverse-frequency weights (normalized around 1.0)
        w_real = total / (2.0 * n_real)
        w_fake = total / (2.0 * n_fake)
        return torch.tensor([w_real, w_fake], device=self.device, dtype=torch.float32)
        logger.info(f"   Best F1: {self.best_metrics['val_f1']:.4f}")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, tag: str | None = None):
        """Save model checkpoint"""
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(self.model.state_dict(), best_path)

            # Also keep an immutable copy of the best-at-epoch snapshot so we can recover
            # if later epochs regress or someone changes selection criteria.
            epoch_best_path = self.checkpoint_dir / f"checkpoint_best_epoch_{epoch}.pt"
            if not epoch_best_path.exists():
                try:
                    torch.save(self.model.state_dict(), epoch_best_path)
                except Exception:
                    pass
            logger.debug(f"✓ Saved checkpoint: {best_path}")
            return

        filename = f"checkpoint_epoch_{epoch}.pt"
        if tag:
            filename = f"checkpoint_epoch_{epoch}_{tag}.pt"
        filepath = self.checkpoint_dir / filename
        torch.save(self.model.state_dict(), filepath)
        logger.debug(f"✓ Saved checkpoint: {filepath}")
    
    def _save_history(self):
        """Save training history to CSV"""
        history_file = self.checkpoint_dir / "training_history.csv"
        
        with open(history_file, 'w', newline='') as f:
            fieldnames = [
                'Epoch',
                'Train_Loss', 'Train_Acc', 'Train_F1',
                'Val_Loss', 'Val_Acc', 'Val_F1', 'Val_ROC_AUC',
                'Val_Precision', 'Val_Recall',
                # Threshold-sweep metrics (optional)
                'Val_Acc_thr', 'Val_Best_thr_acc',
                'Val_F1_thr', 'Val_Best_thr_f1',
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in self.training_history:
                v = record.get('val', {}) or {}
                writer.writerow({
                    'Epoch': record['epoch'],
                    'Train_Loss': record['train']['loss'],
                    'Train_Acc': record['train']['accuracy'],
                    'Train_F1': record['train']['f1'],
                    'Val_Loss': v.get('loss', 0.0),
                    'Val_Acc': v.get('accuracy', 0.0),
                    'Val_F1': v.get('f1', 0.0),
                    'Val_ROC_AUC': v.get('auc', 0.0),
                    'Val_Precision': v.get('precision', 0.0),
                    'Val_Recall': v.get('recall', 0.0),
                    'Val_Acc_thr': v.get('accuracy_thr', ''),
                    'Val_Best_thr_acc': v.get('best_thr_accuracy', ''),
                    'Val_F1_thr': v.get('f1_thr', ''),
                    'Val_Best_thr_f1': v.get('best_thr_f1', ''),
                })
        
        logger.info(f"✓ Saved training history: {history_file}")
