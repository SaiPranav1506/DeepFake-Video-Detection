"""
Pretrained Deepfake Detector with Advanced Transfer Learning
Uses pretrained backbones with aggressive feature learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
import numpy as np
from typing import Tuple, Optional, Dict, List


class PretrainedBackboneDetector(nn.Module):
    """
    Deepfake detector using pretrained backbones with minimal additional layers.
    Optimized for small datasets through transfer learning.
    """
    
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False,
        use_temporal_attention: bool = True
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.use_temporal_attention = use_temporal_attention
        
        # Load pretrained backbone
        if backbone_name in ["resnet18", "resnet34", "resnet50"]:
            backbone = getattr(models, backbone_name)(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = backbone.fc.in_features
        
        elif backbone_name.startswith("efficientnet"):
            backbone = timm.create_model(backbone_name, pretrained=pretrained)
            # Remove classifier
            if hasattr(backbone, 'classifier'):
                self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            else:
                self.backbone = backbone
            self.feature_dim = 1280 if "b0" in backbone_name else 1920
        
        elif backbone_name.startswith("vit"):
            backbone = timm.create_model(backbone_name, pretrained=pretrained)
            self.backbone = backbone
            self.feature_dim = backbone.num_features if hasattr(backbone, 'num_features') else 768
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Temporal attention for video sequences
        if use_temporal_attention:
            self.temporal_attention = nn.Sequential(
                nn.Linear(self.feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.feature_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self._init_head_weights()
    
    def _init_head_weights(self):
        """Initialize classification head with small weights"""
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, 0, 0.01)
        nn.init.constant_(self.fc2.bias, 0)
    
    def unfreeze_backbone(self, num_blocks: int = 2):
        """Gradually unfreeze backbone layers"""
        if self.backbone_name.startswith("resnet"):
            # Unfreeze last N blocks
            layers = list(self.backbone.children())
            for layer in layers[-num_blocks:]:
                for param in layer.parameters():
                    param.requires_grad = True
        elif self.backbone_name.startswith("efficientnet"):
            # Unfreeze last N blocks
            if hasattr(self.backbone, 'blocks'):
                blocks = self.backbone.blocks
                for block in blocks[-num_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C, H, W) - batch of video frames
        
        Returns:
            logits: (B, num_classes)
            frame_scores: (B, T) - per-frame confidence scores
        """
        batch_size, num_frames, c, h, w = x.shape
        
        # Process each frame through backbone
        x_flat = x.view(batch_size * num_frames, c, h, w)
        features_flat = self.backbone(x_flat)  # (B*T, feature_dim)
        features_flat = features_flat.view(batch_size * num_frames, -1)
        
        # Reshape back to temporal dimension
        features = features_flat.view(batch_size, num_frames, -1)  # (B, T, feature_dim)
        
        # Apply temporal attention if enabled
        if self.use_temporal_attention:
            attention_weights = self.temporal_attention(features)  # (B, T, 1)
            attention_weights = attention_weights.squeeze(-1)  # (B, T)
            # Normalize attention
            attention_weights = F.softmax(attention_weights, dim=1)  # (B, T)
            
            # Weighted average of features
            frame_scores = attention_weights
            weighted_features = (features * attention_weights.unsqueeze(-1)).sum(dim=1)  # (B, feature_dim)
        else:
            # Simple mean pooling
            weighted_features = features.mean(dim=1)  # (B, feature_dim)
            frame_scores = torch.ones(batch_size, num_frames, device=x.device) / num_frames
        
        # Classification head
        x_class = self.dropout(weighted_features)
        x_class = F.relu(self.fc1(x_class))
        x_class = self.dropout(x_class)
        logits = self.fc2(x_class)  # (B, num_classes)
        
        return logits, frame_scores


class EnsembleDetector(nn.Module):
    """
    Ensemble of pretrained detectors for robust predictions.
    Combines multiple backbones for better generalization.
    """
    
    def __init__(
        self,
        backbone_names: List[str],
        pretrained: bool = True,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        ensemble_method: str = "average"  # "average", "weighted", "voting"
    ):
        super().__init__()
        
        self.models = nn.ModuleList([
            PretrainedBackboneDetector(
                backbone_name=name,
                pretrained=pretrained,
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                use_temporal_attention=True
            )
            for name in backbone_names
        ])
        
        self.ensemble_method = ensemble_method
        
        if ensemble_method == "weighted":
            # Learnable weights for each model
            self.weights = nn.Parameter(torch.ones(len(backbone_names)) / len(backbone_names))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C, H, W) - batch of video frames
        
        Returns:
            ensemble_logits: (B, num_classes)
            ensemble_scores: (B, T)
        """
        logits_list = []
        scores_list = []
        
        for model in self.models:
            logits, scores = model(x)
            logits_list.append(logits)
            scores_list.append(scores)
        
        logits = torch.stack(logits_list, dim=0)  # (num_models, B, num_classes)
        scores = torch.stack(scores_list, dim=0)  # (num_models, B, T)
        
        if self.ensemble_method == "average":
            ensemble_logits = logits.mean(dim=0)
            ensemble_scores = scores.mean(dim=0)
        
        elif self.ensemble_method == "weighted":
            weights = F.softmax(self.weights, dim=0)
            ensemble_logits = (logits * weights.view(-1, 1, 1)).sum(dim=0)
            ensemble_scores = (scores * weights.view(-1, 1, 1)).sum(dim=0)
        
        elif self.ensemble_method == "voting":
            # Hard voting on predicted class
            preds = logits.argmax(dim=-1)  # (num_models, B)
            ensemble_pred = torch.mode(preds, dim=0)[0]  # (B,)
            ensemble_logits = F.one_hot(ensemble_pred, num_classes=2).float()
            ensemble_scores = scores.mean(dim=0)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return ensemble_logits, ensemble_scores


class ProgressiveFineTuner:
    """
    Progressive fine-tuning strategy for small datasets:
    1. Start with frozen backbone
    2. Train classification head
    3. Gradually unfreeze backbone layers
    4. Fine-tune with lower learning rate
    """
    
    def __init__(self, model: nn.Module, stages: int = 3):
        self.model = model
        self.stages = stages
        self.current_stage = 0
    
    def get_stage_config(self) -> Dict:
        """Get configuration for current training stage"""
        configs = [
            {
                "name": "Stage 1: Head training",
                "freeze_backbone": True,
                "learning_rate": 1e-3,
                "epochs": 20,
                "description": "Train only classification head"
            },
            {
                "name": "Stage 2: Partial unfreezing",
                "freeze_backbone": False,
                "learning_rate": 1e-4,
                "epochs": 20,
                "unfreeze_blocks": 2,
                "description": "Unfreeze last 2 blocks, lower LR"
            },
            {
                "name": "Stage 3: Full fine-tuning",
                "freeze_backbone": False,
                "learning_rate": 1e-5,
                "epochs": 20,
                "unfreeze_blocks": -1,  # All blocks
                "description": "Full backbone fine-tuning with very low LR"
            }
        ]
        
        if self.current_stage < len(configs):
            return configs[self.current_stage]
        return configs[-1]
    
    def advance_stage(self):
        """Move to next training stage"""
        self.current_stage = min(self.current_stage + 1, self.stages - 1)
        
        if self.current_stage > 0 and hasattr(self.model, 'unfreeze_backbone'):
            config = self.get_stage_config()
            if config.get("unfreeze_blocks"):
                self.model.unfreeze_backbone(config["unfreeze_blocks"])
