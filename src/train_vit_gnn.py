"""Train script for the ViT+GNN pipeline with safe fallbacks.

This script attempts to use a ViT+GNN model if `timm` and `torch_geometric` are available.
If they are not present it trains a lightweight fallback model so you can verify training works.

Usage:
  python src/train_vit_gnn.py
"""
from __future__ import annotations

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import timm
    pyg_available = False
except Exception:
    timm = None
    pyg_available = False


class FallbackModel(nn.Module):
    def __init__(self, out_classes: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, out_classes))

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.classifier(z)


def train_loop(model, dataloader, device, epochs=3):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        t0 = time.time()
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        t1 = time.time()
        print(f"Epoch {epoch}/{epochs} — loss: {total_loss/total:.4f}, acc: {correct/total:.4f}, time: {t1-t0:.2f}s")


def main():
    device = torch.device("cpu")

    # Small synthetic dataset so training runs quickly
    B = 16
    epochs = 3
    batch_size = 4
    images = torch.randn(B, 3, 64, 64)
    labels = torch.randint(0, 2, (B,))

    ds = TensorDataset(images, labels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    if timm is not None:
        # Try to use the ViT+GNN model from src/run_vit_gnn.py if possible
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("run_vit_gnn", os.path.join(os.path.dirname(__file__), "run_vit_gnn.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # Build a small ViT+GNN if defined
            if hasattr(mod, "ViT_GNN_Model"):
                model = mod.ViT_GNN_Model()
                # Build with a sample input sized to 64x64 -> many ViT models expect 224, but we attempt
                try:
                    sample = torch.randn(1, 3, 64, 64)
                    model.build(sample)
                except Exception:
                    # If ViT fails for 64x64, fallback to 224-sized sample
                    sample = torch.randn(1, 3, 224, 224)
                    model.build(sample)
                # Wrap the ViT+GNN model in an nn.Module so parameters are visible to the optimizer
                class WrapperModule(nn.Module):
                    def __init__(self, base_model: nn.Module):
                        super().__init__()
                        self.base_model = base_model

                    def forward(self, x: torch.Tensor):
                        device_local = x.device
                        Bx = x.shape[0]
                        # Ensure input spatial size matches ViT expected size (commonly 224)
                        expected_size = None
                        try:
                            pe = self.base_model.encoder.vit.patch_embed
                            expected_size = tuple(pe.img_size)
                        except Exception:
                            expected_size = (224, 224)
                        H = x.shape[2]
                        W = x.shape[3]
                        if (H, W) != expected_size:
                            import torch.nn.functional as F
                            x = F.interpolate(x, size=expected_size, mode='bilinear', align_corners=False)
                        tokens = self.base_model.encoder(x)
                        _, N, C = tokens.shape
                        num_nodes = Bx * N
                        # Build block-diag adjacency
                        adj = torch.zeros((num_nodes, num_nodes), device=device_local)
                        for b in range(Bx):
                            s = b * N
                            e = s + N
                            adj[s:e, s:e] = 1.0
                        deg = adj.sum(dim=1, keepdim=True)
                        deg[deg == 0] = 1.0
                        adj = adj / deg
                        return self.base_model(x, adj)

                model_for_train = WrapperModule(model)
            else:
                print("ViT_GNN_Model not found in run_vit_gnn.py; using fallback model.")
                model_for_train = FallbackModel()
        except Exception as e:
            print("Failed to initialize ViT+GNN automatically:", e)
            print("Falling back to simple model.")
            model_for_train = FallbackModel()
    else:
        print("`timm` not available — training fallback model.")
        model_for_train = FallbackModel()

    train_loop(model_for_train, dl, device, epochs=epochs)

    # Save a small checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "vit_gnn_ckpt.pt")
    torch.save({"model_state": model_for_train.state_dict()}, ckpt_path)
    print("Saved checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()
