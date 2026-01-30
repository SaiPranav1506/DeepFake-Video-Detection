"""Simple runner that composes a ViT encoder with a GNN and runs a dummy forward pass.

This file is intentionally robust: if `timm` or `torch_geometric` are missing it prints guidance
and falls back to a tiny MLP so you can still verify the script runs.

Usage:
  python src/run_vit_gnn.py
"""
from __future__ import annotations

import sys
import torch
import torch.nn as nn

try:
    import timm
except Exception:
    timm = None

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
except Exception:
    GCNConv = None


class ViTEncoder(nn.Module):
    def __init__(self, model_name: str = "vit_small_patch16_224", pretrained: bool = True):
        super().__init__()
        if timm is None:
            raise RuntimeError("`timm` is required for ViTEncoder. Install it (pip install timm).")
        # Create a ViT that returns patch/token embeddings
        # We use num_classes=0 to get raw features where supported
        try:
            self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        except Exception:
            # Fallback: create without num_classes and rely on forward_features
            self.vit = timm.create_model(model_name, pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For many timm ViT models forward will produce a feature vector if num_classes=0.
        # We try a few ways to extract patch tokens. The goal is to return (B, N, C)
        if hasattr(self.vit, "forward_features"):
            feats = self.vit.forward_features(x)
            # Some versions return (B, C) pooled features; try to reshape if sequence available
            if feats.ndim == 3:
                return feats
            # If (B, C) return as single-node sequence
            return feats.unsqueeze(1)

        out = self.vit(x)
        if out.ndim == 2:
            return out.unsqueeze(1)
        return out


class SimpleGNN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128, out_channels: int = 2):
        super().__init__()
        if GCNConv is None:
            raise RuntimeError("`torch_geometric` is required for SimpleGNN. Install torch-geometric.")
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


class ViT_GNN_Model(nn.Module):
    def __init__(self, vit_name: str = "vit_small_patch16_224", gnn_hidden: int = 128, out_classes: int = 2):
        super().__init__()
        if timm is None or GCNConv is None:
            raise RuntimeError("Both `timm` and `torch_geometric` are required to use ViT_GNN_Model.")
        self.encoder = ViTEncoder(vit_name)
        # We will infer channels after a dummy forward in build
        self.gnn = None
        self.out_classes = out_classes

    def build(self, sample_input: torch.Tensor):
        tokens = self.encoder(sample_input)
        B, N, C = tokens.shape
        self.gnn = SimpleGNN(in_channels=C, hidden=128, out_channels=self.out_classes)

    def forward(self, images: torch.Tensor, data_batch: Batch):
        # images: (B,3,H,W)
        tokens = self.encoder(images)  # (B, N, C)
        B, N, C = tokens.shape
        # Flatten tokens into a single node set across the batch
        node_feats = tokens.reshape(B * N, C)
        # Expect data_batch.edge_index and data_batch.batch length = num_nodes
        return self.gnn(node_feats, data_batch.edge_index, data_batch.batch)


def make_fully_connected_edge_index(num_nodes: int, device: torch.device):
    # Create undirected fully connected graph (no self-loops)
    rows = []
    cols = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            rows.append(i)
            cols.append(j)
    edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
    return edge_index


def main():
    device = torch.device("cpu")

    # Create a small random batch of images
    B = 2
    images = torch.randn(B, 3, 224, 224, device=device)

    if timm is None or GCNConv is None:
        print("Missing dependencies for full ViT+GNN pipeline.")
        print("Install `timm` (pip install timm) and `torch-geometric` following PyG docs.")
        # Minimal fallback: run a tiny MLP to confirm script runs
        flat = images.flatten(1)
        out = nn.Sequential(nn.Linear(flat.shape[1], 128), nn.ReLU(), nn.Linear(128, 2))(flat)
        print("Fallback output shape:", out.shape)
        return

    # Build model and a dummy PyG Batch with fully connected graph
    model = ViT_GNN_Model()
    # Run a build pass to initialize GNN dims
    model.build(images)

    # Create node-level batch: each image has N nodes (use encoder to get N)
    tokens = model.encoder(images)
    _, N, C = tokens.shape
    num_nodes = B * N
    edge_index = make_fully_connected_edge_index(num_nodes, device)
    # batch vector mapping each node to its example in the batch
    batch_vec = torch.repeat_interleave(torch.arange(B, device=device), N)

    # Create torch_geometric Batch
    try:
        data = Data(x=tokens.reshape(num_nodes, C), edge_index=edge_index)
        batch = Batch.from_data_list([data])
        # The above Batch.from_data_list with a single data treats all nodes as one graph; instead set batch manually
        batch = data
        batch.batch = batch_vec
        out = model(images, batch)
        print("Model output shape:", out.shape)
    except Exception as e:
        print("Failed to run full pipeline:", e)


if __name__ == "__main__":
    main()
