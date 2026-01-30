"""Inference script: load checkpoint and run model on a sample video (.npz).

Usage:
  python src/test_vit_gnn.py path/to/video.npz [--ckpt checkpoints/vit_gnn_ckpt.pt]
"""
from __future__ import annotations

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn


def load_video_npz(path: str):
    data = np.load(path, allow_pickle=True)
    # pick first array-like entry
    if hasattr(data, "files") and len(data.files) > 0:
        arr = data[data.files[0]]
    else:
        # fallback: try loading as array
        arr = np.load(path)
    return arr


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


def build_wrapper_model():
    # Create same wrapper shape used when saving during training
    from importlib import import_module, util
    mod = util.spec_from_file_location("run_vit_gnn", os.path.join(os.path.dirname(__file__), "run_vit_gnn.py"))
    run_mod = util.module_from_spec(mod)
    mod.loader.exec_module(run_mod)

    base = run_mod.ViT_GNN_Model()
    # initialize gnn weights/shapes by running a build pass with a dummy input
    try:
        base.build(torch.randn(1, 3, 224, 224))
    except Exception:
        # if default size fails, try 256
        try:
            base.build(torch.randn(1, 3, 256, 256))
        except Exception:
            pass

    class Wrapper(nn.Module):
        def __init__(self, base_model: nn.Module):
            super().__init__()
            self.base_model = base_model

        def forward(self, x: torch.Tensor):
            device_local = x.device
            Bx = x.shape[0]
            # ensure size
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
            adj = torch.zeros((num_nodes, num_nodes), device=device_local)
            for b in range(Bx):
                s = b * N
                e = s + N
                adj[s:e, s:e] = 1.0
            deg = adj.sum(dim=1, keepdim=True)
            deg[deg == 0] = 1.0
            adj = adj / deg
            return self.base_model(x, adj)

    return Wrapper(base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help=".npz video file")
    parser.add_argument("--ckpt", default=os.path.join("checkpoints", "vit_gnn_ckpt.pt"))
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print("Video not found:", args.video)
        sys.exit(1)

    arr = load_video_npz(args.video)
    # arr may be (T,H,W,3) or (T,3,H,W) or (H,W,3)
    if arr.ndim == 4:
        # T,H,W,C or T,C,H,W
        if arr.shape[-1] == 3:
            frames = arr
        elif arr.shape[1] == 3:
            # T,C,H,W -> convert to T,H,W,C
            frames = np.transpose(arr, (0,2,3,1))
        else:
            frames = arr
    elif arr.ndim == 3:
        # single frame H,W,C or C,H,W
        if arr.shape[-1] == 3:
            frames = arr[np.newaxis]
        elif arr.shape[0] == 3:
            frames = np.transpose(arr, (1,2,0))[np.newaxis]
        else:
            frames = arr[np.newaxis]
    else:
        raise RuntimeError("Unsupported video array shape: " + str(arr.shape))

    # pick middle frame
    T = frames.shape[0]
    idx = T // 2
    frame = frames[idx]
    # convert to float tensor
    if frame.dtype == np.uint8:
        frame = frame.astype(np.float32) / 255.0
    else:
        frame = frame.astype(np.float32)
    # H,W,C -> C,H,W
    img = np.transpose(frame, (2,0,1))
    img_t = torch.from_numpy(img).unsqueeze(0)

    # load checkpoint
    if not os.path.exists(args.ckpt):
        print("Checkpoint not found:", args.ckpt)
        sys.exit(1)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    # determine model type from checkpoint keys
    keys = list(ckpt.get("model_state", ckpt).keys())
    model = None
    if any(k.startswith("base_model") for k in keys):
        # expects ViT wrapper; require timm
        try:
            mod = build_wrapper_model()
            mod.load_state_dict(ckpt["model_state"]) if "model_state" in ckpt else mod.load_state_dict(ckpt)
            model = mod
        except Exception as e:
            print("Failed to build ViT wrapper model:", e)
            sys.exit(1)
    else:
        # fallback model
        m = FallbackModel()
        m.load_state_dict(ckpt["model_state"]) if "model_state" in ckpt else m.load_state_dict(ckpt)
        model = m

    model.eval()
    with torch.no_grad():
        out = model(img_t)
        probs = torch.softmax(out, dim=1)
        pred = probs.argmax(dim=1).item()
    print("Predicted class:", pred)
    print("Probabilities:", probs.squeeze().tolist())


if __name__ == "__main__":
    main()
