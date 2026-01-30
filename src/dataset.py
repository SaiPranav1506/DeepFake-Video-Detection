import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

import random
from io import BytesIO

try:
    from PIL import Image
    import torchvision.transforms as T
except Exception:
    Image = None
    T = None


class VideoFacesDataset(Dataset):
    """Loads per-video .npz produced by data_prepare.py

    Each file is expected to contain array 'faces' shaped (num_faces, H, W, C).
    The label is inferred from filename if possible: filenames containing 'fake' -> 1, 'real' -> 0.
    """

    def __init__(self, data_dir, transform=None, augment=False, image_size=(224, 224), recursive: bool = False):
        self.data_dir = Path(data_dir)
        self.recursive = recursive
        self.files = list(self.data_dir.rglob('*.npz')) if recursive else list(self.data_dir.glob('*.npz'))
        self.transform = transform
        self.augment = augment
        self.image_size = image_size

        # Build a default transform pipeline (train = randomized, val = deterministic)
        # so that all backbones (especially ViT) see a consistent input size.
        if self.transform is None and T is not None:
            if self.augment:
                self.transform = self._build_train_transform(image_size)
            else:
                self.transform = self._build_eval_transform(image_size)

    def __len__(self):
        return len(self.files)

    def infer_label(self, fname):
        s = fname.lower()
        if 'fake' in s or 'deepfake' in s:
            return 1
        if 'real' in s or 'original' in s:
            return 0
        return -1

    def __getitem__(self, idx):
        p = self.files[idx]
        data = np.load(p)
        faces = data['faces']  # (N, H, W, C)
        if 'label' in data:
            label = int(np.array(data['label']).item())
        else:
            label = self.infer_label(p.name)

        if label == -1:
            raise ValueError(
                f"Could not infer label from filename: {p.name}. "
                "Expected 'fake'/'real' (or 'deepfake'/'original') in the filename."
            )
        
        # Apply transform (train transform is randomized, eval transform is deterministic)
        if self.transform:
            faces_aug = []
            for f in faces:
                if Image is not None:
                    img = Image.fromarray(f).convert('RGB')
                    img = self.transform(img)
                    # ensure numpy HWC uint8
                    img = np.array(img)
                else:
                    img = f
                faces_aug.append(img)
            faces = np.stack(faces_aug) if len(faces_aug) > 0 else faces
        
        sample = {'faces': faces, 'label': label, 'file': str(p.name)}
        return sample

    class _RandomJPEGCompression:
        def __init__(self, p: float = 0.5, quality_min: int = 35, quality_max: int = 95):
            self.p = float(p)
            self.quality_min = int(quality_min)
            self.quality_max = int(quality_max)

        def __call__(self, img):
            if Image is None:
                return img
            if random.random() > self.p:
                return img
            q = random.randint(self.quality_min, self.quality_max)
            buf = BytesIO()
            try:
                img.save(buf, format='JPEG', quality=int(q), optimize=True)
                buf.seek(0)
                out = Image.open(buf).convert('RGB')
                return out
            finally:
                try:
                    buf.close()
                except Exception:
                    pass

    class _RandomDownscaleUpscale:
        def __init__(self, p: float = 0.25, min_scale: float = 0.5, max_scale: float = 0.9):
            self.p = float(p)
            self.min_scale = float(min_scale)
            self.max_scale = float(max_scale)

        def __call__(self, img):
            if Image is None:
                return img
            if random.random() > self.p:
                return img
            w, h = img.size
            s = random.uniform(self.min_scale, self.max_scale)
            w2 = max(8, int(w * s))
            h2 = max(8, int(h * s))
            img_small = img.resize((w2, h2), resample=Image.BILINEAR)
            return img_small.resize((w, h), resample=Image.BILINEAR)

    def _build_eval_transform(self, image_size):
        # Deterministic, size-stable preprocessing
        return T.Compose([
            T.Resize(image_size),
        ])

    def _build_train_transform(self, image_size):
        # DFDC-style augmentations: crop/resize jitter + compression artifacts + mild blur
        # Keep it moderate; over-strong aug on tiny datasets can hurt.
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02)], p=0.7),
            T.RandomGrayscale(p=0.05),
            self._RandomDownscaleUpscale(p=0.25, min_scale=0.55, max_scale=0.9),
            self._RandomJPEGCompression(p=0.5, quality_min=35, quality_max=95),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.25),
        ])
