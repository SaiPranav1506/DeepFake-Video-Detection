import os
import numpy as np
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

# Optional fallback backend
try:
    import imageio
except Exception:
    imageio = None

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def sample_video_frames(video_path, sample_rate=5, max_frames=32):
    """Sample frames from a video every `sample_rate` frames up to `max_frames` frames.

    Backends:
    - imageio (preferred on many Linux deploys)
    - OpenCV (cv2)

    Control via env var:
      VIDEO_BACKEND=auto|imageio|opencv

    Uses OpenCV if available, otherwise falls back to imageio (if installed).
    Returns list of RGB numpy arrays.
    """
    frames = []

    backend = (os.environ.get('VIDEO_BACKEND') or 'auto').strip().lower()
    if backend not in ('auto', 'imageio', 'opencv'):
        backend = 'auto'

    def _read_with_imageio() -> list:
        if imageio is None:
            raise RuntimeError("imageio not installed")
        try:
            reader = imageio.get_reader(str(video_path))
        except Exception as e:
            raise RuntimeError(f"Failed to open video with imageio: {e}")
        try:
            for i, frame in enumerate(reader):
                if len(frames) >= max_frames:
                    break
                if i % sample_rate == 0:
                    frames.append(frame)  # imageio frames are already RGB
        finally:
            try:
                reader.close()
            except Exception:
                pass
        return frames

    def _read_with_opencv() -> list:
        if not _HAS_CV2 or cv2 is None:
            raise RuntimeError("opencv not installed")
        cap = cv2.VideoCapture(str(video_path))
        try:
            idx = 0
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % sample_rate == 0:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                idx += 1
        finally:
            try:
                cap.release()
            except Exception:
                pass
        return frames

    # Prefer imageio on auto (more stable on some deploys)
    if backend in ('auto', 'imageio'):
        try:
            return _read_with_imageio()
        except Exception:
            if backend == 'imageio':
                raise

    if backend in ('auto', 'opencv'):
        try:
            return _read_with_opencv()
        except Exception:
            if backend == 'opencv':
                raise

    raise RuntimeError("No video backend available: install opencv-python-headless or imageio")

def normalize_adjacency(A):
    """Compute symmetric normalized adjacency: D^{-1/2} (A+I) D^{-1/2}"""
    A = A.copy().astype(np.float32)
    N = A.shape[0]
    A = A + np.eye(N, dtype=A.dtype)
    deg = np.sum(A, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt
