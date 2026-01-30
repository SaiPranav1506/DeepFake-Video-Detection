import os
import numpy as np
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False
    # fallback to imageio for video reading
    try:
        import imageio
    except Exception:
        imageio = None

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def sample_video_frames(video_path, sample_rate=5, max_frames=32):
    """Sample frames from a video every `sample_rate` frames up to `max_frames` frames.

    Uses OpenCV if available, otherwise falls back to imageio (if installed).
    Returns list of RGB numpy arrays.
    """
    frames = []
    if _HAS_CV2:
        cap = cv2.VideoCapture(str(video_path))
        idx = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_rate == 0:
                # convert BGR -> RGB
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
        return frames

    # fallback: use imageio
    if imageio is None:
        raise RuntimeError("No video backend available: install opencv-python or imageio")
    try:
        reader = imageio.get_reader(str(video_path))
    except Exception as e:
        raise RuntimeError(f"Failed to open video with imageio: {e}")
    idx = 0
    for i, frame in enumerate(reader):
        if len(frames) >= max_frames:
            break
        if i % sample_rate == 0:
            # imageio frames are already RGB
            frames.append(frame)
    reader.close()
    return frames

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
