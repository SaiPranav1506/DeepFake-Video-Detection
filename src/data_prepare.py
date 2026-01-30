"""Prepare dataset: unzip archive, extract frames, detect faces and save per-video .npz files.

Usage example (use forward slashes or escape backslashes on Windows):
    python src/data_prepare.py --archive C:/datasets/archive.zip --outdir data/prepared
"""
import argparse
import zipfile
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import csv

from facenet_pytorch import MTCNN
from utils import ensure_dir, sample_video_frames
from PIL import Image
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False


def _detect_faces_as_arrays(img, mtcnn: MTCNN, size: int) -> list[np.ndarray]:
    """Detect faces using MTCNN and return resized RGB crops as numpy arrays."""
    try:
        pil = img if isinstance(img, Image.Image) else Image.fromarray(img)
        pil = pil.convert('RGB')
    except Exception:
        return []

    try:
        boxes, _ = mtcnn.detect(pil)
    except Exception:
        boxes = None
    if boxes is None:
        return []

    w, h = pil.size
    faces: list[np.ndarray] = []
    for b in boxes:
        try:
            x1, y1, x2, y2 = [int(v) for v in b]
        except Exception:
            continue
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        try:
            crop = pil.crop((x1, y1, x2, y2))
            crop = crop.resize((size, size))
            faces.append(np.array(crop))
        except Exception:
            continue
    return faces


def infer_label_from_path(path: Path):
    # Prefer exact path-part matches (prevents false positives like `...\Deepfake\...`).
    parts = [p.lower() for p in Path(path).parts]
    if 'real' in parts or 'original' in parts:
        return 0
    if 'fake' in parts or 'deepfake' in parts:
        return 1

    # Tokenize each path part on common separators and check tokens.
    def _tokens(s: str) -> list[str]:
        toks: list[str] = []
        cur = []
        for ch in s.lower():
            if ch.isalnum():
                cur.append(ch)
            else:
                if cur:
                    toks.append(''.join(cur))
                    cur = []
        if cur:
            toks.append(''.join(cur))
        return toks

    all_tokens: set[str] = set()
    for part in parts:
        all_tokens.update(_tokens(part))

    if 'real' in all_tokens or 'original' in all_tokens:
        return 0
    if 'fake' in all_tokens or 'deepfake' in all_tokens or 'synthesis' in all_tokens or 'manipulated' in all_tokens:
        return 1
    return None


def extract_archive(archive_path, dest_dir):
    dest_dir = Path(dest_dir)
    ensure_dir(dest_dir)
    with zipfile.ZipFile(archive_path, 'r') as z:
        z.extractall(dest_dir)
    return dest_dir


def detect_and_save(video_path, out_path, mtcnn, sample_rate=5, max_frames=32, size=224):
    frames = sample_video_frames(str(video_path), sample_rate=sample_rate, max_frames=max_frames)
    faces: list[np.ndarray] = []
    for f in frames:
        faces.extend(_detect_faces_as_arrays(f, mtcnn, size))
    if len(faces) == 0:
        return False
    faces = np.stack(faces)
    np.savez_compressed(out_path, faces=faces)
    return True


def _parse_flat_frames_key(p: Path) -> tuple[str, tuple[int, int, str]] | None:
    """Parse DFDC-style frame filenames like `<videoid>_<frame>_<idx>.png`.

    Returns (video_id, sort_key) or None if not matching expected pattern.
    """
    stem = p.stem
    parts = stem.split('_')
    if len(parts) < 2:
        return None
    video_id = parts[0]
    if not video_id:
        return None

    frame_idx = 0
    face_idx = 0
    if len(parts) >= 2 and parts[1].isdigit():
        frame_idx = int(parts[1])
    if len(parts) >= 3 and parts[2].isdigit():
        face_idx = int(parts[2])
    return video_id, (frame_idx, face_idx, p.name)


def _is_flat_frames_layout(img_paths: list[Path]) -> bool:
    """Heuristic: many images, filenames share `<id>_<num>...` pattern.

    Avoid mis-classifying typical per-video frame folders like `000001.jpg`.
    """
    if len(img_paths) < 50:
        return False
    parsed = 0
    checked = 0
    video_ids: set[str] = set()
    # sample at most 500 files to keep this cheap
    step = max(1, len(img_paths) // 500)
    for p in img_paths[::step]:
        checked += 1
        res = _parse_flat_frames_key(p)
        if res is None:
            continue
        parsed += 1
        video_ids.add(res[0])
        if len(video_ids) >= 2 and parsed >= 20:
            # enough signal
            break
    if parsed == 0 or checked == 0:
        return False
    # require that most sampled names match the pattern and that multiple ids exist
    return (parsed / checked) >= 0.8 and len(video_ids) >= 2


def load_labels_csv(labels_csv: str | None) -> dict[str, int]:
    """Load optional CSV mapping of video path -> label (0/1).

    Expected columns: video_path,label
    video_path can be relative or absolute; matching is done by suffix.
    """
    if not labels_csv:
        return {}
    mapping: dict[str, int] = {}
    with open(labels_csv, 'r', newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or 'video_path' not in reader.fieldnames or 'label' not in reader.fieldnames:
            raise ValueError("labels_csv must have headers: video_path,label")
        for row in reader:
            vp = (row.get('video_path') or '').strip()
            lab = (row.get('label') or '').strip()
            if not vp:
                continue
            try:
                mapping[vp.replace('\\', '/')] = int(lab)
            except Exception:
                continue
    return mapping


def resolve_label(path: Path, dataset_root: Path | None, labels_map: dict[str, int]) -> int | None:
    """Resolve label using CSV mapping first, then path inference.

    Important: infer from a path relative to the extracted dataset root.
    This avoids false positives from the repo folder name (e.g. `C:\\Deepfake`).
    """
    if labels_map:
        p_abs = str(path).replace('\\', '/').lower()
        # suffix match allows CSV rows to be relative paths
        for key, val in labels_map.items():
            if p_abs.endswith(key.lower()):
                return int(val)

    # Prefer relative paths for heuristic inference
    p_for_infer = path
    if dataset_root is not None:
        try:
            p_for_infer = path.relative_to(dataset_root)
        except Exception:
            p_for_infer = path

    return infer_label_from_path(p_for_infer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--sample_rate', type=int, default=5)
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--max_videos', type=int, default=None, help='Optional cap when splitting flat frame folders into per-video samples')
    parser.add_argument('--max_files', type=int, default=None, help='Optional cap on number of discovered video files to process')
    parser.add_argument('--frames-are-faces', action='store_true', help='Treat input images in frame folders as already-cropped faces (skip MTCNN)')
    parser.add_argument('--labels_csv', default=None, help='Optional CSV with video_path,label')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # If archive is a directory, treat it as already-extracted media
    archive_path = Path(args.archive)
    if archive_path.is_dir():
        print(f'Using pre-extracted directory: {archive_path}')
        root = archive_path
    else:
        print('Extracting archive...')
        root = extract_archive(args.archive, outdir / 'raw')
    # initialize MTCNN (only used for raw videos or if frames-are-faces is false)
    mtcnn = MTCNN(keep_all=True)

    labels_map = load_labels_csv(args.labels_csv)

    # discover video files under root
    video_exts = ('.mp4', '.mov', '.avi', '.mkv')
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    videos = [p for p in root.rglob('*') if p.suffix.lower() in video_exts]
    if args.max_files is not None:
        try:
            max_files = max(0, int(args.max_files))
        except Exception:
            max_files = 0
        if max_files > 0:
            videos = videos[:max_files]
        else:
            videos = []

    prepared_dir = outdir / 'videos'
    ensure_dir(prepared_dir)

    if len(videos) > 0:
        print(f'Found {len(videos)} video files')
        for v in tqdm(videos):
            rel = v.relative_to(root)
            label = resolve_label(v, root, labels_map)
            prefix = '' if label is None else ('video_fake__' if label == 1 else 'video_real__')
            outname = prefix + str(rel).replace(os.sep, '__') + '.npz'
            outpath = prepared_dir / outname
            frames = sample_video_frames(str(v), sample_rate=args.sample_rate, max_frames=args.max_frames)
            faces: list[np.ndarray] = []
            for f in frames:
                faces.extend(_detect_faces_as_arrays(f, mtcnn, args.size))

            if len(faces) == 0:
                success = False
            else:
                faces = np.stack(faces)
                if label is None:
                    np.savez_compressed(outpath, faces=faces)
                else:
                    np.savez_compressed(outpath, faces=faces, label=np.int64(label))
                success = True

            if not success and outpath.exists():
                outpath.unlink()
    else:
        # look for folders of image frames; each folder becomes one sample
        frame_dirs = [d for d in root.rglob('*') if d.is_dir() and any((f.suffix.lower() in image_exts) for f in d.iterdir())]
        print(f'Found {len(frame_dirs)} frame folders')
        for d in tqdm(frame_dirs):
            # collect images sorted
            imgs = sorted([p for p in d.iterdir() if p.suffix.lower() in image_exts])
            if len(imgs) == 0:
                continue

            label = resolve_label(d, root, labels_map)
            prefix = '' if label is None else ('video_fake__' if label == 1 else 'video_real__')
            rel = d.relative_to(root)

            # DFDC-style "flat frames" layout: many images from many videos mixed in one folder.
            if _is_flat_frames_layout(imgs):
                grouped: dict[str, list[tuple[tuple[int, int, str], Path]]] = {}
                for p in imgs:
                    parsed = _parse_flat_frames_key(p)
                    if parsed is None:
                        continue
                    vid, sort_key = parsed
                    bucket = grouped.get(vid)
                    if bucket is None:
                        bucket = []
                        grouped[vid] = bucket
                    bucket.append((sort_key, p))
                    # keep memory bounded: retain only earliest-ish frames per video
                    if len(bucket) > max(16, args.max_frames * 4):
                        bucket.sort(key=lambda t: t[0])
                        del bucket[args.max_frames :]

                video_ids = sorted(grouped.keys())
                if args.max_videos is not None:
                    video_ids = video_ids[: max(0, int(args.max_videos))]

                for vid in video_ids:
                    items = grouped.get(vid) or []
                    items.sort(key=lambda t: t[0])
                    paths = [p for _, p in items[: args.max_frames]]
                    if len(paths) == 0:
                        continue

                    outname = prefix + str(rel).replace(os.sep, '__') + f'__{vid}.npz'
                    outpath = prepared_dir / outname

                    faces: list[np.ndarray] = []
                    for img_path in paths:
                        try:
                            im = Image.open(img_path).convert('RGB')
                        except Exception:
                            continue
                        if args.frames_are_faces:
                            try:
                                im2 = im.resize((args.size, args.size))
                                faces.append(np.array(im2))
                            except Exception:
                                continue
                        else:
                            faces.extend(_detect_faces_as_arrays(im, mtcnn, args.size))

                    if len(faces) == 0:
                        if outpath.exists():
                            outpath.unlink()
                        continue
                    faces_arr = np.stack(faces)
                    if label is None:
                        np.savez_compressed(outpath, faces=faces_arr)
                    else:
                        np.savez_compressed(outpath, faces=faces_arr, label=np.int64(label))

                continue

            # Standard per-video frame folder: treat the folder as one sample.
            frames: list[np.ndarray] = []
            for p in imgs[: args.max_frames]:
                try:
                    im = Image.open(p).convert('RGB')
                    frames.append(np.array(im))
                except Exception:
                    continue
            if len(frames) == 0:
                continue

            outname = prefix + str(rel).replace(os.sep, '__') + '.npz'
            outpath = prepared_dir / outname

            faces: list[np.ndarray] = []
            for f in frames:
                if args.frames_are_faces:
                    try:
                        im = f if isinstance(f, Image.Image) else Image.fromarray(f)
                        im = im.convert('RGB')
                        im = im.resize((args.size, args.size))
                        faces.append(np.array(im))
                    except Exception:
                        continue
                else:
                    faces.extend(_detect_faces_as_arrays(f, mtcnn, args.size))
            if len(faces) == 0:
                continue
            faces_arr = np.stack(faces)
            if label is None:
                np.savez_compressed(outpath, faces=faces_arr)
            else:
                np.savez_compressed(outpath, faces=faces_arr, label=np.int64(label))

    print('Data preparation finished. Prepared files in:', prepared_dir)


if __name__ == '__main__':
    main()
