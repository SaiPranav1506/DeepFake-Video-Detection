import os

# Render/Gunicorn can intermittently crash (exit 139) when native libs over-thread
# on low-memory containers. Keep thread counts conservative for stability.
os.environ.setdefault('OMP_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
os.environ.setdefault('MKL_NUM_THREADS', os.environ.get('MKL_NUM_THREADS', '1'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', os.environ.get('OPENBLAS_NUM_THREADS', '1'))
os.environ.setdefault('NUMEXPR_NUM_THREADS', os.environ.get('NUMEXPR_NUM_THREADS', '1'))

try:
    import faulthandler
    faulthandler.enable()
except Exception:
    pass
import sys
import csv
import json
import io
import base64
import re
import time
import uuid
from pathlib import Path
from datetime import datetime

import requests
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

if cv2 is not None:
    # Reduce OpenCV internal threading; improves stability on some deployments.
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.dataset import VideoFacesDataset
from src.models import DeepfakeModel, ViTFeatureExtractor, CNNLSTMHybrid
from src.utils import normalize_adjacency, sample_video_frames

try:
    # Optional but recommended face detector (more robust than Haar)
    from facenet_pytorch import MTCNN
except Exception:
    MTCNN = None

try:
    from src.pretrained_detector import PretrainedBackboneDetector, EnsembleDetector
except Exception:
    PretrainedBackboneDetector = None
    EnsembleDetector = None

try:
    from src.enhanced_decision_agent import EnhancedDecisionAgent
except Exception:
    EnhancedDecisionAgent = None

try:
    from src.agent_system import DecisionAgent, MonitoringAgent, ActionAgent, AlertLevel
except Exception:
    DecisionAgent = None
    MonitoringAgent = None
    ActionAgent = None
    AlertLevel = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret-change-me')

# Dev-friendly defaults so UI edits reflect immediately.
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.after_request
def _no_cache_headers(resp):
    # Prevent stale HTML/CSS/JS in browsers during development.
    resp.headers['Cache-Control'] = 'no-store'
    return resp

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Torch thread limits (helps avoid over-parallelization + crashes under Gunicorn).
try:
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '1')))
except Exception:
    pass
try:
    torch.set_num_interop_threads(int(os.environ.get('TORCH_NUM_INTEROP_THREADS', '1')))
except Exception:
    pass
MODEL = None
MODEL_TYPE = None
CHECKPOINT_PATH = None
MODEL_META = {}
LAST_LOAD_STATS = {}
ENHANCED_AGENT = EnhancedDecisionAgent() if EnhancedDecisionAgent is not None else None
AUTOLOAD_ATTEMPTED = False

# UI results are potentially large (justifications, agent output). Flask's default
# cookie-based session cannot reliably hold these without causing 4xx/5xx errors.
# Store results server-side and keep only a small key in the session cookie.
_UI_RESULTS_CACHE: dict[str, dict] = {}
_UI_RESULTS_TTL_SECONDS = int(os.environ.get('UI_RESULTS_TTL_SECONDS', '1800'))  # 30 minutes
_UI_RESULTS_MAX_ITEMS = int(os.environ.get('UI_RESULTS_MAX_ITEMS', '100'))


def _ui_cache_cleanup(now_ts: float | None = None) -> None:
    try:
        now = float(now_ts) if now_ts is not None else time.time()
    except Exception:
        now = time.time()

    # Remove expired
    try:
        expired = []
        for k, v in list(_UI_RESULTS_CACHE.items()):
            ts = v.get('ts')
            try:
                ts_f = float(ts)
            except Exception:
                ts_f = None
            if ts_f is None or (now - ts_f) > float(_UI_RESULTS_TTL_SECONDS):
                expired.append(k)
        for k in expired:
            _UI_RESULTS_CACHE.pop(k, None)
    except Exception:
        return

    # Cap size
    try:
        if len(_UI_RESULTS_CACHE) <= int(_UI_RESULTS_MAX_ITEMS):
            return
        items = sorted(
            [(k, float(v.get('ts') or 0.0)) for k, v in _UI_RESULTS_CACHE.items()],
            key=lambda x: x[1],
        )
        while len(items) > int(_UI_RESULTS_MAX_ITEMS):
            k, _ = items.pop(0)
            _UI_RESULTS_CACHE.pop(k, None)
    except Exception:
        return


def _ui_cache_set(results: list, error: str | None) -> str:
    _ui_cache_cleanup()
    key = uuid.uuid4().hex
    _UI_RESULTS_CACHE[key] = {
        'ts': time.time(),
        'results': results,
        'error': error,
    }
    return key


def _ui_cache_get(key: str | None) -> tuple[list, str | None] | None:
    if not key:
        return None
    _ui_cache_cleanup()
    try:
        payload = _UI_RESULTS_CACHE.get(str(key))
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        return None
    results = payload.get('results') or []
    error = payload.get('error')
    if not isinstance(results, list):
        results = []
    try:
        error = str(error) if error is not None else None
    except Exception:
        error = None
    return results, error

_MTCNN = None


def _get_mtcnn():
    global _MTCNN
    if _MTCNN is not None:
        return _MTCNN
    if MTCNN is None:
        return None
    try:
        _MTCNN = MTCNN(keep_all=True, device=str(DEVICE) if DEVICE.type == 'cuda' else 'cpu')
        return _MTCNN
    except Exception:
        _MTCNN = None
        return None

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'jpg', 'jpeg', 'png', 'bmp'}
USER_DB_PATH = Path('users.json')
SECRETS_DB_PATH = Path('secrets.json')


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _read_csv_rows(path: Path):
    if not path.exists():
        return []
    try:
        with path.open('r', newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            return list(r)
    except Exception:
        return []


def _best_ensemble_history() -> dict | None:
    """Return best recorded ensemble validation metrics from training_history.csv.

    Returns:
        dict with keys: source, best_epoch, val_accuracy, val_f1, val_auc, epochs_ran
    """
    candidates = [
        Path('checkpoints/ensemble/training_history.csv'),
        Path('checkpoints/ensemble_dfdc_smoketest/training_history.csv'),
    ]

    # Prefer larger, more realistic runs when present (e.g., DFDC200).
    try:
        candidates.extend(sorted(Path('checkpoints').glob('ensemble_dfdc200_*/training_history.csv')))
    except Exception:
        pass

    best_overall = None
    for p in candidates:
        rows = _read_csv_rows(p)
        if not rows:
            continue

        # Normalize expected columns
        # Many files use: Epoch, Val_Acc, Val_F1, Val_ROC_AUC
        def get(row, *names):
            for n in names:
                if n in row:
                    return row.get(n)
            return None

        best_row = None
        for row in rows:
            val_acc = _safe_float(get(row, 'Val_Acc', 'val_accuracy', 'Val_Accuracy'))
            val_f1 = _safe_float(get(row, 'Val_F1', 'val_f1', 'Val_F1_Score'))
            val_auc = _safe_float(get(row, 'Val_ROC_AUC', 'val_auc', 'ROC_AUC', 'AUC'))
            epoch = _safe_int(get(row, 'Epoch', 'epoch'))
            score = (
                (val_acc if val_acc is not None else -1.0),
                (val_f1 if val_f1 is not None else -1.0),
                (val_auc if val_auc is not None else -1.0),
            )
            if best_row is None or score > best_row['_score']:
                best_row = {
                    'epoch': epoch,
                    'val_accuracy': val_acc,
                    'val_f1': val_f1,
                    'val_auc': val_auc,
                    '_score': score,
                }

        epochs_ran = max((_safe_int(get(r, 'Epoch', 'epoch')) or 0) for r in rows)
        out = {
            'source': str(p.as_posix()),
            'best_epoch': best_row.get('epoch') if best_row else None,
            'val_accuracy': best_row.get('val_accuracy') if best_row else None,
            'val_f1': best_row.get('val_f1') if best_row else None,
            'val_auc': best_row.get('val_auc') if best_row else None,
            'epochs_ran': epochs_ran,
        }

        if best_overall is None:
            best_overall = out
        else:
            cur = (
                best_overall.get('val_accuracy') or -1.0,
                best_overall.get('val_f1') or -1.0,
                best_overall.get('val_auc') or -1.0,
            )
            nxt = (
                out.get('val_accuracy') or -1.0,
                out.get('val_f1') or -1.0,
                out.get('val_auc') or -1.0,
            )
            if nxt > cur:
                best_overall = out

    return best_overall


def _pick_best_checkpoint_for_autoload() -> tuple[str, str] | None:
    """Pick a reasonable default checkpoint to auto-load.

    Heuristic:
    - Prefer ensemble runs on larger datasets (folder name contains 'dfdc200')
    - Then other ensemble runs (contains 'dfdc')
    - Then local/smoke ensemble
    - Finally, fall back to legacy single-model checkpoints

    Returns:
        (checkpoint_path, model_type)
    """

    def score_folder(p: Path) -> int:
        s = str(p.as_posix()).lower()
        if 'dfdc200' in s:
            return 30
        if 'dfdc' in s:
            return 20
        if 'ensemble' in s:
            return 10
        return 0

    def best_metrics_from_history(csv_path: Path) -> tuple[float, float, float]:
        # Returns (val_acc, val_f1, val_auc)
        try:
            rows = _read_csv_rows(csv_path)
            if not rows:
                return (-1.0, -1.0, -1.0)

            def get(row, *names):
                for n in names:
                    if n in row:
                        return row.get(n)
                return None

            best = (-1.0, -1.0, -1.0)
            for row in rows:
                val_acc = _safe_float(get(row, 'Val_Acc', 'val_accuracy', 'Val_Accuracy'))
                val_f1 = _safe_float(get(row, 'Val_F1', 'val_f1', 'Val_F1_Score'))
                val_auc = _safe_float(get(row, 'Val_ROC_AUC', 'val_auc', 'ROC_AUC', 'AUC'))
                score = (
                    (val_acc if val_acc is not None else -1.0),
                    (val_f1 if val_f1 is not None else -1.0),
                    (val_auc if val_auc is not None else -1.0),
                )
                if score > best:
                    best = score
            return best
        except Exception:
            return (-1.0, -1.0, -1.0)

    def calibration_penalty(folder: Path) -> int:
        """Penalty for suspicious calibration thresholds.

        Some older runs wrote extreme best thresholds (near 0/1). Those often
        correlate with brittle real-world behavior (e.g. flagging most videos as fake).
        """
        try:
            cand = folder / 'calibration_best.json'
            if not cand.exists():
                return 0
            payload = json.loads(cand.read_text(encoding='utf-8'))
            thr = payload.get('best_thr_accuracy')
            thr_f = float(thr) if thr is not None else None
            if thr_f is None:
                return 0
            # Extreme thresholds are a red flag; down-rank the run.
            if thr_f < 0.05 or thr_f > 0.95:
                return 15
        except Exception:
            return 0
        return 0

    # 1) Prefer ensemble checkpoints with training_history.csv (when sane)
    candidates: list[tuple[tuple[int, float, float, float], str, str]] = []
    try:
        for ckpt in Path('checkpoints').glob('ensemble*/checkpoint_best.pt'):
            folder = ckpt.parent
            hist = folder / 'training_history.csv'
            metrics = best_metrics_from_history(hist) if hist.exists() else (-1.0, -1.0, -1.0)
            pri = score_folder(folder)
            pri = max(0, int(pri) - int(calibration_penalty(folder)))
            candidates.append(((pri, *metrics), str(ckpt), 'ensemble_pretrained'))
    except Exception:
        pass

    # 2) Also consider single-backbone pretrained checkpoints.
    # These can generalize better on in-the-wild videos when an ensemble run is brittle.
    try:
        for ckpt in Path('checkpoints').glob('pretrained*/checkpoint_best*.pt'):
            folder = ckpt.parent
            pri = score_folder(folder)
            # Slightly below ensemble of same dataset, but can win if ensemble is penalized.
            pri = int(pri) - 3
            candidates.append(((pri, -1.0, -1.0, -1.0), str(ckpt), 'pretrained'))
    except Exception:
        pass

    if candidates:
        candidates.sort(reverse=True, key=lambda x: x[0])
        return candidates[0][1], candidates[0][2]

    # 3) Fallback to legacy checkpoints (single-model)
    checkpoint_best = list(Path('.').glob('checkpoint_best.pt'))
    if checkpoint_best:
        return str(checkpoint_best[0]), 'vit_gcn'

    checkpoint_rnn = Path('rnn_dummy_checkpoint.pt')
    if checkpoint_rnn.exists():
        return str(checkpoint_rnn), 'cnn_lstm'

    return None


def _resolve_checkpoint_path(p: str) -> Path:
    try:
        path = Path(p)
        if not path.is_absolute():
            base = Path(__file__).resolve().parent
            path = (base / path).resolve()
        return path
    except Exception:
        return Path(p)


def _download_checkpoint(url: str, dest_path: Path) -> bool:
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists() and dest_path.stat().st_size > 0:
            return True

        tmp_path = dest_path.with_suffix(dest_path.suffix + '.tmp')
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

        timeout = int(os.environ.get('MODEL_DOWNLOAD_TIMEOUT', '120'))
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with open(tmp_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        tmp_path.replace(dest_path)
        return dest_path.exists() and dest_path.stat().st_size > 0
    except Exception as e:
        print(f"⚠️ Failed to download checkpoint from {url}: {e}")
        return False


def _build_autoload_candidates() -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []

    env_path = os.environ.get('MODEL_PATH') or os.environ.get('CHECKPOINT_PATH')
    env_url = os.environ.get('MODEL_URL') or os.environ.get('CHECKPOINT_URL')
    env_type = os.environ.get('MODEL_TYPE') or os.environ.get('CHECKPOINT_TYPE') or 'pretrained'

    if env_url:
        default_name = os.environ.get('MODEL_FILENAME') or 'checkpoint_best.pt'
        target = _resolve_checkpoint_path(env_path or os.path.join('checkpoints', default_name))
        if _download_checkpoint(env_url, target):
            candidates.append((str(target), str(env_type)))

    if env_path:
        target = _resolve_checkpoint_path(env_path)
        if target.exists():
            candidates.append((str(target), str(env_type)))
        else:
            print(f"⚠️ MODEL_PATH does not exist: {target}")

    picked = _pick_best_checkpoint_for_autoload()
    if picked:
        candidates.append((str(picked[0]), str(picked[1])))

    # If the top pick fails (e.g., incompatible checkpoint), try other local candidates.
    try:
        for ckpt in Path('checkpoints').glob('pretrained*/checkpoint_best*.pt'):
            candidates.append((str(ckpt), 'pretrained'))
    except Exception:
        pass
    try:
        for ckpt in Path('checkpoints').glob('ensemble*/checkpoint_best.pt'):
            candidates.append((str(ckpt), 'ensemble_pretrained'))
    except Exception:
        pass

    return candidates


def _attempt_autoload(no_autoload: bool = False) -> None:
    global AUTOLOAD_ATTEMPTED, MODEL_META
    if AUTOLOAD_ATTEMPTED or bool(no_autoload):
        return
    AUTOLOAD_ATTEMPTED = True

    candidates = _build_autoload_candidates()
    seen = set()
    loaded = False
    for ckpt_path, model_type in candidates:
        key = (ckpt_path, model_type)
        if key in seen:
            continue
        seen.add(key)
        if model_type == 'ensemble_pretrained':
            # Default backbones can be overridden via ENSEMBLE_BACKBONES env var.
            MODEL_META = {
                'backbones': [
                    x.strip() for x in str(os.environ.get('ENSEMBLE_BACKBONES', 'efficientnet_b0,resnet50')).split(',')
                    if x.strip()
                ]
            }
        else:
            MODEL_META = {}
        ok = load_model(str(ckpt_path), str(model_type))
        if ok:
            print(f"✅ Auto-loaded checkpoint: {ckpt_path} ({model_type})")
            loaded = True
            break
        print(f"⚠️ Auto-load failed, trying next: {ckpt_path} ({model_type})")

    if not loaded:
        print("⚠️ No compatible checkpoint auto-loaded. Please load a model manually.")


def _latest_eval_summary() -> dict | None:
    """Return the best row (by accuracy then f1 then auc) from evaluation_summary.csv."""
    p = Path('evaluation_summary.csv')
    rows = _read_csv_rows(p)
    if not rows:
        return None

    best = None
    for row in rows:
        acc = _safe_float(row.get('accuracy'))
        f1 = _safe_float(row.get('f1'))
        auc = _safe_float(row.get('roc_auc'))
        score = (
            (acc if acc is not None else -1.0),
            (f1 if f1 is not None else -1.0),
            (auc if auc is not None else -1.0),
        )
        if best is None or score > best['_score']:
            row2 = dict(row)
            row2['_score'] = score
            best = row2
    if best:
        best.pop('_score', None)
    return best


def _try_repo_metrics_reply(message: str) -> str | None:
    """Small deterministic 'agent' that answers common metric questions."""
    text = (message or '').strip().lower()
    if not text:
        return None

    wants_best = any(k in text for k in ['best', 'highest', 'max'])
    wants_epochs = any(k in text for k in ['epoch', 'epochs']) and any(k in text for k in ['run', 'ran', 'trained', 'train', 'completed', 'how many', 'number'])
    wants_metrics = any(k in text for k in ['accuracy', 'precision', 'recall', 'f1', 'f1-score', 'f1 score'])

    if not (wants_best or wants_epochs or wants_metrics):
        return None

    best_ens = _best_ensemble_history()
    best_eval = _latest_eval_summary()

    lines = []
    if wants_epochs:
        if best_ens and best_ens.get('epochs_ran'):
            lines.append(f"Ensemble epochs run: {best_ens['epochs_ran']}")
            if best_ens.get('source'):
                lines.append(f"Source: {best_ens['source']}")
        else:
            lines.append("I couldn't find an ensemble training history file to determine epochs run.")

    if wants_best:
        if best_ens:
            lines.append("Best recorded ensemble validation:")
            if best_ens.get('best_epoch') is not None:
                lines.append(f"- Epoch: {best_ens['best_epoch']}")
            if best_ens.get('val_accuracy') is not None:
                lines.append(f"- Val Accuracy: {best_ens['val_accuracy']:.4f}")
            if best_ens.get('val_f1') is not None:
                lines.append(f"- Val F1: {best_ens['val_f1']:.4f}")
            if best_ens.get('val_auc') is not None:
                lines.append(f"- Val AUC: {best_ens['val_auc']:.4f}")
        else:
            lines.append("I couldn't find ensemble training history to compute best values.")

    if wants_metrics and not wants_best:
        # Prefer evaluation_summary.csv for accuracy/precision/recall/f1 if present
        if best_eval:
            lines.append("Best evaluation_summary.csv row:")
            ckpt = best_eval.get('checkpoint')
            mtype = best_eval.get('model_type')
            if ckpt or mtype:
                lines.append(f"- Checkpoint: {ckpt or 'unknown'} ({mtype or 'unknown'})")
            for k, label in [
                ('accuracy', 'Accuracy'),
                ('precision', 'Precision'),
                ('recall', 'Recall'),
                ('f1', 'F1'),
                ('roc_auc', 'ROC AUC'),
            ]:
                v = _safe_float(best_eval.get(k))
                if v is not None:
                    lines.append(f"- {label}: {v:.4f}")
        else:
            # fallback to ensemble history
            if best_ens:
                lines.append("Latest known ensemble validation (best recorded):")
                if best_ens.get('val_accuracy') is not None:
                    lines.append(f"- Val Accuracy: {best_ens['val_accuracy']:.4f}")
                if best_ens.get('val_f1') is not None:
                    lines.append(f"- Val F1: {best_ens['val_f1']:.4f}")
            else:
                return None

    return "\n".join(lines).strip() if lines else None


def _env_str(name: str) -> str | None:
    """Read an env var and normalize common Windows/.env quoting issues."""
    val = os.environ.get(name)
    if val is None:
        return None
    val = str(val).strip()
    # Strip surrounding single/double quotes if present
    if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
        val = val[1:-1].strip()
    return val or None


FIREBASE_API_KEY = _env_str('FIREBASE_API_KEY')
FIREBASE_DATABASE_URL = _env_str('FIREBASE_DATABASE_URL')


def _firebase_rtdb_base() -> str | None:
    if not FIREBASE_DATABASE_URL:
        return None
    return FIREBASE_DATABASE_URL.rstrip('/')


def _firebase_id_token() -> str | None:
    return session.get('firebase_id_token')


def _rtdb_url(path: str, auth: bool = True) -> str:
    base = _firebase_rtdb_base()
    if not base:
        raise RuntimeError('FIREBASE_DATABASE_URL is not set')
    p = path.strip('/')
    url = f"{base}/{p}.json"
    if auth:
        token = _firebase_id_token()
        if not token:
            raise RuntimeError('Not authenticated with Firebase (missing idToken)')
        url = f"{url}?auth={token}"
    return url


def _rtdb_get(path: str):
    resp = requests.get(_rtdb_url(path), timeout=15)
    if resp.status_code >= 400:
        raise RuntimeError(f"RTDB GET failed: {resp.status_code} {resp.text[:200]}")
    return resp.json() if resp.content else None


def _rtdb_put(path: str, data) -> None:
    resp = requests.put(_rtdb_url(path), json=data, timeout=15)
    if resp.status_code >= 400:
        raise RuntimeError(f"RTDB PUT failed: {resp.status_code} {resp.text[:200]}")


def _rtdb_patch(path: str, data: dict) -> None:
    resp = requests.patch(_rtdb_url(path), json=data, timeout=15)
    if resp.status_code >= 400:
        raise RuntimeError(f"RTDB PATCH failed: {resp.status_code} {resp.text[:200]}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_chat_reply(message: str) -> str:
    """Lightweight, local chatbot reply without external API calls."""
    text = message.lower()
    if not text:
        return "I'm here to help with deepfake checks and uploads. Ask me anything."

    guidance_parts = []
    if any(k in text for k in ['upload', 'video', 'media', 'file']):
        guidance_parts.append("To check a file, go to the dashboard, click the upload area, pick your video, and wait for the result.")
    if any(k in text for k in ['result', 'verdict', 'fake', 'real', 'yes or no', 'deepfake']):
        guidance_parts.append("The detector returns Yes if it flags deepfake signals, No if it looks authentic, plus a short description and confidence.")
    if any(k in text for k in ['model', 'checkpoint', 'load']):
        guidance_parts.append("You can load a checkpoint in the dashboard Model tab before uploading for best accuracy.")
    if any(k in text for k in ['error', 'fail', 'issue', 'problem']):
        guidance_parts.append("If you hit an error, share the exact message and I can suggest a fix. Common fixes: use MP4/WebM, keep under 500MB, and ensure a model is loaded.")

    if guidance_parts:
        return " ".join(guidance_parts)

    return "I can guide uploads, model loading, and reading results. Ask about uploads, verdicts, or errors." 


def _normalize_chat_context(ctx: dict | None) -> dict:
    if not isinstance(ctx, dict):
        return {}
    allowed = {
        'original_filename',
        'prediction',
        'verdict_yes_no',
        'confidence',
        'prob_fake',
        'prob_real',
        'num_faces',
        'agent',
    }
    out = {k: ctx.get(k) for k in allowed if k in ctx}
    if isinstance(out.get('agent'), dict):
        a = out['agent']
        out['agent'] = {
            'alert_level': a.get('alert_level'),
            'explanation': a.get('explanation'),
        }
    else:
        out.pop('agent', None)
    return out


def _is_model_question(message: str) -> bool:
    text = (message or '').lower()
    if not text:
        return False
    keywords = [
        'what model',
        'which model',
        'model are you using',
        'what ai',
        'which ai',
        'what llm',
        'which llm',
        'what is the model',
        'what model do you use',
        'which model do you use',
    ]
    return any(k in text for k in keywords)


def _chat_model_label(api_key_source: str | None) -> str:
    # Keep this truthful: only claim Gemini when we actually call it.
    if api_key_source == 'user':
        return 'Gemini 1.5 Flash (user key)'
    if api_key_source == 'server':
        return 'Gemini 1.5 Flash (server key)'
    return 'Local assistant (no external API)'


def _detector_model_label() -> str:
    if MODEL is None:
        return 'Not loaded'
    ckpt = CHECKPOINT_PATH or 'Unknown checkpoint'
    mtype = MODEL_TYPE or 'Unknown model type'
    return f"{mtype} ({ckpt})"


def _detector_device_label() -> str:
    try:
        return str(DEVICE)
    except Exception:
        return 'Unknown'


def _model_info_reply(api_key_source: str | None) -> str:
    chat_model = _chat_model_label(api_key_source)
    det_model = _detector_model_label()
    det_device = _detector_device_label()
    return (
        "Chat model: " + chat_model + "\n"
        "Detection model: " + det_model + "\n"
        "Device: " + det_device + "\n\n"
        "Note: the chatbot and the deepfake detector are separate; enabling Gemini only affects chat replies."
    )


def generate_chat_reply_with_context(message: str, context: dict | None = None) -> str:
    """Local chat reply that can explain the current detection context."""
    msg = (message or '').strip()
    ctx = _normalize_chat_context(context)
    if not ctx:
        return generate_chat_reply(msg)

    # If user asks about numbers/meaning, provide a direct explanation.
    low = msg.lower()
    asks_meaning = any(k in low for k in ['what', 'mean', 'meaning', 'explain', 'score', 'faces', 'detected', 'confidence'])

    num_faces = ctx.get('num_faces')
    prob_fake = ctx.get('prob_fake')
    prob_real = ctx.get('prob_real')
    confidence = ctx.get('confidence')
    verdict_yes_no = ctx.get('verdict_yes_no')
    prediction = ctx.get('prediction')

    try:
        num_faces_i = int(num_faces) if num_faces is not None else None
    except Exception:
        num_faces_i = None
    try:
        prob_fake_f = float(prob_fake) if prob_fake is not None else None
    except Exception:
        prob_fake_f = None
    try:
        prob_real_f = float(prob_real) if prob_real is not None else None
    except Exception:
        prob_real_f = None
    try:
        confidence_f = float(confidence) if confidence is not None else None
    except Exception:
        confidence_f = None

    agent = ctx.get('agent') if isinstance(ctx.get('agent'), dict) else None
    agent_level = (agent or {}).get('alert_level')
    agent_expl = (agent or {}).get('explanation')

    if asks_meaning:
        parts = []
        if ctx.get('original_filename'):
            parts.append(f"File: {ctx.get('original_filename')}")
        if num_faces_i is not None:
            parts.append(
                f"Faces Detected: {num_faces_i} (the app found {num_faces_i} face crops/frames to analyze; more is usually better up to the app's limit)."
            )
        if prob_fake_f is not None:
            parts.append(
                f"Detection Score: {prob_fake_f * 100:.2f}% (this is the model's fake probability — closer to 100% means more likely deepfake)."
            )
        if prob_real_f is not None:
            parts.append(f"Prob Real: {prob_real_f * 100:.2f}%")
        if verdict_yes_no or prediction:
            parts.append(f"Verdict: {verdict_yes_no or ''} {('(' + prediction + ')') if prediction else ''}".strip())
        if confidence_f is not None:
            parts.append(
                f"Model Confidence: {confidence_f * 100:.2f}% (this is confidence in the chosen class — not the same as fake probability if the model predicts Real)."
            )
        if agent_level or agent_expl:
            parts.append(f"Agent: {agent_level or 'N/A'} — {agent_expl or ''}".strip())

        parts.append("If the score is around 50%, it's uncertain — try a clearer clip (good lighting, stable face) or a different checkpoint for a stronger result.")
        return "\n".join([p for p in parts if p])

    # Otherwise, fall back to generic app guidance.
    return generate_chat_reply(msg)


def _load_secrets_db():
    if not SECRETS_DB_PATH.exists():
        return {}
    try:
        return json.loads(SECRETS_DB_PATH.read_text())
    except Exception:
        return {}


def _save_secrets_db(data: dict):
    SECRETS_DB_PATH.write_text(json.dumps(data, indent=2))


def _firebase_request(endpoint: str, payload: dict) -> dict:
    if not FIREBASE_API_KEY:
        raise RuntimeError('FIREBASE_API_KEY is not set')
    # Basic validation to catch common misconfigurations early.
    if not FIREBASE_API_KEY.startswith('AIza'):
        raise RuntimeError('INVALID_FIREBASE_API_KEY (expected Firebase Web API key, usually starts with "AIza")')
    url = f"https://identitytoolkit.googleapis.com/v1/{endpoint}?key={FIREBASE_API_KEY}"
    resp = requests.post(url, json=payload, timeout=15)
    data = resp.json() if resp.content else {}
    if resp.status_code >= 400:
        # Normalize Firebase error shape
        err = (data.get('error') or {}).get('message') or 'FIREBASE_AUTH_ERROR'
        # Make the most common setup errors actionable.
        if 'API key not valid' in err or err in ('INVALID_API_KEY', 'API_KEY_INVALID'):
            raise RuntimeError('INVALID_FIREBASE_API_KEY (set FIREBASE_API_KEY to your Firebase project\'s Web API key)')
        if err == 'CONFIGURATION_NOT_FOUND':
            raise RuntimeError('FIREBASE_AUTH_CONFIGURATION_NOT_FOUND (check project, API key, and that Email/Password sign-in is enabled)')
        raise RuntimeError(err)
    return data


def _firebase_signup(email: str, password: str) -> dict:
    return _firebase_request('accounts:signUp', {
        'email': email,
        'password': password,
        'returnSecureToken': True
    })


def _firebase_login(email: str, password: str) -> dict:
    return _firebase_request('accounts:signInWithPassword', {
        'email': email,
        'password': password,
        'returnSecureToken': True
    })


def _firebase_store_user_profile(uid: str, email: str, username: str) -> None:
    # Store minimal profile in RTDB under /users/<uid>
    if not _firebase_rtdb_base():
        return
    _rtdb_patch(f"users/{uid}", {
        'email': email,
        'username': username,
        'created_at': datetime.now().isoformat()
    })


def _get_user_gemini_key() -> str | None:
    username = session.get('username')
    if not username:
        return None
    data = _load_secrets_db()
    user = data.get(username) or {}
    return user.get('gemini_api_key')


def _get_global_gemini_key() -> str | None:
    # Optional server-wide key for Gemini chat (kept on the server; never sent to clients).
    # Supports common env var names.
    return _env_str('GEMINI_API_KEY') or _env_str('GOOGLE_API_KEY')


def _get_user_notification_phone() -> str | None:
    username = session.get('username')
    if not username:
        return None
    data = _load_secrets_db()
    user = data.get(username) or {}
    return user.get('notification_phone')


def _validate_phone(phone: str) -> bool:
    # Accept E.164-like formats: + and digits, 8-15 digits total
    if not phone:
        return False
    if not re.fullmatch(r"\+?\d{8,15}", phone):
        return False
    return True


def _log_agent_notification(phone: str, message: str) -> str:
    log_dir = Path('logs') / 'agent_actions'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'notifications.jsonl'
    entry = {
        'timestamp': datetime.now().isoformat(),
        'phone': phone,
        'message': message
    }
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')
    return f"Notification logged for {phone}"


class WebActionAgent(ActionAgent):
    def __init__(self, get_phone_fn):
        super().__init__()
        self._get_phone_fn = get_phone_fn

    def _notify_admin(self, result):
        phone = None
        try:
            phone = self._get_phone_fn()
        except Exception:
            phone = None

        if phone and _validate_phone(phone):
            msg = f"CRITICAL deepfake alert for {result.video_id} ({result.confidence:.1%}): {result.explanation}"
            return _log_agent_notification(phone, msg)
        return super()._notify_admin(result)


# Minimal web agent pipeline (decision + monitoring + action)
WEB_DECISION_AGENT = DecisionAgent() if DecisionAgent else None
WEB_MONITORING_AGENT = MonitoringAgent() if MonitoringAgent else None
WEB_ACTION_AGENT = WebActionAgent(_get_user_notification_phone) if ActionAgent else None


def _run_web_agent_pipeline(res: dict, video_id: str) -> dict | None:
    """Run decision/monitoring/action agents for a prediction result.

    Returns a JSON-serializable dict or None if agents aren't available or res has error.
    """
    if not (WEB_DECISION_AGENT and WEB_MONITORING_AGENT and WEB_ACTION_AGENT):
        return None
    if not isinstance(res, dict) or 'error' in res:
        return None
    # If the core model abstained, do not escalate to deepfake alerts.
    # Treat as a warning-level event with no automated actions.
    if bool(res.get('abstained')):
        return {
            'alert_level': 'WARNING',
            'explanation': 'Model abstained due to low confidence/borderline score. Manual review recommended.',
            'actions_taken': [],
            'monitoring': None,
        }
    try:
        prob_real = float(res.get('prob_real', 0.0))
        prob_fake = float(res.get('prob_fake', 0.0))
        probs = torch.tensor([prob_real, prob_fake])
        # We don't have per-frame scores in the web pipeline, so use a small placeholder.
        frame_scores = torch.zeros(8)
        prediction_data = {
            'video_id': video_id,
            'logits': torch.log(probs + 1e-6),
            'frame_scores': frame_scores,
            'probs': probs,
            # IMPORTANT: keep agent decisions aligned with the app's thresholded verdict.
            'pred_class': res.get('pred_class'),
            'confidence': res.get('confidence'),
            'threshold': res.get('threshold'),
        }
        decision = WEB_DECISION_AGENT.process(prediction_data)
        metrics = WEB_MONITORING_AGENT.process(decision)
        actions = WEB_ACTION_AGENT.process(decision)
        return {
            'alert_level': decision.alert_level.name,
            'explanation': decision.explanation,
            'actions_taken': actions.get('actions_taken', []),
            'monitoring': {
                'total_processed': metrics.get('total_processed'),
                'alerts_by_level': metrics.get('alerts_by_level'),
            },
        }
    except Exception as e:
        return {'error': f'Agent pipeline failed: {e}'}


def generate_chat_reply_gemini(message: str, api_key: str) -> str:
    # Basic guardrail: block disallowed content
    harmful_terms = ['hate', 'racist', 'sexist', 'violence', 'kill', 'nsfw']
    if any(t in message.lower() for t in harmful_terms):
        return "Sorry, I can't assist with that."
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            "You are a helpful assistant for a deepfake detection web app. "
            "Be direct and explain the app's metrics when asked. "
            "If the user provides a JSON 'current detection context', use it. "
            "Explain: 'Faces Detected' = how many face crops/frames were analyzed, "
            "'Detection Score' (prob_fake) = fake probability, and 'Confidence' = confidence in the chosen class. "
            "Give actionable next steps when results are uncertain (~40-60%)."
        )
        resp = model.generate_content([prompt, message])
        text = getattr(resp, 'text', None) or (resp.candidates[0].content.parts[0].text if getattr(resp, 'candidates', None) else '')
        return text.strip() or "I'm here to help with uploads and results."
    except Exception:
        # Fallback to local guidance
        return generate_chat_reply(message)


def _gemini_generate_english_report(results: list[dict], api_key: str, user_notes: str | None = None) -> str:
    """Generate a plain-English report from one or more prediction results.

    IMPORTANT: api_key is used for this request only and is not persisted.
    """
    # Basic guardrail: block disallowed content
    harmful_terms = ['hate', 'racist', 'sexist', 'violence', 'kill', 'nsfw']
    if user_notes and any(t in user_notes.lower() for t in harmful_terms):
        return "Sorry, I can't assist with that."

    # Keep payload compact and predictable.
    cleaned = []
    for item in (results or []):
        if not isinstance(item, dict):
            continue
        r = item.get('result') if isinstance(item.get('result'), dict) else (item if 'prediction' in item else {})
        cleaned.append({
            'original_filename': item.get('original_filename') or item.get('filename') or r.get('filename'),
            'prediction': r.get('prediction') or r.get('verdict_yes_no'),
            'verdict_yes_no': r.get('verdict_yes_no'),
            'confidence': r.get('confidence'),
            'prob_fake': r.get('prob_fake'),
            'prob_real': r.get('prob_real'),
            'num_faces': r.get('num_faces'),
            'description': r.get('description') or r.get('simple_message'),
            'agent': r.get('agent') if isinstance(r.get('agent'), dict) else None,
        })

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        system = (
            "You write clear, non-technical English reports for a deepfake detection app. "
            "Use the provided JSON results. Do not invent extra measurements. "
            "If data is missing, say it's unavailable. "
            "Keep it concise and actionable."
        )

        instruction = (
            "Write an English report for the following video analysis results.\n\n"
            "Format:\n"
            "1) Overall Summary (2-4 sentences)\n"
            "2) Per-Video Findings (one short paragraph per file)\n"
            "3) Confidence & Limitations (bullets)\n"
            "4) Recommended Next Steps (bullets)\n\n"
            "Explain fields briefly:\n"
            "- prob_fake is the model's fake probability\n"
            "- confidence is confidence in the chosen class\n"
            "- num_faces is how many face crops/frames were analyzed\n"
        )

        payload = {
            'results': cleaned,
            'user_notes': (user_notes or '').strip() or None,
        }

        resp = model.generate_content([
            system,
            instruction,
            "JSON input:",
            json.dumps(payload, indent=2, default=str),
        ])
        text = getattr(resp, 'text', None) or (
            resp.candidates[0].content.parts[0].text if getattr(resp, 'candidates', None) else ''
        )
        return (text or '').strip()
    except Exception:
        return "Report generation failed (Gemini not configured or unavailable)."

def get_training_metrics():
    pred_files = sorted(Path('.').glob('preds_epoch_*.csv'), key=lambda x: int(x.stem.split('_')[-1]))
    if not pred_files:
        return {}
    
    metrics = {}
    for csv_file in pred_files:
        epoch = int(csv_file.stem.split('_')[-1])
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if rows:
            labels = [int(r['label']) for r in rows if r['label'] in ('0', '1')]
            preds = [int(r['pred']) for r in rows if r['pred'] in ('0', '1')]
            probs = [float(r['prob']) for r in rows if r['prob']]
            
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
            
            if len(labels) > 0:
                acc = accuracy_score(labels, preds)
                prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
                cm = confusion_matrix(labels, preds).tolist()
                try:
                    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else None
                except Exception:
                    auc = None
                
                metrics[epoch] = {
                    'accuracy': float(acc),
                    'precision': float(prec),
                    'recall': float(rec),
                    'f1': float(f1),
                    'auc': float(auc) if auc else None,
                    'confusion_matrix': cm,
                    'total_samples': len(labels)
                }
    
    return metrics

def load_model(checkpoint_path, model_type='vit_gcn'):
    global MODEL, MODEL_TYPE, CHECKPOINT_PATH, MODEL_META, LAST_LOAD_STATS
    try:
        # Load checkpoint once up-front so we can infer architecture when needed.
        ckpt = None
        state_dict = None
        fake_idx_detected = None
        try:
            ckpt = torch.load(checkpoint_path, map_location=DEVICE)
            if isinstance(ckpt, dict):
                state_dict = ckpt.get('model_state') or ckpt.get('state_dict') or ckpt
            else:
                state_dict = ckpt

            # Best-effort: detect whether class index 0/1 is "fake".
            def _detect_fake_idx_from_ckpt(obj) -> int | None:
                try:
                    if not isinstance(obj, dict):
                        return None

                    def _norm(s: str) -> str:
                        return str(s).strip().lower().replace('-', '_').replace(' ', '_')

                    def _find_in_class_to_idx(d: dict) -> int | None:
                        for k, v in d.items():
                            kk = _norm(k)
                            if 'fake' in kk or 'deepfake' in kk:
                                try:
                                    vi = int(v)
                                except Exception:
                                    continue
                                return vi
                        return None

                    # Common patterns
                    for key in ('class_to_idx', 'class2idx', 'label_to_idx', 'label2idx'):
                        m = obj.get(key)
                        if isinstance(m, dict):
                            out = _find_in_class_to_idx(m)
                            if out is not None:
                                return out

                    for key in ('idx_to_class', 'idx2class', 'idx_to_label', 'idx2label'):
                        m = obj.get(key)
                        if isinstance(m, (dict, list, tuple)):
                            if isinstance(m, dict):
                                items = list(m.items())
                                for k, v in items:
                                    try:
                                        idx = int(k)
                                    except Exception:
                                        continue
                                    vv = _norm(v)
                                    if 'fake' in vv or 'deepfake' in vv:
                                        return idx
                            else:
                                for idx, v in enumerate(list(m)):
                                    vv = _norm(v)
                                    if 'fake' in vv or 'deepfake' in vv:
                                        return idx

                    for key in ('classes', 'class_names', 'labels', 'label_names'):
                        m = obj.get(key)
                        if isinstance(m, (list, tuple)):
                            for idx, v in enumerate(list(m)):
                                vv = _norm(v)
                                if 'fake' in vv or 'deepfake' in vv:
                                    return idx

                    # Some checkpoints stash metadata under 'meta' or 'metadata'
                    for key in ('meta', 'metadata'):
                        sub = obj.get(key)
                        if isinstance(sub, dict):
                            out = _detect_fake_idx_from_ckpt(sub)
                            if out is not None:
                                return out
                except Exception:
                    return None
                return None

            fake_idx_detected = _detect_fake_idx_from_ckpt(ckpt)
        except Exception as e:
            ckpt = None
            state_dict = None
            print(f"⚠️ Warning: could not read checkpoint: {e}")

        def _normalize_state_dict_keys(sd: dict) -> dict:
            """Normalize common checkpoint key prefixes (e.g., DataParallel 'module.')."""
            if not isinstance(sd, dict) or not sd:
                return sd
            prefixes = ('module.', 'model.', 'net.')
            out = {}
            for k, v in sd.items():
                if not isinstance(k, str):
                    out[k] = v
                    continue
                nk = k
                changed = True
                while changed:
                    changed = False
                    for pfx in prefixes:
                        if nk.startswith(pfx):
                            nk = nk[len(pfx):]
                            changed = True
                out[nk] = v
            return out

        def _infer_ensemble_count(sd: dict) -> int | None:
            try:
                idxs = set()
                for k in sd.keys():
                    if not isinstance(k, str) or not k.startswith('models.'):
                        continue
                    parts = k.split('.')
                    if len(parts) >= 2 and parts[1].isdigit():
                        idxs.add(int(parts[1]))
                if not idxs:
                    return None
                return max(idxs) + 1
            except Exception:
                return None

        def _compat_score(m, sd: dict) -> tuple[int, int, int, int]:
            """Score how compatible a model is with a checkpoint without loading tensors.

            Returns a tuple where higher is better:
              (matched_keys, -mismatched_shapes, -missing_keys, -unexpected_keys)
            """
            try:
                msd = m.state_dict()
            except Exception:
                return (0, -10**9, -10**9, -10**9)

            matched = 0
            mismatched = 0
            for k, v in sd.items():
                if k in msd:
                    try:
                        if tuple(msd[k].shape) == tuple(v.shape):
                            matched += 1
                        else:
                            mismatched += 1
                    except Exception:
                        mismatched += 1

            missing = len([k for k in msd.keys() if k not in sd])
            unexpected = len([k for k in sd.keys() if k not in msd])
            return (matched, -mismatched, -missing, -unexpected)

        def _safe_load_state_dict(m, sd: dict):
            """Load only keys with matching shapes to avoid size-mismatch RuntimeErrors."""
            msd = m.state_dict()
            filtered = {}
            for k, v in sd.items():
                if k not in msd:
                    continue
                try:
                    if tuple(msd[k].shape) == tuple(v.shape):
                        filtered[k] = v
                except Exception:
                    continue
            return m.load_state_dict(filtered, strict=False)

        def _load_stats(m, sd: dict) -> dict:
            try:
                msd = m.state_dict()
            except Exception:
                return {
                    'matched': 0,
                    'mismatched': 0,
                    'missing': None,
                    'unexpected': None,
                    'model_keys': None,
                    'ckpt_keys': len(sd) if isinstance(sd, dict) else None,
                    'match_ratio': None,
                }

            matched = 0
            mismatched = 0
            for k, v in sd.items():
                if k in msd:
                    try:
                        if tuple(msd[k].shape) == tuple(v.shape):
                            matched += 1
                        else:
                            mismatched += 1
                    except Exception:
                        mismatched += 1

            missing = len([k for k in msd.keys() if k not in sd])
            unexpected = len([k for k in sd.keys() if k not in msd])
            model_keys = len(msd)
            match_ratio = (matched / float(model_keys)) if model_keys else None
            return {
                'matched': int(matched),
                'mismatched': int(mismatched),
                'missing': int(missing),
                'unexpected': int(unexpected),
                'model_keys': int(model_keys),
                'ckpt_keys': int(len(sd)),
                'match_ratio': float(match_ratio) if match_ratio is not None else None,
            }

        def _infer_backbone_from_checkpoint_name(p: str) -> str | None:
            try:
                name = str(Path(p).name).lower()
            except Exception:
                return None
            known = [
                'efficientnet_b0',
                'resnet18',
                'resnet34',
                'resnet50',
                'vit_base_patch16_224',
            ]
            for k in known:
                if k in name:
                    return k
            # Light fallback for common patterns
            if 'efficientnet' in name:
                return 'efficientnet_b0'
            if 'resnet' in name:
                return 'resnet50'
            if 'vit' in name:
                return 'vit_base_patch16_224'
            return None

        def _infer_single_backbone(sd: dict) -> str | None:
            try:
                keys = [k for k in sd.keys() if isinstance(k, str)]
            except Exception:
                return None

            # ViT/timm signatures
            if any('backbone.patch_embed' in k or 'backbone.blocks.' in k for k in keys):
                # Default timm ViT used in this repo
                return 'vit_base_patch16_224'
            # EfficientNet-like signatures (support Sequential-wrapped variants used by this repo)
            if any('conv_stem' in k or '.blocks.' in k or 'conv_dw' in k or 'se.conv' in k for k in keys) and any(k.startswith('backbone') for k in keys):
                # Prefer the common baseline
                return 'efficientnet_b0'
            # torchvision resnet signatures
            if any('.layer1.' in k or '.layer2.' in k or '.layer3.' in k or '.layer4.' in k for k in keys):
                return 'resnet50'
            # torchvision resnet wrapped as nn.Sequential will have numeric child modules
            if any(k.startswith('backbone.0.') or k.startswith('backbone.1.') for k in keys):
                return 'resnet50'
            return None

        # Normalize keys early to maximize compatibility checks/loads.
        if isinstance(state_dict, dict) and state_dict:
            state_dict = _normalize_state_dict_keys(state_dict)

        chosen_backbone: str | None = None
        chosen_backbones: list[str] | None = None

        if model_type == 'cnn_lstm':
            model = CNNLSTMHybrid(input_channels=3, hidden_size=256, num_layers=2, num_classes=2, dropout=0.3)
        elif model_type in ('pretrained', 'ensemble_pretrained'):
            if PretrainedBackboneDetector is None:
                raise RuntimeError('PretrainedBackboneDetector is not available (missing dependencies?)')

            # Caller may provide metadata via MODEL_META populated by /api/load-model.
            meta = MODEL_META or {}
            backbone = str(meta.get('backbone') or _infer_backbone_from_checkpoint_name(str(checkpoint_path)) or 'efficientnet_b0')
            backbones = meta.get('backbones')
            if model_type == 'ensemble_pretrained':
                if EnsembleDetector is None:
                    raise RuntimeError('EnsembleDetector is not available')
                if not isinstance(backbones, (list, tuple)) or not backbones:
                    backbones = [x.strip() for x in str(os.environ.get('ENSEMBLE_BACKBONES', 'efficientnet_b0,resnet50')).split(',') if x.strip()]

                # Try to auto-detect the best matching ensemble architecture from the checkpoint.
                # This avoids loading a checkpoint into the wrong backbone combination (which produces garbage outputs).
                chosen_backbones = list(backbones)
                if isinstance(state_dict, dict) and state_dict:
                    n_models = _infer_ensemble_count(state_dict) or len(chosen_backbones)
                    pool = ['efficientnet_b0', 'resnet50', 'resnet34', 'resnet18', 'vit_base_patch16_224']
                    presets_1 = [[x] for x in pool]
                    presets_2 = [
                        ['efficientnet_b0', 'resnet50'],
                        ['efficientnet_b0', 'resnet34'],
                        ['efficientnet_b0', 'resnet18'],
                        ['resnet34', 'resnet50'],
                        ['resnet18', 'resnet50'],
                        ['efficientnet_b0', 'vit_base_patch16_224'],
                    ]
                    presets_3 = [
                        ['efficientnet_b0', 'resnet50', 'resnet34'],
                        ['efficientnet_b0', 'resnet50', 'resnet18'],
                        ['efficientnet_b0', 'resnet34', 'resnet18'],
                        ['efficientnet_b0', 'resnet50', 'vit_base_patch16_224'],
                    ]

                    candidates = []
                    if int(n_models) == 1:
                        candidates = presets_1
                    elif int(n_models) == 2:
                        candidates = presets_2
                    elif int(n_models) == 3:
                        candidates = presets_3
                    else:
                        # As a fallback, try a truncated/padded version of provided backbones.
                        bb = list(chosen_backbones)
                        if len(bb) >= int(n_models):
                            candidates = [bb[: int(n_models)]]
                        else:
                            # pad with common pool
                            for x in pool:
                                if len(bb) >= int(n_models):
                                    break
                                if x not in bb:
                                    bb.append(x)
                            candidates = [bb]

                    # Always test the requested backbones first.
                    if chosen_backbones not in candidates and len(chosen_backbones) == int(n_models):
                        candidates.insert(0, chosen_backbones)

                    best_choice = None
                    best_score = None
                    for cand in candidates:
                        if len(cand) != int(n_models):
                            continue
                        try:
                            m = EnsembleDetector(
                                backbone_names=list(cand),
                                pretrained=False,
                                num_classes=2,
                                dropout_rate=0.5,
                                ensemble_method='weighted',
                            )
                            cur = _compat_score(m, state_dict)
                            if best_score is None or cur > best_score:
                                best_score = cur
                                best_choice = (cand, cur)
                        except Exception:
                            continue

                    if best_choice is not None:
                        chosen_backbones = list(best_choice[0])
                        print(
                            f"✅ Auto-selected ensemble backbones: {chosen_backbones} (compat={best_choice[1]})"
                        )

                model = EnsembleDetector(
                    backbone_names=list(chosen_backbones),
                    pretrained=False,
                    num_classes=2,
                    dropout_rate=0.5,
                    ensemble_method='weighted',
                )
                chosen_backbones = list(chosen_backbones)
            else:
                # Auto-detect backbone for single pretrained detector if possible.
                if isinstance(state_dict, dict) and state_dict:
                    # First, prefer an explicit hint from the checkpoint filename when present.
                    hinted = _infer_backbone_from_checkpoint_name(str(checkpoint_path))
                    if hinted:
                        backbone = hinted
                    inferred = _infer_single_backbone(state_dict)
                    if inferred:
                        backbone = inferred
                        print(f"✅ Auto-detected pretrained backbone: {backbone}")
                model = PretrainedBackboneDetector(
                    backbone_name=backbone,
                    pretrained=False,
                    num_classes=2,
                    dropout_rate=0.5,
                    use_temporal_attention=True,
                )
                chosen_backbone = str(backbone)
        else:
            # If checkpoint contains model_config, use it to reconstruct the right backbone
            model_cfg = None
            try:
                ckpt_peek = torch.load(checkpoint_path, map_location=DEVICE)
                if isinstance(ckpt_peek, dict):
                    model_cfg = ckpt_peek.get('model_config')
            except Exception:
                model_cfg = None

            if isinstance(model_cfg, dict) and model_cfg:
                model = DeepfakeModel(**model_cfg)
            else:
                model = DeepfakeModel()
        
        if isinstance(state_dict, dict) and state_dict:
            try:
                # Capture compatibility stats before loading.
                stats = _load_stats(model, state_dict)
                inc = _safe_load_state_dict(model, state_dict)
                missing = len(getattr(inc, 'missing_keys', []) or [])
                unexpected = len(getattr(inc, 'unexpected_keys', []) or [])
                stats['missing'] = int(missing)
                stats['unexpected'] = int(unexpected)
                LAST_LOAD_STATS = {
                    **stats,
                    'checkpoint': str(checkpoint_path),
                    'model_type': str(model_type),
                    'backbone': chosen_backbone,
                    'backbones': chosen_backbones,
                    'fake_class_index_detected': (int(fake_idx_detected) if fake_idx_detected is not None else None),
                }
                print(f"✅ Checkpoint loaded (safe) (missing={missing}, unexpected={unexpected})")

                # Fail fast if checkpoint is clearly incompatible.
                # A bad load usually leads to "everything real" or "everything fake".
                if model_type in ('pretrained', 'ensemble_pretrained'):
                    mr = stats.get('match_ratio')
                    if mr is not None and float(mr) < 0.80:
                        raise RuntimeError(f"Incompatible checkpoint (match_ratio={mr:.3f}).")
            except Exception as e:
                print(f"⚠️ Warning loading checkpoint weights (even after safe filter): {e}")
                print("Using model with random initialization for inference")
                LAST_LOAD_STATS = {
                    'checkpoint': str(checkpoint_path),
                    'model_type': str(model_type),
                    'error': str(e),
                }
                # For pretrained models, a random-init model is worse than returning an error.
                if model_type in ('pretrained', 'ensemble_pretrained'):
                    return False
        else:
            print("⚠️ No state_dict found in checkpoint; using random initialization")
            LAST_LOAD_STATS = {
                'checkpoint': str(checkpoint_path),
                'model_type': str(model_type),
                'error': 'No state_dict found',
            }
            if model_type in ('pretrained', 'ensemble_pretrained'):
                return False
        
        model.to(DEVICE)
        model.eval()
        
        MODEL = model
        MODEL_TYPE = model_type
        CHECKPOINT_PATH = checkpoint_path
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def imagenet_normalize(frames: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to float frames in [0,1]."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device, dtype=frames.dtype)
    std = torch.tensor([0.229, 0.224, 0.225], device=frames.device, dtype=frames.dtype)
    if frames.dim() == 4:  # (T, C, H, W)
        return (frames - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    if frames.dim() == 5:  # (B, T, C, H, W)
        return (frames - mean.view(1, 1, 3, 1, 1)) / std.view(1, 1, 3, 1, 1)
    raise ValueError(f"Unsupported frames shape for normalization: {tuple(frames.shape)}")


def _load_calibration_threshold(checkpoint_path: str) -> float | None:
    """Try to load calibration_best.json next to a checkpoint (ensemble trainer output)."""
    try:
        p = Path(checkpoint_path)
        cand = p.parent / 'calibration_best.json'
        if cand.exists():
            payload = json.loads(cand.read_text(encoding='utf-8'))
            thr = payload.get('best_thr_accuracy')
            if thr is not None:
                thr_f = float(thr)
                # Basic sanity bound
                if 0.0 <= thr_f <= 1.0:
                    return thr_f
                return None
    except Exception:
        return None
    return None


def _env_float(name: str) -> float | None:
    """Best-effort float env var parsing."""
    try:
        v = os.environ.get(name)
    except Exception:
        v = None
    if v is None:
        return None
    v = str(v).strip()
    if not v:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _env_int(name: str) -> int | None:
    """Best-effort int env var parsing."""
    try:
        v = os.environ.get(name)
    except Exception:
        v = None
    if v is None:
        return None
    v = str(v).strip()
    if not v:
        return None
    return _safe_int(v)


def _get_fake_class_index(num_classes: int = 2) -> int:
    """Return the model-logit index that corresponds to the 'fake/deepfake' class.

    Semantics in the app are always: pred_class=1 means fake, pred_class=0 means real.
    This setting only maps which *logit/probability index* is treated as fake.
    """
    try:
        nc = int(num_classes)
    except Exception:
        nc = 2
    nc = max(1, nc)

    # 1) explicit override
    idx = _env_int('FAKE_CLASS_INDEX')
    if idx is not None:
        try:
            idx = int(idx)
        except Exception:
            idx = None
        if idx is not None and 0 <= idx < nc:
            return int(idx)

    # 2) detected from checkpoint metadata (best-effort)
    try:
        det = LAST_LOAD_STATS.get('fake_class_index_detected')
    except Exception:
        det = None
    if det is not None:
        try:
            det_i = int(det)
        except Exception:
            det_i = None
        if det_i is not None and 0 <= det_i < nc:
            return int(det_i)

    # 3) default
    return 1 if nc > 1 else 0


def _get_fake_class_index_source(num_classes: int = 2) -> str:
    """Explain where fake class index came from (env/detected/default)."""
    try:
        nc = int(num_classes)
    except Exception:
        nc = 2
    nc = max(1, nc)
    env_idx = _env_int('FAKE_CLASS_INDEX')
    if env_idx is not None:
        try:
            env_i = int(env_idx)
        except Exception:
            env_i = None
        if env_i is not None and 0 <= env_i < nc:
            return 'env'
    try:
        det = LAST_LOAD_STATS.get('fake_class_index_detected')
    except Exception:
        det = None
    if det is not None:
        try:
            det_i = int(det)
        except Exception:
            det_i = None
        if det_i is not None and 0 <= det_i < nc:
            return 'detected'
    return 'default'


def _get_detection_threshold_fallback(default: float = 0.5) -> float:
    """Return an override threshold if provided, otherwise a safe default."""
    thr_override = _env_float('DETECT_FAKE_THRESHOLD')
    if thr_override is not None and 0.0 <= float(thr_override) <= 1.0:
        return float(thr_override)
    return float(default)

def extract_faces_from_video(video_path, max_frames=16):
    """Extract face crops from a video.

    For pretrained/ensemble models in this repo, the time/sequence dimension is treated
    as a list of face crops sampled across frames.

    Uses MTCNN (facenet-pytorch) when available; falls back to Haar cascade.
    """

    try:
        sample_rate = int(os.environ.get('VIDEO_SAMPLE_RATE', '5'))
    except Exception:
        sample_rate = 5
    sample_rate = max(1, int(sample_rate))

    try:
        face_size = int(os.environ.get('FACE_SIZE', '224'))
    except Exception:
        face_size = 224
    face_size = 224 if int(face_size) <= 0 else int(face_size)

    detector = (os.environ.get('FACE_DETECTOR') or 'mtcnn').strip().lower()
    keep_all = str(os.environ.get('KEEP_ALL_FACES', '')).strip().lower() in ('1', 'true', 'yes', 'y')

    # Sample frames first (works even without OpenCV via imageio fallback).
    try:
        frames = sample_video_frames(str(video_path), sample_rate=sample_rate, max_frames=max(8, int(max_frames)))
    except Exception as e:
        print(f"Error sampling frames: {e}")
        frames = []

    if not frames:
        return np.array([])

    faces: list[np.ndarray] = []

    # Preferred: MTCNN
    mtcnn = _get_mtcnn() if detector in ('mtcnn', 'auto') else None
    if mtcnn is not None:
        for fr in frames:
            try:
                pil = Image.fromarray(fr).convert('RGB')
                boxes, _ = mtcnn.detect(pil)
            except Exception:
                boxes = None

            if boxes is None or len(boxes) == 0:
                continue

            boxes_list = list(boxes)
            if not keep_all:
                def area(b):
                    x1, y1, x2, y2 = [float(v) for v in b]
                    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
                boxes_list = [max(boxes_list, key=area)]

            w, h = pil.size
            for b in boxes_list:
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
                    crop = pil.crop((x1, y1, x2, y2)).resize((face_size, face_size))
                    faces.append(np.array(crop))
                except Exception:
                    continue
                if len(faces) >= int(max_frames):
                    break
            if len(faces) >= int(max_frames):
                break

        return np.array(faces) if faces else np.array([])

    # Fallback: Haar cascade (OpenCV)
    if cv2 is None:
        return np.array([])

    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception:
        face_cascade = None
    if face_cascade is None:
        return np.array([])

    for fr in frames:
        try:
            # fr is RGB already
            gray = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
            rects = face_cascade.detectMultiScale(gray, 1.1, 4)
        except Exception:
            rects = []
        if rects is None or len(rects) == 0:
            continue

        rects_list = list(rects)
        if not keep_all:
            rects_list = [max(rects_list, key=lambda r: float(r[2]) * float(r[3]))]

        for (x, y, w, h) in rects_list:
            try:
                crop = fr[y:y + h, x:x + w]
                crop = cv2.resize(crop, (face_size, face_size))
                faces.append(crop)
            except Exception:
                continue
            if len(faces) >= int(max_frames):
                break
        if len(faces) >= int(max_frames):
            break

    return np.array(faces) if faces else np.array([])

def predict_video(video_path):
    try:
        if MODEL is None:
            return {'error': 'Model not loaded. Please load a checkpoint first.'}

        # If confidence is below this value, return an "UNSURE" verdict.
        # This reduces overconfident wrong labels on random/out-of-domain videos.
        try:
            abstain_conf = float(os.environ.get('DETECT_ABSTAIN_CONF', '0.60'))
        except Exception:
            abstain_conf = 0.60

        # Optional abstain when probability is too close to the decision threshold.
        # Useful to reduce false positives on borderline real videos.
        try:
            abstain_margin = float(os.environ.get('DETECT_ABSTAIN_MARGIN', '0.0'))
        except Exception:
            abstain_margin = 0.0
        abstain_margin = max(0.0, min(0.5, float(abstain_margin)))

        # Pretrained/ensemble-pretrained inference path
        if MODEL_TYPE in ('pretrained', 'ensemble_pretrained'):
            try:
                max_frames = int(os.environ.get('MAX_FRAMES', '8'))
            except Exception:
                max_frames = 8
            max_frames = max(1, min(64, int(max_frames)))

            faces = extract_faces_from_video(video_path, max_frames=max_frames)
            num_faces = int(len(faces))
            if len(faces) == 0:
                return {'error': 'No faces detected in video'}

            # If we only found 1 face crop/frame, results are often unstable.
            # Prefer "Uncertain" over a hard label.
            try:
                min_faces = int(os.environ.get('MIN_FACES', '2'))
            except Exception:
                min_faces = 2
            min_faces = max(1, int(min_faces))
            if num_faces < min_faces:
                return {
                    'prediction': 'Uncertain',
                    'verdict_yes_no': 'Unsure',
                    'description': (
                        f"Not enough faces/frames detected for a stable decision (num_faces={num_faces}, min_faces={min_faces}). "
                        "Try a clearer face shot, better lighting, or a longer clip."
                    ),
                    'pred_class': None,
                    'confidence': None,
                    'prob_real': None,
                    'prob_fake': None,
                    'num_faces': int(num_faces),
                    'abstained': True,
                }

            # Convert to (1, T, C, H, W) float in [0,1] with ImageNet normalization.
            faces_tensor = torch.from_numpy(faces).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
            faces_tensor = imagenet_normalize(faces_tensor)
            faces_tensor = faces_tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits, frame_scores = MODEL(faces_tensor)
                probs = torch.softmax(logits, dim=1)
            fake_idx = _get_fake_class_index(int(probs.shape[1]))
            real_idx = (1 - int(fake_idx)) if int(probs.shape[1]) == 2 else 0
            prob_fake = float(probs[0, int(fake_idx)].item())
            prob_real = float(probs[0, int(real_idx)].item())

            thr = _load_calibration_threshold(CHECKPOINT_PATH)
            if thr is None:
                thr = _get_detection_threshold_fallback(0.5)
            else:
                thr = float(thr)

            # Allow explicit override regardless of calibration.
            thr = float(_get_detection_threshold_fallback(thr))

            # Guardrail: some calibration sweeps produce extreme thresholds (near 0/1).
            # By default we ignore those for better behavior on random/out-of-domain videos.
            allow_extreme_thr = str(os.environ.get('ALLOW_EXTREME_CALIBRATION_THRESHOLD', '')).strip().lower() in ('1', 'true', 'yes', 'y')
            if not allow_extreme_thr and (float(thr) < 0.05 or float(thr) > 0.95):
                thr = 0.5
            is_fake = bool(prob_fake >= float(thr))
            pred_class = 1 if is_fake else 0
            confidence = float(prob_fake if is_fake else prob_real)

            description = (
                f"Pretrained detector (thr={thr:.2f})" if MODEL_TYPE == 'pretrained' else f"Ensemble pretrained detector (thr={thr:.2f})"
            )

            # Optional enhanced decision agent (ensemble only)
            agent_payload = None
            disable_enhanced = str(os.environ.get('DISABLE_ENHANCED_AGENT', '')).strip().lower() in ('1', 'true', 'yes', 'y')
            if not disable_enhanced and ENHANCED_AGENT is not None and hasattr(MODEL, 'models'):
                try:
                    # Keep agent decision aligned with the same threshold we use for prob_fake.
                    old_decision_threshold = getattr(ENHANCED_AGENT, 'decision_threshold', 0.5)
                    try:
                        ENHANCED_AGENT.decision_threshold = float(thr)
                    except Exception:
                        pass

                    individual_logits = []
                    with torch.no_grad():
                        for m in MODEL.models:
                            l, _ = m(faces_tensor)
                            individual_logits.append(l.squeeze(0))

                    # Keep agent class-index mapping aligned with app mapping.
                    try:
                        setattr(ENHANCED_AGENT, 'fake_class_index', int(fake_idx))
                    except Exception:
                        pass

                    # Simple disagreement-based uncertainty
                    ind_probs = [float(torch.softmax(l, dim=0)[int(fake_idx)].item()) for l in individual_logits]
                    uncertainty = float(np.std(ind_probs)) if len(ind_probs) >= 2 else 0.0

                    pred = ENHANCED_AGENT.process_ensemble_output(
                        ensemble_logits=logits,
                        individual_logits=individual_logits,
                        frame_scores=frame_scores.squeeze(0) if hasattr(frame_scores, 'dim') and frame_scores.dim() > 1 else frame_scores,
                        video_id=str(Path(video_path).name),
                        uncertainty=uncertainty,
                    )
                    agent_payload = {
                        'is_fake': bool(pred.is_fake),
                        'ensemble_prob': float(pred.ensemble_prob),
                        'confidence': float(pred.confidence),
                        'alert_level': getattr(pred.alert_level, 'name', str(pred.alert_level)),
                        'uncertainty': float(pred.uncertainty),
                        'explanation': str(pred.explanation),
                    }
                    # Prefer agent explanation for UI
                    description = agent_payload.get('explanation') or description
                    pred_class = int(agent_payload.get('is_fake'))
                    confidence = float(agent_payload.get('confidence', confidence))
                except Exception:
                    agent_payload = None
                finally:
                    try:
                        ENHANCED_AGENT.decision_threshold = old_decision_threshold
                    except Exception:
                        pass

            # Optional abstain on borderline probability
            if abstain_margin > 0.0 and abs(float(prob_fake) - float(thr)) <= float(abstain_margin):
                return {
                    'prediction': 'Uncertain',
                    'verdict_yes_no': 'Unsure',
                    'description': (
                        f"Borderline score (prob_fake={prob_fake * 100:.1f}%, thr={thr:.2f} ± {abstain_margin:.2f}). "
                        "Manual review recommended.\n\n" + (description or '')
                    ),
                    'pred_class': None,
                    'confidence': float(confidence),
                    'prob_real': float(prob_real),
                    'prob_fake': float(prob_fake),
                    'num_faces': int(num_faces),
                    'threshold': float(thr),
                    'enhanced_agent': agent_payload,
                    'abstained': True,
                }

            # Abstain on low confidence
            if confidence < float(abstain_conf):
                return {
                    'prediction': 'Uncertain',
                    'verdict_yes_no': 'Unsure',
                    'description': (
                        f"Low confidence ({confidence * 100:.1f}%). This video may be out-of-domain "
                        "(different compression, face quality, lighting, or manipulation type). Manual review recommended.\n\n" +
                        (description or '')
                    ),
                    'pred_class': None,
                    'confidence': float(confidence),
                    'prob_real': float(prob_real),
                    'prob_fake': float(prob_fake),
                    'num_faces': int(num_faces),
                    'threshold': float(thr),
                    'enhanced_agent': agent_payload,
                    'abstained': True,
                }

            return {
                'prediction': 'Deepfake' if pred_class == 1 else 'Real',
                'verdict_yes_no': 'Yes' if pred_class == 1 else 'No',
                'description': description,
                'pred_class': int(pred_class),
                'confidence': float(confidence),
                'prob_real': float(prob_real),
                'prob_fake': float(prob_fake),
                'num_faces': int(num_faces),
                'threshold': float(thr),
                'enhanced_agent': agent_payload,
            }
        
        faces = extract_faces_from_video(video_path)
        num_faces = int(len(faces))
        if len(faces) == 0:
            return {'error': 'No faces detected in video'}
        
        max_nodes = 16
        if len(faces) < max_nodes:
            pad = max_nodes - len(faces)
            pads = np.repeat(faces[-1][None], pad, axis=0)
            faces = np.concatenate([faces, pads], axis=0)
        elif len(faces) > max_nodes:
            idxs = np.linspace(0, len(faces) - 1, max_nodes).astype(int)
            faces = faces[idxs]
        
        faces_tensor = torch.from_numpy(faces).permute(0, 3, 1, 2).float() / 255.0
        faces_tensor = faces_tensor.unsqueeze(0).to(DEVICE)
        
        if MODEL_TYPE == 'cnn_lstm':
            with torch.no_grad():
                output = MODEL(faces_tensor)
        else:
            N = faces_tensor.shape[1]
            A = np.zeros((1, N, N), dtype=np.float32)
            for i in range(N - 1):
                A[0, i, i + 1] = 1.0
                A[0, i + 1, i] = 1.0
            A_norm = normalize_adjacency(A[0])
            A_norm = torch.from_numpy(A_norm).float().unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = MODEL(faces_tensor, A_norm)
        
        probs = torch.softmax(output, dim=1)
        fake_idx = _get_fake_class_index(int(probs.shape[1]))
        real_idx = (1 - int(fake_idx)) if int(probs.shape[1]) == 2 else 0
        prob_fake = float(probs[0, int(fake_idx)].item())
        prob_real = float(probs[0, int(real_idx)].item())

        thr = float(_get_detection_threshold_fallback(0.5))
        is_fake = bool(prob_fake >= float(thr))
        pred_class = 1 if is_fake else 0
        confidence = float(prob_fake if is_fake else prob_real)

        # Optional abstain on borderline probability
        if abstain_margin > 0.0 and abs(float(prob_fake) - float(thr)) <= float(abstain_margin):
            return {
                'prediction': 'Uncertain',
                'verdict_yes_no': 'Unsure',
                'description': (
                    f"Borderline score (prob_fake={prob_fake * 100:.1f}%, thr={thr:.2f} ± {abstain_margin:.2f}). "
                    "Manual review recommended."
                ),
                'pred_class': None,
                'confidence': float(confidence),
                'prob_real': float(prob_real),
                'prob_fake': float(prob_fake),
                'num_faces': int(num_faces),
                'threshold': float(thr),
                'abstained': True,
            }

        if confidence < float(abstain_conf):
            return {
                'prediction': 'Uncertain',
                'verdict_yes_no': 'Unsure',
                'description': (
                    f"Low confidence ({confidence * 100:.1f}%). This video may be out-of-domain. "
                    "Manual review recommended."
                ),
                'pred_class': None,
                'confidence': float(confidence),
                'prob_real': float(prob_real),
                'prob_fake': float(prob_fake),
                'num_faces': int(num_faces),
                'abstained': True,
            }
        
        verdict_yes_no = 'Yes' if pred_class == 1 else 'No'
        description = (
            'Detected indicators of synthetic manipulation in facial frames.'
            if pred_class == 1 else
            'No strong signs of manipulation detected; appears authentic.'
        )

        return {
            'prediction': 'Deepfake' if pred_class == 1 else 'Real',
            'verdict_yes_no': verdict_yes_no,
            'description': description,
            'pred_class': int(pred_class),
            'confidence': float(confidence),
            'prob_real': float(prob_real),
            'prob_fake': float(prob_fake),
            'num_faces': num_faces,
            'threshold': float(thr),
        }
    except Exception as e:
        return {'error': str(e)}


def _simple_english_message(result: dict | None, filename: str | None = None) -> str:
    """Return a short, school-level English message for UI/server output."""
    if not isinstance(result, dict):
        return "Sorry, I could not check this video."

    if result.get('error'):
        return f"Sorry, I could not check this video. Error: {result.get('error')}"

    pred = (result.get('prediction') or result.get('verdict_yes_no') or 'Unknown')
    pred_s = str(pred).strip().lower()

    # Normalize common labels
    if pred_s in ('yes', 'deepfake', 'fake'):
        label = 'Fake'
    elif pred_s in ('no', 'real', 'original'):
        label = 'Real'
    elif pred_s in ('unsure', 'uncertain', 'unknown'):
        label = 'Not sure'
    else:
        label = 'Not sure'

    conf = result.get('confidence', None)
    try:
        conf_pct = int(round(float(conf) * 100)) if conf is not None else None
    except Exception:
        conf_pct = None

    # Small, friendly explanation
    if label == 'Fake':
        base = "This video looks FAKE (a deepfake)."
    elif label == 'Real':
        base = "This video looks REAL."
    else:
        base = "I am NOT SURE about this video."

    if conf_pct is not None:
        base += f" Confidence: {conf_pct}%."

    if result.get('abstained'):
        base += " The model is not confident, so please double-check manually."

    try:
        nfaces = int(result.get('num_faces') or 0)
    except Exception:
        nfaces = 0

    if nfaces <= 0:
        base += " I could not clearly find a face in the video."

    if filename:
        return f"File: {filename}\n{base}"
    return base


def _ensure_exact_word_count(text: str, target_words: int = 200) -> str:
    """Ensure output is exactly target_words words using whitespace tokenization."""
    if not isinstance(text, str):
        text = str(text)

    tokens = " ".join(text.strip().split()).split()
    if not tokens:
        tokens = ["No", "explanation", "available."]

    if len(tokens) > target_words:
        tokens = tokens[:target_words]
        if tokens and tokens[-1][-1] not in ('.', '!', '?'):
            tokens[-1] = tokens[-1] + '.'
        return " ".join(tokens)

    if len(tokens) < target_words:
        padding_sentences = [
            "Please treat this result as a helpful signal, not a final verdict.",
            "If something looks suspicious, check the source and compare with other copies.",
            "Higher quality video usually gives a more reliable score.",
            "When in doubt, ask for a human review and keep an audit trail.",
        ]
        padding_tokens: list[str] = []
        for s in padding_sentences:
            padding_tokens.extend(s.split())

        i = 0
        while len(tokens) < target_words:
            tokens.append(padding_tokens[i % len(padding_tokens)])
            i += 1

    if tokens and tokens[-1][-1] not in ('.', '!', '?'):
        tokens[-1] = tokens[-1] + '.'
    return " ".join(tokens[:target_words])


def _simple_english_justification_200_words(result: dict | None, filename: str | None = None) -> str:
    """Generate a simple-English justification of the model output (exactly 200 words)."""
    if not isinstance(result, dict):
        base = "I could not create a justification because the prediction data is missing."
        return _ensure_exact_word_count(base, 200)

    if result.get('error'):
        base = (
            "I could not create a justification because the system hit an error while checking the video. "
            f"The error was: {result.get('error')}. "
            "This usually means the file could not be read, the model is not loaded, or the video format is not supported. "
            "Try a different file, or re-upload a smaller and clearer clip, then run the check again."
        )
        return _ensure_exact_word_count(base, 200)

    pred = (result.get('prediction') or result.get('verdict_yes_no') or 'Unknown')
    pred_s = str(pred).strip().lower()
    if pred_s in ('yes', 'deepfake', 'fake'):
        label = 'Fake'
    elif pred_s in ('no', 'real', 'original'):
        label = 'Real'
    elif pred_s in ('unsure', 'uncertain', 'unknown'):
        label = 'Not sure'
    else:
        label = 'Not sure'

    try:
        conf_pct = round(float(result.get('confidence') or 0.0) * 100, 2)
    except Exception:
        conf_pct = 0.0

    try:
        prob_fake_pct = round(float(result.get('prob_fake') or 0.0) * 100, 2)
    except Exception:
        prob_fake_pct = 0.0

    try:
        prob_real_pct = round(float(result.get('prob_real') or 0.0) * 100, 2)
    except Exception:
        prob_real_pct = 0.0

    try:
        faces = int(result.get('num_faces') or 0)
    except Exception:
        faces = 0

    abstained = bool(result.get('abstained'))
    uncertainty_note = " The model flagged low confidence, so a manual check is recommended." if abstained else ""

    name_part = f" for the file {filename}" if filename else ""

    base = (
        f"This is a simple explanation of why the system predicted {label}{name_part}. "
        f"The model predicted {label} with about {conf_pct}% confidence. "
        "Confidence is not a guarantee. It is a score based on patterns the model learned from many examples. "
        f"In this run, the model assigned about {prob_fake_pct}% probability to fake and {prob_real_pct}% to real. "
        f"It detected {faces} face(s) in the video. "
        "The detector checks many frames and looks for visual cues that can appear in manipulated clips. "
        "Examples include odd skin texture, strange edges around the face, lighting that does not match, or small flickers between frames. "
        "It also checks whether these cues stay consistent over time, not just in one frame. "
        "Video quality matters a lot. Strong compression, low light, fast motion, filters, and screen recordings can change pixels and confuse the model. "
        "If the face is tiny, blurred, or partly covered, the output can be less reliable."
        f"{uncertainty_note} "
        "Use this result as a warning sign, not final proof. For high stakes decisions, verify the source, compare with an original upload, and review key frames." 
    )
    return _ensure_exact_word_count(base, 200)

@app.route('/dashboard')
def dashboard():
    metrics = get_training_metrics()
    checkpoints = sorted(Path('.').glob('checkpoint_epoch_*.pt'), key=lambda x: int(x.stem.split('_')[-1]))
    checkpoint_best = list(Path('.').glob('checkpoint_best.pt'))

    return render_template('dashboard.html', 
                         metrics=metrics,
                         checkpoints=[str(c.name) for c in checkpoints],
                         checkpoint_best=str(checkpoint_best[0].name) if checkpoint_best else None,
                         current_model=CHECKPOINT_PATH,
                         model_type=MODEL_TYPE,
                         username=session.get('username'))


@app.route('/')
def index():
    # If logged in, go to the main dashboard; otherwise show the simple multi-upload UI.
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    return redirect(url_for('simple_ui'))


@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def api_login():
    data = request.get_json() or {}
    username = data.get('username')
    email = (data.get('email') or '').strip() or (username or '').strip()
    password = data.get('password')
    if not email or not password:
        return jsonify({'success': False, 'error': 'Email and password required'}), 400

    firebase_misconfigured = False

    # Prefer Firebase Auth if configured
    if FIREBASE_API_KEY:
        try:
            fb = _firebase_login(email=email, password=password)
            session['logged_in'] = True
            session['username'] = email
            session['firebase_uid'] = fb.get('localId')
            session['firebase_id_token'] = fb.get('idToken')
            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        except Exception as e:
            msg = str(e)
            # If Firebase credentials/config are wrong, allow local fallback.
            if 'INVALID_FIREBASE_API_KEY' in msg or 'FIREBASE_AUTH_CONFIGURATION_NOT_FOUND' in msg:
                firebase_misconfigured = True
            else:
                if 'EMAIL_NOT_FOUND' in msg or 'USER_DISABLED' in msg:
                    return jsonify({'success': False, 'error': 'User not found. Please sign up.', 'redirect': url_for('signup')}), 404
                if 'INVALID_PASSWORD' in msg:
                    return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
                return jsonify({'success': False, 'error': f'Login failed: {msg}'}), 400

    # Fallback: local users.json
    users = []
    if USER_DB_PATH.exists():
        try:
            users = json.loads(USER_DB_PATH.read_text())
        except Exception:
            users = []

    user = next((u for u in users if u.get('email') == email or u.get('username') == email), None)
    if user is None:
        if firebase_misconfigured:
            return jsonify({
                'success': False,
                'error': 'Firebase is misconfigured (invalid API key/config). Fix FIREBASE_API_KEY in your environment or remove it to use local auth.',
                'redirect': url_for('signup')
             }), 400
        return jsonify({'success': False, 'error': 'User not found. Please sign up.', 'redirect': url_for('signup')}), 404
    if user.get('password') != password:
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

    session['logged_in'] = True
    session['username'] = user.get('username') or email
    return jsonify({'success': True, 'redirect': url_for('dashboard')})


@app.route('/signup', methods=['GET'])
def signup():
    return render_template('signup.html')


@app.route('/signup', methods=['POST'])
def api_signup():
    data = request.get_json() or {}
    username = data.get('username')
    email = (data.get('email') or '').strip()
    password = data.get('password')
    password_confirm = data.get('password_confirm')
    if not username or not email or not password:
        return jsonify({'success': False, 'error': 'All fields required'}), 400
    if password != password_confirm:
        return jsonify({'success': False, 'error': 'Passwords do not match'}), 400

    firebase_misconfigured = False

    # Prefer Firebase Auth if configured
    if FIREBASE_API_KEY:
        try:
            fb = _firebase_signup(email=email, password=password)
            session['logged_in'] = True
            session['username'] = email
            session['firebase_uid'] = fb.get('localId')
            session['firebase_id_token'] = fb.get('idToken')
            try:
                if fb.get('localId'):
                    _firebase_store_user_profile(uid=fb.get('localId'), email=email, username=username)
            except Exception:
                pass
            return jsonify({'success': True, 'message': 'Signup successful', 'redirect': url_for('dashboard')})
        except Exception as e:
            msg = str(e)
            if 'INVALID_FIREBASE_API_KEY' in msg or 'FIREBASE_AUTH_CONFIGURATION_NOT_FOUND' in msg:
                firebase_misconfigured = True
            else:
                if 'EMAIL_EXISTS' in msg:
                    return jsonify({'success': False, 'error': 'Email already exists. Please log in.'}), 409
                return jsonify({'success': False, 'error': f'Signup failed: {msg}'}), 400

    # Fallback: local users.json
    users = []
    if USER_DB_PATH.exists():
        try:
            users = json.loads(USER_DB_PATH.read_text())
        except Exception:
            users = []
    if any(u.get('username') == username for u in users) or any(u.get('email') == email for u in users):
        return jsonify({'success': False, 'error': 'User already exists'}), 409
    users.append({'username': username, 'email': email, 'password': password})
    USER_DB_PATH.write_text(json.dumps(users, indent=2))
    session['logged_in'] = True
    session['username'] = username
    return jsonify({'success': True, 'message': 'Signup successful', 'redirect': url_for('dashboard')})


@app.route('/ui')
def simple_ui():
    # Keep a single canonical UI entry point.
    # /ui is retained for compatibility but immediately redirects to /results.
    return redirect(url_for('results'))


@app.route('/health')
def health():
    """Health check endpoint for platform deployers (fast, no model load)."""
    return 'ok', 200

@app.route('/about')
def about():
    # About page with deepfake information
    return render_template('about.html')


@app.route('/ui/predict', methods=['POST'])
def ui_predict():
    """Form-based UI flow: upload on one page (/ui) and render results on another (/results)."""
    enable_agent = str(os.environ.get('UI_ENABLE_AGENT', '')).strip().lower() in ('1', 'true', 'yes', 'y')
    enable_justification = str(os.environ.get('UI_ENABLE_JUSTIFICATION', '')).strip().lower() in ('1', 'true', 'yes', 'y')
    if MODEL is None:
        # Keep session cookie small.
        session.pop('ui_last_results', None)
        session.pop('ui_last_error', None)
        session['ui_last_key'] = _ui_cache_set([], 'Model not loaded')
        return redirect(url_for('results'))

    # Accept multiple files from the form.
    files = request.files.getlist('files')
    if not files:
        # Backward compatibility if someone submits a single file field named "file".
        single = request.files.get('file')
        files = [single] if single else []

    if not files or all((f is None or not getattr(f, 'filename', '')) for f in files):
        session.pop('ui_last_results', None)
        session.pop('ui_last_error', None)
        session['ui_last_key'] = _ui_cache_set([], 'No file selected')
        return redirect(url_for('results'))

    results = []
    first_error = None

    for file in files:
        if file is None or not file.filename:
            continue
        if not allowed_file(file.filename):
            first_error = first_error or f"File type not allowed: {file.filename}"
            continue

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{datetime.now().timestamp()}_{filename}")

        try:
            file.save(filepath)
            result = predict_video(filepath)

            agent_result = None
            if enable_agent:
                try:
                    agent_result = _run_web_agent_pipeline(result, filename)
                except Exception:
                    agent_result = None

            # Add simple English summary (non-technical)
            try:
                simple_msg = _simple_english_message(result, filename=filename)
            except Exception:
                simple_msg = None

            # Add 200-word justification (simple English) (opt-in; can be slow on small hosts)
            justification = None
            if enable_justification:
                try:
                    justification = _simple_english_justification_200_words(result, filename=filename)
                except Exception:
                    justification = None

            out = dict(result) if isinstance(result, dict) else {'error': 'Unexpected result type'}
            # Add model metadata for debugging consistency issues
            try:
                out.setdefault('model_type', MODEL_TYPE)
            except Exception:
                pass
            try:
                out.setdefault('checkpoint_path', CHECKPOINT_PATH)
            except Exception:
                pass
            if simple_msg and not out.get('error'):
                out['simple_message'] = simple_msg
            if justification:
                out['justification'] = justification
            if agent_result is not None:
                out['agent'] = agent_result

            results.append({
                'original_filename': file.filename,
                'filename': filename,
                'result': out,
            })
        except Exception as e:
            first_error = first_error or str(e)
        finally:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                pass

    # Store results server-side to avoid oversized cookie sessions (can cause 502 on proxies).
    session.pop('ui_last_results', None)
    session.pop('ui_last_error', None)
    session['ui_last_key'] = _ui_cache_set(results, first_error)
    return redirect(url_for('results'))


@app.route('/predict', methods=['POST'])
def predict_compat():
    """Compatibility route.

    Some UIs or older deployments submit the analysis form to /predict. Keep this
    endpoint working by delegating to the current /ui/predict implementation.
    """
    return ui_predict()


@app.route('/ui/results')
def ui_results():
    cached = _ui_cache_get(session.get('ui_last_key'))
    if cached is not None:
        results, error = cached
    else:
        # Backward compatibility: if older deployments stored results directly in the session.
        results = session.get('ui_last_results') or []
        error = session.get('ui_last_error')
    return render_template('ui_results.html', results=results, error=error)


@app.route('/results')
@app.route('/results', methods=['POST'])
def results():
    """Results page (preferred short URL).

    Supports:
    - GET: show last cached results
    - POST: accept uploads and run analysis, then redirect back to GET /results
    """
    if request.method == 'POST':
        # Process exactly like the old /ui/predict form handler, but without
        # ever navigating users to /predict or /ui/predict in the address bar.
        return ui_predict()
    return ui_results()


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


def _load_uploads_db():
    # Prefer Firebase RTDB if configured and user has Firebase session
    if session.get('logged_in') and _firebase_rtdb_base() and session.get('firebase_uid') and session.get('firebase_id_token'):
        uid = session.get('firebase_uid')
        try:
            data = _rtdb_get(f"uploads/{uid}")
            if isinstance(data, dict):
                return list(data.values())
            return []
        except Exception:
            # fall back to local
            pass

    db_path = Path('uploads.json')
    if not db_path.exists():
        return []
    try:
        return json.loads(db_path.read_text())
    except Exception:
        return []


def _save_uploads_db(items):
    # If using Firebase RTDB, write each item under /uploads/<uid>/<id>
    if session.get('logged_in') and _firebase_rtdb_base() and session.get('firebase_uid') and session.get('firebase_id_token'):
        uid = session.get('firebase_uid')
        try:
            for it in items:
                it_id = it.get('id')
                if it_id is None:
                    continue
                _rtdb_put(f"uploads/{uid}/{it_id}", it)
            return
        except Exception:
            # fall back to local
            pass

    db_path = Path('uploads.json')
    db_path.write_text(json.dumps(items, indent=2))


@app.route('/api/uploads', methods=['GET'])
def api_get_uploads():
    if not session.get('logged_in'):
        return jsonify([])
    items = _load_uploads_db()
    # return latest first
    items = sorted(items, key=lambda x: x.get('uploaded_at', ''), reverse=True)
    return jsonify(items)


@app.route('/api/chat', methods=['POST'])
def api_chat():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    data = request.get_json() or {}
    message = (data.get('message') or '').strip()
    context = _normalize_chat_context(data.get('context'))
    if not message:
        return jsonify({'success': False, 'error': 'Message is required'}), 400

    user_key = _get_user_gemini_key()
    server_key = _get_global_gemini_key()
    api_key = user_key or server_key
    api_key_source = 'user' if user_key else ('server' if server_key else None)

    # Deterministic handling so users always get a correct answer.
    if _is_model_question(message):
        return jsonify({'success': True, 'reply': _model_info_reply(api_key_source=api_key_source)})

    # Deterministic metrics answers (accuracy/precision/recall/f1, best values, epochs run)
    metrics_reply = _try_repo_metrics_reply(message)
    if metrics_reply:
        return jsonify({'success': True, 'reply': metrics_reply})

    if api_key:
        # Add current detection context to improve relevance.
        ctx_text = json.dumps(context, indent=2, default=str) if context else ''
        msg = message
        if ctx_text:
            msg = f"Current detection context (JSON):\n{ctx_text}\n\nUser message: {message}"
        reply = generate_chat_reply_gemini(msg, api_key)
    else:
        reply = generate_chat_reply_with_context(message, context)
    return jsonify({'success': True, 'reply': reply})


@app.route('/api/chat-public', methods=['POST'])
def api_chat_public():
    """Public chat endpoint used by the simple /ui page (no login required)."""
    data = request.get_json() or {}
    message = (data.get('message') or '').strip()
    context = _normalize_chat_context(data.get('context'))
    if not message:
        return jsonify({'success': False, 'error': 'Message is required'}), 400

    # Basic guardrail: block disallowed content
    harmful_terms = ['hate', 'racist', 'sexist', 'violence', 'kill', 'nsfw']
    if any(t in message.lower() for t in harmful_terms):
        return jsonify({'success': True, 'reply': "Sorry, I can't assist with that."})

    server_key = _get_global_gemini_key()
    api_key_source = 'server' if server_key else None

    if _is_model_question(message):
        return jsonify({'success': True, 'reply': _model_info_reply(api_key_source=api_key_source)})

    metrics_reply = _try_repo_metrics_reply(message)
    if metrics_reply:
        return jsonify({'success': True, 'reply': metrics_reply})

    if server_key:
        ctx_text = json.dumps(context, indent=2, default=str) if context else ''
        msg = message
        if ctx_text:
            msg = f"Current detection context (JSON):\n{ctx_text}\n\nUser message: {message}"
        reply = generate_chat_reply_gemini(msg, server_key)
    else:
        reply = generate_chat_reply_with_context(message, context)
    return jsonify({'success': True, 'reply': reply})


@app.route('/api/gemini-report-public', methods=['POST'])
def api_gemini_report_public():
    """Public endpoint for generating an English report from detection results.

    Accepts an optional user-provided Gemini API key (not persisted).
    Falls back to server GEMINI_API_KEY/GOOGLE_API_KEY if present.
    """
    data = request.get_json() or {}
    user_notes = (data.get('notes') or '').strip() or None

    # Do NOT accept an API key from the UI. Use server-side configuration only.
    api_key = _get_global_gemini_key()
    if not api_key:
        return jsonify({
            'success': False,
            'error': 'Gemini is not configured on the server. Set GEMINI_API_KEY (or GOOGLE_API_KEY) and restart the server.'
        }), 400

    results = data.get('results')
    if not isinstance(results, list):
        # Fall back to the last /ui results stored in session.
        results = session.get('ui_last_results') or []

    report = _gemini_generate_english_report(results=results, api_key=api_key, user_notes=user_notes)
    if not report:
        return jsonify({'success': False, 'error': 'No report generated'}), 500
    return jsonify({'success': True, 'report': report})


@app.route('/api/chat-config', methods=['GET', 'POST'])
def api_chat_config():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    if request.method == 'GET':
        key = _get_user_gemini_key()
        redacted = None
        if key:
            redacted = f"***{key[-4:]}" if len(key) >= 4 else "***"
        return jsonify({'success': True, 'configured': bool(key), 'redacted_key': redacted})

    # POST
    data = request.get_json() or {}
    new_key = (data.get('gemini_api_key') or '').strip()
    if not new_key:
        return jsonify({'success': False, 'error': 'API key is required'}), 400
    username = session.get('username')
    store = _load_secrets_db()
    user_entry = store.get(username) or {}
    user_entry['gemini_api_key'] = new_key
    store[username] = user_entry
    _save_secrets_db(store)
    return jsonify({'success': True, 'message': 'Gemini API key saved'})


@app.route('/api/agent-config', methods=['GET', 'POST'])
def api_agent_config():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    if request.method == 'GET':
        phone = _get_user_notification_phone()
        redacted = None
        if phone:
            redacted = f"***{phone[-4:]}" if len(phone) >= 4 else "***"
        return jsonify({'success': True, 'configured': bool(phone), 'redacted_phone': redacted})

    data = request.get_json() or {}
    phone = (data.get('notification_phone') or '').strip()
    if not _validate_phone(phone):
        return jsonify({'success': False, 'error': 'Invalid phone number. Use digits with optional leading + (8-15 digits).'}), 400

    username = session.get('username')
    store = _load_secrets_db()
    user_entry = store.get(username) or {}
    user_entry['notification_phone'] = phone
    store[username] = user_entry
    _save_secrets_db(store)
    return jsonify({'success': True, 'message': 'Notification phone saved'})

@app.route('/result/<int:upload_id>')
def result_page(upload_id):
    # Render result page for a specific upload
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('result.html', upload_id=upload_id)

@app.route('/api/result/<int:upload_id>')
def api_result(upload_id):
    # Return a single upload record by ID
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    items = _load_uploads_db()
    upload = next((it for it in items if int(it.get('id', 0)) == int(upload_id)), None)
    if upload is None:
        return jsonify({'success': False, 'error': 'Result not found'}), 404
    return jsonify({'success': True, 'upload': upload})


@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File type not allowed'}), 400

    filename = secure_filename(file.filename)
    saved_name = f"{session.get('username','anon')}_{datetime.now().timestamp()}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
    file.save(filepath)

    # Run prediction if model available
    res = predict_video(filepath) if MODEL is not None else {'error': 'Model not loaded'}

    upload_id = int(datetime.now().timestamp() * 1000)
    # Run agent pipeline (decision/monitoring/action) using prediction probs
    agent_result = _run_web_agent_pipeline(res, saved_name)

    items = _load_uploads_db()
    upload_id = (max([it.get('id', 0) for it in items]) + 1) if items else 1
    if 'error' in res:
        details_text = f"Error: {res.get('error')}\nFile: {filename}"
    else:
        verdict_text = 'Yes' if int(res.get('pred_class', 0)) == 1 else 'No'
        pred_label = res.get('prediction', 'Unknown')
        conf = float(res.get('confidence', 0.0)) * 100.0
        faces = int(res.get('num_faces', 0))
        desc = res.get('description', '')
        details_lines = [
            f"Verdict (Deepfake?): {verdict_text}",
            f"Prediction: {pred_label}",
            f"Confidence: {conf:.2f}%",
            f"Faces Detected: {faces}",
            f"Notes: {desc}"
        ]
        if agent_result and 'error' not in agent_result:
            details_lines.extend([
                "--- Agent ---",
                f"Alert Level: {agent_result.get('alert_level')}",
                f"Agent Explanation: {agent_result.get('explanation')}",
                f"Agent Actions: {', '.join(agent_result.get('actions_taken') or []) or 'None'}",
            ])
        elif agent_result and 'error' in agent_result:
            details_lines.extend(["--- Agent ---", agent_result.get('error')])
        details_text = "\n".join(details_lines)

    record = {
        'id': upload_id,
        'filename': saved_name,
        'original_filename': filename,
        'uploaded_at': datetime.now().isoformat(),
        'processed': 'error' not in res,
        'is_fake': int(res.get('pred_class')) if res.get('pred_class') is not None else None,
        'confidence': float(res.get('confidence')) if res.get('confidence') is not None else None,
        'prob': float(res.get('prob_fake')) if res.get('prob_fake') is not None else None,
        'num_faces': int(res.get('num_faces')) if res.get('num_faces') is not None else 0,
        'detection_details': details_text,
        'agent': agent_result
    }
    items.append(record)
    _save_uploads_db(items)

    return jsonify({'success': True, 'upload_id': upload_id, 'result': res, 'agent': agent_result})

@app.route('/api/metrics')
def api_metrics():
    metrics = get_training_metrics()
    return jsonify(metrics)

@app.route('/api/load-model', methods=['POST'])
def api_load_model():
    global MODEL_META
    data = request.json
    checkpoint = data.get('checkpoint')
    model_type = data.get('model_type', 'vit_gcn')

    # Persist extra metadata for load_model() (e.g., pretrained backbone list)
    MODEL_META = {
        'backbone': data.get('backbone'),
        'backbones': data.get('backbones'),
    }
    
    if not checkpoint or not os.path.exists(checkpoint):
        return jsonify({'error': 'Checkpoint not found'}), 404
    
    success = load_model(checkpoint, model_type)
    if success:
        return jsonify({'success': True, 'message': f'Model loaded from {checkpoint}', 'load_stats': LAST_LOAD_STATS})
    else:
        return jsonify({'error': 'Failed to load model', 'load_stats': LAST_LOAD_STATS}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 400

    enable_agent = str(os.environ.get('API_ENABLE_AGENT', '')).strip().lower() in ('1', 'true', 'yes', 'y')
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{datetime.now().timestamp()}_{filename}")
        file.save(filepath)
        
        result = predict_video(filepath)

        agent_result = None
        if enable_agent:
            try:
                agent_result = _run_web_agent_pipeline(result, filename)
            except Exception:
                agent_result = None

        # Add simple English summary (non-technical)
        try:
            simple_msg = _simple_english_message(result, filename=filename)
            if isinstance(result, dict) and not result.get('error'):
                result = dict(result)
                result['simple_message'] = simple_msg
        except Exception:
            pass
        
        os.remove(filepath)
        # Preserve existing response keys; add agent output when available.
        if agent_result is not None:
            out = dict(result)
            out['agent'] = agent_result
            return jsonify(out)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info')
def api_model_info():
    # Note: pred_class semantics are stable (0=real, 1=fake). FAKE_CLASS_INDEX only maps logits.
    fake_idx = _get_fake_class_index(2)
    return jsonify({
        'loaded': MODEL is not None,
        'checkpoint': CHECKPOINT_PATH,
        'model_type': MODEL_TYPE,
        'device': str(DEVICE),
        'fake_class_index': int(fake_idx),
        'fake_class_index_source': _get_fake_class_index_source(2),
        'load_stats': LAST_LOAD_STATS,
    })

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DeepfakeGuard server')
    parser.add_argument('--host', default=os.environ.get('HOST', '127.0.0.1'), help='Bind host (use 0.0.0.0 for LAN)')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '5000')), help='Bind port')
    parser.add_argument(
        '--debug',
        action='store_true',
        default=str(os.environ.get('DEBUG', '')).strip().lower() in ('1', 'true', 'yes', 'y'),
        help='Enable Flask debug mode',
    )
    parser.add_argument(
        '--no-autoload',
        action='store_true',
        default=str(os.environ.get('NO_AUTOLOAD', '')).strip().lower() in ('1', 'true', 'yes', 'y'),
        help='Disable auto-loading the best checkpoint on startup',
    )
    args = parser.parse_args()

    # Auto-load a checkpoint on startup (prefer DFDC ensemble runs when available)
    _attempt_autoload(no_autoload=bool(args.no_autoload))

    print(f"Starting server on http://{args.host}:{args.port} (debug={bool(args.debug)})")
    app.run(host=str(args.host), port=int(args.port), debug=bool(args.debug))

else:
    # WSGI/production import path (e.g., Render, Gunicorn)
    _attempt_autoload(
        no_autoload=str(os.environ.get('NO_AUTOLOAD', '')).strip().lower() in ('1', 'true', 'yes', 'y')
    )
