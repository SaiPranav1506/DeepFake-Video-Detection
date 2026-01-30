import argparse
from pathlib import Path
import inspect
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, roc_auc_score

from dataset import VideoFacesDataset
from models import DeepfakeModel, ViTFeatureExtractor
from utils import normalize_adjacency
try:
    from RNNModel import LogicRNNLSTM, create_model as create_rnn_model
except Exception:
    # import for module path when executing as package
    from .RNNModel import LogicRNNLSTM, create_model as create_rnn_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('src.evaluate')


def _infer_timm_vit_model_name_from_state_dict(sd: dict) -> str | None:
    """Best-effort inference of timm ViT variant from a DeepfakeModel checkpoint.

    Our checkpoints may only store a raw `state_dict` (no config). In that case the
    default model constructor (vit_base) can mismatch (e.g. checkpoint trained with
    vit_tiny), causing shape errors and meaningless evaluation.
    """
    if not isinstance(sd, dict):
        return None

    # Typical timm ViT keys when wrapped as: DeepfakeModel.vit (ViTFeatureExtractor).vit (timm model)
    embed_dim = None
    for k in (
        'vit.vit.patch_embed.proj.weight',
        'vit.vit.cls_token',
        'vit.vit.pos_embed',
    ):
        t = sd.get(k)
        if hasattr(t, 'shape'):
            # patch_embed.proj.weight: (embed_dim, 3, 16, 16)
            if k.endswith('proj.weight') and len(t.shape) >= 1:
                embed_dim = int(t.shape[0])
                break
            # cls_token/pos_embed: (..., embed_dim)
            if len(t.shape) >= 1:
                embed_dim = int(t.shape[-1])
                break

    if embed_dim is None:
        return None

    # Common patch16 ViT families in timm
    mapping = {
        192: 'vit_tiny_patch16_224',
        384: 'vit_small_patch16_224',
        768: 'vit_base_patch16_224',
        1024: 'vit_large_patch16_224',
    }
    return mapping.get(embed_dim)


def collate_batch(batch, max_nodes=16, image_size=(224, 224)):
    B = len(batch)
    nodes = []
    labels = []
    files = []
    lengths = []
    for item in batch:
        faces = item['faces']
        M = faces.shape[0]
        if M >= max_nodes:
            idxs = np.linspace(0, M - 1, max_nodes).astype(int)
            sel = faces[idxs]
        else:
            pad = max_nodes - M
            if M == 0:
                sel = np.zeros((max_nodes, image_size[0], image_size[1], 3), dtype=np.uint8)
            else:
                pads = np.repeat(faces[-1][None], pad, axis=0)
                sel = np.concatenate([faces, pads], axis=0)
        nodes.append(sel)
        labels.append(item['label'] if item['label'] is not None else -1)
        files.append(item.get('file', ''))
        lengths.append(M)

    nodes = np.stack(nodes)
    nodes = torch.from_numpy(nodes).permute(0, 1, 4, 2, 3).float() / 255.0
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    N = nodes.shape[1]
    A = np.zeros((B, N, N), dtype=np.float32)
    for b in range(B):
        for i in range(N - 1):
            A[b, i, i + 1] = 1.0
            A[b, i + 1, i] = 1.0
    A_norm = np.stack([normalize_adjacency(A[b]) for b in range(B)])
    A_norm = torch.from_numpy(A_norm).float()
    return nodes, A_norm, labels, files, lengths


def evaluate_gcn(model, loader, device):
    """Evaluate the GCN+ViT model (existing DeepfakeModel).

    Returns dict of metrics and list of (file, label, pred, prob).
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    all_files = []
    with torch.no_grad():
        for batch in loader:
            images, A_norm, labels, files, lengths = collate_batch(batch)
            images = images.to(device)
            A_norm = A_norm.to(device)
            out = model(images, A_norm)
            if out.dim() == 1:
                probs = out.cpu().numpy()
                preds = (probs >= 0.5).astype(int)
            else:
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds = out.argmax(dim=1).cpu().numpy()

            labels = labels.numpy()

            # filter unknown labels
            mask = labels >= 0
            if mask.sum() == 0:
                continue
            all_preds.extend(preds[mask].tolist())
            all_probs.extend(probs[mask].tolist())
            all_labels.extend(labels[mask].tolist())
            for f in np.array(files)[mask].tolist():
                all_files.append(f)

    return _compute_and_print_metrics(all_labels, all_preds, all_probs, all_files)


def evaluate_rnn(model, loader, device, feature_extractor=None, embed_proj=None):
    """Evaluate the Logic RNN+LSTM model.

    The dataset yields per-video face images; we use a ViT (or fallback) feature
    extractor to convert each face to an embedding sequence for the RNN.
    """
    model.eval()
    if feature_extractor is None:
        feature_extractor = ViTFeatureExtractor()
    feature_extractor.eval()
    feature_extractor.to(device)

    all_preds = []
    all_probs = []
    all_labels = []
    all_files = []
    with torch.no_grad():
        for batch in loader:
            images, A_norm, labels, files, lengths = collate_batch(batch)
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W).to(device)
            feats = feature_extractor(images)  # (B*N, F)
            Fdim = feats.shape[-1]
            feats = feats.view(B, N, Fdim)

            # Optionally project embeddings to the RNN input size
            if embed_proj is not None:
                feats_flat = feats.view(B * N, Fdim)
                feats_proj = embed_proj(feats_flat)
                feats = feats_proj.view(B, N, -1)

            labels = labels.numpy()

            # Run RNN (expects float tensors)
            seq = feats
            out = model(seq, lengths.to(device))
            # out expected shape (B,1) or (B)
            probs = out.squeeze(1).cpu().numpy() if out.dim() > 1 else out.cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            mask = labels >= 0
            if mask.sum() == 0:
                continue
            all_preds.extend(preds[mask].tolist())
            all_probs.extend(probs[mask].tolist())
            all_labels.extend(labels[mask].tolist())
            for f in np.array(files)[mask].tolist():
                all_files.append(f)

    return _compute_and_print_metrics(all_labels, all_preds, all_probs, all_files)


def _compute_and_print_metrics(
    all_labels,
    all_preds,
    all_probs,
    all_files,
    threshold: float = 0.5,
    sweep_thresholds: bool = False,
    opt_metric: str = 'accuracy',
):
    if len(all_labels) == 0:
        print('No labeled samples found in dataset. Cannot compute metrics.')
        return {}

    labels_unique = set(all_labels)
    probs_arr = None
    try:
        probs_arr = np.array(all_probs, dtype=np.float32)
    except Exception:
        probs_arr = None

    # If we have probabilities, optionally override preds via thresholding.
    if probs_arr is not None and len(probs_arr) == len(all_labels):
        if sweep_thresholds and len(labels_unique) >= 2:
            opt_metric = (opt_metric or 'accuracy').lower()
            best_thr = None
            best_val = -1.0
            # small grid sweep (fast)
            for thr in np.linspace(0.05, 0.95, 19):
                preds_thr = (probs_arr >= float(thr)).astype(int)
                acc_thr = accuracy_score(all_labels, preds_thr)
                prec_thr, rec_thr, f1_thr, _ = precision_recall_fscore_support(
                    all_labels, preds_thr, average='binary', zero_division=0
                )
                score = acc_thr if opt_metric == 'accuracy' else f1_thr
                if score > best_val:
                    best_val = float(score)
                    best_thr = float(thr)

            if best_thr is not None:
                threshold = best_thr
                all_preds = (probs_arr >= float(threshold)).astype(int).tolist()
                print(f"Using best threshold={threshold:.2f} (optimized for {opt_metric})")
        else:
            all_preds = (probs_arr >= float(threshold)).astype(int).tolist()

    acc = accuracy_score(all_labels, all_preds)
    # handle case with single class gracefully
    # use zero_division=0 to avoid warnings turning into exceptions
    if len(labels_unique) < 2:
        # Only one class present in ground truth: precision/recall/F1 (binary) undefined
        prec, rec, f1 = 0.0, 0.0, 0.0
    else:
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

    print('Results:')
    print(f'  Accuracy:  {acc:.4f}')
    print(f'  Precision: {prec:.4f}')
    print(f'  Recall:    {rec:.4f}')
    print(f'  F1 score:  {f1:.4f}')
    print('\nClassification report:')
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))
    print('Confusion matrix:')
    print(confusion_matrix(all_labels, all_preds))
    # ROC AUC for binary (only if both classes present and probs are valid)
    auc = None
    try:
        if len(labels_unique) >= 2 and probs_arr is not None and all([not np.isnan(p) for p in probs_arr.tolist()]):
            auc = roc_auc_score(all_labels, all_probs)
            print(f'ROC AUC:    {auc:.4f}')
        else:
            print('ROC AUC not available (need both classes present and valid probabilities).')
    except Exception:
        print('ROC AUC not available (error computing AUC).')

    # One-line trainer-style summary (matches src.ensemble_trainer output structure)
    try:
        auc_val = float(auc) if auc is not None else 0.5
        logger.info(f"✓ Results - Acc: {acc:.4f}, AUC: {auc_val:.4f}, F1: {f1:.4f}")
    except Exception:
        pass

    results = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': auc,
        'predictions': list(zip(all_files, all_labels, all_preds, all_probs))
    }
    return results


def _build_gcn_model_from_checkpoint(ckpt, vit_model_name_override: str | None = None):
    """Build a DeepfakeModel instance that matches the checkpoint.

    Newer checkpoints store a `model_config` dict (e.g., selecting CLIP/DINOv2 backbone).
    If we ignore it and instantiate the default model, most keys won't load and metrics
    will look like random guessing.
    """
    model_cfg = {}
    if isinstance(ckpt, dict) and isinstance(ckpt.get('model_config'), dict):
        model_cfg = dict(ckpt['model_config'])

    # If ckpt is a raw state_dict (common in older training code), infer ViT model name.
    sd = None
    try:
        sd = ckpt.get('model_state', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else None
    except Exception:
        sd = None

    vit_override = (vit_model_name_override or '').strip() or None
    if vit_override is not None:
        model_cfg['vit_model_name'] = vit_override
    elif 'vit_model_name' not in model_cfg:
        inferred = _infer_timm_vit_model_name_from_state_dict(sd) if isinstance(sd, dict) else None
        if inferred:
            model_cfg['vit_model_name'] = inferred

    # If the checkpoint carries model weights, we don't need to download pretrained backbone weights.
    # This also makes evaluation work in offline environments.
    backbone = str(model_cfg.get('backbone', '') or '').lower()
    if 'dinov2' in backbone:
        model_cfg['dinov2_pretrained'] = False
    if 'clip' in backbone:
        model_cfg['clip_pretrained'] = False
    # timm ViT path is already controlled by pretrained_vit; keep it explicit
    if 'pretrained_vit' in model_cfg:
        model_cfg['pretrained_vit'] = bool(model_cfg.get('pretrained_vit', False))

    # Only pass supported kwargs to DeepfakeModel
    try:
        sig = inspect.signature(DeepfakeModel.__init__)
        allowed = set(sig.parameters.keys())
        allowed.discard('self')
        model_cfg = {k: v for k, v in model_cfg.items() if k in allowed}
    except Exception:
        # If signature introspection fails for any reason, fall back to best-effort
        pass

    return DeepfakeModel(**model_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--recursive_data', action='store_true', help='Load .npz recursively under --data_dir')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model_type', choices=['gcn', 'rnn'], default='gcn', help='Which model architecture to evaluate')
    parser.add_argument('--out_csv', default=None, help='Optional path to save per-file predictions (CSV)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold for prob_fake when computing preds (default: 0.5)')
    parser.add_argument('--sweep_thresholds', action='store_true', help='Sweep thresholds on probs and pick best by --opt_metric')
    parser.add_argument('--opt_metric', choices=['accuracy', 'f1'], default='accuracy', help='Metric to optimize when sweeping thresholds')
    parser.add_argument('--vit_model_name', default=None, help='timm ViT model name override (e.g. vit_tiny_patch16_224). Useful when checkpoint was trained with a non-default ViT.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    ds = VideoFacesDataset(args.data_dir, recursive=bool(args.recursive_data))
    if len(ds) == 0:
        print('No data found in', args.data_dir)
        return

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=lambda x: x)
    device = torch.device(args.device)
    results = None
    if args.model_type == 'gcn':
        ckpt = torch.load(args.checkpoint, map_location=device)

        model = _build_gcn_model_from_checkpoint(ckpt, vit_model_name_override=args.vit_model_name)

        # Normalize checkpoint dict extraction
        sd = ckpt.get('model_state', ckpt.get('state_dict', ckpt))

        # If checkpoint is a container with unexpected prefixes (e.g. 'base_model.'), try to strip them
        if isinstance(sd, dict):
            # common prefix patterns to strip
            prefix_candidates = ['base_model.encoder.', 'base_model.', 'model.']
            for p in prefix_candidates:
                # if many keys start with this prefix, strip it
                if any(k.startswith(p) for k in sd.keys()):
                    sd = {k[len(p):] if k.startswith(p) else k: v for k, v in sd.items()}
                    break

        # Attempt strict load, then fall back to non-strict if necessary
        try:
            model.load_state_dict(sd)
        except Exception as e:
            try:
                model.load_state_dict(sd, strict=False)
                print('Warning: loaded checkpoint with strict=False (some keys mismatched)')
            except Exception as e2:
                # At this point evaluation would be meaningless (random weights). Fail fast.
                print('Error: failed to load checkpoint into model:', e2)
                print('Tip: pass --vit_model_name to match training (e.g. vit_tiny_patch16_224).')
                return

        model.to(device)
        results = evaluate_gcn(model, loader, device)
    else:
        # load checkpoint first to infer RNN config (input_size, hidden_size, num_layers)
        ckpt = torch.load(args.checkpoint, map_location=device)
        sd = ckpt.get('model_state', ckpt.get('state_dict', ckpt))

        # attempt to infer RNN params from checkpoint keys
        inferred_input = None
        inferred_hidden = None
        inferred_layers = None
        layer_idxs = set()
        for k, v in sd.items():
            if k.startswith('logic_cells.') and k.endswith('.and_gate.weight'):
                # key format: logic_cells.{i}.and_gate.weight
                parts = k.split('.')
                try:
                    idx = int(parts[1])
                    layer_idxs.add(idx)
                    out_dim, in_dim = v.shape
                    inferred_hidden = out_dim
                    inferred_input = in_dim - out_dim
                except Exception:
                    pass
        if len(layer_idxs) > 0:
            inferred_layers = max(layer_idxs) + 1

        # create feature extractor to learn its output dim
        feat_extractor = ViTFeatureExtractor()
        try:
            feat_dim = feat_extractor.out_dim
        except Exception:
            feat_dim = None
        if feat_dim is None:
            dummy = torch.zeros(1, 3, 224, 224)
            try:
                with torch.no_grad():
                    od = feat_extractor(dummy)
                feat_dim = od.shape[-1]
            except Exception:
                feat_dim = 1024

        # decide final RNN config: prefer checkpoint values when available
        rnn_cfg = {
            'input_size': int(inferred_input) if inferred_input is not None else int(feat_dim),
            'hidden_size': int(inferred_hidden) if inferred_hidden is not None else 512,
            'num_layers': int(inferred_layers) if inferred_layers is not None else 2,
            'dropout': 0.5,
        }

        model = create_rnn_model(rnn_cfg)
        # load state dict (support different container keys)
        try:
            if 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
            elif 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                model.load_state_dict(ckpt)
        except Exception as e:
            # If loading fails due to size mismatches, continue with the
            # freshly-initialized model (useful for smoke tests with dummy
            # or mismatched checkpoints). Print a helpful warning.
            print('Warning: failed to load checkpoint into RNN model:', e)
            print('Proceeding with randomly initialized RNN (this is fine for smoke tests).')

        model.to(device)
        feat_extractor.to(device)

        # if feature dim != rnn input, create a projection layer to map embeddings
        embed_proj = None
        if feat_dim != rnn_cfg['input_size']:
            import torch.nn as nn
            embed_proj = nn.Linear(int(feat_dim), int(rnn_cfg['input_size']))
            embed_proj.to(device)

        results = evaluate_rnn(model, loader, device, feature_extractor=feat_extractor, embed_proj=embed_proj)

    if args.out_csv and results and 'predictions' in results:
        import csv
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['file', 'label', 'pred', 'prob'])
            for row in results['predictions']:
                w.writerow(row)

    # Re-print metrics using threshold logic if predictions are available
    try:
        if isinstance(results, dict) and results.get('predictions'):
            files, labels, preds, probs = zip(*results['predictions'])
            _compute_and_print_metrics(
                list(labels),
                list(preds),
                list(probs),
                list(files),
                threshold=float(args.threshold),
                sweep_thresholds=bool(args.sweep_thresholds),
                opt_metric=str(args.opt_metric),
            )
    except Exception:
        pass

    return results


if __name__ == '__main__':
    main()
