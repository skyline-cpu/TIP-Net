#!/usr/bin/env python3
"""
test.py  —  Evaluation entry point for TIP-Net.

Outputs:
  - Video-level metrics (ACC, AUC, Precision, Recall, F1)
  - Snippet-level localisation metrics (AP@0.5, mAP)
  - JSON results file
  - Optional temporal anomaly score visualisations
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from data.dataset   import DeepFakeSnippetDataset, collate_fn
from data.transforms import get_val_transform, get_xception_transform
from model.tipnet   import TIPNet
from utils.train_utils import (
    compute_metrics, compute_loc_metrics, load_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser('TIP-Net Testing')
    p.add_argument('--frames_dir',       type=str, required=True)
    p.add_argument('--checkpoint',       type=str, required=True)
    p.add_argument('--annotation_file',  type=str, default=None,
                   help='JSON with per-video frame-level labels for localisation eval.')
    # Model
    p.add_argument('--backbone',         type=str, default='xception')
    p.add_argument('--snippet_dim',      type=int, default=512)
    p.add_argument('--proj_dim',         type=int, default=128)
    p.add_argument('--num_snippets',     type=int, default=16)
    p.add_argument('--frames_per_snippet', type=int, default=8)
    p.add_argument('--num_prototypes',   type=int, default=8)
    p.add_argument('--tmm_order',        type=int, default=2)
    p.add_argument('--lstm_layers',      type=int, default=2)
    p.add_argument('--attention_heads',  type=int, default=4)
    p.add_argument('--image_size',       type=int, default=224)
    # Eval
    p.add_argument('--batch_size',       type=int, default=4)
    p.add_argument('--num_workers',      type=int, default=4)
    p.add_argument('--threshold',        type=float, default=0.5)
    p.add_argument('--save_dir',         type=str, default='./results')
    p.add_argument('--visualize',        action='store_true')
    p.add_argument('--num_vis_samples',  type=int, default=10)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Inference loop
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device, threshold: float = 0.5):
    model.eval()
    all_vid_true, all_vid_pred, all_vid_prob = [], [], []
    all_anomaly_scores  = []   # list of np arrays [T]
    all_snippet_scores  = []   # list of np arrays [T]
    all_snippet_gt      = []   # list of np arrays [T]
    all_temporal_mt     = []
    all_spatial_dt      = []
    all_video_ids       = []

    for batch in tqdm(loader, desc='Inference'):
        snippets   = batch['snippets'].to(device, non_blocking=True)
        labels     = batch['labels'].to(device,   non_blocking=True)
        snippet_gt = batch['snippet_gt']          # [B, T]  CPU

        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(snippets, labels, is_training=False)

        probs = torch.softmax(outputs['video_logits'], dim=1)[:, 1]
        preds = (probs > threshold).long()

        all_vid_true.extend(labels.cpu().tolist())
        all_vid_pred.extend(preds.cpu().tolist())
        all_vid_prob.extend(probs.cpu().tolist())
        all_video_ids.extend(batch['video_ids'])

        all_anomaly_scores.extend(outputs['anomaly_scores'].cpu().numpy())
        all_snippet_scores.extend(outputs['snippet_scores'].cpu().numpy())
        all_snippet_gt.extend(snippet_gt.numpy())
        all_temporal_mt.extend(outputs['temporal_mt'].cpu().numpy())
        all_spatial_dt.extend(outputs['spatial_dt'].cpu().numpy())

    return {
        'vid_true':       np.array(all_vid_true),
        'vid_pred':       np.array(all_vid_pred),
        'vid_prob':       np.array(all_vid_prob),
        'anomaly_scores': all_anomaly_scores,
        'snippet_scores': all_snippet_scores,
        'snippet_gt':     all_snippet_gt,
        'temporal_mt':    all_temporal_mt,
        'spatial_dt':     all_spatial_dt,
        'video_ids':      all_video_ids,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Snippet → Frame score mapping
# ──────────────────────────────────────────────────────────────────────────────
def expand_to_frames(snippet_scores: np.ndarray, num_frames: int) -> np.ndarray:
    """Linearly interpolate snippet scores to frame resolution."""
    T = len(snippet_scores)
    if T == 0:
        return np.zeros(num_frames)
    # Assign each snippet score to its frame window
    frame_scores = np.zeros(num_frames, dtype=np.float32)
    window = num_frames / T
    for t, s in enumerate(snippet_scores):
        start = int(t * window)
        end   = min(int((t + 1) * window), num_frames)
        frame_scores[start:end] = s
    return frame_scores


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────
def visualise_anomaly_scores(results: dict, save_dir: str,
                             num_samples: int = 10):
    os.makedirs(save_dir, exist_ok=True)

    vid_true    = results['vid_true']
    video_ids   = results['video_ids']
    anomaly     = results['anomaly_scores']
    temporal_mt = results['temporal_mt']
    spatial_dt  = results['spatial_dt']

    fake_idx = np.where(vid_true == 1)[0][:num_samples // 2]
    real_idx = np.where(vid_true == 0)[0][:num_samples // 2]
    indices  = np.concatenate([fake_idx, real_idx])

    for idx in indices:
        vid_id = video_ids[idx]
        label  = 'Fake' if vid_true[idx] == 1 else 'Real'
        At     = anomaly[idx]
        mt     = temporal_mt[idx]
        dt     = spatial_dt[idx]
        T      = len(At)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        t_axis = np.arange(T)

        axes[0].plot(t_axis, At, 'o-', color='crimson', lw=2, ms=5)
        axes[0].axhline(0.5, color='k', ls='--', lw=1, alpha=0.5)
        axes[0].set_ylabel('Fused At', fontsize=11)
        axes[0].set_title(f'{vid_id}  (GT: {label})', fontsize=13)
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_axis, mt, 's-', color='steelblue', lw=2, ms=5)
        axes[1].set_ylabel('Temporal mt', fontsize=11)
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t_axis, dt, '^-', color='darkorange', lw=2, ms=5)
        axes[2].set_ylabel('Spatial dt', fontsize=11)
        axes[2].set_xlabel('Snippet index', fontsize=11)
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{vid_id}_{label}.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()

    log.info(f'Visualisations saved to {save_dir}')


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Transforms ───────────────────────────────────────────────────────────
    if 'xception' in args.backbone.lower():
        size = 299
        val_tf = get_xception_transform(size=size, train=False)
    else:
        size = args.image_size
        val_tf = get_val_transform(size=size)

    # ── Dataset ───────────────────────────────────────────────────────────────
    test_ds = DeepFakeSnippetDataset(
        frames_root=args.frames_dir,
        split='val',           # test on the held-out split
        num_snippets=args.num_snippets,
        frames_per_snippet=args.frames_per_snippet,
        transform=val_tf,
        split_ratio=0.0,       # use ALL videos as test set
        annotation_file=args.annotation_file,
        sampling='uniform',
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn,
    )
    log.info(f'Test videos: {len(test_ds)}')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TIPNet(
        backbone_name    = args.backbone,
        backbone_pretrained = False,
        snippet_dim      = args.snippet_dim,
        proj_dim         = args.proj_dim,
        num_prototypes   = args.num_prototypes,
        tmm_order        = args.tmm_order,
        lstm_layers      = args.lstm_layers,
        attention_heads  = args.attention_heads,
    ).to(device)

    load_checkpoint(model, args.checkpoint, strict=True)
    log.info(f'Checkpoint loaded: {args.checkpoint}')

    # ── Inference ────────────────────────────────────────────────────────────
    results = run_inference(model, test_loader, device, args.threshold)

    # ── Video-level metrics ───────────────────────────────────────────────────
    vid_metrics = compute_metrics(
        results['vid_true'], results['vid_pred'], results['vid_prob'])

    log.info('\n' + '=' * 60)
    log.info('Video-level Metrics:')
    for k, v in vid_metrics.items():
        log.info(f'  {k.upper():12s}: {v:.4f}')

    # ── Snippet / Frame localisation metrics ─────────────────────────────────
    snip_gt_arrays = [np.array(g) for g in results['snippet_gt']]
    loc_metrics    = compute_loc_metrics(
        results['anomaly_scores'],
        snip_gt_arrays,
        thresholds=[0.3, 0.5, 0.7],
    )
    log.info('\nLocalisation Metrics:')
    for k, v in loc_metrics.items():
        log.info(f'  {k:20s}: {v:.4f}')
    log.info('=' * 60)

    # ── Per-video results ─────────────────────────────────────────────────────
    per_video = []
    for i, vid_id in enumerate(results['video_ids']):
        entry = {
            'video_id':      vid_id,
            'true_label':    int(results['vid_true'][i]),
            'pred_label':    int(results['vid_pred'][i]),
            'prob_fake':     float(results['vid_prob'][i]),
            'anomaly_scores': results['anomaly_scores'][i].tolist(),
            'snippet_scores': results['snippet_scores'][i].tolist(),
            'temporal_mt':    results['temporal_mt'][i].tolist(),
            'spatial_dt':     results['spatial_dt'][i].tolist(),
        }
        per_video.append(entry)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        'video_metrics':      vid_metrics,
        'localisation':       loc_metrics,
        'per_video_results':  per_video,
    }
    json_path = os.path.join(args.save_dir, 'test_results.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    log.info(f'Results saved → {json_path}')

    # ── Visualisations ────────────────────────────────────────────────────────
    if args.visualize:
        visualise_anomaly_scores(
            results,
            save_dir=os.path.join(args.save_dir, 'visualizations'),
            num_samples=args.num_vis_samples,
        )


if __name__ == '__main__':
    main()