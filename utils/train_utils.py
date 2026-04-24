"""
utils/train_utils.py  —  Training utilities.
"""

from __future__ import annotations

import math
import os
import json
import logging
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
)

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Running average
# ──────────────────────────────────────────────────────────────────────────────
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count if self.count else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Learning-rate scheduler: warmup → cosine annealing
# ──────────────────────────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    """Step-based LR scheduler (call .step() once per optimiser.step())."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 base_lr: float, min_lr: float = 1e-6):
        self.opt          = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.base_lr      = base_lr
        self.min_lr       = min_lr
        self._step        = 0

    def step(self):
        self._step += 1
        s = self._step
        if s <= self.warmup_steps:
            lr = self.base_lr * s / max(self.warmup_steps, 1)
        else:
            progress = (s - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress))

        for pg in self.opt.param_groups:
            pg['lr'] = lr

    def get_lr(self) -> float:
        if not self.opt.param_groups:
            return 0.0
        return self.opt.param_groups[0]['lr']


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    try:
        acc  = accuracy_score(y_true, y_pred)
        auc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
    except Exception as e:
        log.warning(f'Metric computation error: {e}')
        acc = auc = prec = rec = f1 = 0.0

    return dict(acc=acc, auc=auc, precision=prec, recall=rec, f1=f1)


def compute_snippet_metrics(y_true_vid, y_pred_snip) -> dict:
    """
    y_true_vid: [B]  video-level labels
    y_pred_snip: [B, T] snippet binary predictions
    Expands video label to all snippets, then computes snippet-level metrics.
    """
    B, T = y_pred_snip.shape
    y_true = np.repeat(np.asarray(y_true_vid), T)
    y_pred = np.asarray(y_pred_snip).reshape(-1)
    return compute_metrics(y_true, (y_pred > 0.5).astype(int), y_pred)


# ──────────────────────────────────────────────────────────────────────────────
# Localisation metrics (AP / AR)
# ──────────────────────────────────────────────────────────────────────────────
def compute_ap(scores: np.ndarray, gt: np.ndarray) -> float:
    """Area under Precision-Recall curve (snippet level)."""
    sorted_idx = np.argsort(-scores)
    gt_sorted  = gt[sorted_idx]
    n_pos = gt.sum()
    if n_pos == 0:
        return 0.0
    tp, ap = 0, 0.0
    for i, g in enumerate(gt_sorted):
        tp += g
        if g:
            ap += tp / (i + 1)
    return ap / n_pos


def compute_loc_metrics(anomaly_scores: list[np.ndarray],
                        gt_labels: list[np.ndarray],
                        thresholds=(0.5,)) -> dict:
    """
    Compute localisation metrics averaged over all videos.

    anomaly_scores: list of [T] arrays
    gt_labels:      list of [T] arrays (0/1 per snippet)
    """
    metrics: dict = {}
    for thr in thresholds:
        tp_sum = fp_sum = fn_sum = tn_sum = 0
        for sc, gt in zip(anomaly_scores, gt_labels):
            pred = (sc > thr).astype(int)
            tp_sum += ((pred == 1) & (gt == 1)).sum()
            fp_sum += ((pred == 1) & (gt == 0)).sum()
            fn_sum += ((pred == 0) & (gt == 1)).sum()
            tn_sum += ((pred == 0) & (gt == 0)).sum()

        prec = tp_sum / max(tp_sum + fp_sum, 1)
        rec  = tp_sum / max(tp_sum + fn_sum, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-6)
        metrics[f'loc_prec@{thr}'] = float(prec)
        metrics[f'loc_rec@{thr}']  = float(rec)
        metrics[f'loc_f1@{thr}']   = float(f1)

    aps = [compute_ap(s, g) for s, g in zip(anomaly_scores, gt_labels)]
    metrics['loc_mAP'] = float(np.mean(aps)) if aps else 0.0

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch: int, save_path: str,
                    is_best: bool = False, **extra):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    state = dict(epoch=epoch,
                 state_dict=model.state_dict(),
                 optimizer=optimizer.state_dict(),
                 **extra)
    torch.save(state, save_path)
    if is_best:
        best = save_path.replace('.pth', '_best.pth')
        torch.save(state, best)
        log.info(f'Best checkpoint saved → {best}')


def load_checkpoint(model, checkpoint_path: str,
                    optimizer=None, strict: bool = True) -> int:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    sd   = ckpt.get('state_dict', ckpt)
    model.load_state_dict(sd, strict=strict)
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt.get('epoch', 0)
    log.info(f'Loaded checkpoint from epoch {epoch}: {checkpoint_path}')
    return epoch


# ──────────────────────────────────────────────────────────────────────────────
# Early stopping
# ──────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.0,
                 mode: str = 'max'):
        self.patience    = patience
        self.min_delta   = min_delta
        self.mode        = mode
        self.counter     = 0
        self.best_score  = None
        self.early_stop  = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (score > self.best_score + self.min_delta
                    if self.mode == 'max'
                    else score < self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# ──────────────────────────────────────────────────────────────────────────────
# Training history logger
# ──────────────────────────────────────────────────────────────────────────────
class TrainingHistory:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.history: dict[str, list] = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.history.setdefault(k, []).append(
                float(v) if isinstance(v, (int, float)) else v)

    def save(self):
        os.makedirs(os.path.dirname(self.save_path) or '.', exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(self.history, f, indent=2)