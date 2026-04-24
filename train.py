#!/usr/bin/env python3
"""
train.py  —  Training entry point for TIP-Net.

Usage:
    python train.py --config configs/default.yaml [overrides ...]
    python train.py --data_root ./dataset --backbone xception --epochs 100

Key features:
  - AMP (FP16) for memory efficiency
  - Gradient accumulation (effective batch size)
  - Backbone freeze schedule
  - Periodic prototype update (K-Means)
  - Early stopping
  - Balanced class sampling
  - Training history JSON
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from data.dataset  import DeepFakeSnippetDataset, collate_fn, make_balanced_sampler
from data.transforms import get_train_transform, get_val_transform, get_xception_transform
from model.tipnet  import TIPNet
from losses.losses import TIPNetLoss
from utils.train_utils import (
    AverageMeter, WarmupCosineScheduler,
    compute_metrics, compute_snippet_metrics,
    save_checkpoint, load_checkpoint,
    EarlyStopping, TrainingHistory,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────
def load_config(config_path: str | None, overrides: list[str]) -> dict:
    """Merge YAML config with CLI overrides (key=value pairs)."""
    cfg: dict = {}
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    # Flatten nested cfg for CLI convenience
    flat: dict = {}
    for section, vals in cfg.items():
        if isinstance(vals, dict):
            flat.update(vals)
        else:
            flat[section] = vals
    # Apply CLI overrides
    for ov in overrides:
        if '=' in ov:
            k, v = ov.split('=', 1)
            # Try type coercion
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    if v.lower() in ('true', 'false'):
                        v = v.lower() == 'true'
            flat[k] = v
    return flat


def parse_args():
    p = argparse.ArgumentParser('TIP-Net Training')
    p.add_argument('--config', type=str, default='configs/default.yaml')
    # Shortcuts for common args
    p.add_argument('--data_root',        type=str)
    p.add_argument('--frames_dir',       type=str)
    p.add_argument('--backbone',         type=str)
    p.add_argument('--epochs',           type=int)
    p.add_argument('--batch_size',       type=int)
    p.add_argument('--lr',               type=float)
    p.add_argument('--snippet_dim',      type=int)
    p.add_argument('--num_snippets',     type=int)
    p.add_argument('--frames_per_snippet', type=int)
    p.add_argument('--num_prototypes',   type=int)
    p.add_argument('--tmm_order',        type=int)
    p.add_argument('--gamma1',           type=float)
    p.add_argument('--gamma2',           type=float)
    p.add_argument('--gamma3',           type=float)
    p.add_argument('--top_k',            type=int)
    p.add_argument('--save_dir',         type=str)
    p.add_argument('--log_dir',          type=str)
    p.add_argument('--resume',           type=str)
    p.add_argument('--max_videos',       type=int)
    p.add_argument('--num_workers',      type=int)
    p.add_argument('--no_amp',           action='store_true')
    # Pass remaining args as key=value overrides
    known, extras = p.parse_known_args()
    return known, extras


# ──────────────────────────────────────────────────────────────────────────────
# Build helpers
# ──────────────────────────────────────────────────────────────────────────────
def build_transforms(cfg: dict, backbone_name: str):
    size = cfg.get('image_size', 224)
    if 'xception' in backbone_name.lower():
        size = 299
        train_tf = get_xception_transform(size=size, train=True)
        val_tf   = get_xception_transform(size=size, train=False)
    else:
        train_tf = get_train_transform(size=size)
        val_tf   = get_val_transform(size=size)
    return train_tf, val_tf


def build_datasets(cfg: dict, train_tf, val_tf):
    frames_dir  = cfg.get('frames_dir', cfg.get('data_root', './frames'))
    T           = cfg.get('num_snippets', 16)
    L           = cfg.get('frames_per_snippet', 8)
    split_ratio = cfg.get('split_ratio', 0.8)
    max_videos  = cfg.get('max_videos', None)

    train_ds = DeepFakeSnippetDataset(
        frames_root=frames_dir,
        split='train',
        num_snippets=T,
        frames_per_snippet=L,
        transform=train_tf,
        split_ratio=split_ratio,
        max_videos=max_videos,
        sampling='uniform',
    )
    val_ds = DeepFakeSnippetDataset(
        frames_root=frames_dir,
        split='val',
        num_snippets=T,
        frames_per_snippet=L,
        transform=val_tf,
        split_ratio=split_ratio,
        max_videos=max_videos,
        sampling='uniform',
    )
    return train_ds, val_ds


def build_loaders(train_ds, val_ds, cfg: dict):
    bs          = cfg.get('batch_size', 8)
    num_workers = cfg.get('num_workers', 4)
    balance     = cfg.get('balance_dataset', True)

    if balance and len(train_ds) > 0:
        sampler = make_balanced_sampler(train_ds)
        train_loader = DataLoader(
            train_ds, batch_size=bs, sampler=sampler,
            num_workers=num_workers, pin_memory=cfg.get('pin_memory', True),
            collate_fn=collate_fn, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            num_workers=num_workers, pin_memory=cfg.get('pin_memory', True),
            collate_fn=collate_fn, drop_last=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=max(bs // 2, 1), shuffle=False,
        num_workers=num_workers, pin_memory=cfg.get('pin_memory', True),
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def build_model(cfg: dict, device) -> TIPNet:
    model = TIPNet(
        backbone_name       = cfg.get('backbone', cfg.get('name', 'xception')),
        backbone_pretrained = cfg.get('pretrained', True),
        snippet_dim         = cfg.get('snippet_dim', 512),
        proj_dim            = cfg.get('proj_dim', 128),
        num_prototypes      = cfg.get('num_prototypes', 8),
        memory_size         = cfg.get('memory_size', 4096),
        tmm_order           = cfg.get('tmm_order', 2),
        lstm_layers         = cfg.get('lstm_layers', 2),
        attention_heads     = cfg.get('attention_heads', 4),
        temperature         = cfg.get('temperature', 0.07),
        dropout             = cfg.get('dropout', 0.3),
        frame_chunk         = cfg.get('frame_chunk', 64),
    )
    return model.to(device)


def build_criterion(cfg: dict) -> TIPNetLoss:
    return TIPNetLoss(
        gamma1          = cfg.get('gamma1', 0.5),
        gamma2          = cfg.get('gamma2', 1.0),
        gamma3          = cfg.get('gamma3', 0.8),
        gamma4          = cfg.get('gamma4', 0.3),
        top_k           = cfg.get('top_k', 5),
        temperature     = cfg.get('temperature', 0.07),
        label_smoothing = cfg.get('label_smoothing', 0.05),
        use_contrastive = cfg.get('enable', True),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Epoch loops
# ──────────────────────────────────────────────────────────────────────────────
def _forward_batch(model, batch, device, is_training=True):
    """Move batch to device and run forward pass."""
    snippets = batch['snippets'].to(device, non_blocking=True)
    labels   = batch['labels'].to(device,   non_blocking=True)
    outputs  = model(snippets, labels, is_training=is_training)
    # Attach prototype tensors so the loss function can access them
    outputs['real_prototypes'] = model.pcl.real_prototypes
    outputs['aux_prototypes']  = model.pcl.aux_prototypes
    return outputs, labels


def train_one_epoch(
    model, loader, criterion, optimizer, scheduler,
    scaler, device, cfg, epoch
):
    model.train()
    meters = {k: AverageMeter()
              for k in ('total', 'cls', 'sim', 'proto', 'loc', 'ecl')}

    y_true, y_pred, y_prob = [], [], []
    snip_true, snip_pred = [], []

    accum_steps = cfg.get('accumulate_grad', 1)
    use_amp     = cfg.get('amp', True) and scaler is not None
    grad_clip   = cfg.get('grad_clip', 1.0)

    optimizer.zero_grad()
    t0 = time.time()

    for step, batch in enumerate(tqdm(loader, desc=f'Train {epoch}', leave=False)):
        with autocast(enabled=use_amp):
            outputs, labels = _forward_batch(model, batch, device, is_training=True)
            loss, loss_dict = criterion(outputs, labels)
            loss = loss / accum_steps   # scale for accumulation

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Update meters
        B = labels.shape[0]
        for k, v in loss_dict.items():
            if k in meters:
                meters[k].update(v, B)

        # Collect predictions for metrics
        with torch.no_grad():
            probs = torch.softmax(outputs['video_logits'], dim=1)[:, 1]
            preds = (probs > 0.5).long()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

            # Snippet-level (use snippet_scores)
            ss = (outputs['snippet_scores'] > 0.5).long()
            snip_true.extend(labels.cpu().tolist())
            snip_pred.extend(ss.cpu().tolist())

    elapsed = time.time() - t0
    metrics = compute_metrics(y_true, y_pred, y_prob)

    log.info(
        f'[Train {epoch}] '
        f'loss={meters["total"].avg:.4f} '
        f'cls={meters["cls"].avg:.4f} '
        f'sim={meters["sim"].avg:.4f} '
        f'proto={meters["proto"].avg:.4f} '
        f'loc={meters["loc"].avg:.4f} '
        f'| ACC={metrics["acc"]:.4f} AUC={metrics["auc"]:.4f} '
        f'F1={metrics["f1"]:.4f} '
        f'| mem={model.get_memory_status()} '
        f'| {elapsed:.0f}s'
    )

    return {**metrics, **{f'loss_{k}': v.avg for k, v in meters.items()}}


@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    meters = {k: AverageMeter()
              for k in ('total', 'cls', 'sim', 'proto', 'loc')}

    y_true, y_pred, y_prob = [], [], []
    snip_scores_all, snip_gt_all = [], []

    for batch in tqdm(loader, desc='Val', leave=False):
        outputs, labels = _forward_batch(model, batch, device, is_training=False)
        _, loss_dict = criterion(outputs, labels)

        B = labels.shape[0]
        for k, v in loss_dict.items():
            if k in meters:
                meters[k].update(v, B)

        probs = torch.softmax(outputs['video_logits'], dim=1)[:, 1]
        preds = (probs > 0.5).long()
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())

        snip_scores_all.extend(outputs['anomaly_scores'].cpu().numpy())
        snip_gt_all.extend(batch['snippet_gt'].numpy())

    metrics = compute_metrics(y_true, y_pred, y_prob)
    log.info(
        f'[Val]  loss={meters["total"].avg:.4f} '
        f'| ACC={metrics["acc"]:.4f} AUC={metrics["auc"]:.4f} '
        f'Prec={metrics["precision"]:.4f} Rec={metrics["recall"]:.4f} '
        f'F1={metrics["f1"]:.4f}'
    )

    return {**metrics, **{f'val_loss_{k}': v.avg for k, v in meters.items()}}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args, extras = parse_args()
    cfg = load_config(args.config, extras)

    # CLI shortcuts override config
    cli_map = {
        'data_root':          args.data_root,
        'frames_dir':         args.frames_dir,
        'name':               args.backbone,
        'epochs':             args.epochs,
        'batch_size':         args.batch_size,
        'lr':                 args.lr,
        'snippet_dim':        args.snippet_dim,
        'num_snippets':       args.num_snippets,
        'frames_per_snippet': args.frames_per_snippet,
        'num_prototypes':     args.num_prototypes,
        'tmm_order':          args.tmm_order,
        'gamma1':             args.gamma1,
        'gamma2':             args.gamma2,
        'gamma3':             args.gamma3,
        'top_k':              args.top_k,
        'save_dir':           args.save_dir,
        'log_dir':            args.log_dir,
        'max_videos':         args.max_videos,
        'num_workers':        args.num_workers,
    }
    for k, v in cli_map.items():
        if v is not None:
            cfg[k] = v
    if args.no_amp:
        cfg['amp'] = False

    # Resolve frames_dir
    if 'frames_dir' not in cfg:
        cfg['frames_dir'] = cfg.get('data_root', './frames')

    save_dir = cfg.get('save_dir', './checkpoints')
    log_dir  = cfg.get('log_dir',  './logs')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    # File handler for training log
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setLevel(logging.INFO)
    log.addHandler(fh)

    log.info(f'Config: {cfg}')

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    # ── Data ─────────────────────────────────────────────────────────────────
    backbone_name = cfg.get('backbone', cfg.get('name', 'xception'))
    train_tf, val_tf = build_transforms(cfg, backbone_name)
    train_ds, val_ds = build_datasets(cfg, train_tf, val_tf)
    train_loader, val_loader = build_loaders(train_ds, val_ds, cfg)

    log.info(f'Train batches: {len(train_loader)}  Val batches: {len(val_loader)}')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg, device)
    log.info(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Optional: freeze backbone for first N epochs
    freeze_epochs = cfg.get('freeze_epochs', 0)
    if freeze_epochs > 0:
        model.freeze_backbone(True)
        log.info(f'Backbone frozen for first {freeze_epochs} epochs')

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    lr           = float(cfg.get('lr', 1e-4))
    min_lr       = float(cfg.get('min_lr', 1e-6))
    weight_decay = float(cfg.get('weight_decay', 1e-4))
    epochs       = int(cfg.get('epochs', 100))
    warmup_ep    = int(cfg.get('warmup_epochs', 5))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay)

    total_steps  = epochs * len(train_loader)
    warmup_steps = warmup_ep * len(train_loader)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps, total_steps, lr, min_lr)

    use_amp = cfg.get('amp', True) and torch.cuda.is_available()
    scaler  = GradScaler() if use_amp else None

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion = build_criterion(cfg)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(model, args.resume, optimizer)
        log.info(f'Resumed from epoch {start_epoch}')

    # ── Training loop ─────────────────────────────────────────────────────────
    history  = TrainingHistory(os.path.join(log_dir, 'history.json'))
    stopper  = EarlyStopping(patience=cfg.get('patience', 15), mode='max')
    best_auc = 0.0

    update_proto_freq = int(cfg.get('update_proto_freq', 5))
    eval_freq         = int(cfg.get('eval_freq', 1))

    for epoch in range(start_epoch + 1, epochs + 1):
        # Unfreeze backbone after freeze_epochs
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            model.freeze_backbone(False)
            # Reinit optimizer with all parameters
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr * 0.1, weight_decay=weight_decay)
            log.info('Backbone unfrozen')

        # ── Train ────────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, cfg, epoch)

        # ── Prototype update ─────────────────────────────────────────────────
        if epoch % update_proto_freq == 0:
            updated = model.update_prototypes()
            if updated:
                log.info(f'[Epoch {epoch}] Prototypes updated via K-Means')

        # ── Validation ───────────────────────────────────────────────────────
        val_metrics = {}
        if epoch % eval_freq == 0:
            val_metrics = validate(model, val_loader, criterion, device, cfg)
            val_auc     = val_metrics.get('auc', 0.0)

            # Save best
            is_best = val_auc > best_auc
            if is_best:
                best_auc = val_auc
            save_checkpoint(
                model, optimizer, epoch,
                os.path.join(save_dir, f'checkpoint_ep{epoch:03d}.pth'),
                is_best=is_best,
                val_auc=val_auc,
            )

            # Early stopping
            if stopper(val_auc):
                log.info(f'Early stopping at epoch {epoch}')
                break

        # ── History ──────────────────────────────────────────────────────────
        history.update(
            epoch=epoch,
            lr=scheduler.get_lr(),
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}':   v for k, v in val_metrics.items()},
        )
        history.save()

    log.info(f'Training complete. Best val AUC: {best_auc:.4f}')


if __name__ == '__main__':
    main()