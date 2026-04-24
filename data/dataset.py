"""
data/dataset.py  —  Dataset classes for TIP-Net.

Supports:
  - Frame folder layout (from preprocess.py)
  - Variable-length video → fixed-T snippet sequences
  - Uniform / random / dense frame sampling strategies
  - Optional per-video snippet-level ground-truth (for evaluation)
"""

from __future__ import annotations

import os
import json
import logging
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: load one snippet (L frames → stacked tensor [L, C, H, W])
# ──────────────────────────────────────────────────────────────────────────────
def _load_snippet(frame_paths: List[str], transform) -> torch.Tensor:
    """Load L frames, apply transform, return [L, C, H, W]."""
    frames = []
    for fp in frame_paths:
        try:
            img = Image.open(fp).convert('RGB')
            if transform is not None:
                img = transform(img)
            frames.append(img)
        except Exception as e:
            log.debug(f'Failed to load {fp}: {e}')
            # Fallback: zero frame matching the transform output shape
            if frames:
                frames.append(torch.zeros_like(frames[0]))
            else:
                frames.append(torch.zeros(3, 224, 224))
    return torch.stack(frames, dim=0)  # [L, C, H, W]


# ──────────────────────────────────────────────────────────────────────────────
# Core dataset
# ──────────────────────────────────────────────────────────────────────────────
class DeepFakeSnippetDataset(Dataset):
    """
    Video snippet dataset for weakly supervised DeepFake detection.

    Each item returns:
        snippets   : Tensor [T, L, C, H, W]   — snippet sequence
        label      : int                        — video-level label (0=real, 1=fake)
        video_id   : str
        num_frames : int                        — actual frames in the video

    T (num_snippets) and L (frames_per_snippet) are configurable.
    Frame sampling within each snippet: uniformly from the snippet window.
    """

    def __init__(
        self,
        frames_root: str,
        split: str = 'train',
        num_snippets: int = 16,
        frames_per_snippet: int = 8,
        transform=None,
        split_ratio: float = 0.8,
        seed: int = 42,
        max_videos: Optional[int] = None,
        annotation_file: Optional[str] = None,
        sampling: str = 'uniform',      # 'uniform' | 'random' | 'dense'
        classes: Tuple[str, ...] = ('real', 'fake'),
        min_frames: int = 8,
    ):
        super().__init__()
        self.num_snippets      = num_snippets
        self.frames_per_snippet = frames_per_snippet
        self.transform         = transform
        self.split             = split
        self.sampling          = sampling

        # ── Load per-video frame annotation if available ──────────────────────
        self.frame_labels: Dict[str, List[int]] = {}
        if annotation_file and os.path.exists(annotation_file):
            with open(annotation_file) as f:
                ann = json.load(f)
            # Accept two formats:
            #   {"video_id": [0,0,1,1,...]}
            #   {"video_id": {"frames": [0,0,1,1,...]}}
            for vid, val in ann.items():
                if isinstance(val, dict):
                    self.frame_labels[vid] = val.get('frames', [])
                else:
                    self.frame_labels[vid] = val

        # ── Scan video frame folders ─────────────────────────────────────────
        all_videos: List[Dict[str, Any]] = []
        frames_root = Path(frames_root)

        for cls_name in classes:
            label = 0 if cls_name == 'real' else 1
            cls_dir = frames_root / cls_name
            if not cls_dir.exists():
                log.warning(f'Class dir not found: {cls_dir}')
                continue

            # Each sub-folder is one video
            for vid_dir in sorted(cls_dir.iterdir()):
                if not vid_dir.is_dir():
                    continue
                frame_files = sorted(
                    [f for f in vid_dir.iterdir()
                     if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
                )
                if len(frame_files) < min_frames:
                    continue
                all_videos.append({
                    'path':       vid_dir,
                    'label':      label,
                    'video_id':   vid_dir.name,
                    'frame_files': frame_files,
                })

        if max_videos:
            all_videos = all_videos[:max_videos]

        # ── Train / val split ─────────────────────────────────────────────────
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(all_videos))
        n_train = int(len(all_videos) * split_ratio)
        if split == 'train':
            self.videos = [all_videos[i] for i in idx[:n_train]]
        else:
            self.videos = [all_videos[i] for i in idx[n_train:]]

        n_real = sum(1 for v in self.videos if v['label'] == 0)
        n_fake = sum(1 for v in self.videos if v['label'] == 1)
        log.info(f'[{split}] {len(self.videos)} videos | real={n_real} fake={n_fake}')

    def __len__(self) -> int:
        return len(self.videos)

    def _sample_frames_for_snippet(
        self,
        frame_files: List[Path],
        snippet_idx: int,
        num_snippets: int,
    ) -> List[Path]:
        """
        Return self.frames_per_snippet frame paths for snippet_idx.

        Splits the video into num_snippets windows, then picks L frames
        from the window using the configured sampling strategy.
        """
        N = len(frame_files)
        L = self.frames_per_snippet
        T = num_snippets

        # Window boundaries for this snippet
        window_size  = N / T
        start        = int(snippet_idx * window_size)
        end          = int((snippet_idx + 1) * window_size)
        end          = min(end, N)
        window       = frame_files[start:end]

        if len(window) == 0:
            # Fallback: repeat last frame
            window = [frame_files[-1]]

        if self.sampling == 'uniform':
            # Evenly spaced L frames within the window
            picks = np.linspace(0, len(window) - 1, L, dtype=int)
        elif self.sampling == 'random':
            picks = sorted(random.choices(range(len(window)), k=L))
        elif self.sampling == 'dense':
            # Take L consecutive frames from a random start
            if len(window) >= L:
                s = random.randint(0, len(window) - L)
                picks = list(range(s, s + L))
            else:
                picks = np.linspace(0, len(window) - 1, L, dtype=int)
        else:
            picks = np.linspace(0, len(window) - 1, L, dtype=int)

        return [window[p] for p in picks]

    def __getitem__(self, idx: int):
        info        = self.videos[idx]
        frame_files = info['frame_files']
        label       = info['label']
        video_id    = info['video_id']
        T           = self.num_snippets

        snippets = []
        for t in range(T):
            fp = self._sample_frames_for_snippet(frame_files, t, T)
            snippets.append(_load_snippet([str(p) for p in fp], self.transform))

        # [T, L, C, H, W]
        snippets_tensor = torch.stack(snippets, dim=0)

        # Optional snippet-level ground-truth (for evaluation only)
        if video_id in self.frame_labels:
            raw_labels = np.array(self.frame_labels[video_id])
            N = len(frame_files)
            window_size = N / T
            snippet_gt = []
            for t in range(T):
                s = int(t * window_size)
                e = min(int((t + 1) * window_size), N)
                chunk = raw_labels[s:e]
                # Majority vote
                snippet_gt.append(int(chunk.mean() >= 0.5) if len(chunk) > 0 else label)
            snippet_gt = torch.tensor(snippet_gt, dtype=torch.long)
        else:
            # Use video label for all snippets (bag assumption)
            snippet_gt = torch.full((T,), label, dtype=torch.long)

        return {
            'snippets':   snippets_tensor,            # [T, L, C, H, W]
            'label':      torch.tensor(label, dtype=torch.long),
            'snippet_gt': snippet_gt,                  # [T]
            'video_id':   video_id,
            'num_frames': len(frame_files),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Collate function (handles variable lengths if needed — here T is fixed)
# ──────────────────────────────────────────────────────────────────────────────
def collate_fn(batch: List[Dict]) -> Dict:
    """Stack batch items into model-ready tensors."""
    snippets   = torch.stack([b['snippets']   for b in batch], dim=0)  # [B,T,L,C,H,W]
    labels     = torch.stack([b['label']      for b in batch], dim=0)  # [B]
    snippet_gt = torch.stack([b['snippet_gt'] for b in batch], dim=0)  # [B,T]
    video_ids  = [b['video_id']  for b in batch]
    num_frames = [b['num_frames'] for b in batch]

    return {
        'snippets':   snippets,
        'labels':     labels,
        'snippet_gt': snippet_gt,
        'video_ids':  video_ids,
        'num_frames': num_frames,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Balanced sampler helper
# ──────────────────────────────────────────────────────────────────────────────
def make_balanced_sampler(dataset: DeepFakeSnippetDataset):
    """Weighted random sampler to balance real/fake in training."""
    from torch.utils.data import WeightedRandomSampler

    labels = [v['label'] for v in dataset.videos]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    weights = torch.tensor(weights, dtype=torch.float)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)