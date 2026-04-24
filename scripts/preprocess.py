#!/usr/bin/env python3
"""
preprocess.py  —  Extract frames from raw video files.

Input directory layout expected:
    data_root/
        real/
            video001.mp4
            video002.avi
            ...
        fake/
            video001.mp4
            ...

Output layout (mirrors input, replaces video files with frame folders):
    frames_dir/
        real/
            video001/
                0000.jpg
                0001.jpg
                ...
        fake/
            video001/
                ...

Optional: face detection + crop using facenet-pytorch MTCNN.
"""

import os
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
from tqdm import tqdm

try:
    from facenet_pytorch import MTCNN
    import torch
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}


# ──────────────────────────────────────────────────────────────────────────────
# Face detector (shared across workers via global to avoid re-init overhead)
# ──────────────────────────────────────────────────────────────────────────────
_mtcnn = None

def _get_mtcnn(device='cpu'):
    global _mtcnn
    if _mtcnn is None and MTCNN_AVAILABLE:
        _mtcnn = MTCNN(
            select_largest=True,
            keep_all=False,
            device=device,
            post_process=False,
        )
    return _mtcnn


def crop_face(frame_bgr: np.ndarray, margin: float = 0.3,
              fallback_center: bool = True) -> np.ndarray:
    """Detect and crop face region. Falls back to center crop on failure."""
    if not MTCNN_AVAILABLE:
        return _center_crop(frame_bgr)

    mtcnn = _get_mtcnn()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    try:
        boxes, _ = mtcnn.detect(rgb)
    except Exception:
        boxes = None

    if boxes is None or len(boxes) == 0:
        return _center_crop(frame_bgr) if fallback_center else frame_bgr

    x1, y1, x2, y2 = boxes[0].astype(int)
    h, w = frame_bgr.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    x1 = max(0, x1 - mx);  y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx);  y2 = min(h, y2 + my)
    cropped = frame_bgr[y1:y2, x1:x2]
    return cropped if cropped.size > 0 else frame_bgr


def _center_crop(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return frame[y0:y0+s, x0:x0+s]


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction logic
# ──────────────────────────────────────────────────────────────────────────────
def extract_frames(video_path: Path, out_dir: Path, args) -> dict:
    """
    Extract frames from a single video.

    Strategy:
      - If args.max_frames is given, uniformly sample that many frames.
      - Otherwise extract every N-th frame (stride = args.stride).
      - Optionally apply face crop.
      - Resize to args.size × args.size.
      - Save as JPEG (quality = args.quality).

    Returns metadata dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already done (idempotent)
    existing = list(out_dir.glob('*.jpg'))
    if existing and not args.overwrite:
        return {'video': str(video_path), 'frames': len(existing), 'status': 'skip'}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {'video': str(video_path), 'frames': 0, 'status': 'error_open'}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if args.max_frames and args.max_frames > 0:
        # Uniform sampling: pick exactly max_frames indices
        indices = set(np.linspace(0, max(total - 1, 0),
                                  min(args.max_frames, total),
                                  dtype=int).tolist())
    else:
        # Stride-based sampling
        indices = set(range(0, total, max(1, args.stride)))

    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in indices:
            if args.face_crop:
                frame = crop_face(frame, margin=args.face_margin)

            if args.size:
                frame = cv2.resize(frame, (args.size, args.size),
                                   interpolation=cv2.INTER_LANCZOS4)

            fname = out_dir / f'{saved:05d}.jpg'
            cv2.imwrite(str(fname), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            saved += 1

        frame_idx += 1

    cap.release()
    return {'video': str(video_path), 'frames': saved,
            'total_frames': total, 'fps': fps, 'status': 'ok'}


def _worker(task):
    """Unpack args for multiprocessing."""
    video_path, out_dir, args = task
    try:
        return extract_frames(video_path, out_dir, args)
    except Exception as e:
        return {'video': str(video_path), 'frames': 0, 'status': f'error: {e}'}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='Extract frames from DeepFake video datasets.'
    )
    # I/O
    p.add_argument('--data_root',  type=str, required=True,
                   help='Root dir containing real/ and fake/ sub-folders with videos.')
    p.add_argument('--frames_dir', type=str, required=True,
                   help='Output root dir for extracted frame folders.')
    p.add_argument('--classes', nargs='+', default=['real', 'fake'],
                   help='Sub-folder class names to process.')

    # Sampling
    p.add_argument('--max_frames', type=int, default=None,
                   help='If set, uniformly sample this many frames per video. '
                        'Recommended: 64-256 depending on video length.')
    p.add_argument('--stride', type=int, default=4,
                   help='Frame stride when max_frames is not set (default: every 4th frame).')

    # Processing
    p.add_argument('--size', type=int, default=224,
                   help='Resize frame to size×size. 0 = no resize.')
    p.add_argument('--quality', type=int, default=90,
                   help='JPEG quality (1-100).')
    p.add_argument('--face_crop', action='store_true',
                   help='Detect & crop face region before resize (requires facenet-pytorch).')
    p.add_argument('--face_margin', type=float, default=0.3,
                   help='Margin around face box as fraction of box size.')

    # Misc
    p.add_argument('--num_workers', type=int, default=4,
                   help='Parallel worker processes.')
    p.add_argument('--overwrite', action='store_true',
                   help='Re-extract even if output folder is non-empty.')
    p.add_argument('--min_frames', type=int, default=8,
                   help='Skip videos with fewer extracted frames than this.')

    return p.parse_args()


def main():
    args = parse_args()

    if args.face_crop and not MTCNN_AVAILABLE:
        log.warning('facenet-pytorch not installed; --face_crop has no effect.')

    data_root  = Path(args.data_root)
    frames_dir = Path(args.frames_dir)

    # Collect all video files
    tasks = []
    for cls in args.classes:
        class_in  = data_root  / cls
        class_out = frames_dir / cls

        if not class_in.exists():
            log.warning(f'Class directory not found: {class_in}')
            continue

        # Support both direct videos and one-level sub-folder layout
        video_files = []
        for ext in VIDEO_EXTS:
            video_files.extend(class_in.glob(f'*{ext}'))
            video_files.extend(class_in.glob(f'**/*{ext}'))  # nested
        video_files = sorted(set(video_files))

        if not video_files:
            log.warning(f'No video files found in {class_in}')
            continue

        for vf in video_files:
            # Preserve sub-folder hierarchy in output
            rel = vf.relative_to(class_in)
            stem = rel.with_suffix('')       # remove extension
            out_dir = class_out / stem
            tasks.append((vf, out_dir, args))

        log.info(f'Found {len(video_files)} videos in {cls}/')

    if not tasks:
        log.error('No videos to process. Check --data_root and class folders.')
        return

    log.info(f'Total videos to process: {len(tasks)}')

    # Run extraction
    results = []
    if args.num_workers <= 1:
        for task in tqdm(tasks, desc='Extracting'):
            results.append(_worker(task))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {pool.submit(_worker, t): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc='Extracting'):
                results.append(fut.result())

    # Summary
    ok      = [r for r in results if r['status'] == 'ok']
    skipped = [r for r in results if r['status'] == 'skip']
    errors  = [r for r in results if r['status'] not in ('ok', 'skip')]
    too_few = [r for r in ok if r['frames'] < args.min_frames]

    log.info(f'\n{"="*60}')
    log.info(f'Done. OK: {len(ok)}  Skipped: {len(skipped)}  Errors: {len(errors)}')
    log.info(f'Videos with < {args.min_frames} frames: {len(too_few)}')
    if errors:
        log.warning('Error details:')
        for r in errors[:10]:
            log.warning(f'  {r["video"]}  →  {r["status"]}')
    total_frames = sum(r.get('frames', 0) for r in ok)
    log.info(f'Total frames saved: {total_frames:,}')
    log.info(f'Output: {frames_dir}')


if __name__ == '__main__':
    main()