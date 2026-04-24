"""
model/tipnet.py  —  TIP-Net: Temporal Inconsistency & Prototype Contrastive Network.

Architecture overview (memory-efficient design):
  [Video: T snippets × L frames]
         ↓  Frame encoder (backbone, shared weights)
  [Snippet features: B×T×D after mean-pooling over L]
         ↓  Temporal Refinement Head (1D Conv)
  ┌──────┬──────────┬────────────────────────────────┐
  │ TMM  │ PCL      │ MSA-L                           │
  │      │          │                                  │
  │ mt   │ zt, dt   │ ht, αt, At                      │
  └──────┴──────────┴────────────────────────────────┘
         ↓  MIL aggregation (attention-weighted)
  [Video logit]  +  [Snippet logits]

Memory optimisations:
  - Frames are encoded in micro-batches inside forward() to avoid OOM.
  - Backbone can be frozen.
  - Mixed precision (AMP) is handled outside in the training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

from .backbone import build_backbone


# ══════════════════════════════════════════════════════════════════════════════
# 1. Temporal Refinement Head  (lightweight 1-D conv residual block)
# ══════════════════════════════════════════════════════════════════════════════
class TemporalRefinementHead(nn.Module):
    """
    Applies two 1-D conv layers with a residual connection over the
    snippet sequence to suppress content fluctuations while preserving
    semantically meaningful temporal variations.

    Input / output: [B, T, D]
    """

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        p = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=p),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size, padding=p),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        z = x.transpose(1, 2)          # [B, D, T]
        z = self.act(self.net(z) + z)  # residual
        return z.transpose(1, 2)       # [B, T, D]


# ══════════════════════════════════════════════════════════════════════════════
# 2. Temporal Inconsistency Mining (TIM / TMM)
# ══════════════════════════════════════════════════════════════════════════════
class TemporalInconsistencyMining(nn.Module):
    """
    Computes the k-th order discrete temporal difference of inter-snippet
    cosine similarities, then normalises per video.

    Returns:
        mt   [B, T]   normalised temporal inconsistency prior
        sim  [B, T-1] raw cosine similarities between adjacent snippets
    """

    def __init__(self, order: int = 2):
        super().__init__()
        self.order = order  # k in the paper (ablation target)

    def _kth_difference(self, s: torch.Tensor, k: int) -> torch.Tensor:
        """
        s: [B, T-1] similarity sequence
        Returns: [B, T-1-k] k-th order difference using the discrete Laplacian
        """
        delta = s
        for _ in range(k):
            delta = delta[:, 1:] - delta[:, :-1]
        return delta

    def forward(self, x: torch.Tensor):
        """x: [B, T, D] ℓ2-normalised snippet embeddings"""
        B, T, D = x.shape
        x_n = F.normalize(x, dim=2)

        # Adjacent cosine similarities  [B, T-1]
        sim = (x_n[:, :-1] * x_n[:, 1:]).sum(dim=2)

        if T <= 2 or self.order == 0:
            m_raw = torch.zeros(B, T, device=x.device)
        else:
            delta = self._kth_difference(sim, self.order)  # [B, T-1-k]
            m_abs = delta.abs()
            # Pad back to length T
            pad_front = self.order // 2 + 1
            pad_back  = T - m_abs.shape[1] - pad_front
            pad_back  = max(pad_back, 0)
            m_raw = F.pad(m_abs, (pad_front, pad_back), value=0.0)[:, :T]

        # Per-video min-max normalisation
        m_min = m_raw.min(dim=1, keepdim=True)[0]
        m_max = m_raw.max(dim=1, keepdim=True)[0]
        mt = (m_raw - m_min) / (m_max - m_min + 1e-6)

        return mt, sim  # [B,T], [B,T-1]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Prototype-guided Contrastive Learning (PCL)
# ══════════════════════════════════════════════════════════════════════════════
class PrototypeGuidedContrastive(nn.Module):
    """
    Models the real feature manifold via K prototypes obtained by
    K-Means clustering over a dynamic FIFO memory bank of real snippets.

    Projection head: Linear → LayerNorm → ReLU → Linear → LayerNorm → ℓ2-norm

    Returns:
        zt  [B, T, proj_dim]   projected & normalised snippet embeddings
        dt  [B, T]             spatial anomaly score (deviation from real manifold)
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 128,
        num_prototypes: int = 8,
        temperature: float = 0.07,
        memory_size: int = 4096,
    ):
        super().__init__()
        self.proj_dim       = proj_dim
        self.num_prototypes = num_prototypes
        self.temperature    = temperature
        self.memory_size    = memory_size

        # ── Projection head ─────────────────────────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # ── Memory bank (FIFO, non-trainable buffers) ────────────────────────
        self.register_buffer('memory_bank',
                             F.normalize(torch.randn(memory_size, proj_dim), dim=1))
        self.register_buffer('memory_ptr',    torch.zeros(1, dtype=torch.long))
        self.register_buffer('memory_filled', torch.zeros(1, dtype=torch.long))

        # ── Real prototypes ──────────────────────────────────────────────────
        self.register_buffer('real_prototypes',
                             F.normalize(torch.randn(num_prototypes, proj_dim), dim=1))

        # Small set of auxiliary learnable prototypes
        self.aux_prototypes = nn.Parameter(
            F.normalize(torch.randn(num_prototypes // 2 + 1, proj_dim), dim=1))

    # ── Memory update (called during training, no grad) ─────────────────────
    @torch.no_grad()
    def update_memory(
        self,
        zt: torch.Tensor,    # [B, T, proj_dim] already normalised
        mt: torch.Tensor,    # [B, T] temporal prior
        labels: torch.Tensor,  # [B]
        threshold: float = 0.3,
    ):
        """Push reliable real snippets (low mutation score) into the bank."""
        B, T, D = zt.shape
        zt_flat = zt.reshape(-1, D)                          # [B*T, D]
        mt_flat = mt.reshape(-1)                             # [B*T]
        lbl_exp = labels.unsqueeze(1).expand(-1, T).reshape(-1)  # [B*T]

        mask = (lbl_exp == 0) & (mt_flat < threshold)
        if mask.sum() == 0:
            return

        selected = F.normalize(zt_flat[mask], dim=1)
        n = min(selected.shape[0], self.memory_size)
        ptr = int(self.memory_ptr)

        if ptr + n > self.memory_size:
            tail = self.memory_size - ptr
            self.memory_bank[ptr:]      = selected[:tail]
            self.memory_bank[:n - tail] = selected[tail:n]
            self.memory_ptr[0]          = n - tail
        else:
            self.memory_bank[ptr:ptr + n] = selected[:n]
            self.memory_ptr[0]            = (ptr + n) % self.memory_size

        self.memory_filled[0] = min(int(self.memory_filled) + n, self.memory_size)

    @torch.no_grad()
    def update_prototypes(self):
        """K-Means on the memory bank to refresh real prototypes."""
        filled = int(self.memory_filled)
        if filled < self.num_prototypes * 10:
            return False

        feats_np = self.memory_bank[:filled].cpu().float().numpy()
        try:
            km = KMeans(n_clusters=self.num_prototypes,
                        random_state=42, n_init=10, max_iter=300)
            km.fit(feats_np)
            centers = torch.from_numpy(km.cluster_centers_).to(
                self.memory_bank.device, dtype=self.memory_bank.dtype)
            self.real_prototypes.copy_(F.normalize(centers, dim=1))
            return True
        except Exception as e:
            print(f'[PCL] Prototype update failed: {e}')
            return False

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """x: [B, T, D]"""
        B, T, D = x.shape

        zt = self.proj(x)                          # [B, T, proj_dim]
        zt = F.normalize(zt, dim=2)

        # Similarity to real prototypes
        # real_prototypes: [K, proj_dim]
        sim_to_real = torch.einsum(
            'btd,kd->btk', zt, self.real_prototypes)   # [B, T, K]
        max_sim = sim_to_real.max(dim=2)[0]             # [B, T]
        dt = 1.0 - max_sim                              # [B, T]

        return zt, dt


# ══════════════════════════════════════════════════════════════════════════════
# 4. Multi-Source Attention and Bootstrapping Localisation (MSA-L / MAB)
# ══════════════════════════════════════════════════════════════════════════════
class MultiSourceAttentionLocalisation(nn.Module):
    """
    1. Bi-LSTM for contextual temporal modelling.
    2. Multi-source attention: fuses ht (context), mt (temporal prior),
       dt (spatial anomaly) to produce attention weights αt.
    3. Bootstrapping: fused anomaly score At = λαt + (1-λ)mt.

    Returns:
        ht    [B, T, D]  contextualised snippet features
        alpha [B, T]     attention weights
        At    [B, T]     fused anomaly scores (for pseudo-label generation)
    """

    def __init__(
        self,
        dim: int,
        lstm_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.3,
        lam: float = 0.5,      # λ in At = λαt + (1-λ)mt
    ):
        super().__init__()
        self.lam = lam

        # Bi-LSTM (outputs dim because bidirectional doubles hidden_size)
        assert dim % 2 == 0, 'snippet_dim must be even for Bi-LSTM'
        self.bilstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_drop = nn.Dropout(p=dropout)

        # Multi-head self-attention over contextual features
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Attention scoring: combines ht, mt, dt
        self.W_h = nn.Linear(dim, dim)
        self.W_m = nn.Linear(1, dim, bias=False)
        self.W_d = nn.Linear(1, dim, bias=False)
        self.v   = nn.Linear(dim, 1, bias=False)

        self.drop = nn.Dropout(p=dropout)

    def forward(
        self,
        x:  torch.Tensor,   # [B, T, D]
        mt: torch.Tensor,   # [B, T]
        dt: torch.Tensor,   # [B, T]
    ):
        # ── Contextual encoding ───────────────────────────────────────────
        ht, _ = self.bilstm(x)                     # [B, T, D]
        ht    = self.lstm_drop(ht)

        # Multi-head self-attention to model long-range dependencies
        ht_att, _ = self.mha(ht, ht, ht)          # [B, T, D]
        ht = ht + ht_att                            # residual

        # ── Attention score ───────────────────────────────────────────────
        mt_e = mt.unsqueeze(2)   # [B, T, 1]
        dt_e = dt.unsqueeze(2)   # [B, T, 1]

        et = torch.tanh(
            self.W_h(ht) +
            self.W_m(mt_e) +
            self.W_d(dt_e)
        )                                           # [B, T, D]
        et = self.v(self.drop(et)).squeeze(2)       # [B, T]
        alpha = F.softmax(et, dim=1)                # [B, T]

        # ── Fused anomaly score ───────────────────────────────────────────
        At = self.lam * alpha + (1.0 - self.lam) * mt  # [B, T]

        return ht, alpha, At


# ══════════════════════════════════════════════════════════════════════════════
# 5. Snippet Prediction Head
# ══════════════════════════════════════════════════════════════════════════════
class SnippetPredictionHead(nn.Module):
    """Lightweight head: contextualised snippet → manipulation probability."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(dim // 2, 1),   # binary: real / fake
            nn.Sigmoid(),
        )

    def forward(self, ht: torch.Tensor) -> torch.Tensor:
        """ht: [B, T, D] → [B, T] ∈ (0,1)"""
        return self.net(ht).squeeze(2)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Video-level Classifier
# ══════════════════════════════════════════════════════════════════════════════
class VideoClassifier(nn.Module):
    def __init__(self, dim: int, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(dim // 2, num_classes),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """v: [B, D] → [B, 2]"""
        return self.net(v)


# ══════════════════════════════════════════════════════════════════════════════
# 7. TIP-Net  (full model)
# ══════════════════════════════════════════════════════════════════════════════
class TIPNet(nn.Module):
    """
    Complete TIP-Net model.

    Key design choices for memory efficiency:
      - Backbone is shared across all B×T×L frames but processed in chunks.
      - snippet_proj converts backbone output to a smaller snippet_dim.
      - All per-snippet operations are O(B×T×D), not O(B×T×L×C×H×W).
      - Memory bank and prototypes are CPU↔GPU managed.

    Output dict keys (see forward):
        video_logits   [B, 2]
        snippet_scores [B, T]   ∈(0,1)
        attention      [B, T]
        anomaly_scores [B, T]   fused At
        temporal_mt    [B, T]
        spatial_dt     [B, T]
        proj_features  [B, T, proj_dim]
        context_feats  [B, T, snippet_dim]
    """

    def __init__(
        self,
        backbone_name:   str   = 'xception',
        backbone_pretrained: bool = True,
        snippet_dim:     int   = 512,
        proj_dim:        int   = 128,
        num_prototypes:  int   = 8,
        memory_size:     int   = 4096,
        tmm_order:       int   = 2,
        lstm_layers:     int   = 2,
        attention_heads: int   = 4,
        temperature:     float = 0.07,
        dropout:         float = 0.3,
        frame_chunk:     int   = 64,   # process at most N frames at once
    ):
        super().__init__()
        self.frame_chunk = frame_chunk

        # ── Frame encoder ────────────────────────────────────────────────────
        self.backbone, backbone_dim = build_backbone(
            backbone_name, pretrained=backbone_pretrained, dropout=0.0)

        # ── Snippet projection (backbone_dim → snippet_dim) ──────────────────
        self.snippet_proj = nn.Sequential(
            nn.Linear(backbone_dim, snippet_dim),
            nn.LayerNorm(snippet_dim),
            nn.ReLU(inplace=True),
        )

        # ── Temporal refinement ──────────────────────────────────────────────
        self.temporal_refine = TemporalRefinementHead(snippet_dim)

        # ── TIM ─────────────────────────────────────────────────────────────
        self.tim = TemporalInconsistencyMining(order=tmm_order)

        # ── PCL ─────────────────────────────────────────────────────────────
        self.pcl = PrototypeGuidedContrastive(
            in_dim=snippet_dim,
            proj_dim=proj_dim,
            num_prototypes=num_prototypes,
            temperature=temperature,
            memory_size=memory_size,
        )

        # ── MSA-L ────────────────────────────────────────────────────────────
        self.msal = MultiSourceAttentionLocalisation(
            dim=snippet_dim,
            lstm_layers=lstm_layers,
            attention_heads=attention_heads,
            dropout=dropout,
        )

        # ── Heads ────────────────────────────────────────────────────────────
        self.snippet_head  = SnippetPredictionHead(snippet_dim, dropout=dropout)
        self.video_cls     = VideoClassifier(snippet_dim, num_classes=2, dropout=dropout)

    # ── Snippet feature extraction (memory-safe) ─────────────────────────────
    def _encode_snippets(self, snippets: torch.Tensor) -> torch.Tensor:
        """
        snippets: [B, T, L, C, H, W]
        Returns:  [B, T, snippet_dim]

        Processes frames in chunks of self.frame_chunk to avoid OOM.
        """
        B, T, L, C, H, W = snippets.shape

        # Flatten to [B*T*L, C, H, W]
        flat = snippets.reshape(B * T * L, C, H, W)

        # Encode in chunks
        chunks = flat.split(self.frame_chunk, dim=0)
        feats  = []
        for chunk in chunks:
            feats.append(self.backbone(chunk))         # [chunk, backbone_dim]
        feats = torch.cat(feats, dim=0)                # [B*T*L, backbone_dim]

        # Reshape → snippet average pooling
        feats = feats.reshape(B, T, L, -1)
        feats = feats.mean(dim=2)                      # [B, T, backbone_dim]

        # Project + refine
        feats = self.snippet_proj(feats)               # [B, T, snippet_dim]
        feats = self.temporal_refine(feats)            # [B, T, snippet_dim]
        return feats

    # ── Main forward ─────────────────────────────────────────────────────────
    def forward(
        self,
        snippets:    torch.Tensor,              # [B, T, L, C, H, W]
        labels:      torch.Tensor | None = None,  # [B]
        is_training: bool               = True,
    ) -> dict:

        # 1. Extract snippet features
        X = self._encode_snippets(snippets)     # [B, T, D]

        # 2. Temporal Inconsistency Mining
        mt, sim = self.tim(X)                   # [B,T], [B,T-1]

        # 3. Prototype-guided Contrastive
        zt, dt = self.pcl(X)                    # [B,T,proj], [B,T]

        # 4. MSA-L
        ht, alpha, At = self.msal(X, mt, dt)   # [B,T,D], [B,T], [B,T]

        # 5. Snippet prediction
        s_loc = self.snippet_head(ht)           # [B, T]

        # 6. MIL video-level aggregation
        v_agg = (alpha.unsqueeze(2) * ht).sum(dim=1)  # [B, D]
        vid_logits = self.video_cls(v_agg)             # [B, 2]

        # 7. Update memory bank during training
        if is_training and labels is not None:
            with torch.no_grad():
                self.pcl.update_memory(zt.detach(), mt.detach(),
                                       labels, threshold=0.3)

        return {
            'video_logits':   vid_logits,   # [B, 2]
            'snippet_scores': s_loc,         # [B, T]
            'attention':      alpha,         # [B, T]
            'anomaly_scores': At,            # [B, T]
            'temporal_mt':    mt,            # [B, T]
            'spatial_dt':     dt,            # [B, T]
            'similarity':     sim,           # [B, T-1]
            'proj_features':  zt,            # [B, T, proj_dim]
            'context_feats':  ht,            # [B, T, D]
        }

    @torch.no_grad()
    def update_prototypes(self) -> bool:
        return self.pcl.update_prototypes()

    def get_memory_status(self) -> str:
        filled = int(self.pcl.memory_filled)
        total  = self.pcl.memory_size
        return f'{filled}/{total}'

    def freeze_backbone(self, freeze: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def get_snippet_to_frame_scores(
        self, snippet_scores: torch.Tensor, num_frames_list: list[int]
    ) -> list[np.ndarray]:
        """
        Map snippet-level scores back to frame-level scores.

        snippet_scores: [B, T]
        num_frames_list: list of actual frame counts per video

        Returns: list of numpy arrays, each of length num_frames[i]
        """
        B, T = snippet_scores.shape
        results = []
        for b in range(B):
            scores_t = snippet_scores[b].cpu().numpy()   # [T]
            N = num_frames_list[b]
            # Expand each snippet score to its window of frames
            frame_scores = np.zeros(N, dtype=np.float32)
            window_size  = N / T
            for t in range(T):
                s = int(t * window_size)
                e = min(int((t + 1) * window_size), N)
                frame_scores[s:e] = scores_t[t]
            results.append(frame_scores)
        return results