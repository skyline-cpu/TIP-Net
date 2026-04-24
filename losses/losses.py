"""
losses/losses.py  —  All loss functions for TIP-Net.

Includes:
  1. SimilarityConsistencyLoss   (Lsim)
  2. PrototypeContrastiveLoss    (Lproto)
  3. BootstrappingLocalisationLoss (Lloc)
  4. ECLReweightedLoss           (Lcontrast)  — from the contrastive pre-training branch
  5. TIPNetLoss                  — combined loss with configurable weights
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1. Similarity Consistency Loss  (Lsim)
# ══════════════════════════════════════════════════════════════════════════════
class SimilarityConsistencyLoss(nn.Module):
    """
    Real videos: encourage high, smooth similarity evolution.
    Fake videos: encourage at least one local similarity drop.
    """

    def __init__(self, real_threshold: float = 0.7,
                 fake_drop_threshold: float = 0.5,
                 smoothness_weight: float = 0.1):
        super().__init__()
        self.real_thr   = real_threshold
        self.fake_thr   = fake_drop_threshold
        self.smooth_w   = smoothness_weight

    def forward(self, sim: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        sim:    [B, T-1]  inter-snippet cosine similarities
        labels: [B]
        """
        B = sim.shape[0]
        loss = sim.new_zeros(1)

        for i in range(B):
            s_i = sim[i]                              # [T-1]
            lbl = labels[i].item()

            if lbl == 0:  # real
                # 1a. Penalise low similarity
                low_sim = F.relu(self.real_thr - s_i).mean()
                # 1b. Penalise abrupt changes (smoothness)
                smooth  = s_i.diff().abs().mean() if s_i.shape[0] > 1 \
                          else s_i.new_zeros(1)
                loss    = loss + low_sim + self.smooth_w * smooth

            else:  # fake
                # 2. Encourage at least one similarity drop
                min_sim = s_i.min()
                loss    = loss + F.relu(min_sim - self.fake_thr)

        return loss / B


# ══════════════════════════════════════════════════════════════════════════════
# 2. Prototype Contrastive Loss  (Lproto)
# ══════════════════════════════════════════════════════════════════════════════
class PrototypeContrastiveLoss(nn.Module):
    """
    InfoNCE-style loss: push snippet projections toward real prototypes,
    weighted by spatial anomaly dt (so real snippets inside fake videos
    contribute less gradient).

    Lproto = -1/T Σ_t  d_t · log [ exp(z_t·p_k/τ) / Σ_{p∈P_all} exp(z_t·p/τ) ]
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(
        self,
        zt:         torch.Tensor,   # [B, T, proj_dim]
        dt:         torch.Tensor,   # [B, T]
        prototypes: torch.Tensor,   # [K, proj_dim]
        aux_protos: torch.Tensor | None = None,  # [K', proj_dim] learnable
    ) -> torch.Tensor:
        B, T, D = zt.shape

        # Build P_all = P_real ∪ P_aux
        if aux_protos is not None:
            P_all = torch.cat([prototypes,
                               F.normalize(aux_protos, dim=1)], dim=0)  # [K+K', D]
        else:
            P_all = prototypes                                             # [K, D]

        K_all = P_all.shape[0]
        K_real = prototypes.shape[0]

        # [B*T, D]
        zt_flat = zt.reshape(B * T, D)
        dt_flat = dt.reshape(B * T).clamp(0, 1)   # weights ∈ [0, 1]

        # Logits [B*T, K_all]
        logits = torch.matmul(zt_flat, P_all.T) / self.tau

        # Positive: any real prototype (uniform distribution)
        log_denom = torch.logsumexp(logits, dim=1)              # [B*T]
        log_nums  = torch.logsumexp(logits[:, :K_real], dim=1)  # [B*T]
        nce       = log_denom - log_nums                         # [B*T]

        # Weight by dt (higher dt = more anomalous → stronger gradient signal)
        loss = (dt_flat * nce).mean()
        return loss


# ══════════════════════════════════════════════════════════════════════════════
# 3. Bootstrapping Localisation Loss  (Lloc)
# ══════════════════════════════════════════════════════════════════════════════
class BootstrappingLocalisationLoss(nn.Module):
    """
    For fake videos: select Top-K snippets by anomaly score → pseudo-labels = 1.
    For real videos: all snippets get pseudo-label = 0.
    Supervise the snippet prediction head with BCE.
    """

    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(
        self,
        snippet_scores: torch.Tensor,  # [B, T]  ∈ (0,1)
        anomaly_scores: torch.Tensor,  # [B, T]  fused At
        labels:         torch.Tensor,  # [B]
    ) -> torch.Tensor:
        B, T = snippet_scores.shape
        loss = snippet_scores.new_zeros(1)

        for i in range(B):
            ss  = snippet_scores[i]     # [T]
            At  = anomaly_scores[i]     # [T]
            lbl = labels[i].item()

            if lbl == 1:  # fake
                k = min(self.top_k, T)
                _, top_idx = At.topk(k)
                pseudo = torch.zeros(T, device=ss.device)
                pseudo[top_idx] = 1.0
            else:          # real
                pseudo = torch.zeros(T, device=ss.device)

            # BCE
            loss = loss + F.binary_cross_entropy(ss, pseudo)

        return loss / B


# ══════════════════════════════════════════════════════════════════════════════
# 4. ECL-Style Reweighted Contrastive Loss  (Lcontrast)
# ══════════════════════════════════════════════════════════════════════════════
class ECLReweightedLoss(nn.Module):
    """
    Supervised contrastive loss with adaptive re-weighting for fake samples.

    Fake snippets that are semantically close to real prototypes are down-
    weighted (they are likely authentic frames within a partially faked video).

    Input:
        features: [B, T, proj_dim]  — snippet projections
        labels:   [B]               — video-level labels
        prototypes: [K, proj_dim]   — real prototypes (for weight estimation)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        gamma: float = 1.5,
    ):
        super().__init__()
        self.tau   = temperature
        self.tau0  = base_temperature
        self.gamma = gamma

    def forward(
        self,
        features:   torch.Tensor,   # [B, T, proj_dim]
        labels:     torch.Tensor,   # [B]
        prototypes: torch.Tensor,   # [K, proj_dim]
    ) -> torch.Tensor:
        B, T, D = features.shape
        device   = features.device

        # ── Flatten to snippet level  [B*T, D]  with snippet labels ─────────
        feat_flat = features.reshape(B * T, D)
        # Snippet-level labels: inherit from video label
        lbl_flat  = labels.unsqueeze(1).expand(-1, T).reshape(-1)  # [B*T]

        N = feat_flat.shape[0]

        # ── Similarity matrix  [N, N] ─────────────────────────────────────
        sim = torch.matmul(feat_flat, feat_flat.T) / self.tau  # [N, N]

        # Numeric stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - sim_max.detach()

        # ── Positive mask (same label) ────────────────────────────────────
        lbl_col = lbl_flat.unsqueeze(1)
        mask = torch.eq(lbl_col, lbl_col.T).float()   # [N, N]
        # Mask out self-contrast
        eye  = torch.eye(N, device=device)
        logits_mask = 1.0 - eye
        mask         = mask * logits_mask

        # ── Log-probabilities ─────────────────────────────────────────────
        exp_logits   = torch.exp(logits) * logits_mask
        log_prob     = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        pos_count = mask.sum(1).clamp(min=1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count   # [N]

        # ── Adaptive re-weighting for fake snippets ───────────────────────
        fake_mask = (lbl_flat == 1)
        if fake_mask.any() and prototypes is not None:
            fake_feats = feat_flat[fake_mask]                      # [Nf, D]
            # Avg similarity to real prototypes
            avg_sim = torch.matmul(fake_feats, prototypes.T).mean(dim=1)  # [Nf]
            # High similarity → authentic → low weight
            weights = (1.0 - avg_sim.clamp(0, 1)).pow(self.gamma)
            # Clamp to avoid zero
            weights = weights.clamp(min=0.05)
            mean_log_prob_pos = mean_log_prob_pos.clone()
            mean_log_prob_pos[fake_mask] = (
                mean_log_prob_pos[fake_mask] * weights)

        loss = -(self.tau / self.tau0) * mean_log_prob_pos
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Combined TIP-Net Loss
# ══════════════════════════════════════════════════════════════════════════════
class TIPNetLoss(nn.Module):
    """
    Ltotal = Lcls
           + γ1 · Lsim
           + γ2 · Lproto
           + γ3 · Lloc
           + γ4 · Lcontrast   (optional ECL-style)

    All sub-losses return a scalar.
    """

    def __init__(
        self,
        gamma1: float = 0.5,
        gamma2: float = 1.0,
        gamma3: float = 0.8,
        gamma4: float = 0.3,
        top_k:  int   = 5,
        temperature: float = 0.07,
        label_smoothing: float = 0.05,
        use_contrastive: bool = True,
    ):
        super().__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.gamma4 = gamma4
        self.use_contrastive = use_contrastive

        self.cls_loss   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.sim_loss   = SimilarityConsistencyLoss()
        self.proto_loss = PrototypeContrastiveLoss(temperature=temperature)
        self.loc_loss   = BootstrappingLocalisationLoss(top_k=top_k)
        if use_contrastive:
            self.ecl_loss = ECLReweightedLoss(temperature=temperature)

    def forward(
        self,
        outputs: dict,
        labels:  torch.Tensor,   # [B]
    ) -> tuple[torch.Tensor, dict]:
        """
        outputs keys (from TIPNet.forward):
            video_logits, snippet_scores, attention, anomaly_scores,
            temporal_mt, spatial_dt, similarity, proj_features,
            context_feats
        Also expects:
            outputs['real_prototypes']  [K, D]
            outputs['aux_prototypes']   [K', D]  (optional)
        """
        # ── Cls loss ─────────────────────────────────────────────────────────
        L_cls = self.cls_loss(outputs['video_logits'], labels)

        # ── Similarity consistency ────────────────────────────────────────────
        L_sim = self.sim_loss(outputs['similarity'], labels)

        # ── Prototype contrastive ─────────────────────────────────────────────
        L_proto = self.proto_loss(
            zt=outputs['proj_features'],
            dt=outputs['spatial_dt'],
            prototypes=outputs['real_prototypes'],
            aux_protos=outputs.get('aux_prototypes'),
        )

        # ── Bootstrapping localisation ────────────────────────────────────────
        L_loc = self.loc_loss(
            snippet_scores=outputs['snippet_scores'],
            anomaly_scores=outputs['anomaly_scores'],
            labels=labels,
        )

        total = L_cls \
              + self.gamma1 * L_sim \
              + self.gamma2 * L_proto \
              + self.gamma3 * L_loc

        loss_dict = {
            'total': total.item(),
            'cls':   L_cls.item(),
            'sim':   L_sim.item(),
            'proto': L_proto.item(),
            'loc':   L_loc.item(),
        }

        # ── ECL-style contrastive (optional) ──────────────────────────────────
        if self.use_contrastive:
            L_ecl = self.ecl_loss(
                features=outputs['proj_features'],
                labels=labels,
                prototypes=outputs['real_prototypes'],
            )
            total = total + self.gamma4 * L_ecl
            loss_dict['ecl']   = L_ecl.item()
            loss_dict['total'] = total.item()

        return total, loss_dict