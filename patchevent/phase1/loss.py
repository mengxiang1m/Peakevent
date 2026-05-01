"""
Phase1Loss
==========
Phase-1 Patch Encoder 复合损失函数（优化版）：

    L = λ_bce · FocalBCE(has_apex)     # Focal Loss替代BCE，改善正负样本不均衡
      + λ_d   · SmoothL1(d_to_apex)    # Smooth L1替代MSE，对异常值更鲁棒
      + λ_kl  · WeightedKL(phase_dist) # 加权KL，强调稀有阶段(rising/apex/falling)
      + λ_off · MaskedCE(apex_offset)  # 仅对has_apex=1的patch计算，带label smoothing
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Phase1Loss(nn.Module):
    """
    Parameters
    ----------
    lambda_bce : weight for has_apex focal-BCE loss
    lambda_d   : weight for d_to_apex Smooth-L1 loss
    lambda_kl  : weight for phase_dist weighted-KL-divergence loss
    lambda_off : weight for apex_offset Masked-CE loss
    kl_phase_weights : per-class weights [bg, rising, apex, falling] for KL
    focal_gamma : focal loss gamma (0=standard BCE, 2=standard focal)
    offset_label_smoothing : label smoothing for apex_offset CE
    """

    def __init__(
        self,
        lambda_bce: float = 1.0,
        lambda_d:   float = 0.5,
        lambda_kl:  float = 1.0,
        lambda_off: float = 0.5,
        lambda_recon: float = 0.0,
        lambda_npp: float = 0.0,
        kl_phase_weights: list | None = None,
        focal_gamma: float = 2.0,
        offset_label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.l_bce = lambda_bce
        self.l_d   = lambda_d
        self.l_kl  = lambda_kl
        self.l_off = lambda_off
        self.l_recon = lambda_recon
        self.l_npp = lambda_npp
        self.focal_gamma = focal_gamma
        self.offset_label_smoothing = offset_label_smoothing

        self.smooth_l1 = nn.SmoothL1Loss()
        if kl_phase_weights is None:
            kl_phase_weights = [1.0, 1.0, 1.0, 1.0]
        self.register_buffer(
            'kl_weights',
            torch.tensor(kl_phase_weights, dtype=torch.float32)
        )

    def _focal_bce(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal Binary Cross-Entropy: 降低易分类样本的权重，关注难分类样本。"""
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = pred * target + (1 - pred) * (1 - target)  # p_t
        focal_weight = (1 - pt) ** self.focal_gamma
        return (focal_weight * bce).mean()

    def forward(self, pred: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        device = pred['has_apex'].device

        # ── 1. Focal BCE : has_apex ───────────────────────────────────────────
        l_bce = self._focal_bce(pred['has_apex'], batch['has_apex'].to(device))

        # ── 2. Smooth L1 : d_to_apex ─────────────────────────────────────────
        l_d = self.smooth_l1(pred['d_to_apex'], batch['min_d'].to(device))

        # ── 3. Weighted KL : phase distribution ──────────────────────────────
        B, N, C = pred['phase_dist'].shape
        log_pred_phase = torch.log(pred['phase_dist'].clamp(min=1e-8))
        true_phase = batch['phase_dist'].to(device).clamp(min=0.0)
        true_phase = true_phase / true_phase.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        kl_per_class = true_phase * (torch.log(true_phase.clamp(min=1e-8)) - log_pred_phase)
        w = self.kl_weights.to(device)
        l_kl = (kl_per_class * w).sum(dim=-1).mean() / w.sum()

        # ── 4. Masked CE with label smoothing : apex_offset ───────────────────
        apex_mask = batch['has_apex'].to(device) > 0.5
        if apex_mask.any():
            pred_off = pred['apex_offset'][apex_mask]
            true_off = batch['apex_offset'].to(device)[apex_mask].long().clamp(0, 7)
            l_off = F.cross_entropy(pred_off, true_off,
                                     label_smoothing=self.offset_label_smoothing)
        else:
            l_off = torch.tensor(0.0, device=device)

        total = (self.l_bce * l_bce
                 + self.l_d * l_d
                 + self.l_kl * l_kl
                 + self.l_off * l_off)

        components = {
            'bce': l_bce.item(),
            'd':   l_d.item(),
            'kl':  l_kl.item(),
            'off': l_off.item() if isinstance(l_off, torch.Tensor) else l_off,
        }

        # ── 5. Next-Patch Prediction (NPP) ───────────────────────────────────
        if self.l_npp > 0 and 'npp_pred' in pred and 'patches' in pred:
            npp_pred = pred['npp_pred'][:, :-1, :]   # (B, N-1, patch_len) predict from patch i
            npp_target = pred['patches'][:, 1:, :]   # (B, N-1, patch_len) target is patch i+1
            l_npp = F.mse_loss(npp_pred, npp_target)
            total = total + self.l_npp * l_npp
            components['npp'] = l_npp.item()

        # ── 6. Masked Patch Reconstruction (MPR) ───────────────────────────────
        if self.l_recon > 0 and 'recon' in pred and 'mask_bool' in pred and 'patches' in pred:
            recon = pred['recon']           # (B, N, patch_len)
            patches = pred['patches']       # (B, N, patch_len)
            mask_bool = pred['mask_bool']   # (B, N)
            if mask_bool.any():
                l_recon = F.mse_loss(recon[mask_bool], patches[mask_bool])
            else:
                l_recon = torch.tensor(0.0, device=device)
            total = total + self.l_recon * l_recon
            components['recon'] = l_recon.item() if isinstance(l_recon, torch.Tensor) else l_recon

        return total, components
