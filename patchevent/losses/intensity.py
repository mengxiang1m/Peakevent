"""Intensity-related loss functions."""

from __future__ import annotations

import torch

from patchevent.models.event_tokenizer import EventTokenizer

def compute_int_cls_loss(
    int_cls_logits: torch.Tensor,
    gt_tokens: torch.Tensor,
    sigma: float = 2.0,
    mode: str = 'ordinal',
    tokenizer=None,
):
    """Compute loss for decoupled INT head: ordinal / ce / mix."""
    tok = tokenizer if tokenizer is not None else EventTokenizer()
    non_pad = (gt_tokens != tok.PAD_ID).float()
    gt_bins = (gt_tokens - tok.INT_OFFSET).clamp(0, tok.N_INT - 1)
    denom = non_pad.sum().clamp(min=1.0)

    # CE on 100-bin logits.
    ce = torch.nn.functional.cross_entropy(
        int_cls_logits.reshape(-1, tok.N_INT), gt_bins.reshape(-1), reduction='none',
    ).reshape_as(gt_bins).float()
    ce = (ce * non_pad).sum() / denom

    if mode == 'ce':
        return ce

    bins = torch.arange(tok.N_INT, device=int_cls_logits.device).view(1, 1, -1)
    center = gt_bins.unsqueeze(-1).float()
    soft_target = torch.exp(-0.5 * ((bins - center) / max(float(sigma), 1e-6)) ** 2)
    soft_target = soft_target / soft_target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    log_probs = torch.nn.functional.log_softmax(int_cls_logits, dim=-1)
    ord_loss = -(soft_target * log_probs).sum(dim=-1)
    ord_loss = (ord_loss * non_pad).sum() / denom

    if mode == 'mix':
        return 0.5 * ce + 0.5 * ord_loss
    return ord_loss

def _compute_intensity_loss(
    predicted_intensity: torch.Tensor,
    x_future: torch.Tensor,
    norm_mean: float,
    event_apex: torch.Tensor,
    event_valid: torch.Tensor,
) -> torch.Tensor:
    """MSE loss on predicted intensity ratio vs ground truth intensity ratio.

    Args:
        predicted_intensity: (B, max_events) — predicted intensity ratio
        x_future: (B, pred_len) — ground truth future values (raw)
        norm_mean: normalization mean (used to compute intensity ratio)
        event_apex: (B, max_events) — apex positions
        event_valid: (B, max_events) — valid mask
    """
    pred_len = x_future.size(1)
    apex_idx = event_apex.clamp(0, pred_len - 1).long()
    gt_raw_at_apex = x_future.float().gather(1, apex_idx)
    gt_intensity = gt_raw_at_apex / max(norm_mean, 1e-8)
    diff = (predicted_intensity - gt_intensity) * event_valid.float()
    n_valid = event_valid.float().sum()
    if n_valid.item() <= 0:
        return predicted_intensity.new_tensor(0.0)
    return (diff * diff).sum() / n_valid


# Backward compatibility alias
compute_intensity_loss = _compute_intensity_loss


__all__ = ["compute_int_cls_loss", "compute_intensity_loss", "_compute_intensity_loss"]
