"""Event localization loss terms."""

from __future__ import annotations

import torch

def _compute_event_local_mse_loss_gpu(
    pred_norm: torch.Tensor,
    gt_norm: torch.Tensor,
    event_onset: torch.Tensor,
    event_dur: torch.Tensor,
    event_valid: torch.Tensor,
    event_apex: torch.Tensor | None = None,
    apex_weight: float = 1.0,
) -> torch.Tensor:
    """GPU-vectorised event-local MSE (no Python loop).

    Args:
        pred_norm, gt_norm: (B, pred_len)
        event_onset, event_dur: (B, max_events) int on device
        event_valid: (B, max_events) bool
        event_apex: (B, max_events) int, optional — apex positions for extra weighting
        apex_weight: extra weight multiplier for apex positions (1.0 = no boost)
    """
    pred_len = pred_norm.size(1)
    # t: (1, 1, pred_len)
    t = torch.arange(pred_len, device=pred_norm.device, dtype=torch.float32).view(1, 1, -1)
    onset_f = event_onset.float().unsqueeze(-1)       # (B, me, 1)
    end_f = (event_onset + event_dur).float().unsqueeze(-1)
    # event_mask: (B, me, pred_len) → any-event mask: (B, pred_len)
    event_mask = (t >= onset_f) & (t < end_f) & event_valid.unsqueeze(-1)
    mask = event_mask.any(dim=1).float()               # (B, pred_len)

    # apex boosting: add extra weight at apex positions
    if event_apex is not None and apex_weight > 1.0:
        apex_f = event_apex.float().unsqueeze(-1)     # (B, me, 1)
        apex_hit = (t == apex_f) & event_valid.unsqueeze(-1)  # (B, me, pred_len)
        apex_mask = apex_hit.any(dim=1).float()        # (B, pred_len)
        weight = mask + apex_mask * (apex_weight - 1.0)
    else:
        weight = mask

    n_points = weight.sum()
    if n_points.item() <= 0:
        return torch.nn.functional.mse_loss(pred_norm, gt_norm)
    diff = pred_norm - gt_norm
    return (diff * diff * weight).sum() / n_points


# Backward compatibility alias
compute_event_local_mse_loss_gpu = _compute_event_local_mse_loss_gpu


__all__ = ["compute_event_local_mse_loss_gpu", "_compute_event_local_mse_loss_gpu"]
