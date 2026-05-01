"""Value prediction losses."""

from __future__ import annotations

import torch

from patchevent.losses.event_localization import (
    compute_event_local_mse_loss_gpu as _compute_event_local_mse_loss_gpu,
)

def _compute_apex_mape_loss(
    pred_norm: torch.Tensor,
    x_future: torch.Tensor,
    norm_mean: float,
    norm_std: float,
    event_apex: torch.Tensor,
    event_valid: torch.Tensor,
) -> torch.Tensor:
    """Direct MAPE loss at apex positions in raw space.

    This matches the evaluation metric exactly:
      MAPE = mean( |pred_raw[apex] - gt_raw[apex]| / gt_raw[apex] )
    """
    pred_len = pred_norm.size(1)
    apex_idx = event_apex.clamp(0, pred_len - 1).long()    # (B, me)
    # gather pred and gt at apex positions
    pred_at_apex = pred_norm.gather(1, apex_idx) * norm_std + norm_mean   # (B, me) raw
    gt_at_apex = x_future.float().gather(1, apex_idx)                     # (B, me) raw
    # smooth MAPE: avoid division by zero
    eps = 1e-6
    ape = torch.abs(pred_at_apex - gt_at_apex) / torch.clamp(torch.abs(gt_at_apex), min=eps)
    ape = ape * event_valid.float()
    n_valid = event_valid.float().sum()
    if n_valid.item() <= 0:
        return pred_norm.new_tensor(0.0)
    return ape.sum() / n_valid


def _compute_dense_value_loss(
    model,
    future_values_norm: torch.Tensor,
    x_future: torch.Tensor,
    events_batch: list[list[dict]] | None = None,
    use_event_local: bool = False,
    event_onset: torch.Tensor | None = None,
    event_dur: torch.Tensor | None = None,
    event_apex: torch.Tensor | None = None,
    event_valid: torch.Tensor | None = None,
    apex_weight: float = 1.0,
    apex_mape_weight: float = 0.0,
) -> torch.Tensor:
    gt_future_norm = (x_future.float() - model.norm_mean) / model.norm_std
    if use_event_local and event_onset is not None:
        mse = _compute_event_local_mse_loss_gpu(
            pred_norm=future_values_norm,
            gt_norm=gt_future_norm,
            event_onset=event_onset,
            event_dur=event_dur,
            event_valid=event_valid,
            event_apex=event_apex,
            apex_weight=apex_weight,
        )
        if apex_mape_weight > 0 and event_apex is not None:
            mape = _compute_apex_mape_loss(
                pred_norm=future_values_norm,
                x_future=x_future,
                norm_mean=model.norm_mean,
                norm_std=model.norm_std,
                event_apex=event_apex,
                event_valid=event_valid,
            )
            return mse + apex_mape_weight * mape
        return mse
    return torch.nn.functional.mse_loss(future_values_norm, gt_future_norm)

def _merge_value_loss(loss: torch.Tensor, value_loss: torch.Tensor | None, value_weight: float) -> torch.Tensor:
    if value_loss is None or value_weight <= 0:
        return loss
    return (loss + value_weight * value_loss) / (1.0 + value_weight)


# Backward compatibility aliases
compute_apex_mape_loss = _compute_apex_mape_loss
compute_dense_value_loss = _compute_dense_value_loss
merge_value_loss = _merge_value_loss


__all__ = [
    "compute_apex_mape_loss",
    "compute_dense_value_loss",
    "merge_value_loss",
    "_compute_apex_mape_loss",
    "_compute_dense_value_loss",
    "_merge_value_loss",
]
