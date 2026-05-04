"""Validation and evaluation loops."""

from __future__ import annotations

import torch

from patchevent.losses.intensity import _compute_intensity_loss, compute_int_cls_loss
from patchevent.losses.matching import hungarian_match_loss as _hungarian_match_loss
from patchevent.losses.value_losses import _compute_dense_value_loss, _merge_value_loss
from patchevent.models.event_tokenizer import EventTokenizer
from patchevent.models.non_ar_decoder import NonAutoRegressiveDecoder
from patchevent.training.output_adapter import _unpack_model_outputs

@torch.no_grad()
def validate(model, loader, device, args=None):
    """验证集loss计算。"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=EventTokenizer.PAD_ID)
    is_non_ar = isinstance(model, NonAutoRegressiveDecoder)
    is_hybrid = bool(getattr(model, 'is_hybrid_event_decoder', False))

    finetune_int_head = getattr(args, 'finetune_int_head', False) if args is not None else False
    int_head_loss_mode = getattr(args, 'int_head_loss', 'ordinal') if args is not None else 'ordinal'
    int_ord_sigma = max(float(getattr(args, 'int_ordinal_sigma', 2.0)), 1e-6) if args is not None else 2.0
    use_event_local_value_loss = bool(getattr(args, 'use_independent_dense_int', False)) if args is not None else False

    for batch in loader:
        x = batch['x'].to(device)
        x_future = batch['x_future'].to(device)
        target_ids = batch['target_ids'].to(device)
        count_logits = None

        if is_hybrid:
            preds = model(x)
            loss = _hungarian_match_loss(preds, target_ids, model.tokenizer, device, args=args)
            count_logits = preds.get('count_logits')
        elif is_non_ar:
            preds = model(x)
            loss = _hungarian_match_loss(preds, target_ids, model.tokenizer, device, args=args)
        else:
            fwd_kwargs = {}
            if 'hours_future' in batch:
                fwd_kwargs['hours_future'] = batch['hours_future'].to(device)
                fwd_kwargs['weekdays_future'] = batch['weekdays_future'].to(device)
            model_out = model(x, target_ids, **fwd_kwargs)
            labels = target_ids[:, 1:]
            logits, _, int_cls_logits, int_positions, future_values_norm, events_batch, event_onset, event_dur, event_apex, event_valid, predicted_intensity, _, _ = _unpack_model_outputs(model_out)
            value_loss = None
            intensity_loss = None
            if future_values_norm is not None:
                value_loss = _compute_dense_value_loss(
                    model=model,
                    future_values_norm=future_values_norm,
                    x_future=x_future,
                    events_batch=events_batch,
                    use_event_local=use_event_local_value_loss,
                    event_onset=event_onset,
                    event_dur=event_dur,
                    event_apex=event_apex,
                    event_valid=event_valid,
                    apex_weight=float(getattr(args, 'apex_weight', 1.0)) if args is not None else 1.0,
                    apex_mape_weight=float(getattr(args, 'apex_mape_weight', 0.0)) if args is not None else 0.0,
                )
            if predicted_intensity is not None and event_apex is not None:
                intensity_loss = _compute_intensity_loss(
                    predicted_intensity=predicted_intensity,
                    x_future=x_future,
                    norm_mean=float(model.norm_mean.item()),
                    event_apex=event_apex,
                    event_valid=event_valid,
                )

            if finetune_int_head:
                if int_cls_logits is None:
                    raise ValueError('finetune_int_head requires use_decoupled_int_head')
                gt_tokens = labels[:, int_positions]
                loss = compute_int_cls_loss(
                    int_cls_logits, gt_tokens, sigma=int_ord_sigma, mode=int_head_loss_mode,
                    tokenizer=model.tokenizer,
                )
                loss = _merge_value_loss(loss, value_loss, float(getattr(args, 'value_loss_weight', 1.0)))
                if intensity_loss is not None:
                    int_loss_w = float(getattr(args, 'intensity_loss_weight', 1.0)) if args is not None else 1.0
                    loss = loss + int_loss_w * intensity_loss
            else:
                loss = ce_loss(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss = _merge_value_loss(loss, value_loss, float(getattr(args, 'value_loss_weight', 1.0)))
                if intensity_loss is not None:
                    int_loss_w = float(getattr(args, 'intensity_loss_weight', 1.0)) if args is not None else 1.0
                    loss = loss + int_loss_w * intensity_loss

        if count_logits is not None:
            onset_lo = model.tokenizer.ONSET_OFFSET
            onset_hi = onset_lo + model.tokenizer.N_ONSET
            is_onset = (target_ids >= onset_lo) & (target_ids < onset_hi)
            max_event_count = int(getattr(model, 'max_event_count', 12))
            max_events = int(getattr(model, 'max_events', max_event_count))
            gt_count = is_onset.sum(dim=1).clamp(max=min(max_event_count, max_events)).long()
            count_loss_w = getattr(args, 'hybrid_count_loss_weight', None) if args is not None else None
            if count_loss_w is None:
                count_loss_w = getattr(args, 'count_loss_weight', 0.5) if args is not None else 0.5
            loss = loss + float(count_loss_w) * torch.nn.functional.cross_entropy(count_logits, gt_count)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


__all__ = ["validate"]
