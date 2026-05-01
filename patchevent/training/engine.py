"""Training engine loops."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from patchevent.losses.intensity import _compute_intensity_loss, compute_int_cls_loss
from patchevent.losses.matching import hungarian_match_loss as _hungarian_match_loss
from patchevent.losses.value_losses import _compute_dense_value_loss, _merge_value_loss
from patchevent.models.event_tokenizer import EventTokenizer
from patchevent.models.non_ar_decoder import NonAutoRegressiveDecoder
from patchevent.training.output_adapter import _unpack_model_outputs

def train_one_epoch(model, loader, optimizer, scheduler, device, args, global_step: int):
    """训练一个epoch，支持属性级加权loss和position-aware label smoothing。"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    label_smoothing = getattr(args, 'label_smoothing', 0.0)
    no_pos_aware = getattr(args, 'no_pos_aware_smoothing', False)
    if no_pos_aware and label_smoothing > 0:
        ce_loss = torch.nn.CrossEntropyLoss(
            ignore_index=EventTokenizer.PAD_ID, label_smoothing=label_smoothing,
        )
        label_smoothing = 0.0
    else:
        ce_loss = torch.nn.CrossEntropyLoss(
            ignore_index=EventTokenizer.PAD_ID, label_smoothing=0.0,
        )

    # 属性级加权
    attr_weights_str = getattr(args, 'attr_loss_weights', None)
    attr_w = [float(w) for w in attr_weights_str.split(',')] if attr_weights_str else None

    # 位置约束掩码 (用于position-aware smoothing)
    pos_valid_mask = getattr(model, 'pos_valid_mask', None)
    use_event_local_value_loss = bool(getattr(args, 'use_independent_dense_int', False))

    is_non_ar = isinstance(model, NonAutoRegressiveDecoder)
    is_hybrid = bool(getattr(model, 'is_hybrid_event_decoder', False))

    for step, batch in enumerate(loader):
        x = batch['x'].to(device)
        x_future = batch['x_future'].to(device)
        target_ids = batch['target_ids'].to(device)

        use_int_reg = getattr(args, 'use_int_regression', False)
        int_reg_w = getattr(args, 'int_reg_weight', 1.0)
        use_int_ord = getattr(args, 'use_int_ordinal_loss', False) and (not use_int_reg) and model.tokenizer.has_intensity
        int_ord_sigma = max(float(getattr(args, 'int_ordinal_sigma', 2.0)), 1e-6)
        finetune_int_head = getattr(args, 'finetune_int_head', False)
        int_head_loss_mode = getattr(args, 'int_head_loss', 'ordinal')
        value_loss_weight = float(getattr(args, 'value_loss_weight', 1.0))
        count_logits = None
        pos_reg_preds = None

        if is_hybrid:
            preds = model(x)
            loss = _hungarian_match_loss(preds, target_ids, model.tokenizer, device, args=args)
        elif is_non_ar:
            preds = model(x)
            loss = _hungarian_match_loss(preds, target_ids, model.tokenizer, device, args=args)
        else:
            fwd_kwargs = {}
            if 'hours_future' in batch:
                fwd_kwargs['hours_future'] = batch['hours_future'].to(device)
                fwd_kwargs['weekdays_future'] = batch['weekdays_future'].to(device)
            model_out = model(x, target_ids, **fwd_kwargs)
            logits, int_preds, int_cls_logits, int_positions, future_values_norm, events_batch, event_onset, event_dur, event_apex, event_valid, predicted_intensity, count_logits, pos_reg_preds = _unpack_model_outputs(model_out)
            labels = target_ids[:, 1:]
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
                    apex_weight=float(getattr(args, 'apex_weight', 1.0)),
                    apex_mape_weight=float(getattr(args, 'apex_mape_weight', 0.0)),
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
                if int_positions is None:
                    int_positions = model.tokenizer.get_attr_positions(labels.size(1), 'intensity', device)
                gt_tokens = labels[:, int_positions]
                loss = compute_int_cls_loss(
                    int_cls_logits, gt_tokens, sigma=int_ord_sigma, mode=int_head_loss_mode,
                    tokenizer=model.tokenizer,
                )
                loss = _merge_value_loss(loss, value_loss, value_loss_weight)
            elif attr_w is not None:
                B, T_out, V = logits.shape
                loss = torch.tensor(0.0, device=device)
                total_w = 0.0
                n_cls_attrs = model.tokenizer.event_size - (1 if use_int_reg and model.tokenizer.has_intensity else 0)
                if len(attr_w) < n_cls_attrs:
                    raise ValueError(f'attr_loss_weights expects at least {n_cls_attrs} weights, got {len(attr_w)}')
                for r in range(n_cls_attrs):
                    positions_r = torch.arange(r, T_out, model.tokenizer.event_size, device=device)
                    positions_r = positions_r[positions_r < T_out]
                    if len(positions_r) == 0:
                        continue
                    pos_logits = logits[:, positions_r, :]
                    pos_labels = labels[:, positions_r]
                    if use_int_ord and r == model.tokenizer.intensity_pos:
                        tok = model.tokenizer
                        cls_logits = pos_logits[:, :, tok.INT_OFFSET:tok.INT_OFFSET + tok.N_INT]
                        non_pad = (pos_labels != tok.PAD_ID).float()
                        gt_bins = (pos_labels - tok.INT_OFFSET).clamp(0, tok.N_INT - 1)
                        bins = torch.arange(tok.N_INT, device=device).view(1, 1, -1)
                        center = gt_bins.unsqueeze(-1).float()
                        soft_target = torch.exp(-0.5 * ((bins - center) / int_ord_sigma) ** 2)
                        soft_target = soft_target / soft_target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                        log_probs = torch.nn.functional.log_softmax(cls_logits, dim=-1)
                        per_pos = -(soft_target * log_probs).sum(dim=-1)
                        loss_r = (per_pos * non_pad).sum() / non_pad.sum().clamp(min=1.0)
                    else:
                        onset_focal_gamma = float(getattr(args, 'onset_focal_gamma', 0.0))
                        if onset_focal_gamma > 0 and r == 0:
                            # Focal loss for onset positions: (1-p_t)^gamma * CE
                            log_probs_focal = F.log_softmax(pos_logits, dim=-1)
                            probs_focal = log_probs_focal.exp()
                            flat_labels = pos_labels.reshape(-1)
                            flat_logprobs = log_probs_focal.reshape(-1, V)
                            flat_probs = probs_focal.reshape(-1, V)
                            valid = flat_labels != EventTokenizer.PAD_ID
                            if valid.any():
                                gt_probs = flat_probs[valid, flat_labels[valid]]
                                gt_logprobs = flat_logprobs[valid, flat_labels[valid]]
                                focal_weight = (1 - gt_probs) ** onset_focal_gamma
                                loss_r = -(focal_weight * gt_logprobs).mean()
                            else:
                                loss_r = torch.tensor(0.0, device=device)
                        else:
                            loss_r = ce_loss(pos_logits.reshape(-1, V), pos_labels.reshape(-1))

                    # position-aware label smoothing
                    if (label_smoothing > 0 and pos_valid_mask is not None
                            and not (use_int_ord and r == model.tokenizer.intensity_pos)):
                        log_probs = torch.nn.functional.log_softmax(pos_logits, dim=-1)
                        t_indices = positions_r.clamp(max=pos_valid_mask.size(0) - 1)
                        valid_counts = pos_valid_mask[t_indices].sum(dim=-1).float()
                        valid_masks = pos_valid_mask[t_indices]
                        log_probs_safe = log_probs.masked_fill(~valid_masks.unsqueeze(0), 0.0)
                        smooth_loss = -log_probs_safe.sum(dim=-1)
                        smooth_loss = smooth_loss / valid_counts.unsqueeze(0)
                        non_pad = (pos_labels != EventTokenizer.PAD_ID).float()
                        smooth_loss = (smooth_loss * non_pad).sum() / non_pad.sum().clamp(min=1)
                        loss_r = (1 - label_smoothing) * loss_r + label_smoothing * smooth_loss

                    loss = loss + attr_w[r] * loss_r
                    total_w += attr_w[r]

                # intensity regression loss at r=3 positions
                if use_int_reg and model.tokenizer.has_intensity:
                    tok = model.tokenizer
                    pos_int = model.tokenizer.get_attr_positions(T_out, 'intensity', device)
                    if len(pos_int) > 0:
                        pred_int = int_preds[:, pos_int]           # (B, n_int)
                        gt_tokens = labels[:, pos_int]             # (B, n_int)
                        non_pad = (gt_tokens != tok.PAD_ID).float()
                        gt_bins = (gt_tokens - tok.INT_OFFSET).float()
                        gt_int = tok.INT_MIN + gt_bins * tok.INT_STEP  # normalized intensity
                        huber = torch.nn.functional.smooth_l1_loss(
                            pred_int * non_pad, gt_int * non_pad, reduction='sum',
                        )
                        n_valid = non_pad.sum().clamp(min=1)
                        loss_int = huber / n_valid
                        loss = loss + int_reg_w * loss_int
                        total_w += int_reg_w

                loss = loss / total_w
                loss = _merge_value_loss(loss, value_loss, value_loss_weight)
                if intensity_loss is not None:
                    int_loss_w = float(getattr(args, 'intensity_loss_weight', 1.0))
                    loss = loss + int_loss_w * intensity_loss
            else:
                loss = ce_loss(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss = _merge_value_loss(loss, value_loss, value_loss_weight)
                if intensity_loss is not None:
                    int_loss_w = float(getattr(args, 'intensity_loss_weight', 1.0))
                    loss = loss + int_loss_w * intensity_loss

        # Count prediction loss (方案A)
        if count_logits is not None:
            onset_lo = model.tokenizer.ONSET_OFFSET
            onset_hi = onset_lo + model.tokenizer.N_ONSET
            is_onset = (target_ids >= onset_lo) & (target_ids < onset_hi)
            gt_count = is_onset.sum(dim=1).clamp(max=model.max_event_count).long()
            count_loss_w = float(getattr(args, 'count_loss_weight', 0.5))
            count_loss = F.cross_entropy(count_logits, gt_count)
            loss = loss + count_loss_w * count_loss

        # Position regression loss (方案C')
        if pos_reg_preds is not None:
            labels = target_ids[:, 1:]
            tok = model.tokenizer
            es = tok.event_size
            T_out = labels.size(1)
            pos_reg_target = getattr(model, 'pos_reg_target', 'both')
            # Onset positions (k % event_size == 0)
            onset_pos = torch.arange(0, T_out, es, device=device)
            onset_pos = onset_pos[onset_pos < T_out]
            # Apex positions (k % event_size == 2)
            apex_pos = torch.arange(2, T_out, es, device=device)
            apex_pos = apex_pos[apex_pos < T_out]
            if pos_reg_target == 'onset':
                pos_indices = onset_pos
            elif pos_reg_target == 'apex':
                pos_indices = apex_pos
            else:
                pos_indices = torch.cat([onset_pos, apex_pos])
            if len(pos_indices) > 0:
                pred_vals = pos_reg_preds[:, pos_indices]  # (B, n_pos)
                gt_tokens = labels[:, pos_indices]  # (B, n_pos)
                # Convert GT tokens to continuous hour values
                gt_hours = torch.zeros_like(gt_tokens, dtype=torch.float32)
                if pos_reg_target == 'onset':
                    gt_hours = (gt_tokens - tok.ONSET_OFFSET).float()
                elif pos_reg_target == 'apex':
                    gt_hours = (gt_tokens - tok.APEX_OFFSET).float()
                else:
                    n_onset = len(onset_pos)
                    gt_hours[:, :n_onset] = (gt_tokens[:, :n_onset] - tok.ONSET_OFFSET).float()
                    gt_hours[:, n_onset:] = (gt_tokens[:, n_onset:] - tok.APEX_OFFSET).float()
                # Normalize to [0, 1] for stable loss
                gt_hours = gt_hours / tok.pred_len
                pred_vals_scaled = torch.sigmoid(pred_vals)  # map to [0, 1]
                # Mask: only valid onset/apex tokens (exclude PAD, EOS, and out-of-range tokens)
                if pos_reg_target == 'onset':
                    valid_mask = (gt_tokens >= tok.ONSET_OFFSET) & (gt_tokens < tok.ONSET_OFFSET + tok.N_ONSET)
                elif pos_reg_target == 'apex':
                    valid_mask = (gt_tokens >= tok.APEX_OFFSET) & (gt_tokens < tok.APEX_OFFSET + tok.N_APEX)
                else:
                    n_onset = len(onset_pos)
                    onset_valid = (gt_tokens[:, :n_onset] >= tok.ONSET_OFFSET) & (gt_tokens[:, :n_onset] < tok.ONSET_OFFSET + tok.N_ONSET)
                    apex_valid = (gt_tokens[:, n_onset:] >= tok.APEX_OFFSET) & (gt_tokens[:, n_onset:] < tok.APEX_OFFSET + tok.N_APEX)
                    valid_mask = torch.cat([onset_valid, apex_valid], dim=1)
                non_pad = valid_mask.float()
                pos_reg_loss = F.smooth_l1_loss(
                    pred_vals_scaled * non_pad, gt_hours * non_pad, reduction='sum'
                ) / non_pad.sum().clamp(min=1.0)
                pos_reg_w = float(getattr(args, 'pos_reg_weight', 0.3))
                loss = loss + pos_reg_w * pos_reg_loss

        loss.backward()
        clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        global_step += 1

        total_loss += loss.item()
        n_batches += 1

        if (step + 1) % 50 == 0:
            print(f'    step {step + 1}/{len(loader)}  loss={loss.item():.4f}  global_step={global_step}')

    return total_loss / max(n_batches, 1), global_step


__all__ = ["train_one_epoch"]
