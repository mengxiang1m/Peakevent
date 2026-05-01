"""Hungarian matching losses for event decoders."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from patchevent.models.event_tokenizer import StructuredEventTokenizer


def _target_events_from_ids(
    target_ids: torch.Tensor,
    tokenizer: StructuredEventTokenizer,
    device: torch.device,
) -> list[torch.Tensor]:
    """Convert padded token ids into per-sample event tensors.

    Returns a list of tensors shaped (n_events, 4): onset, duration, apex,
    intensity_ratio. Invalid or incomplete trailing tokens are skipped.
    """
    events_per_batch: list[torch.Tensor] = []
    ids_cpu = target_ids.detach().cpu()
    for b in range(ids_cpu.size(0)):
        ids = ids_cpu[b].tolist()
        i = 0
        while i < len(ids) and ids[i] in (tokenizer.PAD_ID, tokenizer.BOS_ID):
            i += 1
        events = []
        while i < len(ids) and ids[i] not in (tokenizer.EOS_ID, tokenizer.PAD_ID):
            if i + 2 >= len(ids):
                break
            onset_id, dur_id, apex_id = ids[i], ids[i + 1], ids[i + 2]
            onset = onset_id - tokenizer.ONSET_OFFSET
            duration = (dur_id - tokenizer.DUR_OFFSET) + 3
            apex = apex_id - tokenizer.APEX_OFFSET
            if not (0 <= onset < tokenizer.N_ONSET and 3 <= duration <= 5 and 0 <= apex < tokenizer.N_APEX):
                break
            intensity = 1.0
            step = 3
            if tokenizer.has_intensity:
                if i + 3 >= len(ids):
                    break
                int_id = ids[i + 3]
                int_bin = int_id - tokenizer.INT_OFFSET
                if not (0 <= int_bin < tokenizer.N_INT):
                    break
                intensity = tokenizer.INT_MIN + int_bin * tokenizer.INT_STEP
                step = 4
            events.append([float(onset), float(duration), float(apex), float(intensity)])
            i += step
        if events:
            events_per_batch.append(torch.tensor(events, dtype=torch.float32, device=device))
        else:
            events_per_batch.append(torch.empty(0, 4, dtype=torch.float32, device=device))
    return events_per_batch


def _weighted_objectness_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    no_object_weight: float,
    focal_gamma: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if focal_gamma > 0:
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        bce = ((1 - pt) ** focal_gamma) * bce
    weights = torch.where(targets > 0.5, torch.ones_like(targets), torch.full_like(targets, no_object_weight))
    return (weights * bce).mean()


def _hybrid_loss_on_prediction_set(
    preds: dict,
    prefix: str,
    target_events: list[torch.Tensor],
    device: torch.device,
    pred_len: int,
    weights: dict,
    reuse_matches: list[tuple[list[int], list[int]]] | None = None,
) -> tuple[torch.Tensor, list[tuple[list[int], list[int]]]]:
    from scipy.optimize import linear_sum_assignment

    obj = preds[f"{prefix}obj_logits"]
    onset = preds[f"{prefix}onset"]
    apex = preds[f"{prefix}apex"]
    duration = preds[f"{prefix}duration"]
    intensity = preds[f"{prefix}intensity"]
    B, K = obj.shape

    total = obj.new_tensor(0.0)
    matches: list[tuple[list[int], list[int]]] = []
    scale_t = max(float(pred_len - 1), 1.0)

    for b in range(B):
        gt = target_events[b]
        n_gt = int(gt.size(0))
        obj_target = torch.zeros(K, device=device)

        if n_gt > 0:
            if reuse_matches is None:
                obj_cost = -F.logsigmoid(obj[b]).unsqueeze(1)
                loc_cost = (
                    weights["match_onset"] * torch.cdist(onset[b].unsqueeze(-1), gt[:, 0:1], p=1) / scale_t
                    + weights["match_apex"] * torch.cdist(apex[b].unsqueeze(-1), gt[:, 2:3], p=1) / scale_t
                    + weights["match_duration"] * torch.cdist(duration[b].unsqueeze(-1), gt[:, 1:2], p=1) / 2.0
                    + weights["match_intensity"] * torch.cdist(intensity[b].unsqueeze(-1), gt[:, 3:4], p=1)
                )
                cost = (obj_cost + loc_cost).detach().cpu().numpy()
                row_idx, col_idx = linear_sum_assignment(cost)
                row_list = [int(v) for v in row_idx]
                col_list = [int(v) for v in col_idx]
            else:
                row_list, col_list = reuse_matches[b]

            if row_list:
                q_idx = torch.tensor(row_list, dtype=torch.long, device=device)
                g_idx = torch.tensor(col_list, dtype=torch.long, device=device)
                matched_gt = gt[g_idx]
                obj_target[q_idx] = 1.0
                total = total + weights["onset"] * F.smooth_l1_loss(onset[b, q_idx] / scale_t, matched_gt[:, 0] / scale_t)
                total = total + weights["apex"] * F.smooth_l1_loss(apex[b, q_idx] / scale_t, matched_gt[:, 2] / scale_t)
                total = total + weights["duration"] * F.smooth_l1_loss((duration[b, q_idx] - 3.0) / 2.0, (matched_gt[:, 1] - 3.0) / 2.0)
                total = total + weights["intensity"] * F.smooth_l1_loss(intensity[b, q_idx], matched_gt[:, 3])
            matches.append((row_list, col_list))
        else:
            matches.append(([], []))

        total = total + weights["objectness"] * _weighted_objectness_loss(
            obj[b],
            obj_target,
            no_object_weight=weights["no_object"],
            focal_gamma=weights["object_focal_gamma"],
        )

    return total / max(B, 1), matches


def hybrid_hungarian_loss(
    preds: dict,
    target_ids: torch.Tensor,
    tokenizer: StructuredEventTokenizer,
    device: torch.device,
    args=None,
) -> torch.Tensor:
    """Hybrid DETR-AR loss: set matching plus refined event supervision."""
    target_events = _target_events_from_ids(target_ids, tokenizer, device)
    weights = {
        "objectness": float(getattr(args, "hybrid_objectness_weight", 1.0)),
        "no_object": float(getattr(args, "hybrid_no_object_weight", 0.15)),
        "onset": float(getattr(args, "hybrid_onset_weight", 2.0)),
        "apex": float(getattr(args, "hybrid_apex_weight", 2.0)),
        "duration": float(getattr(args, "hybrid_duration_weight", 0.5)),
        "intensity": float(getattr(args, "hybrid_intensity_weight", 0.5)),
        "match_onset": float(getattr(args, "hybrid_match_onset_weight", 2.0)),
        "match_apex": float(getattr(args, "hybrid_match_apex_weight", 2.0)),
        "match_duration": float(getattr(args, "hybrid_match_duration_weight", 0.5)),
        "match_intensity": float(getattr(args, "hybrid_match_intensity_weight", 0.2)),
        "object_focal_gamma": float(getattr(args, "hybrid_object_focal_gamma", 0.0)),
    }
    pred_len = int(getattr(tokenizer, "pred_len", 96))

    refined_loss, matches = _hybrid_loss_on_prediction_set(
        preds, "pred_", target_events, device, pred_len, weights
    )
    refine_weight = float(getattr(args, "hybrid_refine_loss_weight", 1.0))
    total = refine_weight * refined_loss

    proposal_weight = float(getattr(args, "hybrid_proposal_loss_weight", 0.3))
    if proposal_weight > 0 and "proposal_obj_logits" in preds:
        proposal_loss, _ = _hybrid_loss_on_prediction_set(
            preds, "proposal_", target_events, device, pred_len, weights, reuse_matches=matches
        )
        total = total + proposal_weight * proposal_loss

    return total


def _legacy_non_ar_hungarian_loss(
    preds: dict,
    target_ids: torch.Tensor,
    tokenizer: StructuredEventTokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Legacy DETR ablation loss retained for old checkpoints/scripts."""
    from scipy.optimize import linear_sum_assignment
    B = target_ids.size(0)
    total_loss = torch.tensor(0.0, device=device)

    for b in range(B):
        # 解析GT事件
        gt_ids = target_ids[b].tolist()
        gt_events = []
        i = 0
        while i < len(gt_ids) and gt_ids[i] in (tokenizer.PAD_ID, tokenizer.BOS_ID):
            i += 1
        while i + 3 < len(gt_ids) and gt_ids[i] != tokenizer.EOS_ID and gt_ids[i] != tokenizer.PAD_ID:
            gt_events.append(gt_ids[i:i+4])
            i += 4
        n_gt = len(gt_events)
        n_pred = preds['onset_logits'].size(1)

        # 计算cost矩阵
        cost = torch.zeros(n_pred, max(n_gt, 1), device=device)
        for j in range(n_gt):
            onset_gt = gt_events[j][0] - tokenizer.ONSET_OFFSET
            dur_gt = gt_events[j][1] - tokenizer.DUR_OFFSET
            apex_gt = gt_events[j][2] - tokenizer.APEX_OFFSET
            int_gt = gt_events[j][3] - tokenizer.INT_OFFSET
            for q in range(n_pred):
                c = -F.log_softmax(preds['onset_logits'][b, q], dim=-1)[onset_gt]
                c += -F.log_softmax(preds['dur_logits'][b, q], dim=-1)[dur_gt]
                c += -F.log_softmax(preds['apex_logits'][b, q], dim=-1)[apex_gt]
                c += -F.log_softmax(preds['int_logits'][b, q], dim=-1)[int_gt]
                c += -torch.log(torch.sigmoid(preds['exist_logits'][b, q]).clamp(min=1e-6))
                cost[q, j] = c.detach()

        if n_gt > 0:
            row_idx, col_idx = linear_sum_assignment(cost[:, :n_gt].cpu().numpy())
        else:
            row_idx, col_idx = [], []

        # 匹配的query: 分类loss
        for q, j in zip(row_idx, col_idx):
            onset_gt = gt_events[j][0] - tokenizer.ONSET_OFFSET
            dur_gt = gt_events[j][1] - tokenizer.DUR_OFFSET
            apex_gt = gt_events[j][2] - tokenizer.APEX_OFFSET
            int_gt = gt_events[j][3] - tokenizer.INT_OFFSET
            total_loss += F.cross_entropy(preds['onset_logits'][b, q:q+1], torch.tensor([onset_gt], device=device))
            total_loss += F.cross_entropy(preds['dur_logits'][b, q:q+1], torch.tensor([dur_gt], device=device))
            total_loss += F.cross_entropy(preds['apex_logits'][b, q:q+1], torch.tensor([apex_gt], device=device))
            total_loss += F.cross_entropy(preds['int_logits'][b, q:q+1], torch.tensor([int_gt], device=device))
            total_loss += F.binary_cross_entropy_with_logits(preds['exist_logits'][b, q:q+1], torch.ones(1, device=device))

        # 未匹配的query: 不存在
        matched_set = set(row_idx)
        for q in range(n_pred):
            if q not in matched_set:
                total_loss += F.binary_cross_entropy_with_logits(preds['exist_logits'][b, q:q+1], torch.zeros(1, device=device))

    return total_loss / B


def hungarian_match_loss(
    preds: dict,
    target_ids: torch.Tensor,
    tokenizer: StructuredEventTokenizer,
    device: torch.device,
    args=None,
) -> torch.Tensor:
    """Dispatch Hungarian loss by prediction format."""
    if "pred_obj_logits" in preds:
        return hybrid_hungarian_loss(preds, target_ids, tokenizer, device, args=args)
    return _legacy_non_ar_hungarian_loss(preds, target_ids, tokenizer, device)


# Backward compatibility alias
_hungarian_match_loss = hungarian_match_loss


__all__ = [
    "hungarian_match_loss",
    "hybrid_hungarian_loss",
    "_hungarian_match_loss",
]
