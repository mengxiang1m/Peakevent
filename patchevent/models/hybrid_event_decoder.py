"""Hybrid DETR-AR event decoder.

The decoder keeps DETR's set prediction and Hungarian matching as the training
core, then applies a causal event refinement block over query embeddings sorted
by predicted onset. The AR behavior is internal to this module; it is not a
separate autoregressive baseline.
"""

from __future__ import annotations

import json as _json

import torch
import torch.nn as nn
import torch.nn.functional as F

from patchevent.models.backbone import PatchBackboneEncoder
from patchevent.models.event_tokenizer import StructuredEventTokenizer


class HybridEventDecoder(nn.Module):
    """End-to-end Hybrid DETR-AR decoder for structured event prediction."""

    is_hybrid_event_decoder = True

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        patch_len: int = 8,
        patch_stride: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        max_events: int = 10,
        refine_layers: int = 1,
        exist_threshold: float = 0.5,
        norm_mean: float = 0.0,
        norm_std: float = 1.0,
        event_schema: str = "quad",
        backbone_layers: int = 2,
        refine_order: str = "onset",
        use_causal_refine_mask: bool = True,
        hybrid_nms_onset_radius: int = 0,
        use_count_head: bool = False,
        use_count_decoding: bool = True,
        max_event_count: int = 12,
        time_head: str = "structured",
    ):
        super().__init__()
        if event_schema != "quad":
            raise ValueError("HybridEventDecoder requires event_schema='quad'")
        if refine_order not in ("onset", "query"):
            raise ValueError("refine_order must be 'onset' or 'query'")
        if time_head not in ("structured", "independent"):
            raise ValueError("time_head must be 'structured' or 'independent'")
        self.tokenizer = StructuredEventTokenizer(pred_len=pred_len, event_schema=event_schema)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.max_events = int(max_events)
        self.exist_threshold = float(exist_threshold)
        self.intensity_from_values = "apex"
        self.refine_order = str(refine_order)
        self.use_causal_refine_mask = bool(use_causal_refine_mask)
        self.hybrid_nms_onset_radius = int(hybrid_nms_onset_radius)
        self.use_count_head = bool(use_count_head)
        self.use_count_decoding = bool(use_count_decoding)
        self.max_event_count = int(max_event_count)
        self.time_head = str(time_head)

        self.register_buffer("norm_mean", torch.tensor(float(norm_mean), dtype=torch.float32))
        self.register_buffer("norm_std", torch.tensor(max(float(norm_std), 1e-6), dtype=torch.float32))

        self.backbone = PatchBackboneEncoder(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=patch_stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=backbone_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.event_queries = nn.Parameter(torch.randn(max_events, d_model) * 0.02)

        proposal_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.proposal_decoder = nn.TransformerDecoder(proposal_layer, num_layers=n_layers)

        if refine_layers > 0:
            refine_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.refine_encoder = nn.TransformerEncoder(refine_layer, num_layers=refine_layers)
        else:
            self.refine_encoder = None
        self.refine_norm = nn.LayerNorm(d_model)

        self.obj_head = nn.Linear(d_model, 1)
        loc_out_dim = 5 if self.time_head == "structured" else 3
        self.localization_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, loc_out_dim),
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        if self.use_count_head:
            self.count_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, self.max_event_count + 1),
            )
        else:
            self.count_head = None

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"[HybridEventDecoder] trainable: {trainable:,} || total: {total:,} "
            f"|| queries: {max_events} || refine_layers: {refine_layers} "
            f"|| refine_order: {self.refine_order} || causal_refine: {self.use_causal_refine_mask} "
            f"|| nms_onset_radius: {self.hybrid_nms_onset_radius} "
            f"|| count_head: {self.use_count_head} || count_decode: {self.use_count_decoding} "
            f"|| time_head: {self.time_head}"
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.norm_mean) / self.norm_std.clamp(min=1e-6)

    def _decode_heads(self, hidden: torch.Tensor) -> dict:
        loc_raw = self.localization_head(hidden)
        int_raw = self.intensity_head(hidden).squeeze(-1)

        extra = {}
        if self.time_head == "structured":
            onset_raw = loc_raw[..., 0]
            apex_offset_raw = loc_raw[..., 1]
            duration_logits = loc_raw[..., 2:5]
            dur_values = torch.tensor([3.0, 4.0, 5.0], device=hidden.device, dtype=hidden.dtype)
            duration_probs = F.softmax(duration_logits, dim=-1)
            duration = (duration_probs * dur_values.view(1, 1, 3)).sum(dim=-1)
            onset = torch.sigmoid(onset_raw) * float(self.pred_len - 1)
            apex_offset = torch.sigmoid(apex_offset_raw) * (duration - 1.0).clamp(min=1.0)
            apex = (onset + apex_offset).clamp(max=float(self.pred_len - 1))
            extra.update({
                "duration_logits": duration_logits,
                "duration_probs": duration_probs,
                "apex_offset": apex_offset,
            })
        else:
            onset_raw, apex_raw, dur_raw = loc_raw.unbind(dim=-1)
            onset = torch.sigmoid(onset_raw) * float(self.pred_len - 1)
            apex = torch.sigmoid(apex_raw) * float(self.pred_len - 1)
            duration = 3.0 + torch.sigmoid(dur_raw) * 2.0
        intensity_max = self.tokenizer.INT_MIN + (self.tokenizer.N_INT - 1) * self.tokenizer.INT_STEP
        intensity = self.tokenizer.INT_MIN + torch.sigmoid(int_raw) * (intensity_max - self.tokenizer.INT_MIN)
        outputs = {
            "obj_logits": self.obj_head(hidden).squeeze(-1),
            "onset": onset,
            "apex": apex,
            "duration": duration,
            "intensity": intensity,
        }
        outputs.update(extra)
        return outputs

    def _target_events_from_ids(self, target_ids: torch.Tensor | None) -> list[torch.Tensor] | None:
        if target_ids is None:
            return None
        events_per_batch: list[torch.Tensor] = []
        ids_cpu = target_ids.detach().cpu()
        tok = self.tokenizer
        for b in range(ids_cpu.size(0)):
            ids = ids_cpu[b].tolist()
            i = 0
            while i < len(ids) and ids[i] in (tok.PAD_ID, tok.BOS_ID):
                i += 1
            events = []
            while i < len(ids) and ids[i] not in (tok.EOS_ID, tok.PAD_ID):
                if i + 3 >= len(ids):
                    break
                onset_id, dur_id, apex_id, int_id = ids[i], ids[i + 1], ids[i + 2], ids[i + 3]
                onset = onset_id - tok.ONSET_OFFSET
                duration = (dur_id - tok.DUR_OFFSET) + 3
                apex = apex_id - tok.APEX_OFFSET
                int_bin = int_id - tok.INT_OFFSET
                if not (
                    0 <= onset < tok.N_ONSET
                    and 3 <= duration <= 5
                    and 0 <= apex < tok.N_APEX
                    and 0 <= int_bin < tok.N_INT
                ):
                    break
                intensity = tok.INT_MIN + int_bin * tok.INT_STEP
                events.append([float(onset), float(duration), float(apex), float(intensity)])
                i += 4
            if events:
                events_per_batch.append(torch.tensor(events, dtype=torch.float32, device=target_ids.device))
            else:
                events_per_batch.append(torch.empty(0, 4, dtype=torch.float32, device=target_ids.device))
        return events_per_batch

    def _matched_refine_order(self, proposal: dict, target_ids: torch.Tensor | None) -> torch.Tensor | None:
        if target_ids is None or not self.training or self.refine_order != "onset":
            return None
        target_events = self._target_events_from_ids(target_ids)
        if target_events is None:
            return None

        from scipy.optimize import linear_sum_assignment

        obj = proposal["obj_logits"]
        onset = proposal["onset"]
        apex = proposal["apex"]
        duration = proposal["duration"]
        intensity = proposal["intensity"]
        B, K = obj.shape
        scale_t = max(float(self.pred_len - 1), 1.0)
        orders = []
        for b in range(B):
            gt = target_events[b]
            n_gt = int(gt.size(0))
            base_order = onset[b].detach().argsort().tolist()
            if n_gt == 0:
                orders.append(base_order)
                continue

            obj_cost = -F.logsigmoid(obj[b]).unsqueeze(1)
            loc_cost = (
                2.0 * torch.cdist(onset[b].unsqueeze(-1), gt[:, 0:1], p=1) / scale_t
                + 2.0 * torch.cdist(apex[b].unsqueeze(-1), gt[:, 2:3], p=1) / scale_t
                + 0.5 * torch.cdist(duration[b].unsqueeze(-1), gt[:, 1:2], p=1) / 2.0
                + 0.2 * torch.cdist(intensity[b].unsqueeze(-1), gt[:, 3:4], p=1)
            )
            cost = (obj_cost + loc_cost).detach().cpu().numpy()
            row_idx, col_idx = linear_sum_assignment(cost)
            matched = sorted(
                [(int(row), int(col)) for row, col in zip(row_idx, col_idx)],
                key=lambda item: float(gt[item[1], 0].item()),
            )
            matched_rows = [row for row, _ in matched]
            matched_set = set(matched_rows)
            unmatched_rows = [q for q in base_order if q not in matched_set]
            orders.append(matched_rows + unmatched_rows)
        return torch.tensor(orders, dtype=torch.long, device=obj.device)

    def _causal_refine(
        self,
        hidden: torch.Tensor,
        proposal: dict,
        target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.refine_encoder is None:
            return self.refine_norm(hidden)

        B, K, D = hidden.shape
        matched_order = self._matched_refine_order(proposal, target_ids)
        if matched_order is not None:
            order = matched_order
        elif self.refine_order == "onset":
            order = proposal["onset"].detach().argsort(dim=1)
        else:
            order = torch.arange(K, device=hidden.device).unsqueeze(0).expand(B, -1)
        inv_order = torch.empty_like(order)
        ranks = torch.arange(K, device=hidden.device).unsqueeze(0).expand(B, -1)
        inv_order.scatter_(1, order, ranks)

        sorted_hidden = hidden.gather(1, order.unsqueeze(-1).expand(-1, -1, D))
        causal_mask = None
        if self.use_causal_refine_mask:
            causal_mask = torch.triu(
                torch.ones(K, K, device=hidden.device, dtype=torch.bool),
                diagonal=1,
            )
        refined_sorted = self.refine_encoder(sorted_hidden, mask=causal_mask)
        refined = refined_sorted.gather(1, inv_order.unsqueeze(-1).expand(-1, -1, D))
        return self.refine_norm(refined)

    def forward(self, x: torch.Tensor, target_ids: torch.Tensor | None = None, **kwargs) -> dict:
        x_norm = self._normalize(x)
        memory = self.backbone(x_norm)["z"]
        B = x.size(0)
        queries = self.event_queries.unsqueeze(0).expand(B, -1, -1)
        proposal_hidden = self.proposal_decoder(queries, memory)
        proposal = self._decode_heads(proposal_hidden)

        refined_hidden = self._causal_refine(proposal_hidden, proposal, target_ids=target_ids)
        refined = self._decode_heads(refined_hidden)
        count_logits = None
        if self.count_head is not None:
            count_logits = self.count_head(refined_hidden.mean(dim=1))

        outputs = {
            "pred_obj_logits": refined["obj_logits"],
            "pred_onset": refined["onset"],
            "pred_apex": refined["apex"],
            "pred_duration": refined["duration"],
            "pred_intensity": refined["intensity"],
            "proposal_obj_logits": proposal["obj_logits"],
            "proposal_onset": proposal["onset"],
            "proposal_apex": proposal["apex"],
            "proposal_duration": proposal["duration"],
            "proposal_intensity": proposal["intensity"],
            "query_hidden": refined_hidden,
            "proposal_hidden": proposal_hidden,
            "memory": memory,
        }
        if "duration_logits" in refined:
            outputs["pred_duration_logits"] = refined["duration_logits"]
            outputs["proposal_duration_logits"] = proposal["duration_logits"]
        if "apex_offset" in refined:
            outputs["pred_apex_offset"] = refined["apex_offset"]
            outputs["proposal_apex_offset"] = proposal["apex_offset"]
        if count_logits is not None:
            outputs["count_logits"] = count_logits
        return outputs

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_new_tokens: int = 60, return_aux: bool = False, **kwargs):
        nms_radius = int(kwargs.get("hybrid_nms_onset_radius", self.hybrid_nms_onset_radius))
        preds = self.forward(x)
        obj_prob = torch.sigmoid(preds["pred_obj_logits"])
        onset = preds["pred_onset"]
        order = onset.argsort(dim=1)
        count_pred = None
        if self.use_count_head and self.use_count_decoding and "count_logits" in preds:
            count_pred = preds["count_logits"].argmax(dim=-1).clamp(min=0, max=self.max_events)
        results: list[str] = []
        for b in range(x.size(0)):
            candidates = []
            if count_pred is not None:
                k_pred = int(count_pred[b].item())
                selected = obj_prob[b].argsort(descending=True)[:k_pred].tolist()
                selected = sorted(selected, key=lambda q: float(onset[b, q].item()))
            else:
                selected = order[b].tolist()
            for q_t in selected:
                if count_pred is None and obj_prob[b, q_t].item() < self.exist_threshold:
                    continue
                on = int(round(float(preds["pred_onset"][b, q_t].item())))
                ap = int(round(float(preds["pred_apex"][b, q_t].item())))
                if "pred_duration_logits" in preds:
                    du = int(torch.argmax(preds["pred_duration_logits"][b, q_t]).item()) + 3
                else:
                    du = int(round(float(preds["pred_duration"][b, q_t].item())))
                it = float(preds["pred_intensity"][b, q_t].item())
                on = max(0, min(self.pred_len - 1, on))
                du = max(3, min(5, du))
                ap = max(on, min(self.pred_len - 1, on + du - 1, ap))
                candidates.append({
                    "prob": float(obj_prob[b, q_t].item()),
                    "event": [on, du, ap, round(it, 2)],
                })
            if nms_radius > 0 and candidates:
                kept = []
                for item in sorted(candidates, key=lambda v: v["prob"], reverse=True):
                    on = int(item["event"][0])
                    if all(abs(on - int(prev["event"][0])) > nms_radius for prev in kept):
                        kept.append(item)
                candidates = kept
            events = [item["event"] for item in sorted(candidates, key=lambda v: v["event"][0])]
            results.append(_json.dumps({"peak_events": events}, separators=(",", ":")))
        if return_aux:
            return results, {}
        return results

    def get_param_groups(self, lr: float = 1e-3, encoder_lr: float | None = None) -> list[dict]:
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": lr}]


__all__ = ["HybridEventDecoder"]
