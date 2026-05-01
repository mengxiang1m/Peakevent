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
    ):
        super().__init__()
        if event_schema != "quad":
            raise ValueError("HybridEventDecoder requires event_schema='quad'")
        self.tokenizer = StructuredEventTokenizer(pred_len=pred_len, event_schema=event_schema)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.max_events = int(max_events)
        self.exist_threshold = float(exist_threshold)
        self.intensity_from_values = "apex"

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
        self.localization_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"[HybridEventDecoder] trainable: {trainable:,} || total: {total:,} "
            f"|| queries: {max_events} || refine_layers: {refine_layers}"
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.norm_mean) / self.norm_std.clamp(min=1e-6)

    def _decode_heads(self, hidden: torch.Tensor) -> dict:
        loc_raw = self.localization_head(hidden)
        onset_raw, apex_raw, dur_raw = loc_raw.unbind(dim=-1)
        int_raw = self.intensity_head(hidden).squeeze(-1)

        onset = torch.sigmoid(onset_raw) * float(self.pred_len - 1)
        apex = torch.sigmoid(apex_raw) * float(self.pred_len - 1)
        duration = 3.0 + torch.sigmoid(dur_raw) * 2.0
        intensity_max = self.tokenizer.INT_MIN + (self.tokenizer.N_INT - 1) * self.tokenizer.INT_STEP
        intensity = self.tokenizer.INT_MIN + torch.sigmoid(int_raw) * (intensity_max - self.tokenizer.INT_MIN)
        return {
            "obj_logits": self.obj_head(hidden).squeeze(-1),
            "onset": onset,
            "apex": apex,
            "duration": duration,
            "intensity": intensity,
        }

    def _causal_refine(self, hidden: torch.Tensor, proposal_onset: torch.Tensor) -> torch.Tensor:
        if self.refine_encoder is None:
            return self.refine_norm(hidden)

        B, K, D = hidden.shape
        order = proposal_onset.detach().argsort(dim=1)
        inv_order = torch.empty_like(order)
        ranks = torch.arange(K, device=hidden.device).unsqueeze(0).expand(B, -1)
        inv_order.scatter_(1, order, ranks)

        sorted_hidden = hidden.gather(1, order.unsqueeze(-1).expand(-1, -1, D))
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

        refined_hidden = self._causal_refine(proposal_hidden, proposal["onset"])
        refined = self._decode_heads(refined_hidden)

        return {
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

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_new_tokens: int = 60, return_aux: bool = False, **kwargs):
        preds = self.forward(x)
        obj_prob = torch.sigmoid(preds["pred_obj_logits"])
        onset = preds["pred_onset"]
        order = onset.argsort(dim=1)
        results: list[str] = []
        for b in range(x.size(0)):
            events = []
            for q_t in order[b].tolist():
                if obj_prob[b, q_t].item() < self.exist_threshold:
                    continue
                on = int(round(float(preds["pred_onset"][b, q_t].item())))
                ap = int(round(float(preds["pred_apex"][b, q_t].item())))
                du = int(round(float(preds["pred_duration"][b, q_t].item())))
                it = float(preds["pred_intensity"][b, q_t].item())
                on = max(0, min(self.pred_len - 1, on))
                ap = max(0, min(self.pred_len - 1, ap))
                du = max(3, min(5, du))
                events.append([on, du, ap, round(it, 2)])
            results.append(_json.dumps({"peak_events": events}, separators=(",", ":")))
        if return_aux:
            return results, {}
        return results

    def get_param_groups(self, lr: float = 1e-3, encoder_lr: float | None = None) -> list[dict]:
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": lr}]


__all__ = ["HybridEventDecoder"]
