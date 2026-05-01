"""DETR-style non-autoregressive decoder for ablation."""

from __future__ import annotations

import json as _json

import torch
import torch.nn as nn

from patchevent.phase1.model import PatchEncoder
from patchevent.models.event_tokenizer import StructuredEventTokenizer

class NonAutoRegressiveDecoder(nn.Module):
    """
    DETR-style并行事件预测 (消融: 证明AR解码有效性)。
    固定数量的learnable event queries并行预测所有事件。
    训练: Hungarian matching; 推理: existence阈值过滤。
    """

    def __init__(self, encoder_ckpt_path: str, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 3, d_ff: int = 256,
                 dropout: float = 0.2, max_events: int = 8,
                 pred_len: int = 96,
                 encoder_mode: str = 'frozen', exist_threshold: float = 0.5):
        super().__init__()
        self.tokenizer = StructuredEventTokenizer(pred_len=pred_len)
        self.encoder_mode = encoder_mode
        self.max_events = max_events
        self.exist_threshold = exist_threshold

        ckpt = torch.load(encoder_ckpt_path, map_location='cpu', weights_only=False)
        enc_args = ckpt['args']
        d_patch = enc_args.get('d_model', 128)

        self.encoder = PatchEncoder(
            seq_len=enc_args.get('seq_len', 96), patch_len=enc_args.get('patch_len', 8),
            stride=enc_args.get('stride', 4), d_model=d_patch,
            n_heads=enc_args.get('n_heads', 4), e_layers=enc_args.get('e_layers', 2),
            d_ff=enc_args.get('d_ff', 256), dropout=enc_args.get('dropout', 0.1),
        )
        if encoder_mode != 'scratch':
            self.encoder.load_state_dict(ckpt['model_state'], strict=False)
        if encoder_mode == 'frozen':
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

        self.register_buffer('norm_mean', torch.tensor(ckpt['mean'], dtype=torch.float32))
        self.register_buffer('norm_std', torch.tensor(ckpt['std'], dtype=torch.float32))
        self.n_patches = self.encoder.n_patches

        self.patch_proj = nn.Sequential(nn.Linear(d_patch, d_model), nn.LayerNorm(d_model))
        self.event_queries = nn.Parameter(torch.randn(max_events, d_model) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # 每个query的预测头
        self.onset_head = nn.Linear(d_model, 96)
        self.dur_head = nn.Linear(d_model, 3)
        self.apex_head = nn.Linear(d_model, 96)
        self.int_head = nn.Linear(d_model, 100)
        self.exist_head = nn.Linear(d_model, 1)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f'[NonAutoRegressiveDecoder] trainable: {trainable:,} || total: {total:,} || max_events: {max_events}')

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_mode == 'frozen':
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> dict:
        x_norm = (x.float() - self.norm_mean) / self.norm_std
        with torch.no_grad() if self.encoder_mode == 'frozen' else torch.enable_grad():
            enc_out = self.encoder(x_norm)
        z = enc_out['z'].float()
        memory = self.patch_proj(z)

        B = x.size(0)
        queries = self.event_queries.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(queries, memory)

        return {
            'onset_logits': self.onset_head(out),
            'dur_logits': self.dur_head(out),
            'apex_logits': self.apex_head(out),
            'int_logits': self.int_head(out),
            'exist_logits': self.exist_head(out).squeeze(-1),
        }

    @torch.no_grad()
    def generate(self, x: torch.Tensor, **kwargs) -> list[str]:
        preds = self.forward(x)
        B = x.size(0)
        results = []
        for b in range(B):
            exist_prob = torch.sigmoid(preds['exist_logits'][b])
            events = []
            for q in range(self.max_events):
                if exist_prob[q].item() < self.exist_threshold:
                    continue
                onset = preds['onset_logits'][b, q].argmax().item()
                dur = preds['dur_logits'][b, q].argmax().item() + 3
                apex = preds['apex_logits'][b, q].argmax().item()
                int_bin = preds['int_logits'][b, q].argmax().item()
                intensity = round(self.tokenizer.INT_MIN + int_bin * self.tokenizer.INT_STEP, 2)
                events.append([onset, dur, apex, intensity])
            events.sort(key=lambda e: e[0])
            results.append(_json.dumps({"peak_events": events}, separators=(',', ':')))
        return results

    def get_param_groups(self, lr: float = 1e-3, encoder_lr: float | None = None) -> list[dict]:
        return [{'params': [p for p in self.parameters() if p.requires_grad], 'lr': lr}]


__all__ = ["NonAutoRegressiveDecoder"]
