"""Encoder alternatives for PatchEvent ablations."""

from __future__ import annotations

import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    """1D CNN编码器，直接在原始序列上提取特征后池化为N个tokens。"""

    def __init__(self, seq_len=96, n_tokens=23, d_model=128, dropout=0.0):
        super().__init__()
        self.n_patches = n_tokens
        self.convs = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> dict:
        out = self.convs(x.unsqueeze(1))   # (B, d_model, seq_len)
        out = self.pool(out)                # (B, d_model, n_tokens)
        z = self.norm(out.transpose(1, 2))  # (B, n_tokens, d_model)
        return {'z': z}


class LSTMEncoder(nn.Module):
    """BiLSTM编码器，序列建模后池化为N个tokens。"""

    def __init__(self, seq_len=96, n_tokens=23, d_model=128, dropout=0.0):
        super().__init__()
        self.n_patches = n_tokens
        hidden = max(d_model // 2, 32)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.proj = nn.Linear(hidden * 2, d_model)
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> dict:
        h, _ = self.lstm(x.unsqueeze(-1))       # (B, L, 2H)
        h = self.proj(h)                        # (B, L, d_model)
        h = self.pool(h.transpose(1, 2))        # (B, d_model, n_tokens)
        z = self.norm(h.transpose(1, 2))        # (B, n_tokens, d_model)
        return {'z': z}


class MLPEncoder(nn.Module):
    """Per-patch MLP编码器，不做跨patch交互。"""

    def __init__(self, seq_len=96, patch_len=8, stride=4, d_model=128, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.mlp = nn.Sequential(
            nn.Linear(patch_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> dict:
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, N, patch_len)
        z = self.mlp(patches)                                # (B, N, d_model)
        return {'z': z}


__all__ = ["CNNEncoder", "LSTMEncoder", "MLPEncoder"]
