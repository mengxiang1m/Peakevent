"""Lightweight end-to-end patch backbone for event forecasting."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPatchPosition(nn.Module):
    """Non-learned sinusoidal positions for patch tokens."""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1)])


class PatchBackboneEncoder(nn.Module):
    """Patch encoder without Phase-1 pretraining heads.

    This is the new mainline encoder: it is trained end-to-end with the
    Hybrid DETR-AR decoder and does not expose encoder-only metrics.
    """

    def __init__(
        self,
        seq_len: int = 96,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if patch_len <= 0 or stride <= 0:
            raise ValueError("patch_len and stride must be positive")
        if seq_len < patch_len:
            raise ValueError("seq_len must be >= patch_len")
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.n_patches = (self.seq_len - self.patch_len) // self.stride + 1
        self.d_model = int(d_model)

        self.patch_embed = nn.Linear(self.patch_len, d_model)
        self.pos_enc = SinusoidalPatchPosition(d_model, max_len=self.n_patches + 4, dropout=dropout)
        self.embed_norm = nn.LayerNorm(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> dict:
        """Encode normalized input series into patch memory."""
        patches = x.float().unfold(dimension=1, size=self.patch_len, step=self.stride)
        z = self.patch_embed(patches)
        z = self.embed_norm(self.pos_enc(z))
        z = self.encoder(z)
        z = self.out_norm(z)
        return {"z": z, "patches": patches}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["z"]


__all__ = ["PatchBackboneEncoder", "SinusoidalPatchPosition"]
