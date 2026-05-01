"""
PatchEncoder
============
Phase-1 峰值感知 Patch Encoder。

Pipeline
--------
x  (B, L)
  ↓  unfold(patch_len, stride) → (B, N, patch_len)
  ↓  Linear(patch_len → d_model) + sinusoidal PE + LayerNorm
z  (B, N, d_model)
  ↓  Transformer Encoder  (e_layers × EncoderLayer + FullAttention)
z  (B, N, d_model)
  ↓  4 prediction heads
outputs: has_apex (B,N), d_to_apex (B,N), phase_dist (B,N,4), apex_offset (B,N)
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 使用内部layers目录 ─────────────────────────────────────────────────────
_pkg = os.path.dirname(os.path.abspath(__file__))
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)  # TODO(R-future): migrate to patchevent package import

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class _SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding (non-learnable)."""

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1)])


class PatchEncoder(nn.Module):
    """
    Phase-1 Patch Encoder for peak-aware pretraining.

    Parameters
    ----------
    seq_len   : input window length in time steps (hours)
    patch_len : number of time steps per patch
    stride    : patch stride
    d_model   : embedding dimension
    n_heads   : number of attention heads
    e_layers  : number of Transformer encoder layers
    d_ff      : feedforward hidden dimension
    dropout   : dropout rate

    Forward input
    -------------
    x : (B, seq_len) float — normalised load sequence

    Forward output (dict)
    ---------------------
    has_apex    : (B, N)    ∈ [0, 1]    — probability of apex in patch
    d_to_apex   : (B, N)   ≥ 0         — predicted min distance to nearest apex
    phase_dist  : (B, N, 4) sums to 1  — [background, rising, apex, falling]
    apex_offset : (B, N)   raw logit   — apex position in patch (0-7), masked in loss
    """

    def __init__(
        self,
        seq_len: int   = 96,
        patch_len: int = 8,
        stride: int    = 4,
        d_model: int   = 128,
        n_heads: int   = 4,
        e_layers: int  = 2,
        d_ff: int      = 256,
        dropout: float = 0.1,
        mask_ratio: float = 0.0,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.patch_len = patch_len
        self.stride    = stride
        self.n_patches = (seq_len - patch_len) // stride + 1  # 23 for defaults
        self.mask_ratio = mask_ratio

        # ── Patch embedding ───────────────────────────────────────────────────
        self.patch_embed = nn.Linear(patch_len, d_model, bias=True)
        self.pos_enc     = _SinusoidalPE(d_model, max_len=self.n_patches + 4, dropout=dropout)
        self.embed_norm  = nn.LayerNorm(d_model)

        # ── Masked Patch Reconstruction (MPR) ────────────────────────────────
        if mask_ratio > 0:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.mask_token, std=0.02)
            self.recon_head = nn.Linear(d_model, patch_len)

        # ── Transformer Encoder ───────────────────────────────────────────────
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation='gelu',
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # ── Prediction heads ──────────────────────────────────────────────────
        # Head 1: binary classification  →  BCE loss
        self.has_apex_head    = nn.Linear(d_model, 1)
        # Head 2: distance regression    →  MSE loss
        self.d_to_apex_head   = nn.Linear(d_model, 1)
        # Head 3: phase distribution     →  KL-divergence loss
        self.phase_dist_head  = nn.Linear(d_model, 4)
        # Head 4: apex offset classification → Cross-Entropy (8 classes for patch positions 0-7)
        self.apex_offset_head = nn.Linear(d_model, 8)
        # Head 5: Next-Patch Prediction (NPP) — predict next patch raw values from current z
        self.npp_head = nn.Linear(d_model, patch_len)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, use_mask: bool = False) -> dict:
        """
        Args
        ----
        x        : (B, L) — normalised load sequence
        use_mask : if True and mask_ratio > 0, apply random patch masking for MPR

        Returns
        -------
        dict with:
          has_apex    (B, N)    ∈ [0,1]
          d_to_apex   (B, N)   ≥ 0
          phase_dist  (B, N, 4) sums to 1
          apex_offset (B, N)   raw (used masked in loss)
          recon       (B, N, patch_len)  — only present when use_mask=True
          mask_bool   (B, N)             — only present when use_mask=True
          patches     (B, N, patch_len)  — only present when use_mask=True
        """
        # ── tokenise into patches ─────────────────────────────────────────────
        # x.unfold: (B, L) → (B, N, patch_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)

        # ── embed ─────────────────────────────────────────────────────────────
        z = self.patch_embed(patches)          # (B, N, d_model)
        z = self.pos_enc(z)
        z = self.embed_norm(z)

        # ── optional masking for MPR ──────────────────────────────────────────
        mask_bool = None
        if use_mask and self.mask_ratio > 0 and self.training:
            B, N, D = z.shape
            n_mask = max(1, int(N * self.mask_ratio))
            mask_bool = torch.zeros(B, N, dtype=torch.bool, device=z.device)
            for b in range(B):
                idx = torch.randperm(N, device=z.device)[:n_mask]
                mask_bool[b, idx] = True
            z = z.clone()
            z[mask_bool] = self.mask_token.squeeze(0).squeeze(0)

        # ── encode ───────────────────────────────────────────────────────────
        z, _ = self.encoder(z, attn_mask=None)  # (B, N, d_model)

        # ── heads ─────────────────────────────────────────────────────────────
        result = {
            'has_apex':    torch.sigmoid(self.has_apex_head(z)).squeeze(-1),      # (B, N)
            'd_to_apex':   F.relu(self.d_to_apex_head(z)).squeeze(-1),            # (B, N)
            'phase_dist':  F.softmax(self.phase_dist_head(z), dim=-1),            # (B, N, 4)
            'apex_offset': self.apex_offset_head(z),                              # (B, N, 8) raw logits
            'z':           z,                                                     # (B, N, d_model)
            'npp_pred':    self.npp_head(z),                                      # (B, N, patch_len)
            'patches':     patches,                                               # (B, N, patch_len)
        }

        # ── MPR reconstruction head ───────────────────────────────────────────
        if use_mask and self.mask_ratio > 0 and mask_bool is not None:
            result['recon'] = self.recon_head(z)       # (B, N, patch_len)
            result['mask_bool'] = mask_bool             # (B, N)
            result['patches'] = patches                 # (B, N, patch_len)

        return result

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch embeddings z ∈ (B, N, d_model) — for downstream use."""
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        z = self.embed_norm(self.pos_enc(self.patch_embed(patches)))
        z, _ = self.encoder(z, attn_mask=None)
        return z
