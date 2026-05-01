from __future__ import annotations

import torch
import torch.nn as nn


class TupleGuidedValueHead(nn.Module):
    """PeakFocus-inspired tuple-guided dense future value head.

    The head predicts a horizon-aligned dense future sequence by:
    1. creating learned future queries,
    2. attending them to encoder memory,
    3. modulating the attended context with tuple guidance via
       tanh(guidance) * sigmoid(context),
    4. projecting the fused hidden states to scalar future values.
    """

    def __init__(
        self,
        d_model: int,
        pred_len: int,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.future_queries = nn.Parameter(torch.randn(self.pred_len, d_model) * 0.02)
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(d_model, 1)

    def forward(
        self,
        memory: torch.Tensor,
        tuple_guidance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
        memory: (B, N_mem, d_model)
        tuple_guidance: (B, pred_len, d_model)

        Returns:
            future_norm: (B, pred_len)
            hidden: (B, pred_len, d_model)
        """
        if tuple_guidance.dim() != 3:
            raise ValueError('tuple_guidance must be 3D: (B, pred_len, d_model)')
        if tuple_guidance.size(1) != self.pred_len:
            raise ValueError(
                f'tuple_guidance length mismatch: got {tuple_guidance.size(1)}, expected {self.pred_len}'
            )

        batch_size = memory.size(0)
        queries = self.future_queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.query_norm(queries + tuple_guidance)

        context, _ = self.cross_attn(queries, memory, memory, need_weights=False)
        gated_context = torch.tanh(tuple_guidance) * torch.sigmoid(self.context_proj(context))
        hidden = self.context_norm(queries + context + gated_context)
        hidden = self.out_norm(hidden + self.ffn(hidden))
        future_norm = self.proj(hidden).squeeze(-1)
        return future_norm, hidden


class IndependentDenseForecastHead(nn.Module):
    """Independent dense 96->pred_len forecasting with peak-event cross-attention."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_temporal: bool = False,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.d_model = int(d_model)
        self.use_temporal = use_temporal
        self.dense_encoder = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.pred_len * self.d_model),
        )
        if use_temporal:
            self.hour_embed = nn.Embedding(24, d_model)
            self.weekday_embed = nn.Embedding(7, d_model)
            self.temporal_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.peak_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(d_model, 1)
        self.apex_intensity_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )

    def forward(
        self,
        x_norm: torch.Tensor,
        peak_memory: torch.Tensor,
        peak_guidance: torch.Tensor,
        peak_key_padding_mask: torch.Tensor | None = None,
        hours_future: torch.Tensor | None = None,
        weekdays_future: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
        x_norm: (B, seq_len)
        peak_memory: (B, N_events, d_model)
        peak_guidance: (B, pred_len, d_model)
        peak_key_padding_mask: (B, N_events), True means padding
        hours_future: (B, pred_len) int — hour of day for each future step
        weekdays_future: (B, pred_len) int — day of week for each future step

        Returns:
            future_norm: (B, pred_len)
            hidden: (B, pred_len, d_model)
        """
        if x_norm.dim() != 2:
            raise ValueError('x_norm must be 2D: (B, seq_len)')
        if x_norm.size(1) != self.seq_len:
            raise ValueError(f'x_norm length mismatch: got {x_norm.size(1)}, expected {self.seq_len}')
        if peak_memory.dim() != 3:
            raise ValueError('peak_memory must be 3D: (B, N_events, d_model)')
        if peak_memory.size(-1) != self.d_model:
            raise ValueError(f'peak_memory dim mismatch: got {peak_memory.size(-1)}, expected {self.d_model}')
        if peak_guidance.dim() != 3:
            raise ValueError('peak_guidance must be 3D: (B, pred_len, d_model)')
        if peak_guidance.size(1) != self.pred_len:
            raise ValueError(
                f'peak_guidance length mismatch: got {peak_guidance.size(1)}, expected {self.pred_len}'
            )
        if peak_guidance.size(2) != self.d_model:
            raise ValueError(
                f'peak_guidance dim mismatch: got {peak_guidance.size(2)}, expected {self.d_model}'
            )

        batch_size = x_norm.size(0)
        dense_hidden = self.dense_encoder(x_norm).view(batch_size, self.pred_len, self.d_model)
        if self.use_temporal and hours_future is not None and weekdays_future is not None:
            temporal = self.hour_embed(hours_future.long()) + self.weekday_embed(weekdays_future.long())
            gate = self.temporal_gate(temporal)
            dense_hidden = dense_hidden + gate * temporal
        query = self.query_norm(dense_hidden + peak_guidance)
        context, _ = self.peak_cross_attn(
            query,
            peak_memory,
            peak_memory,
            key_padding_mask=peak_key_padding_mask,
            need_weights=False,
        )
        gated_context = torch.tanh(peak_guidance) * torch.sigmoid(self.context_proj(context))
        hidden = self.context_norm(query + context + gated_context)
        hidden = self.out_norm(hidden + self.ffn(hidden))
        future_norm = self.proj(hidden).squeeze(-1)
        return future_norm, hidden

    def predict_apex_intensity(
        self,
        hidden: torch.Tensor,
        apex_positions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict intensity ratio at apex positions from dense hidden.

        Args:
            hidden: (B, pred_len, d_model) — dense hidden states
            apex_positions: (B, max_events) — apex index for each event
            valid_mask: (B, max_events) — True for valid events

        Returns:
            intensity: (B, max_events) — predicted intensity ratio
        """
        B, max_events = apex_positions.shape
        pred_len = hidden.size(1)
        apex_idx = apex_positions.clamp(0, pred_len - 1).long()
        apex_idx_expanded = apex_idx.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        hidden_at_apex = hidden.gather(1, apex_idx_expanded)
        intensity = self.apex_intensity_head(hidden_at_apex).squeeze(-1)
        intensity = intensity * valid_mask.float()
        return intensity


class TupleCrossAttnIntensityHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_int_bins: int,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.numeric_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.guidance_proj = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(d_model, n_int_bins)

    def forward(
        self,
        query_hidden: torch.Tensor,
        memory: torch.Tensor,
        onset_repr: torch.Tensor,
        duration_repr: torch.Tensor,
        apex_repr: torch.Tensor,
        tuple_numeric: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        numeric_hidden = self.numeric_proj(tuple_numeric)
        tuple_guidance = self.guidance_proj(
            torch.cat([onset_repr, duration_repr, apex_repr, numeric_hidden], dim=-1)
        )
        query = self.query_norm(query_hidden + tuple_guidance)
        context, _ = self.cross_attn(query, memory, memory, need_weights=False)
        gated_context = torch.tanh(tuple_guidance) * torch.sigmoid(self.context_proj(context))
        hidden = self.context_norm(query_hidden + context + gated_context)
        hidden = self.out_norm(hidden + self.ffn(hidden))
        return self.proj(hidden), hidden


class IntensityDecoder(nn.Module):
    """Direct intensity prediction from encoder memory + event location.
    
    Bypasses dense curve prediction entirely. Uses cross-attention to
    attend encoder memory conditioned on event location, then directly
    regresses intensity ratio.
    """

    def __init__(
        self,
        d_model: int,
        pred_len: int,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.onset_embed = nn.Embedding(pred_len, d_model)
        self.dur_embed = nn.Embedding(pred_len, d_model)
        self.apex_embed = nn.Embedding(pred_len, d_model)
        self.event_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )

    def forward(
        self,
        encoder_memory: torch.Tensor,
        onset: torch.Tensor,
        duration: torch.Tensor,
        apex: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict intensity ratio directly from encoder memory + event location.

        Args:
            encoder_memory: (B, 6, d_model) — encoder output
            onset: (B, max_events) — onset positions
            duration: (B, max_events) — duration values
            apex: (B, max_events) — apex positions
            valid_mask: (B, max_events) — True for valid events

        Returns:
            intensity: (B, max_events) — predicted intensity ratio
        """
        B, max_events = onset.shape
        onset_clamped = onset.clamp(0, self.pred_len - 1).long()
        dur_clamped = duration.clamp(0, self.pred_len - 1).long()
        apex_clamped = apex.clamp(0, self.pred_len - 1).long()
        onset_emb = self.onset_embed(onset_clamped)
        dur_emb = self.dur_embed(dur_clamped)
        apex_emb = self.apex_embed(apex_clamped)
        event_repr = self.event_proj(torch.cat([onset_emb, dur_emb, apex_emb], dim=-1))
        query = self.query_norm(event_repr)
        memory_expanded = encoder_memory.unsqueeze(1).expand(-1, max_events, -1, -1)
        memory_flat = memory_expanded.reshape(B * max_events, encoder_memory.size(1), -1)
        query_flat = query.reshape(B * max_events, 1, -1)
        context, _ = self.cross_attn(query_flat, memory_flat, memory_flat, need_weights=False)
        context = context.reshape(B, max_events, -1)
        gated_context = torch.tanh(event_repr) * torch.sigmoid(self.context_proj(context))
        hidden = self.context_norm(event_repr + context + gated_context)
        hidden = self.out_norm(hidden + self.ffn(hidden))
        intensity = self.intensity_head(hidden).squeeze(-1)
        intensity = intensity * valid_mask.float()
        return intensity
