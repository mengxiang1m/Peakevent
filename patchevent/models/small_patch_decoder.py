"""Autoregressive PatchEvent decoder model."""

from __future__ import annotations

import json as _json

import torch
import torch.nn as nn
import torch.nn.functional as F

from patchevent.phase1.loss import Phase1Loss
from patchevent.phase1.model import PatchEncoder
from patchevent.phase2.tuple_guided_value_head import (
    IndependentDenseForecastHead,
    TupleCrossAttnIntensityHead,
    TupleGuidedValueHead,
)
from patchevent.models.decomposition import InputSeriesDecomposition
from patchevent.models.encoders import CNNEncoder, LSTMEncoder, MLPEncoder
from patchevent.models.event_tokenizer import StructuredEventTokenizer

class SmallPatchDecoder(nn.Module):
    """
    PatchEncoder(frozen/unfreeze) + Memory Bridge + InputDecomp + AR Decoder。
    支持E5 Full配置及所有消融变体。保留跨域联合训练接口。
    """

    def __init__(
        self,
        encoder_ckpt_path: str,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        max_seq_len: int = 64,
        pred_len: int = 96,
        event_schema: str = 'quad',
        encoder_mode: str = 'frozen',  # frozen/scratch/cnn/lstm/mlp
        embed_dropout: float = 0.0,
        unfreeze_last_n: int = 0,
        use_memory_pos: bool = False,
        use_self_attn_agg: bool = False,
        use_input_decomp: bool = False,
        input_decomp_mode: str = 'full',
        use_raw_bypass: bool = False,
        use_tuple_guided_value: bool = False,
        use_independent_dense_int: bool = False,
        use_apex_cond_int: bool = False,
        use_int_direct_lookup: bool = False,
        use_decoupled_int_head: bool = False,
        use_tuple_crossatt_int: bool = False,
        min_onset_spacing: int = 0,
        override_norm_mean: float | None = None,
        override_norm_std: float | None = None,
        no_pos_valid_mask: bool = False,
        no_causal_mask: bool = False,
        use_int_regression: bool = False,
        use_series_stats: bool = False,
        series_stats_recent_k: int = 8,
        intensity_from_values: str = 'apex',
        # ── Event-Anchor Guided Decoding (方案A) ──
        use_count_head: bool = False,
        max_event_count: int = 12,
        # ── Position Regression Head (方案C') ──
        use_position_regression: bool = False,
        pos_reg_target: str = 'both',  # 'both', 'onset', 'apex'
        # ── Q-Bridge (Q-Former style Memory Bridge) ──
        use_qbridge: bool = False,
        qbridge_n_queries: int = 16,
        qbridge_n_layers: int = 2,
    ):
        super().__init__()
        self.tokenizer = StructuredEventTokenizer(pred_len=pred_len, event_schema=event_schema)
        self.encoder_mode = encoder_mode
        self.use_memory_pos = use_memory_pos
        self.use_self_attn_agg = use_self_attn_agg
        self.use_input_decomp = use_input_decomp
        self.use_raw_bypass = use_raw_bypass
        self.use_tuple_guided_value = use_tuple_guided_value
        self.use_independent_dense_int = use_independent_dense_int
        self.use_apex_cond_int = use_apex_cond_int
        self.use_int_direct_lookup = use_int_direct_lookup
        self.use_decoupled_int_head = use_decoupled_int_head
        self.use_tuple_crossatt_int = use_tuple_crossatt_int
        self.min_onset_spacing = min_onset_spacing
        self.unfreeze_last_n = unfreeze_last_n
        self.no_pos_valid_mask = no_pos_valid_mask
        self.no_causal_mask = no_causal_mask
        self.use_int_regression = use_int_regression
        self.use_series_stats = use_series_stats
        self.series_stats_recent_k = max(1, int(series_stats_recent_k))
        self.intensity_from_values = intensity_from_values
        self.use_count_head = use_count_head
        self.max_event_count = max_event_count
        self.use_position_regression = use_position_regression
        self.pos_reg_target = pos_reg_target
        self.use_qbridge = use_qbridge
        if self.use_tuple_guided_value and self.tokenizer.has_intensity:
            raise ValueError('use_tuple_guided_value requires event_schema="triplet"')
        if self.use_independent_dense_int and self.tokenizer.has_intensity:
            raise ValueError('use_independent_dense_int requires event_schema="triplet"')
        if self.use_tuple_crossatt_int and not self.tokenizer.has_intensity:
            raise ValueError('use_tuple_crossatt_int requires event_schema="quad"')
        if self.use_tuple_guided_value and (
            self.use_apex_cond_int or self.use_int_direct_lookup or self.use_decoupled_int_head or self.use_int_regression
        ):
            raise ValueError('tuple-guided value mode cannot be combined with intensity token heads')
        if self.use_independent_dense_int and (
            self.use_tuple_guided_value or self.use_apex_cond_int or self.use_int_direct_lookup
            or self.use_decoupled_int_head or self.use_int_regression or self.use_tuple_crossatt_int
        ):
            raise ValueError('independent dense intensity mode cannot be combined with other intensity/token heads')
        if self.use_tuple_crossatt_int and (
            self.use_tuple_guided_value or self.use_apex_cond_int or self.use_int_direct_lookup or self.use_decoupled_int_head or self.use_int_regression
        ):
            raise ValueError('tuple cross-att INT head cannot be combined with other intensity modes')
        vocab_size = self.tokenizer.VOCAB_SIZE

        # ── 1. Encoder ────────────────────────────────────────────────────────
        ckpt = torch.load(encoder_ckpt_path, map_location='cpu', weights_only=False)
        enc_args = ckpt['args']
        d_patch = enc_args.get('d_model', 128)

        n_tokens = (enc_args.get('seq_len', 96) - enc_args.get('patch_len', 8)) // enc_args.get('stride', 4) + 1

        if encoder_mode == 'cnn':
            n_tokens = (enc_args.get('seq_len', 96) - enc_args.get('patch_len', 8)) // enc_args.get('stride', 4) + 1
            self.encoder = CNNEncoder(
                seq_len=enc_args.get('seq_len', 96), n_tokens=n_tokens,
                d_model=d_patch, dropout=dropout,
            )
        elif encoder_mode == 'lstm':
            self.encoder = LSTMEncoder(
                seq_len=enc_args.get('seq_len', 96), n_tokens=n_tokens,
                d_model=d_patch, dropout=dropout,
            )
        elif encoder_mode == 'mlp':
            self.encoder = MLPEncoder(
                seq_len=enc_args.get('seq_len', 96),
                patch_len=enc_args.get('patch_len', 8),
                stride=enc_args.get('stride', 4),
                d_model=d_patch, dropout=dropout,
            )
        else:
            # PatchEncoder: frozen / scratch / joint_scratch
            self.encoder = PatchEncoder(
                seq_len=enc_args.get('seq_len', 96),
                patch_len=enc_args.get('patch_len', 8),
                stride=enc_args.get('stride', 4),
                d_model=d_patch,
                n_heads=enc_args.get('n_heads', 4),
                e_layers=enc_args.get('e_layers', 2),
                d_ff=enc_args.get('d_ff', 256),
                dropout=enc_args.get('dropout', 0.1),
                mask_ratio=enc_args.get('mask_ratio', 0.0),
            )
            # scratch: 随机初始化不加载权重; 其他模式加载预训练权重
            if encoder_mode not in ('scratch', 'joint_scratch'):
                self.encoder.load_state_dict(ckpt['model_state'], strict=False)
            if encoder_mode == 'frozen':
                for p in self.encoder.parameters():
                    p.requires_grad_(False)
                self.encoder.eval()

        # 选择性解冻最后N个encoder block
        if unfreeze_last_n > 0 and encoder_mode == 'frozen' and hasattr(self.encoder, 'encoder'):
            attn_layers = self.encoder.encoder.attn_layers
            n_total = len(attn_layers)
            n_unf = min(unfreeze_last_n, n_total)
            for i in range(n_total - n_unf, n_total):
                for p in attn_layers[i].parameters():
                    p.requires_grad_(True)
            cnt = sum(p.numel() for layer in attn_layers[n_total - n_unf:] for p in layer.parameters())
            print(f'[Encoder] unfroze last {n_unf} block(s): {cnt:,} params')

        # 联合训练模式保留Phase1 loss接口
        if encoder_mode in ('joint', 'joint_scratch'):
            self.phase1_loss = Phase1Loss()

        _mean = override_norm_mean if override_norm_mean is not None else ckpt['mean']
        _std = override_norm_std if override_norm_std is not None else ckpt['std']
        self.register_buffer('norm_mean', torch.tensor(_mean, dtype=torch.float32))
        self.register_buffer('norm_std', torch.tensor(_std, dtype=torch.float32))
        self.n_patches = self.encoder.n_patches
        self.patch_len = enc_args.get('patch_len', 8)
        self.patch_stride = enc_args.get('stride', 4)

        # ── 2. Memory Bridge ──────────────────────────────────────────────────
        # Patch Projector
        self.patch_proj = nn.Sequential(
            nn.Linear(d_patch, d_model),
            nn.LayerNorm(d_model),
        )

        # Pre-Transformer Patch Bypass: project normalized raw patches directly.
        if use_raw_bypass:
            self.raw_patch_proj = nn.Sequential(
                nn.Linear(self.patch_len, d_model),
                nn.LayerNorm(d_model),
            )

        # Memory Position Encoding
        if use_memory_pos:
            max_mem_tokens = self.n_patches * (2 if use_raw_bypass else 1) + 4
            self.memory_pos_embed = nn.Embedding(max_mem_tokens, d_model)

        # Self-Attention Aggregation
        if use_self_attn_agg:
            self.self_attn_agg = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, activation='gelu', batch_first=True,
            )

        # Q-Bridge: learnable queries cross-attend to encoder memory (BLIP-2 Q-Former style)
        if use_qbridge:
            self.qbridge_queries = nn.Parameter(torch.randn(qbridge_n_queries, d_model) * 0.02)
            qbridge_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, activation='gelu', batch_first=True,
            )
            self.qbridge_decoder = nn.TransformerDecoder(qbridge_layer, num_layers=qbridge_n_layers)
            self.qbridge_ln = nn.LayerNorm(d_model)
            _qb_params = sum(p.numel() for p in [self.qbridge_queries]) + \
                         sum(p.numel() for p in self.qbridge_decoder.parameters()) + \
                         sum(p.numel() for p in self.qbridge_ln.parameters())
            print(f'[Q-Bridge] {_qb_params:,} params (n_queries={qbridge_n_queries}, n_layers={qbridge_n_layers})')

        # InputSeriesDecomposition (输入级分解)
        if use_input_decomp:
            _seq_len = enc_args.get('seq_len', 96)
            _patch_len = enc_args.get('patch_len', 8)
            _stride = enc_args.get('stride', 4)
            self.input_decomp = InputSeriesDecomposition(
                seq_len=_seq_len, patch_len=_patch_len, stride=_stride,
                d_model=d_model, alpha=0.3, mode=input_decomp_mode,
            )
            print(f'[InputDecomp] {sum(p.numel() for p in self.input_decomp.parameters()):,} params (mode={input_decomp_mode})')

        # Series Statistics Injection (mean/std/recent/max/min -> 1 memory token)
        if use_series_stats:
            self.series_stats_proj = nn.Sequential(
                nn.Linear(5, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            print(f'[SeriesStats] {sum(p.numel() for p in self.series_stats_proj.parameters()):,} params (recent_k={self.series_stats_recent_k})')

        # ── 3. Token Embedding + Positional ──────────────────────────────────
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=self.tokenizer.PAD_ID)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_drop = nn.Dropout(embed_dropout) if embed_dropout > 0 else nn.Identity()
        self.out_drop = nn.Dropout(dropout)

        # ── 4. Transformer Decoder ───────────────────────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # ── 5. Output Head ───────────────────────────────────────────────────
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Apex-conditioned INT head: [decoder_hidden, raw_apex_value] -> INT bins.
        if use_apex_cond_int:
            self.apex_int_head = nn.Sequential(
                nn.Linear(d_model + 1, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, self.tokenizer.N_INT),
            )

        # Decoupled INT head (keeps lm_head untouched for onset/dur/apex tokens).
        if use_decoupled_int_head:
            self.int_cls_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, self.tokenizer.N_INT),
            )

        if use_tuple_crossatt_int:
            self.tuple_crossatt_int_head = TupleCrossAttnIntensityHead(
                d_model=d_model,
                n_int_bins=self.tokenizer.N_INT,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )

        # ── 5b. Intensity Regression Head (optional) ──────────────────────
        if use_int_regression:
            self.int_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
            print(f'[IntRegressionHead] {sum(p.numel() for p in self.int_head.parameters()):,} params')

        if use_tuple_guided_value:
            self.guidance_proj = nn.Sequential(
                nn.Linear(4, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.future_pos_embed = nn.Embedding(pred_len, d_model)
            self.tuple_value_head = TupleGuidedValueHead(
                d_model=d_model,
                pred_len=pred_len,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )

        if use_independent_dense_int:
            _seq_len = int(enc_args.get('seq_len', 96))
            self.indep_peak_feat_proj = nn.Sequential(
                nn.Linear(4, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.indep_peak_pos_embed = nn.Embedding(pred_len, d_model)
            self.indep_future_pos_embed = nn.Embedding(pred_len, d_model)
            self.independent_dense_head = IndependentDenseForecastHead(
                seq_len=_seq_len,
                pred_len=pred_len,
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_temporal=True,
            )

        # ── 5c. Count Prediction Head (方案A) ────────────────────────────
        if use_count_head:
            # Count classifier: memory pooling → predicted count
            self.count_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, max_event_count + 1),  # 0..max_event_count
            )
            # Count embedding: inject predicted count as extra memory token
            # so decoder cross-attention can condition on expected event count
            self.count_embed = nn.Embedding(max_event_count + 1, d_model)
            _cnt_params = sum(p.numel() for p in self.count_head.parameters()) + self.count_embed.weight.numel()
            print(f'[CountHead] {_cnt_params:,} params (max_count={max_event_count}, +count_embed)')

        # ── 5d. Position Regression Head (方案C') ─────────────────────────
        if use_position_regression:
            # Predicts continuous onset/apex positions from decoder hidden states
            # at onset/apex token positions, complementing the discrete CE loss
            self.pos_reg_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
            print(f'[PosRegHead] {sum(p.numel() for p in self.pos_reg_head.parameters()):,} params')

        # ── 6. 位置约束掩码 ──────────────────────────────────────────────────
        self.max_seq_len = max_seq_len
        pos_mask = self.tokenizer.get_position_valid_mask(max_seq_len)
        self.register_buffer('pos_valid_mask', pos_mask)

        # 参数统计
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f'[SmallPatchDecoder] trainable: {trainable:,} || total: {total:,} || trainable%: {trainable/total*100:.2f}')

    def train(self, mode: bool = True):
        """frozen模式encoder始终eval。"""
        super().train(mode)
        if self.encoder_mode == 'frozen':
            self.encoder.eval()
        return self

    def _encode_patches(self, x: torch.Tensor, return_x_norm: bool = False):
        """x: (B, 96) → memory: (B, N, d_model), optional x_norm"""
        x_norm = (x.float() - self.norm_mean) / self.norm_std

        # Encoder
        if self.encoder_mode == 'frozen' and self.unfreeze_last_n == 0:
            with torch.no_grad():
                enc_out = self.encoder(x_norm)
        else:
            enc_out = self.encoder(x_norm)
        z = enc_out['z'].float()  # (B, 23, d_patch)

        # Patch Projector
        memory = self.patch_proj(z)  # (B, 23, d_model)

        # Memory Position Encoding
        if self.use_memory_pos:
            N_mem = memory.size(1)
            mem_positions = torch.arange(N_mem, device=memory.device)
            memory = memory + self.memory_pos_embed(mem_positions).unsqueeze(0)

        # Self-Attention Aggregation
        if self.use_self_attn_agg:
            memory = self.self_attn_agg(memory)

        # Q-Bridge: learnable queries extract task-relevant features from encoder memory
        if self.use_qbridge:
            B = memory.size(0)
            queries = self.qbridge_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d)
            # Cross-attention: queries attend to encoder memory
            queries = self.qbridge_decoder(queries, memory)  # (B, K, d)
            memory = self.qbridge_ln(queries)  # replace memory with compressed queries

        # InputSeriesDecomposition
        if self.use_input_decomp:
            memory = self.input_decomp(x_norm, memory)

        if self.use_raw_bypass:
            raw_patches = x_norm.unfold(1, self.patch_len, self.patch_stride)
            raw_memory = self.raw_patch_proj(raw_patches)
            if self.use_memory_pos:
                start = memory.size(1)
                pos = torch.arange(start, start + raw_memory.size(1), device=memory.device)
                raw_memory = raw_memory + self.memory_pos_embed(pos).unsqueeze(0)
            memory = torch.cat([memory, raw_memory], dim=1)

        if self.use_series_stats:
            stats_token = self._build_series_stats_token(x).unsqueeze(1)
            memory = torch.cat([memory, stats_token], dim=1)

        # Count-conditioned memory injection: predict count → embed → append to memory
        if self.use_count_head:
            pool = memory.mean(dim=1)  # (B, d)
            count_logits = self.count_head(pool)  # (B, max_count+1)
            if self.training:
                # Soft: use Gumbel-softmax for differentiable count embedding
                count_weights = F.softmax(count_logits, dim=-1)  # (B, max_count+1)
                count_token = torch.matmul(count_weights, self.count_embed.weight)  # (B, d)
            else:
                # Hard: argmax during inference
                count_pred = count_logits.argmax(dim=-1)  # (B,)
                count_token = self.count_embed(count_pred)  # (B, d)
            memory = torch.cat([memory, count_token.unsqueeze(1)], dim=1)
            # Store count_logits for loss computation
            self._last_count_logits = count_logits

        if return_x_norm:
            return memory, x_norm
        return memory

    def _encode_patches_with_heads(self, x: torch.Tensor):
        """联合训练模式: 返回memory和encoder四头预测。"""
        x_norm = (x.float() - self.norm_mean) / self.norm_std
        enc_out = self.encoder(x_norm)
        z = enc_out['z'].float()
        memory = self.patch_proj(z)
        if self.use_memory_pos:
            N_mem = memory.size(1)
            mem_positions = torch.arange(N_mem, device=memory.device)
            memory = memory + self.memory_pos_embed(mem_positions).unsqueeze(0)
        if self.use_self_attn_agg:
            memory = self.self_attn_agg(memory)
        if self.use_input_decomp:
            memory = self.input_decomp(x_norm, memory)
        if self.use_raw_bypass:
            raw_patches = x_norm.unfold(1, self.patch_len, self.patch_stride)
            raw_memory = self.raw_patch_proj(raw_patches)
            if self.use_memory_pos:
                start = memory.size(1)
                pos = torch.arange(start, start + raw_memory.size(1), device=memory.device)
                raw_memory = raw_memory + self.memory_pos_embed(pos).unsqueeze(0)
            memory = torch.cat([memory, raw_memory], dim=1)
        if self.use_series_stats:
            stats_token = self._build_series_stats_token(x).unsqueeze(1)
            memory = torch.cat([memory, stats_token], dim=1)
        return memory, enc_out

    def _build_series_stats_token(self, x: torch.Tensor) -> torch.Tensor:
        """Build a single token from raw-series stats for absolute scale cues."""
        x_f = x.float()
        recent_k = min(self.series_stats_recent_k, x_f.size(1))
        recent_level = x_f[:, -recent_k:].mean(dim=1)
        stats = torch.stack([
            x_f.mean(dim=1),
            x_f.std(dim=1, unbiased=False),
            recent_level,
            x_f.max(dim=1).values,
            x_f.min(dim=1).values,
        ], dim=-1)
        return self.series_stats_proj(stats)

    def _events_from_token_batch(self, token_batch: torch.Tensor) -> list[list[dict]]:
        events = []
        token_cpu = token_batch.detach().cpu()
        for i in range(token_cpu.size(0)):
            events.append(self.tokenizer.decode_events(token_cpu[i].tolist()))
        return events

    def _extract_events_gpu(
        self,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """GPU-only event extraction from target_ids — no CPU transfer.

        Returns:
            onset_vals: (B, max_events) int, clamped to [0, pred_len-1]
            dur_vals:   (B, max_events) int, clamped to [3, 5]
            apex_vals:  (B, max_events) int, clamped to [0, pred_len-1]
            valid:      (B, max_events) bool
        """
        B, T = token_ids.shape
        es = self.tokenizer.event_size  # 3 for triplet
        tok = self.tokenizer
        # event tokens start after BOS (index 0); max possible events
        max_ev = max(1, (T - 2) // es)
        idx = torch.arange(max_ev, device=token_ids.device)
        onset_idx = 1 + idx * es
        dur_idx = onset_idx + 1
        apex_idx = onset_idx + 2
        # clamp to sequence length
        onset_idx = onset_idx[onset_idx < T]
        dur_idx = dur_idx[dur_idx < T]
        apex_idx = apex_idx[apex_idx < T]
        n = min(len(onset_idx), len(dur_idx), len(apex_idx))
        onset_idx, dur_idx, apex_idx = onset_idx[:n], dur_idx[:n], apex_idx[:n]

        onset_tok = token_ids[:, onset_idx]          # (B, n)
        dur_tok = token_ids[:, dur_idx]
        apex_tok = token_ids[:, apex_idx]

        onset_vals = onset_tok - tok.ONSET_OFFSET
        dur_vals = (dur_tok - tok.DUR_OFFSET) + 3
        apex_vals = apex_tok - tok.APEX_OFFSET

        valid = (
            (onset_tok != tok.PAD_ID)
            & (onset_tok != tok.EOS_ID)
            & (onset_vals >= 0) & (onset_vals < tok.N_ONSET)
            & (dur_vals >= 3) & (dur_vals <= 5)
            & (apex_vals >= 0) & (apex_vals < tok.N_APEX)
        )
        onset_vals = onset_vals.clamp(0, tok.pred_len - 1)
        dur_vals = dur_vals.clamp(3, 5)
        apex_vals = apex_vals.clamp(0, tok.pred_len - 1)
        return onset_vals, dur_vals, apex_vals, valid

    def _build_independent_peak_memory_gpu(
        self,
        onset_vals: torch.Tensor,
        dur_vals: torch.Tensor,
        apex_vals: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fully-vectorised peak memory/guidance construction on GPU.

        Args — all (B, max_events) tensors on device.
        Returns:
            peak_memory:           (B, max_events, d_model)
            peak_key_padding_mask: (B, max_events) bool, True=pad
            peak_guidance:         (B, pred_len, d_model)
        """
        pred_len = self.tokenizer.pred_len
        pos_denom = float(max(pred_len - 1, 1))
        span_denom = float(max(pred_len, 1))

        onset_f = onset_vals.float()
        dur_f = dur_vals.float()
        apex_f = apex_vals.float()
        apex_offset_f = (apex_f - onset_f).clamp(min=0)

        # (B, max_events, 4)
        numeric = torch.stack([
            onset_f / pos_denom,
            dur_f / span_denom,
            apex_f / pos_denom,
            apex_offset_f / span_denom,
        ], dim=-1)

        # (B, max_events, d_model)
        event_repr = self.indep_peak_feat_proj(numeric)
        event_repr = event_repr + self.indep_peak_pos_embed(apex_vals.clamp(0, pred_len - 1).long())
        peak_memory = event_repr * valid.unsqueeze(-1).float()

        # Gaussian guidance: (B, max_events, pred_len)
        pos = torch.arange(pred_len, device=onset_vals.device, dtype=torch.float32)
        sigma = (dur_f / 2.0).clamp(min=1.0)                       # (B, me)
        gauss = torch.exp(
            -0.5 * ((pos.view(1, 1, -1) - apex_f.unsqueeze(-1)) / sigma.unsqueeze(-1)) ** 2
        )
        gauss = gauss * valid.unsqueeze(-1).float()                 # mask invalid

        # peak_guidance = sum_j gauss[b,j,:] * event_repr[b,j,:] / n_events
        # => bmm: (B, pred_len, me) @ (B, me, d_model) → (B, pred_len, d_model)
        peak_guidance = torch.bmm(gauss.transpose(1, 2), event_repr)
        n_events = valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        peak_guidance = peak_guidance / n_events.unsqueeze(-1)

        # future positional embedding
        positions = torch.arange(pred_len, device=onset_vals.device)
        peak_guidance = peak_guidance + self.indep_future_pos_embed(positions).unsqueeze(0)

        return peak_memory, ~valid, peak_guidance

    def _build_independent_peak_memory(
        self,
        events_batch: list[list[dict]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = len(events_batch)
        pred_len = self.tokenizer.pred_len
        d_model = self.pos_embed.embedding_dim
        max_events = max(1, max((len(events) for events in events_batch), default=0))

        peak_memory = torch.zeros(B, max_events, d_model, device=device)
        peak_valid = torch.zeros(B, max_events, dtype=torch.bool, device=device)
        peak_guidance = torch.zeros(B, pred_len, d_model, device=device)
        pos = torch.arange(pred_len, device=device, dtype=torch.float32)

        pos_denom = float(max(pred_len - 1, 1))
        span_denom = float(max(pred_len, 1))
        for b, events in enumerate(events_batch):
            if len(events) == 0:
                peak_valid[b, 0] = True
                continue
            for j, ev in enumerate(events):
                onset = min(max(int(ev.get('onset', 0)), 0), pred_len - 1)
                duration = max(int(ev.get('duration', 1)), 1)
                apex = min(max(int(ev.get('apex_index', onset)), 0), pred_len - 1)
                apex_offset = max(apex - onset, 0)
                numeric = torch.tensor([
                    float(onset) / pos_denom,
                    float(duration) / span_denom,
                    float(apex) / pos_denom,
                    float(apex_offset) / span_denom,
                ], device=device).unsqueeze(0)
                event_repr = self.indep_peak_feat_proj(numeric).squeeze(0)
                event_repr = event_repr + self.indep_peak_pos_embed(
                    torch.tensor(apex, device=device, dtype=torch.long)
                )
                peak_memory[b, j, :] = event_repr
                peak_valid[b, j] = True

                sigma = max(float(duration) / 2.0, 1.0)
                gauss = torch.exp(-0.5 * ((pos - float(apex)) / sigma) ** 2).unsqueeze(-1)
                peak_guidance[b] = peak_guidance[b] + gauss * event_repr.unsqueeze(0)
            peak_guidance[b] = peak_guidance[b] / float(len(events))

        positions = torch.arange(pred_len, device=device)
        peak_guidance = peak_guidance + self.indep_future_pos_embed(positions).unsqueeze(0)
        peak_key_padding_mask = ~peak_valid
        return peak_memory, peak_key_padding_mask, peak_guidance

    def _predict_independent_future_values_norm(
        self,
        x_norm: torch.Tensor,
        events_batch: list[list[dict]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        peak_memory, peak_key_padding_mask, peak_guidance = self._build_independent_peak_memory(
            events_batch=events_batch,
            device=x_norm.device,
        )
        return self.independent_dense_head(
            x_norm=x_norm,
            peak_memory=peak_memory,
            peak_guidance=peak_guidance,
            peak_key_padding_mask=peak_key_padding_mask,
        )

    def _build_tuple_guidance(self, events_batch: list[list[dict]], device: torch.device) -> torch.Tensor:
        B = len(events_batch)
        pred_len = self.tokenizer.pred_len
        base = torch.zeros(B, pred_len, 4, device=device)
        pos = torch.arange(pred_len, device=device, dtype=torch.float32)
        for b, events in enumerate(events_batch):
            for ev in events:
                onset = min(max(int(ev.get('onset', 0)), 0), pred_len - 1)
                duration = max(int(ev.get('duration', 1)), 1)
                apex = min(max(int(ev.get('apex_index', onset)), 0), pred_len - 1)
                end = min(pred_len, onset + duration)
                if end > onset:
                    base[b, onset:end, 0] = 1.0
                    base[b, onset:end, 3] = torch.maximum(
                        base[b, onset:end, 3],
                        torch.full((end - onset,), float(duration) / float(pred_len), device=device),
                    )
                base[b, onset, 1] = 1.0
                base[b, apex, 2] = 1.0
                sigma = max(float(duration) / 2.0, 1.0)
                gauss = torch.exp(-0.5 * ((pos - float(apex)) / sigma) ** 2)
                base[b, :, 3] = torch.maximum(base[b, :, 3], gauss)
        positions = torch.arange(pred_len, device=device)
        return self.guidance_proj(base) + self.future_pos_embed(positions).unsqueeze(0)

    def _predict_future_values_norm(
        self,
        memory: torch.Tensor,
        events_batch: list[list[dict]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        guidance = self._build_tuple_guidance(events_batch, memory.device)
        return self.tuple_value_head(memory, guidance)

    def _future_norm_to_raw(self, future_norm: torch.Tensor) -> torch.Tensor:
        return future_norm * self.norm_std + self.norm_mean

    def _tuple_numeric_from_tokens(
        self,
        onset_tokens: torch.Tensor,
        duration_tokens: torch.Tensor,
        apex_tokens: torch.Tensor,
    ) -> torch.Tensor:
        pos_denom = float(max(self.tokenizer.pred_len - 1, 1))
        span_denom = float(max(self.tokenizer.pred_len, 1))
        onset = (onset_tokens - self.tokenizer.ONSET_OFFSET).clamp(0, self.tokenizer.N_ONSET - 1).float()
        duration = ((duration_tokens - self.tokenizer.DUR_OFFSET) + 3).clamp(3, 5).float()
        apex = (apex_tokens - self.tokenizer.APEX_OFFSET).clamp(0, self.tokenizer.N_APEX - 1).float()
        apex_offset = (apex - onset).clamp(min=0.0)
        return torch.stack([
            onset / pos_denom,
            duration / span_denom,
            apex / pos_denom,
            apex_offset / span_denom,
        ], dim=-1)

    def _predict_tuple_crossatt_int_logits(
        self,
        memory: torch.Tensor,
        hidden: torch.Tensor,
        decoder_input: torch.Tensor,
        int_positions: torch.Tensor,
    ) -> torch.Tensor | None:
        if len(int_positions) == 0:
            return None
        onset_positions = int_positions - 2
        duration_positions = int_positions - 1
        apex_positions = int_positions
        query_hidden = hidden[:, int_positions, :]
        onset_repr = self.token_embed(decoder_input[:, onset_positions])
        duration_repr = self.token_embed(decoder_input[:, duration_positions])
        apex_repr = self.token_embed(decoder_input[:, apex_positions])
        tuple_numeric = self._tuple_numeric_from_tokens(
            decoder_input[:, onset_positions],
            decoder_input[:, duration_positions],
            decoder_input[:, apex_positions],
        )
        int_logits, _ = self.tuple_crossatt_int_head(
            query_hidden=query_hidden,
            memory=memory,
            onset_repr=onset_repr,
            duration_repr=duration_repr,
            apex_repr=apex_repr,
            tuple_numeric=tuple_numeric,
        )
        return int_logits

    def _events_to_output_json(
        self,
        events_batch: list[list[dict]],
        future_values_raw: torch.Tensor | None = None,
    ) -> list[str]:
        outputs = []
        mean_val = max(float(self.norm_mean.item()), 1e-8)
        future_cpu = future_values_raw.detach().cpu() if future_values_raw is not None else None
        for b, events in enumerate(events_batch):
            packed = []
            for ev in events:
                intensity = ev.get('apex_intensity', float('nan'))
                if future_cpu is not None:
                    onset = min(max(int(ev['onset']), 0), future_cpu.size(1) - 1)
                    duration = max(int(ev['duration']), 1)
                    apex = min(max(int(ev['apex_index']), 0), future_cpu.size(1) - 1)
                    if self.intensity_from_values == 'span_max':
                        end = min(future_cpu.size(1), onset + duration)
                        if end <= onset:
                            intensity = float(future_cpu[b, apex].item()) / mean_val
                        else:
                            intensity = float(future_cpu[b, onset:end].max().item()) / mean_val
                    else:
                        intensity = float(future_cpu[b, apex].item()) / mean_val
                packed.append([
                    int(ev['onset']),
                    int(ev['duration']),
                    int(ev['apex_index']),
                    round(float(intensity), 2),
                ])
            outputs.append(_json.dumps({'peak_events': packed}, separators=(',', ':')))
        return outputs

    def _events_to_output_json_with_intensity(
        self,
        events_batch: list[list[dict]],
        predicted_intensity: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> list[str]:
        """Convert events to JSON using predicted intensity from intensity regression head."""
        outputs = []
        intensity_cpu = predicted_intensity.detach().cpu()
        for b, events in enumerate(events_batch):
            packed = []
            for j, ev in enumerate(events):
                if j < intensity_cpu.size(1):
                    intensity = float(intensity_cpu[b, j].item())
                    if intensity <= 0 or intensity != intensity:  # nan check
                        intensity = 1.0  # fallback
                else:
                    intensity = 1.0  # fallback
                packed.append([
                    int(ev['onset']),
                    int(ev['duration']),
                    int(ev['apex_index']),
                    round(float(intensity), 2),
                ])
            outputs.append(_json.dumps({'peak_events': packed}, separators=(',', ':')))
        return outputs

    def _make_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(T, device=device)

    def forward(self, x: torch.Tensor, target_ids: torch.Tensor, **kwargs):
        """训练forward: x (B,96), target_ids (B,T) → logits (B,T-1,V) [, int_preds (B,T-1)]"""
        if self.use_apex_cond_int or self.use_independent_dense_int:
            memory, x_norm = self._encode_patches(x, return_x_norm=True)
        else:
            memory = self._encode_patches(x)

        decoder_input = target_ids[:, :-1]
        T = decoder_input.size(1)
        positions = torch.arange(T, device=x.device)
        tgt = self.embed_drop(self.token_embed(decoder_input) + self.pos_embed(positions))

        causal_mask = None if self.no_causal_mask else self._make_causal_mask(T, x.device)
        tgt_pad_mask = (decoder_input == self.tokenizer.PAD_ID)

        out = self.decoder(tgt, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_pad_mask)
        hidden = self.out_drop(out)
        logits = self.lm_head(hidden)

        # 位置约束掩码
        if not self.no_pos_valid_mask and T <= self.max_seq_len:
            pos_mask = self.pos_valid_mask[:T]
            logits = logits.masked_fill(~pos_mask.unsqueeze(0), float('-inf'))

        # ── Compute auxiliary outputs (count / pos_reg) once ──────────────
        _aux = {}
        if self.use_count_head:
            _aux['count_logits'] = getattr(self, '_last_count_logits', None)
        if self.use_position_regression:
            pos_preds = self.pos_reg_head(hidden).squeeze(-1)  # (B, T-1)
            _aux['pos_reg_preds'] = pos_preds

        if self.use_tuple_guided_value:
            events_batch = self._events_from_token_batch(target_ids)
            future_values_norm, _ = self._predict_future_values_norm(memory, events_batch)
            return {
                'logits': logits,
                'future_values_norm': future_values_norm,
                'events_batch': events_batch,
                **_aux,
            }

        if self.use_independent_dense_int:
            onset_vals, dur_vals, apex_vals, ev_valid = self._extract_events_gpu(target_ids)
            peak_mem, peak_kpm, peak_guide = self._build_independent_peak_memory_gpu(
                onset_vals, dur_vals, apex_vals, ev_valid,
            )
            future_values_norm, _ = self.independent_dense_head(
                x_norm=x_norm,
                peak_memory=peak_mem,
                peak_guidance=peak_guide,
                peak_key_padding_mask=peak_kpm,
                hours_future=kwargs.get('hours_future'),
                weekdays_future=kwargs.get('weekdays_future'),
            )
            return {
                'logits': logits,
                'future_values_norm': future_values_norm,
                'event_onset': onset_vals,
                'event_dur': dur_vals,
                'event_apex': apex_vals,
                'event_valid': ev_valid,
                **_aux,
            }

        if self.use_tuple_crossatt_int:
            int_positions = self.tokenizer.get_attr_positions(T, 'intensity', x.device)
            if len(int_positions) > 0:
                int_logits = self._predict_tuple_crossatt_int_logits(memory, hidden, decoder_input, int_positions)
                lo = self.tokenizer.INT_OFFSET
                hi = lo + self.tokenizer.N_INT
                logits[:, int_positions, lo:hi] = int_logits

        if self.use_apex_cond_int:
            int_positions = self.tokenizer.get_attr_positions(T, 'intensity', x.device)
            if len(int_positions) > 0:
                apex_tokens = decoder_input[:, int_positions]
                apex_pos = (apex_tokens - self.tokenizer.APEX_OFFSET).clamp(0, self.tokenizer.N_APEX - 1).long()
                apex_val = torch.gather(x_norm, 1, apex_pos).unsqueeze(-1)
                int_hidden = hidden[:, int_positions, :]
                fused = torch.cat([int_hidden, apex_val], dim=-1)
                int_logits = self.apex_int_head(fused)
                lo = self.tokenizer.INT_OFFSET
                hi = lo + self.tokenizer.N_INT
                logits[:, int_positions, lo:hi] = int_logits

        if self.use_int_regression:
            int_preds = self.int_head(hidden).squeeze(-1)  # (B, T-1)
            return logits, int_preds

        if self.use_decoupled_int_head:
            int_positions = self.tokenizer.get_attr_positions(T, 'intensity', x.device)
            int_cls_logits = self.int_cls_head(hidden[:, int_positions, :]) if len(int_positions) > 0 else None
            return logits, int_cls_logits, int_positions

        if _aux:
            _aux['logits'] = logits
            return _aux

        return logits

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_new_tokens: int = 60, return_aux: bool = False, **kwargs):
        """自回归推理: x (B,96) → list[str] JSON事件序列"""
        B = x.size(0)
        device = x.device
        if self.use_apex_cond_int or self.use_int_direct_lookup or self.use_independent_dense_int:
            memory, x_norm = self._encode_patches(x, return_x_norm=True)
        else:
            memory = self._encode_patches(x)

        generated = torch.full((B, 1), self.tokenizer.BOS_ID, device=device, dtype=torch.long)
        last_onset = torch.full((B,), -1, device=device, dtype=torch.long)

        for step in range(max_new_tokens):
            T = generated.size(1)
            if T > self.max_seq_len:
                break
            positions = torch.arange(T, device=device)
            tgt = self.token_embed(generated) + self.pos_embed(positions)
            causal_mask = None if self.no_causal_mask else self._make_causal_mask(T, device)

            out = self.decoder(tgt, memory, tgt_mask=causal_mask)
            next_logits = self.lm_head(out[:, -1, :])

            # 位置约束
            k = T - 1
            if not self.no_pos_valid_mask:
                pos_mask = self.pos_valid_mask[k]
                next_logits = next_logits.masked_fill(~pos_mask.unsqueeze(0), float('-inf'))

            # onset单调+最小间距约束
            if not self.no_pos_valid_mask and k % self.tokenizer.event_size == 0 and (last_onset >= 0).any():
                for b in range(B):
                    if last_onset[b] >= 0:
                        lo = self.tokenizer.ONSET_OFFSET
                        min_next = int(last_onset[b].item()) + max(1, self.min_onset_spacing)
                        hi = self.tokenizer.ONSET_OFFSET + min_next
                        if hi > lo:
                            next_logits[b, lo:hi] = float('-inf')

            # INT position: use regression head if enabled
            is_int_pos = self.tokenizer.has_intensity and k % self.tokenizer.event_size == self.tokenizer.intensity_pos
            if self.use_int_regression and is_int_pos:
                int_val = self.int_head(out[:, -1, :]).squeeze(-1)  # (B,)
                tok = self.tokenizer
                bin_idx = ((int_val - tok.INT_MIN) / tok.INT_STEP).round().long()
                bin_idx = bin_idx.clamp(0, tok.N_INT - 1)
                next_token = (bin_idx + tok.INT_OFFSET).unsqueeze(-1)
            elif self.use_tuple_crossatt_int and is_int_pos and generated.size(1) >= 4:
                int_pos = torch.tensor([generated.size(1) - 1], device=device)
                int_logits = self._predict_tuple_crossatt_int_logits(memory, out, generated, int_pos)
                lo = self.tokenizer.INT_OFFSET
                hi = lo + self.tokenizer.N_INT
                next_logits[:, lo:hi] = int_logits[:, 0, :]
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            elif self.use_int_direct_lookup and is_int_pos:
                apex_token = generated[:, -1]
                apex_pos = (apex_token - self.tokenizer.APEX_OFFSET).clamp(0, self.tokenizer.N_APEX - 1).long()
                # intensity_norm = x_raw[apex] / mean = x_norm[apex] * (std/mean) + 1
                intensity_norm = x_norm[torch.arange(B, device=device), apex_pos] * (self.norm_std / self.norm_mean) + 1.0
                tok = self.tokenizer
                bin_idx = ((intensity_norm - tok.INT_MIN) / tok.INT_STEP).round().long()
                bin_idx = bin_idx.clamp(0, tok.N_INT - 1)
                next_token = (bin_idx + tok.INT_OFFSET).unsqueeze(-1)
            elif self.use_apex_cond_int and is_int_pos and generated.size(1) >= 1:
                apex_token = generated[:, -1]
                apex_pos = (apex_token - self.tokenizer.APEX_OFFSET).clamp(0, self.tokenizer.N_APEX - 1).long()
                apex_val = x_norm[torch.arange(B, device=device), apex_pos].unsqueeze(-1)
                fused = torch.cat([out[:, -1, :], apex_val], dim=-1)
                int_logits = self.apex_int_head(fused)
                lo = self.tokenizer.INT_OFFSET
                hi = lo + self.tokenizer.N_INT
                next_logits[:, lo:hi] = int_logits
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            elif self.use_decoupled_int_head and is_int_pos:
                int_logits = self.int_cls_head(out[:, -1, :])
                lo = self.tokenizer.INT_OFFSET
                hi = lo + self.tokenizer.N_INT
                next_logits[:, lo:hi] = int_logits
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            # 更新last_onset
            if k % self.tokenizer.event_size == 0:
                for b in range(B):
                    tok = next_token[b, 0].item()
                    if self.tokenizer.ONSET_OFFSET <= tok < self.tokenizer.ONSET_OFFSET + self.tokenizer.N_ONSET:
                        last_onset[b] = tok - self.tokenizer.ONSET_OFFSET

            if (next_token.squeeze(-1) == self.tokenizer.EOS_ID).all():
                break

        if self.use_tuple_guided_value or self.use_independent_dense_int:
            events_batch = self._events_from_token_batch(generated)
            if self.use_tuple_guided_value:
                future_values_norm, _ = self._predict_future_values_norm(memory, events_batch)
                future_values_raw = self._future_norm_to_raw(future_values_norm)
                pred_strs = self._events_to_output_json(events_batch, future_values_raw=future_values_raw)
            else:
                onset_v, dur_v, apex_v, ev_v = self._extract_events_gpu(generated)
                pm, kpm, pg = self._build_independent_peak_memory_gpu(onset_v, dur_v, apex_v, ev_v)
                future_values_norm, _ = self.independent_dense_head(
                    x_norm=x_norm, peak_memory=pm, peak_guidance=pg, peak_key_padding_mask=kpm,
                    hours_future=kwargs.get('hours_future'),
                    weekdays_future=kwargs.get('weekdays_future'),
                )
                future_values_raw = self._future_norm_to_raw(future_values_norm)
                pred_strs = self._events_to_output_json(events_batch, future_values_raw=future_values_raw)
            if return_aux:
                return pred_strs, {'future_values': future_values_raw}
            return pred_strs

        pred_strs = [self.tokenizer.decode(generated[i].tolist()) for i in range(B)]
        if return_aux:
            return pred_strs, {}
        return pred_strs

    def get_param_groups(self, lr: float = 1e-3, encoder_lr: float | None = None) -> list[dict]:
        """返回参数组，unfreeze/scratch模式对encoder使用不同lr。"""
        if self.unfreeze_last_n > 0 and encoder_lr is not None:
            enc_params = [p for n, p in self.named_parameters() if p.requires_grad and n.startswith('encoder.')]
            dec_params = [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith('encoder.')]
            groups = []
            if enc_params:
                groups.append({'params': enc_params, 'lr': encoder_lr, 'name': 'encoder_unfrozen'})
            if dec_params:
                groups.append({'params': dec_params, 'lr': lr, 'name': 'decoder'})
            return groups
        if self.encoder_mode in ('scratch', 'joint_scratch') and encoder_lr is not None:
            enc_params = [p for p in self.encoder.parameters() if p.requires_grad]
            dec_params = [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith('encoder.')]
            return [
                {'params': enc_params, 'lr': encoder_lr, 'name': 'encoder'},
                {'params': dec_params, 'lr': lr, 'name': 'decoder'},
            ]
        return [{'params': [p for p in self.parameters() if p.requires_grad], 'lr': lr, 'name': 'decoder'}]


__all__ = ["SmallPatchDecoder"]
