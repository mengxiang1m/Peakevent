"""Model factory for PatchEvent Phase-2."""

from __future__ import annotations

from patchevent.models.hybrid_event_decoder import HybridEventDecoder
from patchevent.models.non_ar_decoder import NonAutoRegressiveDecoder
from patchevent.models.small_patch_decoder import SmallPatchDecoder

def _require_encoder_ckpt(args, decoder_name: str) -> str:
    ckpt = getattr(args, 'encoder_ckpt', None)
    if not ckpt:
        raise ValueError(f'{decoder_name} is a legacy checkpoint-based decoder and requires --encoder_ckpt')
    return ckpt


def build_model(args) -> HybridEventDecoder | SmallPatchDecoder | NonAutoRegressiveDecoder:
    """根据args构建模型。"""
    decoder_type = getattr(args, 'decoder_type', 'hybrid')
    if getattr(args, 'use_non_ar', False):
        decoder_type = 'legacy_detr'

    if decoder_type == 'hybrid':
        return HybridEventDecoder(
            seq_len=getattr(args, 'seq_len', 96),
            pred_len=getattr(args, 'pred_len', 96),
            patch_len=getattr(args, 'patch_len', 8),
            patch_stride=getattr(args, 'patch_stride', 4),
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            d_ff=args.d_ff, dropout=args.dropout,
            max_events=getattr(args, 'max_events', 10),
            refine_layers=getattr(args, 'refine_layers', 1),
            exist_threshold=getattr(args, 'exist_threshold', 0.5),
            norm_mean=getattr(args, 'norm_mean', 0.0),
            norm_std=getattr(args, 'norm_std', 1.0),
            event_schema=getattr(args, 'event_schema', 'quad'),
            backbone_layers=getattr(args, 'backbone_layers', 2),
            refine_order=getattr(args, 'hybrid_refine_order', 'onset'),
            use_causal_refine_mask=not getattr(args, 'hybrid_no_causal_refine_mask', False),
            use_matched_refine_order=getattr(args, 'hybrid_use_matched_refine_order', False),
            hybrid_nms_onset_radius=getattr(args, 'hybrid_nms_onset_radius', 0),
            use_count_head=(
                getattr(args, 'hybrid_use_count_head', False)
                or getattr(args, 'use_count_head', False)
            ),
            use_count_decoding=(
                getattr(args, 'hybrid_use_count_decoding', False)
                and not getattr(args, 'hybrid_no_count_decoding', False)
            ),
            max_event_count=getattr(args, 'max_event_count', 12),
            time_head=getattr(args, 'hybrid_time_head', 'structured'),
        )

    if decoder_type == 'legacy_detr':
        return NonAutoRegressiveDecoder(
            encoder_ckpt_path=_require_encoder_ckpt(args, 'legacy_detr'),
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            d_ff=args.d_ff, dropout=args.dropout,
            max_events=getattr(args, 'max_events', 8),
            pred_len=getattr(args, 'pred_len', 96),
            encoder_mode=getattr(args, 'encoder_mode', 'frozen'),
            exist_threshold=getattr(args, 'exist_threshold', 0.5),
        )

    if decoder_type != 'legacy_ar':
        raise ValueError(f'Unsupported decoder_type: {decoder_type}')

    return SmallPatchDecoder(
        encoder_ckpt_path=_require_encoder_ckpt(args, 'legacy_ar'),
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, dropout=args.dropout,
        max_seq_len=getattr(args, 'max_seq_len', 64),
        pred_len=getattr(args, 'pred_len', 96),
        event_schema=getattr(args, 'event_schema', 'quad'),
        encoder_mode=getattr(args, 'encoder_mode', 'frozen'),
        embed_dropout=getattr(args, 'embed_dropout', 0.0),
        unfreeze_last_n=getattr(args, 'unfreeze_last_n', 0),
        use_memory_pos=getattr(args, 'use_memory_pos', False),
        use_self_attn_agg=getattr(args, 'use_self_attn_agg', False),
        use_input_decomp=getattr(args, 'use_input_decomp', False),
        input_decomp_mode=getattr(args, 'input_decomp_mode', 'full'),
        use_raw_bypass=getattr(args, 'use_raw_bypass', False),
        use_tuple_guided_value=getattr(args, 'use_tuple_guided_value', False),
        use_independent_dense_int=getattr(args, 'use_independent_dense_int', False),
        use_apex_cond_int=getattr(args, 'use_apex_cond_int', False),
        use_int_direct_lookup=getattr(args, 'use_int_direct_lookup', False),
        use_decoupled_int_head=getattr(args, 'use_decoupled_int_head', False),
        use_tuple_crossatt_int=getattr(args, 'use_tuple_crossatt_int', False),
        min_onset_spacing=getattr(args, 'min_onset_spacing', 0),
        override_norm_mean=getattr(args, 'override_norm_mean', None),
        override_norm_std=getattr(args, 'override_norm_std', None),
        no_pos_valid_mask=getattr(args, 'no_pos_valid_mask', False),
        no_causal_mask=getattr(args, 'no_causal_mask', False),
        use_int_regression=getattr(args, 'use_int_regression', False),
        use_series_stats=getattr(args, 'use_series_stats', False),
        series_stats_recent_k=getattr(args, 'series_stats_recent_k', 8),
        intensity_from_values=getattr(args, 'intensity_from_values', 'apex'),
        use_count_head=getattr(args, 'use_count_head', False),
        max_event_count=getattr(args, 'max_event_count', 12),
        use_position_regression=getattr(args, 'use_position_regression', False),
        pos_reg_target=getattr(args, 'pos_reg_target', 'both'),
        use_qbridge=getattr(args, 'use_qbridge', False),
        qbridge_n_queries=getattr(args, 'qbridge_n_queries', 16),
        qbridge_n_layers=getattr(args, 'qbridge_n_layers', 2),
    )


__all__ = ["build_model"]
