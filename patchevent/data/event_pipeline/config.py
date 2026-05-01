"""Shared configuration dataclass for event dataset builders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    version: str = "v1"
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    history_days: int = 30
    min_history_days: int = 7
    enter_quantile: float = 0.95
    exit_quantile: float = 0.90
    merge_gap: int = 1
    min_duration: int = 3
    outlier_z_threshold: float = 3.5
    event_mode: str = "gradient_width"  # gradient_width | anchor_only | peak_label_midpoint | peak_label_midpoint_legacy | dynamic_threshold
    max_flank_hours: int = 10  # hours; limits boundary search at dataset edges
    anchor_fallback_window_hours: int = 6
    # --- gradient_width mode ---
    max_half_hours: int = 3      # max expansion hours per side
    min_half_hours: int = 1      # min expansion hours per side
    rate_frac: float = 0.020     # effective rate threshold = rate_frac * apex_value (per hour)
    warmup_hours: int = -1
    # --- patch-level labels ---
    patch_len: int = 8           # patch length in hours, should be > max event width (6h)
    patch_stride: int = 4        # sliding stride in hours
