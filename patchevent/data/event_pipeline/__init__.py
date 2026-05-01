"""Shared event pipeline utilities."""

from . import detectors, io, labeling, preprocess, reporting, splits
from .config import PipelineConfig
from .detectors import (
    _build_anchor_fallback_event,
    _event_from_range,
    _merge_overlapping_events_preserve_apex,
    compute_dynamic_thresholds,
    detect_events,
    detect_events_anchor_only,
    detect_events_from_peak_labels,
    detect_events_gradient_width,
    detect_events_hybrid,
    merge_close_events,
)
from .io import save_events
from .labeling import attach_point_level_event_fields, build_patch_label_table
from .preprocess import ensure_hourly_timeline, robust_zscore
from .reporting import build_quality_report, report_to_markdown
from .splits import build_split_indices, validate_ratios

__all__ = [
    "preprocess",
    "detectors",
    "labeling",
    "splits",
    "reporting",
    "io",
    "PipelineConfig",
    "robust_zscore",
    "ensure_hourly_timeline",
    "compute_dynamic_thresholds",
    "detect_events_gradient_width",
    "detect_events_anchor_only",
    "detect_events_hybrid",
    "detect_events_from_peak_labels",
    "detect_events",
    "merge_close_events",
    "_event_from_range",
    "_merge_overlapping_events_preserve_apex",
    "_build_anchor_fallback_event",
    "attach_point_level_event_fields",
    "build_patch_label_table",
    "build_split_indices",
    "validate_ratios",
    "build_quality_report",
    "report_to_markdown",
    "save_events",
]
