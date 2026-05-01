"""
Build WLEL event-labeled dataset (v1) for data-first peak/event forecasting.

Outputs:
1) Point-level CSV for model training compatibility and event-aware training.
2) Event-level JSONL/CSV for structured generation tasks.
3) Fixed split index JSON (train/val/test).
4) Data quality report (JSON + Markdown).
5) Metadata JSON with versioned configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # TODO(R-future): migrate to patchevent package import

from patchevent.data.event_pipeline.config import PipelineConfig
from patchevent.data.event_pipeline.detectors import (
    compute_dynamic_thresholds,
    detect_events,
    detect_events_anchor_only,
    detect_events_from_peak_labels,
    detect_events_gradient_width,
    detect_events_hybrid,
)
from patchevent.data.event_pipeline.io import save_events
from patchevent.data.event_pipeline.labeling import attach_point_level_event_fields, build_patch_label_table
from patchevent.data.event_pipeline.preprocess import ensure_hourly_timeline
from patchevent.data.event_pipeline.reporting import build_quality_report, report_to_markdown
from patchevent.data.event_pipeline.splits import build_split_indices, validate_ratios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WLEL event dataset v1")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("dataset/load_data/hf_load_data/hf_load_data_20210101-20251127_mixed_with_peaks_lookahead_3.csv"),
        help="Source WLEL mixed dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/load_data/hf_load_data/wlel_event_v1"),
        help="Output folder for v1 artifacts",
    )

    scalar_args = [
        ("--timestamp-col", str, "date_60min"),
        ("--value-col", str, "value_max"),
        ("--version", str, "v1"),
        ("--train-ratio", float, 0.7),
        ("--val-ratio", float, 0.1),
        ("--test-ratio", float, 0.2),
        ("--history-days", int, 30),
        ("--min-history-days", int, 7),
        ("--enter-quantile", float, 0.95),
        ("--exit-quantile", float, 0.90),
        ("--merge-gap", int, 1),
        ("--min-duration", int, 3),
        ("--outlier-z-threshold", float, 3.5),
        ("--max-half-hours", int, 3),
        ("--min-half-hours", int, 1),
        ("--rate-frac", float, 0.020),
        ("--patch-len", int, 8),
        ("--patch-stride", int, 4),
        ("--anchor-fallback-window-hours", int, 6),
        ("--warmup-hours", int, -1),
    ]
    for name, arg_type, default in scalar_args:
        parser.add_argument(name, type=arg_type, default=default)

    parser.add_argument(
        "--max-flank-hours",
        type=int,
        default=168,
        help="For peak_label_midpoint mode, max search radius at dataset edges (hours).",
    )
    parser.add_argument(
        "--event-mode",
        type=str,
        default="gradient_width",
        choices=["gradient_width", "anchor_only", "peak_label_midpoint", "peak_label_midpoint_legacy", "dynamic_threshold"],
        help=(
            "Event extraction mode: gradient-based variable-width (default), "
            "pure anchor midpoint-cut, hybrid threshold + apex refinement, legacy midpoint-only, or threshold-only."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    config = PipelineConfig(
        version=args.version,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        history_days=args.history_days,
        min_history_days=args.min_history_days,
        enter_quantile=args.enter_quantile,
        exit_quantile=args.exit_quantile,
        merge_gap=args.merge_gap,
        min_duration=args.min_duration,
        outlier_z_threshold=args.outlier_z_threshold,
        event_mode=args.event_mode,
        max_flank_hours=args.max_flank_hours,
        anchor_fallback_window_hours=args.anchor_fallback_window_hours,
        warmup_hours=args.warmup_hours,
        max_half_hours=args.max_half_hours,
        min_half_hours=args.min_half_hours,
        rate_frac=args.rate_frac,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir, docs_dir = output_dir / "data", output_dir / "docs"
    data_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.input_file)
    if args.timestamp_col not in df_raw.columns:
        raise ValueError(f"timestamp column '{args.timestamp_col}' not found in {args.input_file}")
    if args.value_col not in df_raw.columns:
        raise ValueError(f"value column '{args.value_col}' not found in {args.input_file}")

    df_clean, clean_stats = ensure_hourly_timeline(df_raw, args.timestamp_col, args.value_col)
    enter_threshold, exit_threshold = compute_dynamic_thresholds(df_clean, args.timestamp_col, args.value_col, config)
    df_clean["threshold_enter"] = enter_threshold
    df_clean["threshold_exit"] = exit_threshold

    values = df_clean[args.value_col].to_numpy(dtype=float)
    enter_np, exit_np = enter_threshold.to_numpy(dtype=float), exit_threshold.to_numpy(dtype=float)
    warmup_steps = int(config.warmup_hours) if int(config.warmup_hours) >= 0 else int(config.min_history_days) * 24
    warmup_steps = max(0, warmup_steps)
    df_clean["is_warmup"] = (np.arange(len(df_clean)) < warmup_steps).astype(int)

    if config.event_mode == "gradient_width":
        if "is_peak" not in df_clean.columns or int(df_clean["is_peak"].sum()) == 0:
            raise ValueError("event_mode=gradient_width requires an is_peak column with at least one label.")
        events = detect_events_gradient_width(
            values=values,
            is_peak=df_clean["is_peak"].to_numpy(dtype=int),
            config=config,
            warmup_steps=0,
            grad_values=values,
        )
    elif config.event_mode == "anchor_only":
        if "is_peak" not in df_clean.columns or int(df_clean["is_peak"].sum()) == 0:
            raise ValueError("event_mode=anchor_only requires an is_peak column with at least one label.")
        events = detect_events_anchor_only(
            values=values,
            is_peak=df_clean["is_peak"].to_numpy(dtype=int),
            config=config,
            warmup_steps=0,
        )
    elif config.event_mode == "peak_label_midpoint":
        peak_arr = df_clean["is_peak"].to_numpy(dtype=int) if "is_peak" in df_clean.columns else np.zeros(len(df_clean), dtype=int)
        if "is_peak" not in df_clean.columns:
            print("[WARN] event_mode=peak_label_midpoint but no is_peak column found; use threshold-only apex fallback")
        events = detect_events_hybrid(
            values=values,
            is_peak=peak_arr,
            enter_threshold=enter_np,
            exit_threshold=exit_np,
            config=config,
            warmup_steps=warmup_steps,
        )
    elif config.event_mode == "peak_label_midpoint_legacy":
        if "is_peak" not in df_clean.columns or int(df_clean["is_peak"].sum()) == 0:
            print("[WARN] event_mode=peak_label_midpoint_legacy but no is_peak anchors found; fallback to dynamic_threshold")
            events = detect_events(values=values, enter_threshold=enter_np, exit_threshold=exit_np, config=config, warmup_steps=warmup_steps)
        else:
            events = detect_events_from_peak_labels(values=values, is_peak=df_clean["is_peak"].to_numpy(dtype=int), config=config)
    else:
        events = detect_events(values=values, enter_threshold=enter_np, exit_threshold=exit_np, config=config, warmup_steps=warmup_steps)

    df_labeled = attach_point_level_event_fields(df_clean, events, args.timestamp_col)
    split_indices, split_col = build_split_indices(len(df_labeled), config)
    df_labeled["split"] = split_col
    df_labeled["timestamp"] = pd.to_datetime(df_labeled[args.timestamp_col])
    df_labeled["value"] = df_labeled[args.value_col].astype(float)

    output_cols = ["timestamp", "value", "is_peak", "d_to_apex", "is_peak_event", "phase", "split"]
    series_path = data_dir / f"wlel_event_series_{config.version}.csv"
    df_labeled[output_cols].to_csv(series_path, index=False, encoding="utf-8")

    df_patch = build_patch_label_table(df_labeled[output_cols], patch_len=config.patch_len, stride=config.patch_stride)
    patch_path = data_dir / f"wlel_patch_labels_{config.version}.csv"
    df_patch.to_csv(patch_path, index=False, encoding="utf-8")

    save_events(events=events, df=df_labeled, timestamp_col=args.timestamp_col, output_dir=data_dir, version=config.version, dataset_prefix="wlel", default_event_source="threshold")

    split_path = data_dir / f"split_indices_{config.version}.json"
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(split_indices, f, ensure_ascii=False)

    quality_report = build_quality_report(
        df=df_labeled,
        events=events,
        clean_stats=clean_stats,
        config=config,
        timestamp_col=args.timestamp_col,
        value_col=args.value_col,
    )
    with (docs_dir / f"quality_report_{config.version}.json").open("w", encoding="utf-8") as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=2)

    report_md_path = docs_dir / f"quality_report_{config.version}.md"
    report_md_path.write_text(report_to_markdown(quality_report, dataset_name="WLEL"), encoding="utf-8")

    metadata = {
        "version": config.version,
        "source_file": str(args.input_file),
        "output_dir": str(output_dir),
        "schema": {
            "point_level_required": ["timestamp", "value", "is_peak", "event_id", "d_to_apex"],
            "event_level_required": [
                "event_id",
                "onset_idx",
                "end_idx",
                "duration",
                "apex_idx",
                "apex_intensity",
                "onset_time",
                "apex_time",
                "event_source",
            ],
            "split_required": ["train_ids", "val_ids", "test_ids"],
        },
        "config": asdict(config),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (docs_dir / f"metadata_{config.version}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("WLEL event dataset build completed")
    print(f"Source: {args.input_file}")
    print(f"Output dir: {output_dir}")
    print("  ├── data/  → model-ready files")
    print("  └── docs/  → reports & metadata")
    print(f"Point-level CSV:   data/{series_path.name}")
    print(f"Patch-level labels: data/{patch_path.name} ({len(df_patch)} patches, len={config.patch_len}, stride={config.patch_stride})")
    print(f"Event-level JSONL: data/wlel_events_{config.version}.jsonl")
    print(f"Split JSON:        data/{split_path.name}")
    print(f"Quality report:    docs/{report_md_path.name}")
    print(f"Rows: {len(df_labeled)}, Events: {len(events)}")
    print(f"Event mode: {config.event_mode}")
    print("=" * 80)


if __name__ == "__main__":
    main()
