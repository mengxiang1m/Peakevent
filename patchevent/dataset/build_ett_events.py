"""
Build ETT event-labeled dataset from ETTh2_mixed_with_peaks_lookahead_5.csv
Follows the same format as wlel_event_dataset_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # TODO(R-future): migrate to patchevent package import

from patchevent.data.event_pipeline.config import PipelineConfig
from patchevent.data.event_pipeline.detectors import detect_events_gradient_width
from patchevent.data.event_pipeline.io import save_events
from patchevent.data.event_pipeline.labeling import (
    attach_point_level_event_fields,
    build_patch_label_table,
)
from patchevent.data.event_pipeline.preprocess import ensure_hourly_timeline
from patchevent.data.event_pipeline.reporting import (
    build_quality_report,
    report_to_markdown,
)
from patchevent.data.event_pipeline.splits import build_split_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ETT event dataset")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("dataset/ETT-small/ETTh2_mixed_with_peaks_lookahead_5.csv"),
        help="Source ETT dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/ETT-small/ett_event_v1"),
        help="Output folder for artifacts",
    )
    parser.add_argument("--timestamp-col", type=str, default="date_60min")
    parser.add_argument("--value-col", type=str, default="value_max")
    parser.add_argument("--version", type=str, default="v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(version=args.version)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "data"
    docs_dir = output_dir / "docs"
    data_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.input_file)
    print(f"Loaded {len(df_raw)} rows from {args.input_file}")

    df_clean, clean_stats = ensure_hourly_timeline(df_raw, args.timestamp_col, args.value_col)

    values = df_clean[args.value_col].to_numpy(dtype=float)
    events = detect_events_gradient_width(
        values=values,
        is_peak=df_clean["is_peak"].to_numpy(dtype=int),
        config=config,
        warmup_steps=0,
    )

    df_labeled = attach_point_level_event_fields(df_clean, events, args.timestamp_col)

    split_indices, split_col = build_split_indices(len(df_labeled), config)
    df_labeled["split"] = split_col

    df_labeled["timestamp"] = pd.to_datetime(df_labeled[args.timestamp_col])
    df_labeled["value"] = df_labeled[args.value_col].astype(float)

    _OUTPUT_COLS = [
        "timestamp",
        "value",
        "is_peak",
        "d_to_apex",
        "is_peak_event",
        "phase",
        "split",
    ]
    series_path = data_dir / f"ett_event_series_{config.version}.csv"
    df_labeled[_OUTPUT_COLS].to_csv(series_path, index=False, encoding="utf-8")

    df_patch = build_patch_label_table(
        df_labeled[_OUTPUT_COLS],
        patch_len=config.patch_len,
        stride=config.patch_stride,
    )
    patch_path = data_dir / f"ett_patch_labels_{config.version}.csv"
    df_patch.to_csv(patch_path, index=False, encoding="utf-8")

    save_events(
        events=events,
        df=df_labeled,
        timestamp_col=args.timestamp_col,
        output_dir=data_dir,
        version=config.version,
        dataset_prefix="ett",
        default_event_source="anchor_only",
    )

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
    report_md_path = docs_dir / f"quality_report_{config.version}.md"
    report_md_path.write_text(
        report_to_markdown(quality_report, dataset_name="ETT"),
        encoding="utf-8",
    )

    print("=" * 60)
    print("ETT event dataset build completed")
    print(f"Output dir: {output_dir}")
    print(f"  - Point-level CSV: data/{series_path.name}")
    print(f"  - Patch-level labels: data/{patch_path.name}")
    print(f"  - Events JSONL: data/ett_events_{config.version}.jsonl")
    print(f"  - Split JSON: data/{split_path.name}")
    print(f"  - Quality report: docs/{report_md_path.name}")
    print(f"Rows: {len(df_labeled)}, Events: {len(events)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
