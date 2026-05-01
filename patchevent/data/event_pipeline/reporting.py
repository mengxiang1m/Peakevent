"""Shared quality-report helpers for event dataset builders."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .preprocess import robust_zscore


def build_quality_report(
    df: pd.DataFrame,
    events: List[Dict[str, float]],
    clean_stats: Dict[str, float],
    config: PipelineConfig,
    timestamp_col: str,
    value_col: str,
) -> Dict[str, object]:
    values = df[value_col]
    z = robust_zscore(values, clip=None)
    outlier_rate = float(np.mean(np.abs(z) > config.outlier_z_threshold))

    month_counts: Dict[str, int] = {}
    for event in events:
        month = pd.to_datetime(df.loc[int(event["onset_idx"]), timestamp_col]).strftime("%Y-%m")
        month_counts[month] = month_counts.get(month, 0) + 1

    duration_values = [int(e["duration"]) for e in events]
    apex_values = [float(e["apex_intensity"]) for e in events]

    event_source_distribution: Dict[str, int] = {}
    for event in events:
        src = str(event.get("event_source", "threshold"))
        event_source_distribution[src] = event_source_distribution.get(src, 0) + 1

    event_summary = {
        "event_count": len(events),
        "event_density_per_1k_points": round(len(events) / len(df) * 1000, 4),
        "duration_mean": float(np.mean(duration_values)) if duration_values else 0.0,
        "duration_p50": float(np.percentile(duration_values, 50)) if duration_values else 0.0,
        "duration_p90": float(np.percentile(duration_values, 90)) if duration_values else 0.0,
        "duration_max": int(np.max(duration_values)) if duration_values else 0,
        "apex_mean": float(np.mean(apex_values)) if apex_values else 0.0,
        "apex_p90": float(np.percentile(apex_values, 90)) if apex_values else 0.0,
        "apex_max": float(np.max(apex_values)) if apex_values else 0.0,
    }

    warmup_hours_effective = int(config.warmup_hours) if int(config.warmup_hours) >= 0 else int(config.min_history_days) * 24

    if config.event_mode == "peak_label_midpoint":
        boundary_semantics = "threshold_crossing_hysteresis_with_peak_anchor_apex_plus_uncovered_anchor_fallback"
        merge_policy = "gap_based"
        merge_gap_effective = config.merge_gap
    elif config.event_mode == "peak_label_midpoint_legacy":
        boundary_semantics = "valley_to_valley_with_anchor_apex"
        merge_policy = "strict_overlap_only"
        merge_gap_effective = 0
    else:
        boundary_semantics = "threshold_crossing_hysteresis"
        merge_policy = "gap_based"
        merge_gap_effective = config.merge_gap

    report = {
        "version": config.version,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_profile": {
            "rows": int(len(df)),
            "start_time": str(pd.to_datetime(df[timestamp_col]).min()),
            "end_time": str(pd.to_datetime(df[timestamp_col]).max()),
            "missing_rate_value_col": float(df[value_col].isna().mean()),
            "legacy_peak_density": float(df["is_peak"].mean()) if "is_peak" in df.columns else 0.0,
            "event_window_density": float(df["is_peak_event"].mean()),
            "outlier_rate_robust_z": outlier_rate,
        },
        "timeline_cleaning": clean_stats,
        "event_summary": event_summary,
        "event_source_distribution": event_source_distribution,
        "monthly_event_distribution": month_counts,
        "threshold_config": {
            "event_mode": config.event_mode,
            "boundary_semantics": boundary_semantics,
            "merge_policy": merge_policy,
            "merge_gap_effective": merge_gap_effective,
            "peak_anchor_count": int(df["is_peak"].sum()) if "is_peak" in df.columns else 0,
            "history_days": config.history_days,
            "min_history_days": config.min_history_days,
            "enter_quantile": config.enter_quantile,
            "exit_quantile": config.exit_quantile,
            "merge_gap": config.merge_gap,
            "min_duration": config.min_duration,
            "max_flank_hours": config.max_flank_hours,
            "anchor_fallback_window_hours": config.anchor_fallback_window_hours,
            "warmup_hours": config.warmup_hours,
            "warmup_hours_effective": warmup_hours_effective,
        },
    }
    return report


def report_to_markdown(report: Dict[str, object], dataset_name: str = "WLEL") -> str:
    profile = report["data_profile"]
    timeline = report["timeline_cleaning"]
    event_summary = report["event_summary"]
    cfg = report["threshold_config"]
    monthly = report["monthly_event_distribution"]
    source_dist = report.get("event_source_distribution", {})
    top_months = sorted(monthly.items(), key=lambda kv: kv[1], reverse=True)[:12]

    lines = [
        f"# {dataset_name} Data Quality Report ({report['version']})",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Time range: {profile['start_time']} -> {profile['end_time']}",
        f"- Total rows: {profile['rows']}",
        "",
        "## Data Cleaning",
        f"- Duplicate timestamps removed: {timeline['duplicate_timestamp_count_removed']}",
        f"- Missing/inserted points filled: {timeline['inserted_or_missing_points_filled']}",
        f"- Irregular interval ratio (after unification): {timeline['irregular_interval_ratio_after_unification']:.6f}",
        "",
        "## Data Quality",
        f"- Missing rate (value): {profile['missing_rate_value_col']:.6f}",
        f"- Outlier rate (robust z): {profile['outlier_rate_robust_z']:.6f}",
        f"- Legacy peak density (`is_peak`): {profile['legacy_peak_density']:.6f}",
        f"- Event window density (`is_peak_event`): {profile['event_window_density']:.6f}",
        "",
        "## Event Summary",
        f"- Event count: {event_summary['event_count']}",
        f"- Event density per 1k points: {event_summary['event_density_per_1k_points']}",
        f"- Duration mean/p50/p90/max: "
        f"{event_summary['duration_mean']:.3f} / {event_summary['duration_p50']:.3f} / "
        f"{event_summary['duration_p90']:.3f} / {event_summary['duration_max']}",
        f"- Apex mean/p90/max: "
        f"{event_summary['apex_mean']:.3f} / {event_summary['apex_p90']:.3f} / {event_summary['apex_max']:.3f}",
        "",
        "## Threshold Config (locked for v1)",
        f"- event_mode: {cfg['event_mode']}",
        f"- boundary_semantics: {cfg['boundary_semantics']}",
        f"- merge_policy: {cfg['merge_policy']}",
        f"- merge_gap_effective: {cfg['merge_gap_effective']}",
        f"- peak_anchor_count: {cfg['peak_anchor_count']}",
        f"- history_days: {cfg['history_days']}",
        f"- min_history_days: {cfg['min_history_days']}",
        f"- enter_quantile: {cfg['enter_quantile']}",
        f"- exit_quantile: {cfg['exit_quantile']}",
        f"- merge_gap: {cfg['merge_gap']}",
        f"- min_duration: {cfg['min_duration']}",
        f"- max_flank_hours: {cfg['max_flank_hours']}",
        f"- anchor_fallback_window_hours: {cfg['anchor_fallback_window_hours']}",
        f"- warmup_hours(raw): {cfg['warmup_hours']}",
        f"- warmup_hours_effective: {cfg['warmup_hours_effective']}",
        "",
        "## Event Source Distribution",
    ]
    for source, cnt in sorted(source_dist.items(), key=lambda kv: kv[0]):
        lines.append(f"- {source}: {cnt}")

    lines.append("")
    lines.append("## Monthly Event Distribution (top months)")
    for month, cnt in top_months:
        lines.append(f"- {month}: {cnt}")
    return "\n".join(lines) + "\n"
