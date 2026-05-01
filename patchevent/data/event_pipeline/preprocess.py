"""Shared preprocessing helpers for event dataset builders."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def robust_zscore(series: pd.Series, clip: float | None = 5.0) -> np.ndarray:
    median = float(series.median())
    mad = float((series - median).abs().median())
    if mad <= 1e-12:
        return np.zeros(len(series), dtype=float)

    z = 0.6745 * (series.to_numpy() - median) / mad
    if clip is not None:
        z = np.clip(z, -float(clip), float(clip))
    return z


def ensure_hourly_timeline(
    df_raw: pd.DataFrame,
    timestamp_col: str,
    value_col: str,
    method: str = "linear",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = df_raw.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    duplicate_count = int(df.duplicated(subset=[timestamp_col]).sum())
    if duplicate_count > 0:
        df = df.drop_duplicates(subset=[timestamp_col], keep="last").reset_index(drop=True)

    full_index = pd.date_range(df[timestamp_col].min(), df[timestamp_col].max(), freq="h")
    df = df.set_index(timestamp_col).reindex(full_index)
    df.index.name = timestamp_col

    missing_timestamps_count = int(df[value_col].isna().sum())
    for numeric_col in [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]:
        if numeric_col in ("is_peak", "is_peak_seq2peaks"):
            continue
        df[numeric_col] = df[numeric_col].interpolate(method=method, limit_direction="both")

    if "is_peak" in df.columns:
        df["is_peak"] = df["is_peak"].fillna(0).astype(int)
    else:
        df["is_peak"] = 0

    if "is_peak_seq2peaks" in df.columns:
        df["is_peak_seq2peaks"] = df["is_peak_seq2peaks"].fillna(0).astype(int)
    else:
        df["is_peak_seq2peaks"] = 0

    if "value_60min" in df.columns:
        df["value_60min"] = df["value_60min"].interpolate(method=method, limit_direction="both")
    if "value_max" in df.columns:
        df["value_max"] = df["value_max"].interpolate(method=method, limit_direction="both")

    # Keep compatibility columns for existing mixed loader.
    if "date_60min" not in df.columns:
        df["date_60min"] = df.index
    else:
        df["date_60min"] = df.index

    if "date_max" not in df.columns:
        df["date_max"] = df.index
    else:
        df["date_max"] = pd.to_datetime(df["date_max"]).fillna(df.index.to_series())

    # Ensure value column has no NA after interpolation.
    df[value_col] = df[value_col].interpolate(method=method, limit_direction="both")
    if df[value_col].isna().any():
        raise ValueError(f"Value column {value_col} still contains NaN after cleaning.")

    timestamp_diffs = df.index.to_series().diff().dropna().dt.total_seconds().to_numpy()
    expected_seconds = 3600.0
    irregular_ratio = float(np.mean(timestamp_diffs != expected_seconds)) if len(timestamp_diffs) else 0.0

    stats = {
        "rows_after_timeline_unification": int(len(df)),
        "duplicate_timestamp_count_removed": duplicate_count,
        "inserted_or_missing_points_filled": missing_timestamps_count,
        "irregular_interval_ratio_after_unification": irregular_ratio,
    }

    df[timestamp_col] = df.index
    df = df.reset_index(drop=True)
    return df, stats
