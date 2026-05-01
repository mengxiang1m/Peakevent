"""Shared I/O helpers for event dataset builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def save_events(
    events: List[Dict[str, float]],
    df: pd.DataFrame,
    timestamp_col: str,
    output_dir: Path,
    version: str,
    dataset_prefix: str = "wlel",
    default_event_source: str = "threshold",
) -> None:
    events_with_time: List[Dict[str, object]] = []
    ts = pd.to_datetime(df[timestamp_col])
    for idx, event in enumerate(events, start=1):
        s = int(event["onset_idx"])
        a = int(event["apex_idx"])
        event_item = {
            "event_id": idx,
            "onset_idx": s,
            "end_idx": int(event["end_idx"]),
            "duration": int(event["duration"]),
            "apex_idx": a,
            "apex_intensity": float(event["apex_intensity"]),
            "onset_time": ts.iloc[s].isoformat(sep=" "),
            "apex_time": ts.iloc[a].isoformat(sep=" "),
            "event_source": str(event.get("event_source", default_event_source)),
        }
        events_with_time.append(event_item)

    jsonl_path = output_dir / f"{dataset_prefix}_events_{version}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in events_with_time:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    csv_path = output_dir / f"{dataset_prefix}_events_{version}.csv"
    pd.DataFrame(events_with_time).to_csv(csv_path, index=False, encoding="utf-8")
