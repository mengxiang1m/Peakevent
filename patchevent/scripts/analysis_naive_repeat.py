#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks


def load_eval_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def contiguous_segments(binary_array: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous [start, end) segments where value == 1."""
    segments: list[tuple[int, int]] = []
    n = len(binary_array)
    i = 0
    while i < n:
        if binary_array[i] == 1:
            start = i
            while i < n and binary_array[i] == 1:
                i += 1
            segments.append((start, i))
        else:
            i += 1
    return segments


def build_naive_repeat_events(
    series_values: np.ndarray,
    series_is_peak_event: np.ndarray,
    pred_start: int,
    history_len: int = 96,
) -> list[dict]:
    """
    Build naive-repeat events for one sample:
    - Use historical [pred_start-history_len, pred_start) window.
    - Detect historical peaks with scipy.signal.find_peaks (prominence based).
    - Copy historical event duration/apex/intensity and shift to prediction window.
    """
    if pred_start < history_len:
        return []
    if pred_start > len(series_values):
        return []

    hist_start = pred_start - history_len
    hist_values = series_values[hist_start:pred_start]
    hist_flags = series_is_peak_event[hist_start:pred_start]

    if len(hist_values) != history_len or len(hist_flags) != history_len:
        return []

    # Prominence threshold is adaptive to local variation.
    spread = float(np.percentile(hist_values, 95) - np.percentile(hist_values, 5))
    std = float(np.std(hist_values))
    prominence = max(1.0, 0.08 * spread, 0.08 * std)
    peak_rel, peak_props = find_peaks(hist_values, prominence=prominence, distance=2)
    peak_prom = peak_props.get("prominences", np.array([], dtype=float))

    segments = contiguous_segments(hist_flags)
    pred_events: list[dict] = []

    for seg_start_rel, seg_end_rel in segments:
        if seg_end_rel <= seg_start_rel:
            continue

        duration = seg_end_rel - seg_start_rel
        in_seg_mask = (peak_rel >= seg_start_rel) & (peak_rel < seg_end_rel)

        if np.any(in_seg_mask):
            candidates = np.where(in_seg_mask)[0]
            best_idx = int(candidates[np.argmax(peak_prom[candidates])])
            apex_rel = int(peak_rel[best_idx])
        else:
            # Fallback: choose local maximum inside the segment.
            local = hist_values[seg_start_rel:seg_end_rel]
            apex_rel = seg_start_rel + int(np.argmax(local))

        onset_abs = pred_start + seg_start_rel
        apex_abs = pred_start + apex_rel
        intensity = float(series_values[hist_start + apex_rel])

        pred_events.append(
            {
                "onset_idx": int(onset_abs),
                "end_idx": int(onset_abs + duration),
                "duration": int(duration),
                "apex_idx": int(apex_abs),
                "apex_intensity": intensity,
            }
        )

    pred_events.sort(key=lambda x: x["onset_idx"])
    return pred_events


def match_events_onset(
    pred_events: list[dict],
    gt_events: list[dict],
    tolerance: int = 3,
) -> dict:
    n_pred = len(pred_events)
    n_gt = len(gt_events)

    if n_pred == 0 and n_gt == 0:
        return {"tp_pairs": [], "fp_events": [], "fn_events": [], "tp": 0, "fp": 0, "fn": 0}
    if n_pred == 0:
        return {"tp_pairs": [], "fp_events": [], "fn_events": gt_events, "tp": 0, "fp": 0, "fn": n_gt}
    if n_gt == 0:
        return {"tp_pairs": [], "fp_events": pred_events, "fn_events": [], "tp": 0, "fp": n_pred, "fn": 0}

    pred_onset = np.array([e["onset_idx"] for e in pred_events], dtype=np.float64)
    gt_onset = np.array([e["onset_idx"] for e in gt_events], dtype=np.float64)
    cost = np.abs(pred_onset[:, None] - gt_onset[None, :])

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pred = set()
    matched_gt = set()
    tp_pairs: list[tuple[dict, dict]] = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= tolerance:
            matched_pred.add(int(r))
            matched_gt.add(int(c))
            tp_pairs.append((pred_events[int(r)], gt_events[int(c)]))

    fp_events = [pred_events[i] for i in range(n_pred) if i not in matched_pred]
    fn_events = [gt_events[j] for j in range(n_gt) if j not in matched_gt]
    return {
        "tp_pairs": tp_pairs,
        "fp_events": fp_events,
        "fn_events": fn_events,
        "tp": len(tp_pairs),
        "fp": len(fp_events),
        "fn": len(fn_events),
    }


def run_analysis(series_csv: Path, eval_jsonl: Path, history_len: int, tolerance: int) -> None:
    series_df = pd.read_csv(series_csv)
    if "value" not in series_df.columns or "is_peak_event" not in series_df.columns:
        raise ValueError("series CSV must contain columns: value, is_peak_event")

    series_values = series_df["value"].to_numpy(dtype=np.float64)
    series_is_peak_event = series_df["is_peak_event"].to_numpy(dtype=np.int64)

    records = load_eval_records(eval_jsonl)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    onset_errors: list[float] = []
    duration_errors: list[float] = []
    apex_errors: list[float] = []
    intensity_errors: list[float] = []

    for rec in records:
        pred_start = int(rec["pred_start"])
        gt_events = sorted(rec.get("gt_events", []), key=lambda x: x["onset_idx"])
        pred_events = build_naive_repeat_events(
            series_values=series_values,
            series_is_peak_event=series_is_peak_event,
            pred_start=pred_start,
            history_len=history_len,
        )

        match_res = match_events_onset(pred_events, gt_events, tolerance=tolerance)
        total_tp += match_res["tp"]
        total_fp += match_res["fp"]
        total_fn += match_res["fn"]

        for pred_ev, gt_ev in match_res["tp_pairs"]:
            onset_errors.append(abs(pred_ev["onset_idx"] - gt_ev["onset_idx"]))
            duration_errors.append(abs(pred_ev["duration"] - gt_ev["duration"]))
            apex_errors.append(abs(pred_ev["apex_idx"] - gt_ev["apex_idx"]))
            intensity_errors.append(abs(pred_ev["apex_intensity"] - gt_ev["apex_intensity"]))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    onset_mae = float(np.mean(onset_errors)) if onset_errors else float("nan")
    duration_mae = float(np.mean(duration_errors)) if duration_errors else float("nan")
    apex_mae = float(np.mean(apex_errors)) if apex_errors else float("nan")
    intensity_mae = float(np.mean(intensity_errors)) if intensity_errors else float("nan")

    print(
        f"Naive Repeat WLEL: F1={f1:.3f}, Precision={precision:.3f}, "
        f"Recall={recall:.3f}, OnsetMAE={onset_mae:.2f}"
    )
    print(
        f"DurationMAE={duration_mae:.2f}, ApexMAE={apex_mae:.2f}, "
        f"IntensityMAE={intensity_mae:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive Repeat baseline for WLEL (H=96).")
    parser.add_argument(
        "--series_csv",
        type=Path,
        default=Path("dataset/wlel/event_v1/data/wlel_event_series_v1.csv"),
    )
    parser.add_argument(
        "--eval_jsonl",
        type=Path,
        default=Path(
            "phase2_small_decoder_final/checkpoints/wlel_arch_ablation_v2/"
            "G0_full_s42/test_eval/eval_outputs.jsonl"
        ),
    )
    parser.add_argument("--history_len", type=int, default=96)
    parser.add_argument("--tolerance", type=int, default=3)
    args = parser.parse_args()

    run_analysis(
        series_csv=args.series_csv,
        eval_jsonl=args.eval_jsonl,
        history_len=args.history_len,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()
