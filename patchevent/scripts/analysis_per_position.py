#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_eval_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def match_events_onset_with_indices(
    pred_events: list[dict],
    gt_events: list[dict],
    tolerance: int = 3,
) -> dict:
    n_pred = len(pred_events)
    n_gt = len(gt_events)

    if n_pred == 0 or n_gt == 0:
        return {
            "matched_pairs": [],
            "matched_pred_idx": set(),
            "matched_gt_idx": set(),
        }

    pred_onset = np.array([e["onset_idx"] for e in pred_events], dtype=np.float64)
    gt_onset = np.array([e["onset_idx"] for e in gt_events], dtype=np.float64)
    cost = np.abs(pred_onset[:, None] - gt_onset[None, :])

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pairs: list[tuple[int, int]] = []
    matched_pred_idx: set[int] = set()
    matched_gt_idx: set[int] = set()

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= tolerance:
            r_i = int(r)
            c_i = int(c)
            matched_pairs.append((r_i, c_i))
            matched_pred_idx.add(r_i)
            matched_gt_idx.add(c_i)

    return {
        "matched_pairs": matched_pairs,
        "matched_pred_idx": matched_pred_idx,
        "matched_gt_idx": matched_gt_idx,
    }


def ordinal_label(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def run_analysis(eval_jsonl: Path, tolerance: int) -> None:
    records = load_eval_records(eval_jsonl)

    grouped: dict[int, dict] = defaultdict(lambda: {"n_events": 0, "n_matched": 0, "onset_errors": []})

    for rec in records:
        gt_events = sorted(rec.get("gt_events", []), key=lambda x: x["onset_idx"])
        pred_events = sorted(rec.get("pred_events", []), key=lambda x: x["onset_idx"])

        match_res = match_events_onset_with_indices(pred_events, gt_events, tolerance=tolerance)
        gt_to_pred = {gt_idx: pred_idx for pred_idx, gt_idx in match_res["matched_pairs"]}

        for pos, gt_ev in enumerate(gt_events, start=1):
            grouped[pos]["n_events"] += 1
            gt_idx = pos - 1
            if gt_idx in gt_to_pred:
                pred_idx = gt_to_pred[gt_idx]
                pred_ev = pred_events[pred_idx]
                grouped[pos]["n_matched"] += 1
                grouped[pos]["onset_errors"].append(abs(pred_ev["onset_idx"] - gt_ev["onset_idx"]))

    print("Position | n_events | n_matched | recall | onset_MAE")
    for pos in sorted(grouped.keys()):
        n_events = grouped[pos]["n_events"]
        n_matched = grouped[pos]["n_matched"]
        recall = n_matched / n_events if n_events > 0 else 0.0
        onset_mae = float(np.mean(grouped[pos]["onset_errors"])) if grouped[pos]["onset_errors"] else float("nan")
        print(
            f"{ordinal_label(pos):<8} | {n_events:<8} | {n_matched:<9} | "
            f"{recall:.3f} | {onset_mae:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-event-position analysis for WLEL H=96 seed=42.")
    parser.add_argument(
        "--eval_jsonl",
        type=Path,
        default=Path(
            "phase2_small_decoder_final/checkpoints/wlel_arch_ablation_v2/"
            "G0_full_s42/test_eval/eval_outputs.jsonl"
        ),
    )
    parser.add_argument("--tolerance", type=int, default=3)
    args = parser.parse_args()

    run_analysis(eval_jsonl=args.eval_jsonl, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
