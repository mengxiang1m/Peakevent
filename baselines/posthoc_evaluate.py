from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # TODO(R-future): migrate to patchevent package import

from patchevent.data.event_pipeline.config import PipelineConfig
from patchevent.data.event_pipeline.detectors import (
    detect_events_from_peak_labels,
    detect_events_gradient_width,
)


def _make_match_result(tp_pairs: list, fp_events: list, fn_events: list) -> dict:
    return {
        "tp_pairs": tp_pairs,
        "fp_events": fp_events,
        "fn_events": fn_events,
        "tp": len(tp_pairs),
        "fp": len(fp_events),
        "fn": len(fn_events),
        "n_pred": len(tp_pairs) + len(fp_events),
        "n_gt": len(tp_pairs) + len(fn_events),
    }


def match_events(pred_events: list[dict], gt_events: list[dict], tolerance: int = 3) -> dict:
    n_pred = len(pred_events)
    n_gt = len(gt_events)
    if n_pred == 0 and n_gt == 0:
        return _make_match_result([], [], [])
    if n_pred == 0:
        return _make_match_result([], [], gt_events)
    if n_gt == 0:
        return _make_match_result([], pred_events, [])

    pred_apices = np.array([e["apex_index"] for e in pred_events], dtype=np.float64)
    gt_apices = np.array([e["apex_index"] for e in gt_events], dtype=np.float64)
    cost = np.abs(pred_apices[:, None] - gt_apices[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pred = set()
    matched_gt = set()
    tp_pairs = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= tolerance:
            tp_pairs.append((pred_events[r], gt_events[c]))
            matched_pred.add(r)
            matched_gt.add(c)

    fp_events = [pred_events[i] for i in range(n_pred) if i not in matched_pred]
    fn_events = [gt_events[j] for j in range(n_gt) if j not in matched_gt]
    return _make_match_result(tp_pairs, fp_events, fn_events)


def compute_sample_errors(match_result: dict, intensity_scale: float = 1.0) -> dict:
    out = {
        "tp": match_result["tp"],
        "fp": match_result["fp"],
        "fn": match_result["fn"],
        "n_pred": match_result["n_pred"],
        "n_gt": match_result["n_gt"],
        "apex_errors": [],
        "onset_errors": [],
        "duration_errors": [],
        "intensity_apes": [],
    }
    for pred_ev, gt_ev in match_result["tp_pairs"]:
        out["apex_errors"].append(abs(pred_ev["apex_index"] - gt_ev["apex_index"]))
        out["onset_errors"].append(abs(pred_ev["onset"] - gt_ev["onset"]))
        out["duration_errors"].append(abs(pred_ev["duration"] - gt_ev["duration"]))
        pred_int = float(pred_ev["apex_intensity"]) * intensity_scale
        gt_int = float(gt_ev["apex_intensity"]) * intensity_scale
        if gt_int > 0:
            out["intensity_apes"].append(abs(pred_int - gt_int) / gt_int)
    return out


def aggregate_metrics(sample_results: list[dict], parse_statuses: list[str]) -> dict:
    total_tp = sum(r["tp"] for r in sample_results)
    total_fp = sum(r["fp"] for r in sample_results)
    total_fn = sum(r["fn"] for r in sample_results)
    total_pred = sum(r["n_pred"] for r in sample_results)
    total_gt = sum(r["n_gt"] for r in sample_results)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    all_apex = [e for r in sample_results for e in r["apex_errors"]]
    all_onset = [e for r in sample_results for e in r["onset_errors"]]
    all_duration = [e for r in sample_results for e in r["duration_errors"]]
    all_ape = [e for r in sample_results for e in r["intensity_apes"]]

    n_total = len(parse_statuses)
    n_parse_fail = sum(1 for s in parse_statuses if s == "parse_fail")
    n_correct_empty = sum(
        1
        for r, s in zip(sample_results, parse_statuses)
        if r["n_pred"] == 0 and r["n_gt"] == 0 and s != "parse_fail"
    )

    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_pred_events": total_pred,
        "total_gt_events": total_gt,
        "apex_mae": float(np.mean(all_apex)) if all_apex else float("nan"),
        "onset_mae": float(np.mean(all_onset)) if all_onset else float("nan"),
        "duration_mae": float(np.mean(all_duration)) if all_duration else float("nan"),
        "intensity_mape": float(np.mean(all_ape)) if all_ape else float("nan"),
        "parse_success_rate": 1.0 - n_parse_fail / n_total if n_total > 0 else 0.0,
        "correct_empty_rate": n_correct_empty / n_total if n_total > 0 else 0.0,
        "n_samples": n_total,
    }


METHOD_RULES = [
    ("peak_Autoformer", "Seq2Peak"),
    ("PatchTST", "PatchTST"),
    ("iTransformer", "iTransformer"),
    ("TimeMixer", "TimeMixer"),
    ("DLinear", "DLinear"),
    ("Autoformer", "Autoformer"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc event evaluation for baseline forecasts")
    parser.add_argument("--dataset", required=True, choices=["wlel", "ett", "elc"])
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--window_stride", type=int, default=4)
    parser.add_argument("--dataset_config", type=str, default="data/dataset_configs.json")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--tolerance", type=int, default=3)
    parser.add_argument("--distance", type=int, default=6)
    parser.add_argument("--prominence", type=float, default=None)
    parser.add_argument("--prominence_scale", type=float, default=0.05)
    parser.add_argument(
        "--event_mode",
        type=str,
        default="gradient_width",
        choices=["gradient_width", "peak_label_midpoint"],
        help="Boundary extraction mode for predicted events.",
    )
    parser.add_argument("--min_duration", type=int, default=3)
    parser.add_argument("--max_flank_hours", type=int, default=6)
    parser.add_argument(
        "--max_half_hours",
        type=int,
        default=3,
        help="gradient_width mode: max expansion hours per side.",
    )
    parser.add_argument(
        "--min_half_hours",
        type=int,
        default=1,
        help="gradient_width mode: min expansion hours per side.",
    )
    parser.add_argument(
        "--rate_frac",
        type=float,
        default=0.4,
        help="gradient_width mode: relative height cutoff fraction.",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_events(events_path: Path) -> tuple[list[dict], np.ndarray]:
    events = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    events.sort(key=lambda x: x["onset_idx"])
    onset_arr = np.array([int(e["onset_idx"]) for e in events], dtype=np.int64)
    return events, onset_arr


def expected_test_starts(
    split_start: int,
    split_end: int,
    seq_len: int,
    pred_len: int,
    stride: int,
) -> np.ndarray:
    max_start = split_end + 1 - seq_len - pred_len
    if max_start < split_start:
        return np.array([], dtype=np.int64)
    offset = (stride - (split_start % stride)) % stride
    first = split_start + offset
    if first > max_start:
        return np.array([], dtype=np.int64)
    return np.arange(first, max_start + 1, stride, dtype=np.int64)


def parse_method(run_name: str) -> str:
    for token, method in METHOD_RULES:
        if token in run_name:
            return method
    return "Unknown"


def build_gt_window_events(
    events: list[dict],
    onset_arr: np.ndarray,
    pred_start: int,
    pred_end: int,
    pred_len: int,
) -> list[dict]:
    lo = int(np.searchsorted(onset_arr, pred_start, side="left"))
    hi = int(np.searchsorted(onset_arr, pred_end - 1, side="right"))
    window_events = events[lo:hi]
    out = []
    for ev in window_events:
        onset = int(ev["onset_idx"]) - pred_start
        apex = int(ev["apex_idx"]) - pred_start
        if onset < 0 or onset >= pred_len or apex < 0:
            continue
        if apex >= pred_len:
            apex = pred_len - 1
        out.append(
            {
                "onset": onset,
                "duration": int(ev["duration"]),
                "apex_index": apex,
                "apex_intensity": float(ev["apex_intensity"]),
            }
        )
    return out


def build_pred_events(pred_curve: np.ndarray, args: argparse.Namespace) -> list[dict]:
    prominence = args.prominence
    if prominence is None:
        prominence = max(float(np.std(pred_curve)) * args.prominence_scale, 1e-6)
    peaks, _ = find_peaks(pred_curve, distance=args.distance, prominence=prominence)
    is_peak = np.zeros(len(pred_curve), dtype=np.int32)
    is_peak[peaks] = 1

    cfg = PipelineConfig(
        min_duration=args.min_duration,
        max_flank_hours=args.max_flank_hours,
        max_half_hours=args.max_half_hours,
        min_half_hours=args.min_half_hours,
        rate_frac=args.rate_frac,
    )
    if args.event_mode == "gradient_width":
        pred_abs = detect_events_gradient_width(
            values=pred_curve,
            is_peak=is_peak,
            config=cfg,
            warmup_steps=0,
            grad_values=pred_curve,
        )
    else:
        pred_abs = detect_events_from_peak_labels(pred_curve, is_peak, cfg)
    pred_len = len(pred_curve)
    out = []
    for ev in pred_abs:
        onset = int(ev["onset_idx"])
        apex = int(ev["apex_idx"])
        if onset < 0 or onset >= pred_len:
            continue
        apex = min(max(apex, 0), pred_len - 1)
        out.append(
            {
                "onset": onset,
                "duration": int(ev["duration"]),
                "apex_index": apex,
                "apex_intensity": float(ev["apex_intensity"]),
            }
        )
    return out


def evaluate_one_run(
    run_dir: Path,
    events: list[dict],
    onset_arr: np.ndarray,
    expected_starts: np.ndarray,
    args: argparse.Namespace,
) -> dict | None:
    preds_path = run_dir / "per_window_preds.npy"
    if not preds_path.exists():
        return None
    preds = np.load(preds_path)
    if preds.ndim == 3:
        preds = preds[:, :, 0]
    elif preds.ndim != 2:
        raise ValueError(f"Unexpected per_window_preds shape: {preds.shape} at {preds_path}")
    if preds.shape[1] != args.pred_len:
        print(f"[skip] {run_dir.name}: pred_len mismatch {preds.shape[1]} != {args.pred_len}")
        return None

    starts_path = run_dir / "per_window_starts.npy"
    if starts_path.exists():
        starts = np.load(starts_path).astype(np.int64)
    else:
        starts = expected_starts.copy()

    n = min(len(preds), len(starts))
    if n == 0:
        print(f"[skip] {run_dir.name}: no aligned windows")
        return None
    preds = preds[:n]
    starts = starts[:n]

    sample_results = []
    parse_statuses = []
    for pred_curve, window_start in zip(preds, starts):
        pred_start = int(window_start) + args.seq_len
        pred_end = pred_start + args.pred_len
        gt_events = build_gt_window_events(
            events=events,
            onset_arr=onset_arr,
            pred_start=pred_start,
            pred_end=pred_end,
            pred_len=args.pred_len,
        )
        pred_events = build_pred_events(np.asarray(pred_curve, dtype=np.float64), args)
        match_res = match_events(pred_events, gt_events, tolerance=args.tolerance)
        sample_results.append(compute_sample_errors(match_res))
        parse_statuses.append("ok")

    metrics = aggregate_metrics(sample_results, parse_statuses)
    metrics.update(
        {
            "run_name": run_dir.name,
            "method": parse_method(run_dir.name),
            "dataset": args.dataset,
            "pred_len": int(args.pred_len),
            "n_windows_eval": int(n),
            "distance": int(args.distance),
            "prominence": args.prominence,
            "prominence_scale": float(args.prominence_scale),
            "event_mode": args.event_mode,
            "max_half_hours": int(args.max_half_hours),
            "min_half_hours": int(args.min_half_hours),
            "rate_frac": float(args.rate_frac),
        }
    )
    return metrics


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    cfg_path = resolve_path(here, args.dataset_config)
    results_dir = resolve_path(here, args.results_dir)
    out_dir = results_dir / "posthoc" / args.dataset / f"pred{args.pred_len}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    ds_cfg = cfg["datasets"][args.dataset]
    events_path = resolve_path(here, ds_cfg["events_jsonl"])
    events, onset_arr = load_events(events_path)

    split_test = ds_cfg["splits"]["test"]
    expected_starts = expected_test_starts(
        split_start=int(split_test["start"]),
        split_end=int(split_test["end"]),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.window_stride,
    )
    print(
        f"[info] dataset={args.dataset} pred_len={args.pred_len} "
        f"expected_test_windows={len(expected_starts)}"
    )

    rows = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if f"_pl{args.pred_len}_" not in run_dir.name:
            continue
        if args.dataset.lower() not in run_dir.name.lower():
            continue
        if not (run_dir / "per_window_preds.npy").exists():
            continue

        metrics = evaluate_one_run(
            run_dir=run_dir,
            events=events,
            onset_arr=onset_arr,
            expected_starts=expected_starts,
            args=args,
        )
        if metrics is None:
            continue

        out_path = out_dir / f"{run_dir.name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        rows.append(metrics)
        print(
            f"[ok] {run_dir.name} | method={metrics['method']} "
            f"F1={metrics['event_f1']:.4f} OnsetMAE={metrics['onset_mae']:.4f}"
        )

    if not rows:
        print("[done] no matching runs found")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["method", "run_name"]).reset_index(drop=True)
    summary_csv = out_dir / "summary_runs.csv"
    df.to_csv(summary_csv, index=False)
    print(f"[done] saved {summary_csv}")


if __name__ == "__main__":
    main()
