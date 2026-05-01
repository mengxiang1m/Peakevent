#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from tools.peakfocus_to_stsep import peakfocus_to_events
except ImportError:
    from peakfocus_to_stsep import peakfocus_to_events


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate PeakFocus outputs with STSEP metrics')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--setting', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['wlel', 'ett', 'elc'])
    parser.add_argument('--events_path', type=str, required=True)
    parser.add_argument('--pred_len', type=int, required=True)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--peak_threshold', type=float, default=0.4)
    parser.add_argument('--tolerance', type=int, default=3)
    return parser.parse_args()


def load_events_jsonl(events_path):
    events = []
    with open(events_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    events.sort(key=lambda x: int(x['onset_idx']))
    onset_arr = np.array([int(e['onset_idx']) for e in events], dtype=np.int64)
    return events, onset_arr


def build_gt_events_for_window(events, onset_arr, pred_start, pred_len):
    pred_end = pred_start + pred_len
    lo = int(np.searchsorted(onset_arr, pred_start, side='left'))
    hi = int(np.searchsorted(onset_arr, pred_end, side='left'))

    out = []
    for ev in events[lo:hi]:
        onset = int(ev['onset_idx']) - pred_start
        end = int(ev['end_idx']) - pred_start
        apex = int(ev['apex_idx']) - pred_start
        if onset < 0 or onset >= pred_len:
            continue

        out.append(
            {
                'onset_idx': int(onset),
                'end_idx': int(end),
                'duration': int(ev.get('duration', max(0, end - onset))),
                'apex_idx': int(apex),
                'apex_intensity': float(ev['apex_intensity']),
            }
        )
    return out


def match_events(pred_events, gt_events, tolerance=3):
    if not pred_events or not gt_events:
        return [], pred_events, gt_events

    cost = np.array(
        [[abs(p['apex_idx'] - g['apex_idx']) for g in gt_events] for p in pred_events],
        dtype=np.float64,
    )
    row_ind, col_ind = linear_sum_assignment(cost)

    tp_pairs = []
    matched_pred = set()
    matched_gt = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= tolerance:
            tp_pairs.append((pred_events[r], gt_events[c]))
            matched_pred.add(r)
            matched_gt.add(c)

    fp = [pred_events[i] for i in range(len(pred_events)) if i not in matched_pred]
    fn = [gt_events[i] for i in range(len(gt_events)) if i not in matched_gt]
    return tp_pairs, fp, fn


def compute_metrics(tp_total, fp_total, fn_total, onset_errs, apex_errs, dur_errs, intensity_apes):
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        'dataset': None,
        'event_precision': float(precision),
        'event_recall': float(recall),
        'event_f1': float(f1),
        'total_tp': int(tp_total),
        'total_fp': int(fp_total),
        'total_fn': int(fn_total),
        'onset_mae': float(np.mean(onset_errs)) if onset_errs else float('nan'),
        'apex_mae': float(np.mean(apex_errs)) if apex_errs else float('nan'),
        'duration_mae': float(np.mean(dur_errs)) if dur_errs else float('nan'),
        'intensity_mape': float(np.mean(intensity_apes)) if intensity_apes else float('nan'),
    }
    return metrics


def main():
    args = parse_args()

    setting_dir = os.path.join(args.results_dir, args.setting)
    preds_path = os.path.join(setting_dir, 'per_window_preds.npy')
    peak_probs_path = os.path.join(setting_dir, 'per_window_peak_probs.npy')
    starts_path = os.path.join(setting_dir, 'per_window_starts.npy')

    if not os.path.exists(preds_path):
        raise FileNotFoundError(f'Missing file: {preds_path}')
    if not os.path.exists(peak_probs_path):
        raise FileNotFoundError(f'Missing file: {peak_probs_path}')
    if not os.path.exists(starts_path):
        raise FileNotFoundError(f'Missing file: {starts_path}')

    value_preds = np.load(preds_path)
    peak_probs = np.load(peak_probs_path)
    starts = np.load(starts_path).astype(np.int64)

    if value_preds.ndim == 3 and value_preds.shape[-1] == 1:
        value_preds = value_preds[:, :, 0]
    if peak_probs.ndim == 3 and peak_probs.shape[-1] == 1:
        peak_probs = peak_probs[:, :, 0]

    if value_preds.ndim != 2 or peak_probs.ndim != 2:
        raise ValueError(
            f'Expected 2D arrays after squeeze, got preds={value_preds.shape}, peak_probs={peak_probs.shape}'
        )

    n = min(len(value_preds), len(peak_probs), len(starts))
    value_preds = value_preds[:n]
    peak_probs = peak_probs[:n]
    starts = starts[:n]

    events, onset_arr = load_events_jsonl(args.events_path)

    tp_total, fp_total, fn_total = 0, 0, 0
    onset_errs, apex_errs, dur_errs, intensity_apes = [], [], [], []

    for i in range(n):
        pred_events = peakfocus_to_events(
            peak_probs=peak_probs[i],
            value_preds=value_preds[i],
            threshold=args.peak_threshold,
        )

        pred_start = int(starts[i]) + int(args.seq_len)
        gt_events = build_gt_events_for_window(
            events=events,
            onset_arr=onset_arr,
            pred_start=pred_start,
            pred_len=args.pred_len,
        )

        tp_pairs, fp, fn = match_events(pred_events, gt_events, tolerance=args.tolerance)

        tp_total += len(tp_pairs)
        fp_total += len(fp)
        fn_total += len(fn)

        for pred_ev, gt_ev in tp_pairs:
            onset_errs.append(abs(pred_ev['onset_idx'] - gt_ev['onset_idx']))
            apex_errs.append(abs(pred_ev['apex_idx'] - gt_ev['apex_idx']))
            dur_errs.append(abs(pred_ev['duration'] - gt_ev['duration']))

            gt_intensity = float(gt_ev['apex_intensity'])
            if gt_intensity != 0.0:
                intensity_apes.append(abs(float(pred_ev['apex_intensity']) - gt_intensity) / abs(gt_intensity))

    metrics = compute_metrics(
        tp_total=tp_total,
        fp_total=fp_total,
        fn_total=fn_total,
        onset_errs=onset_errs,
        apex_errs=apex_errs,
        dur_errs=dur_errs,
        intensity_apes=intensity_apes,
    )
    metrics['dataset'] = args.dataset
    metrics['setting'] = args.setting
    metrics['n_windows_eval'] = int(n)
    metrics['pred_len'] = int(args.pred_len)
    metrics['seq_len'] = int(args.seq_len)
    metrics['peak_threshold'] = float(args.peak_threshold)
    metrics['tolerance'] = int(args.tolerance)

    print('=' * 72)
    print(f"STSEP Metrics | dataset={args.dataset} | setting={args.setting}")
    print('=' * 72)
    print(f"Event F1      : {metrics['event_f1']:.6f}")
    print(f"Onset MAE     : {metrics['onset_mae']:.6f}")
    print(f"Apex MAE      : {metrics['apex_mae']:.6f}")
    print(f"Duration MAE  : {metrics['duration_mae']:.6f}")
    print(f"Intensity MAPE: {metrics['intensity_mape']:.6f}")
    print(f"TP/FP/FN      : {metrics['total_tp']}/{metrics['total_fp']}/{metrics['total_fn']}")
    print(f"N windows     : {metrics['n_windows_eval']}")

    os.makedirs(setting_dir, exist_ok=True)
    save_path = os.path.join(setting_dir, 'stsep_metrics.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to: {save_path}")


if __name__ == '__main__':
    main()
