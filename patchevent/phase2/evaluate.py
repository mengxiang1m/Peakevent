"""
evaluate.py
===========
Phase-2 Exp-0 评估模块。

主要内容
--------
- match_events(pred_events, gt_events, tolerance=3)
    : 匈牙利算法匹配预测事件与 GT 事件，返回 TP/FP/FN 统计

- compute_metrics(match_result)
    : 由匹配结果计算事件级指标

- evaluate_loader(model, test_loader, device, max_new_tokens, tolerance)
    : 对整个 DataLoader 做生成 + 匹配 + 汇总

CLI 用法
--------
    CLI入口已移除（SmallDecoder使用train.py内置评估流程）。
    可通过 evaluate_loader() 或 predict() 函数在代码中调用。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Optional

import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)  # TODO(R-future): migrate to patchevent package import

from patchevent.phase2.data_utils import parse_event_json, fill_event_intensity_from_values


# ─────────────────────────────────────────────────────────────────────────────
# 事件匹配
# ─────────────────────────────────────────────────────────────────────────────

def match_events(
    pred_events: list[dict],
    gt_events:   list[dict],
    tolerance:   int = 3,
) -> dict:
    """
    用匈牙利算法对预测事件与 GT 事件做最优匹配，再以 tolerance 门控保留有效 TP。

    代价矩阵 C_ij = |pred_i.apex_index - gt_j.apex_idx_rel|
    （gt_events 中的索引已是相对预测窗口的相对索引）

    Parameters
    ----------
    pred_events : parse_event_json() 返回的 list[dict]，字段:
                    onset(int), duration(int), apex_index(int), apex_intensity(float)
    gt_events   : build_event_json() 对应的相对索引事件，字段:
                    onset(int), duration(int), apex_index(int), apex_intensity(float)
                  注：GT 来自 data_utils.parse_event_json(json_str)，字段名与 pred 一致
    tolerance   : apex 相差 ≤ tolerance 的才算 TP（单位：时间步 = 小时），默认 3

    Returns
    -------
    dict 含:
      tp_pairs   : List[(pred_dict, gt_dict)]  — 匹配成功的对 (TP)
      fp_events  : List[pred_dict]             — 多余预测 (FP)
      fn_events  : List[gt_dict]               — 未命中 GT (FN)
      n_pred, n_gt, tp, fp, fn
    """
    N = len(pred_events)
    M = len(gt_events)

    # ── 边界情况 ─────────────────────────────────────────────────────────────
    if N == 0 and M == 0:
        return _make_match_result([], pred_events, gt_events)
    if N == 0:
        return _make_match_result([], [], gt_events)
    if M == 0:
        return _make_match_result([], pred_events, [])

    # ── 构建代价矩阵（N × M）────────────────────────────────────────────────
    cost = np.zeros((N, M), dtype=np.float64)
    pred_apices = np.array([e['apex_index'] for e in pred_events], dtype=np.float64)
    gt_apices   = np.array([e['apex_index'] for e in gt_events],   dtype=np.float64)
    cost = np.abs(pred_apices[:, None] - gt_apices[None, :])  # (N, M) broadcast

    # ── 匈牙利最优匹配 ───────────────────────────────────────────────────────
    row_ind, col_ind = linear_sum_assignment(cost)

    # ── Tolerance 门控 ───────────────────────────────────────────────────────
    matched_pred_idx = set()
    matched_gt_idx   = set()
    tp_pairs = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= tolerance:
            tp_pairs.append((pred_events[r], gt_events[c]))
            matched_pred_idx.add(r)
            matched_gt_idx.add(c)

    fp_events = [pred_events[i] for i in range(N) if i not in matched_pred_idx]
    fn_events = [gt_events[j]   for j in range(M) if j not in matched_gt_idx]

    return _make_match_result(tp_pairs, fp_events, fn_events)


def _make_match_result(
    tp_pairs:  list,
    fp_events: list,
    fn_events: list,
) -> dict:
    return {
        'tp_pairs':  tp_pairs,
        'fp_events': fp_events,
        'fn_events': fn_events,
        'tp': len(tp_pairs),
        'fp': len(fp_events),
        'fn': len(fn_events),
        'n_pred': len(tp_pairs) + len(fp_events),
        'n_gt':   len(tp_pairs) + len(fn_events),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────────────────────────────────────

def compute_sample_errors(match_result: dict, intensity_scale: float = 1.0) -> dict:
    """
    从单个样本的匹配结果计算误差项（用于后续汇总）。

    Returns
    -------
    dict 含:
      tp, fp, fn, n_pred, n_gt
      apex_errors    : List[float]  — 每对 TP 的 |pred.apex_index - gt.apex_index|
      onset_errors   : List[float]
      duration_errors: List[float]
      intensity_apes : List[float]  — abs percentage error
    """
    result = {
        'tp': match_result['tp'],
        'fp': match_result['fp'],
        'fn': match_result['fn'],
        'n_pred': match_result['n_pred'],
        'n_gt':   match_result['n_gt'],
        'apex_errors':     [],
        'onset_errors':    [],
        'duration_errors': [],
        'intensity_apes':  [],
    }
    for pred_ev, gt_ev in match_result['tp_pairs']:
        result['apex_errors'].append(abs(pred_ev['apex_index'] - gt_ev['apex_index']))
        result['onset_errors'].append(abs(pred_ev['onset'] - gt_ev['onset']))
        result['duration_errors'].append(abs(pred_ev['duration'] - gt_ev['duration']))
        pred_int = float(pred_ev['apex_intensity']) * intensity_scale
        gt_int = float(gt_ev['apex_intensity']) * intensity_scale
        # Guard against MAPE explosion: skip when gt_int is near-zero or
        # pred_int is non-finite (can happen with dense value head outputs)
        if gt_int > 1e-4 and np.isfinite(pred_int):
            ape = abs(pred_int - gt_int) / gt_int
            result['intensity_apes'].append(ape)
    return result


def aggregate_metrics(sample_results: list[dict], parse_statuses: list[str]) -> dict:
    """
    汇总所有样本的误差为最终指标。

    Returns
    -------
    dict 含所有报告指标
    """
    total_tp = sum(r['tp']     for r in sample_results)
    total_fp = sum(r['fp']     for r in sample_results)
    total_fn = sum(r['fn']     for r in sample_results)
    total_pred = sum(r['n_pred'] for r in sample_results)
    total_gt   = sum(r['n_gt']   for r in sample_results)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    all_apex     = [e for r in sample_results for e in r['apex_errors']]
    all_onset    = [e for r in sample_results for e in r['onset_errors']]
    all_duration = [e for r in sample_results for e in r['duration_errors']]
    all_ape      = [e for r in sample_results for e in r['intensity_apes']]

    n_total          = len(parse_statuses)
    n_parse_fail     = sum(1 for s in parse_statuses if s == 'parse_fail')
    n_correct_empty  = sum(
        1 for r, s in zip(sample_results, parse_statuses)
        if r['n_pred'] == 0 and r['n_gt'] == 0 and s != 'parse_fail'
    )

    return {
        # 事件级检测指标
        'event_precision':    precision,
        'event_recall':       recall,
        'event_f1':           f1,
        'total_tp':           total_tp,
        'total_fp':           total_fp,
        'total_fn':           total_fn,
        'total_pred_events':  total_pred,
        'total_gt_events':    total_gt,
        # 精度指标（仅 TP 对）
        'apex_mae':           float(np.mean(all_apex))     if all_apex     else float('nan'),
        'onset_mae':          float(np.mean(all_onset))    if all_onset    else float('nan'),
        'duration_mae':       float(np.mean(all_duration)) if all_duration else float('nan'),
        'intensity_mape':     float(np.mean(all_ape))      if all_ape      else float('nan'),
        # 解析质量
        'parse_success_rate': 1.0 - n_parse_fail / n_total if n_total > 0 else 0.0,
        'correct_empty_rate': n_correct_empty / n_total    if n_total > 0 else 0.0,
        'n_samples':          n_total,
    }


def print_metrics(metrics: dict) -> None:
    """打印格式化的指标表格。"""
    print('\n' + '=' * 60)
    print('Phase-2 Exp-0 Evaluation Results')
    print('=' * 60)
    print(f'  Samples evaluated    : {metrics["n_samples"]}')
    print(f'  Parse success rate   : {metrics["parse_success_rate"]:.4f}')
    print(f'  Correct-empty rate   : {metrics["correct_empty_rate"]:.4f}')
    print()
    print(f'  Event Precision      : {metrics["event_precision"]:.4f}')
    print(f'  Event Recall         : {metrics["event_recall"]:.4f}')
    print(f'  Event F1             : {metrics["event_f1"]:.4f}')
    print(f'  Total TP/FP/FN       : {metrics["total_tp"]}/{metrics["total_fp"]}/{metrics["total_fn"]}')
    print()
    print(f'  Apex MAE (hours)     : {metrics["apex_mae"]:.3f}')
    print(f'  Onset MAE (hours)    : {metrics["onset_mae"]:.3f}')
    print(f'  Duration MAE (hours) : {metrics["duration_mae"]:.3f}')
    print(f'  Intensity MAPE       : {metrics["intensity_mape"]:.4f}')
    print('=' * 60 + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# 批量评估
# ─────────────────────────────────────────────────────────────────────────────

def _rel_events_to_gt_format(
    events: list[dict],
    pred_start: int,
    intensity_scale: float = 1.0,
) -> list[dict]:
    """将相对索引的解析事件转换为 GT jsonl 格式（绝对索引）。"""
    results = []
    for ev in events:
        onset_abs = ev['onset'] + pred_start
        apex_abs  = ev['apex_index'] + pred_start
        end_abs   = onset_abs + ev['duration']
        results.append({
            'onset_idx':      onset_abs,
            'end_idx':        end_abs,
            'duration':       ev['duration'],
            'apex_idx':       apex_abs,
            'apex_intensity': float(ev['apex_intensity']) * intensity_scale,
        })
    return results


def evaluate_loader(
    model,
    test_loader,
    device: torch.device,
    max_new_tokens: int = 512,
    tolerance: int = 3,
    verbose: bool = True,
    save_dir: Optional[str] = None,
    intensity_scale: float = 1.0,
    hybrid_nms_onset_radius: int | None = None,
) -> dict:
    """
    对整个 DataLoader 做生成 + 匹配 + 汇总。

    Parameters
    ----------
    model          : PatchLLM，已移到 device
    test_loader    : DataLoader
    device         : 推理设备
    max_new_tokens : generate 最大新 token 数
    tolerance      : 匹配容差（小时）
    save_dir       : 若指定，将 eval_metrics.json + eval_outputs.jsonl 写入该目录

    Returns
    -------
    metrics dict（同 aggregate_metrics 的返回值）
    """
    model.eval()
    sample_results  = []
    parse_statuses  = []
    output_records: list[dict] = []  # 逐样本输出记录

    total = len(test_loader)
    t_eval_start = time.time()
    for step, batch in enumerate(test_loader):
        x              = batch['x'].to(device)
        gt_json_strs   = batch['json_str']
        pred_starts    = batch.get('pred_start', [0] * len(gt_json_strs))

        # 传递可选的时间特征给 generate
        gen_kwargs = {}
        if 'hours' in batch:
            gen_kwargs['hours'] = batch['hours'].to(device)
            gen_kwargs['weekdays'] = batch['weekdays'].to(device)
        if 'hours_future' in batch:
            gen_kwargs['hours_future'] = batch['hours_future'].to(device)
            gen_kwargs['weekdays_future'] = batch['weekdays_future'].to(device)
        if hybrid_nms_onset_radius is not None:
            gen_kwargs['hybrid_nms_onset_radius'] = int(hybrid_nms_onset_radius)

        # 生成
        gen_out = model.generate(x, max_new_tokens=max_new_tokens, return_aux=True, **gen_kwargs)
        pred_aux = {}
        if isinstance(gen_out, tuple) and len(gen_out) == 2:
            pred_strs, pred_aux = gen_out
        else:
            pred_strs = gen_out
        future_values = pred_aux.get('future_values') if isinstance(pred_aux, dict) else None
        intensity_source = getattr(model, 'intensity_from_values', 'apex')
        future_values_cpu = future_values.detach().cpu().numpy() if future_values is not None else None

        for local_idx, (pred_str, gt_str, ps) in enumerate(zip(pred_strs, gt_json_strs, pred_starts)):
            ps = int(ps)
            sample_idx = len(sample_results)

            # 解析预测
            pred_events, status = parse_event_json(pred_str)
            if pred_events is None:
                pred_events = []
            needs_intensity_fill = any(
                not np.isfinite(float(ev.get('apex_intensity', float('nan'))))
                for ev in pred_events
            )
            if future_values_cpu is not None and needs_intensity_fill:
                pred_events = fill_event_intensity_from_values(pred_events, future_values_cpu[local_idx], source=intensity_source)

            # 解析 GT
            gt_events, _ = parse_event_json(gt_str)
            if gt_events is None:
                gt_events = []

            # 匹配
            match_res = match_events(pred_events, gt_events, tolerance=tolerance)
            errors    = compute_sample_errors(match_res, intensity_scale=intensity_scale)

            sample_results.append(errors)
            parse_statuses.append(status)

            # 转为 GT 格式（绝对索引）
            pred_gt_fmt = _rel_events_to_gt_format(
                pred_events, ps, intensity_scale=intensity_scale
            )
            gt_gt_fmt   = _rel_events_to_gt_format(
                gt_events, ps, intensity_scale=intensity_scale
            )

            output_records.append({
                'sample_idx':   sample_idx,
                'pred_start':   ps,
                'parse_status': status,
                'pred_raw':     pred_str,
                'pred_events':  pred_gt_fmt,
                'gt_events':    gt_gt_fmt,
                'pred_future':  future_values_cpu[local_idx].tolist() if future_values_cpu is not None else None,
                'tp': errors['tp'], 'fp': errors['fp'], 'fn': errors['fn'],
            })

        if verbose and (step + 1) % 50 == 0:
            elapsed = time.time() - t_eval_start
            eta = elapsed / (step + 1) * (total - step - 1)
            print(f'  evaluate [{step+1}/{total}]  elapsed={elapsed:.0f}s  eta={eta:.0f}s')

    metrics = aggregate_metrics(sample_results, parse_statuses)
    if verbose:
        print_metrics(metrics)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # 聚合指标
        metrics_path = os.path.join(save_dir, 'eval_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f'[evaluate] metrics saved → {metrics_path}')
        # 逐样本输出（含 pred_events / gt_events，GT 格式绝对索引）
        outputs_path = os.path.join(save_dir, 'eval_outputs.jsonl')
        with open(outputs_path, 'w', encoding='utf-8') as f:
            for rec in output_records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f'[evaluate] outputs saved → {outputs_path}')

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# predict：纯推理，不需要 GT 和评估
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    model,
    x: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 500,
    pred_start: int = 0,
    intensity_scale: float = 1.0,
) -> list[dict]:
    """
    对单条或批量输入做推理，返回解析后的事件列表（GT 格式绝对索引）。

    Parameters
    ----------
    model          : PatchLLM，已加载权重并移到 device
    x              : (96,) 或 (B, 96) 原始负荷值（未归一化）
    device         : 推理设备
    max_new_tokens : 生成最大 token 数
    pred_start     : 预测窗口全局起始索引，用于转绝对索引；
                     若只关心相对索引可传 0

    Returns
    -------
    list[list[dict]]，外层长度 = B，内层每个 dict 为一个事件：
      {onset_idx, end_idx, duration, apex_idx, apex_intensity}
    """
    model.eval()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        gen_out = model.generate(x, max_new_tokens=max_new_tokens, return_aux=True)
        pred_aux = {}
        if isinstance(gen_out, tuple) and len(gen_out) == 2:
            pred_strs, pred_aux = gen_out
        else:
            pred_strs = gen_out
        future_values = pred_aux.get('future_values') if isinstance(pred_aux, dict) else None
        intensity_source = getattr(model, 'intensity_from_values', 'apex')
        future_values_cpu = future_values.detach().cpu().numpy() if future_values is not None else None

    batch_results = []
    for idx, pred_str in enumerate(pred_strs):
        events, _ = parse_event_json(pred_str)
        if events is None:
            events = []
        needs_intensity_fill = any(
            not np.isfinite(float(ev.get('apex_intensity', float('nan'))))
            for ev in events
        )
        if future_values_cpu is not None and needs_intensity_fill:
            events = fill_event_intensity_from_values(events, future_values_cpu[idx], source=intensity_source)
        batch_results.append(
            _rel_events_to_gt_format(events, pred_start, intensity_scale=intensity_scale)
        )
    return batch_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

## CLI入口已移除（原LLM评估代码不适用于SmallDecoder，已删除对phase2_llm_finetune的依赖）
