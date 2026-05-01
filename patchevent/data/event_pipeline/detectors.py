"""Shared event detection helpers for event dataset builders."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig


def compute_dynamic_thresholds(
    df: pd.DataFrame,
    timestamp_col: str,
    value_col: str,
    config: PipelineConfig,
) -> Tuple[pd.Series, pd.Series]:
    values = df[value_col]
    shifted = values.shift(1)
    hour = pd.to_datetime(df[timestamp_col]).dt.hour

    enter_hourly = pd.Series(np.nan, index=df.index, dtype=float)
    exit_hourly = pd.Series(np.nan, index=df.index, dtype=float)

    for h in range(24):
        mask = hour == h
        s = shifted[mask]
        if s.empty:
            continue
        enter_hourly.loc[mask] = (
            s.rolling(window=config.history_days, min_periods=config.min_history_days)
            .quantile(config.enter_quantile)
            .to_numpy()
        )
        exit_hourly.loc[mask] = (
            s.rolling(window=config.history_days, min_periods=config.min_history_days)
            .quantile(config.exit_quantile)
            .to_numpy()
        )

    global_window = max(24 * config.history_days, 24)
    global_min_periods = max(24 * config.min_history_days, 24)
    enter_global = shifted.rolling(global_window, min_periods=global_min_periods).quantile(
        config.enter_quantile
    )
    exit_global = shifted.rolling(global_window, min_periods=global_min_periods).quantile(
        config.exit_quantile
    )

    enter_expand = shifted.expanding(min_periods=1).quantile(config.enter_quantile)
    exit_expand = shifted.expanding(min_periods=1).quantile(config.exit_quantile)

    enter_threshold = enter_hourly.fillna(enter_global).fillna(enter_expand).fillna(values)
    exit_threshold = exit_hourly.fillna(exit_global).fillna(exit_expand).fillna(values)
    exit_threshold = np.minimum(exit_threshold, enter_threshold)
    return enter_threshold, exit_threshold


def _event_from_range(values: np.ndarray, onset_idx: int, end_idx: int) -> Dict[str, float]:
    end_idx = max(onset_idx, end_idx)
    local = values[onset_idx : end_idx + 1]
    apex_offset = int(np.argmax(local))
    apex_idx = onset_idx + apex_offset
    return {
        "onset_idx": int(onset_idx),
        "end_idx": int(end_idx),
        "duration": int(end_idx - onset_idx + 1),
        "apex_idx": int(apex_idx),
        "apex_intensity": float(values[apex_idx]),
    }


def merge_close_events(
    events: List[Dict[str, float]],
    values: np.ndarray,
    merge_gap: int,
) -> List[Dict[str, float]]:
    if not events:
        return events
    merged: List[Dict[str, float]] = [events[0]]
    for event in events[1:]:
        prev = merged[-1]
        gap = event["onset_idx"] - prev["end_idx"] - 1
        if gap <= merge_gap:
            merged[-1] = _event_from_range(
                values,
                onset_idx=int(prev["onset_idx"]),
                end_idx=int(event["end_idx"]),
            )
        else:
            merged.append(event)
    return merged


def detect_events_from_peak_labels(
    values: np.ndarray,
    is_peak: np.ndarray,
    config: PipelineConfig,
) -> List[Dict[str, float]]:
    apexes = np.where(is_peak.astype(int) == 1)[0]
    if len(apexes) == 0:
        return []

    n = len(values)
    events: List[Dict[str, float]] = []
    for k, apex in enumerate(apexes):
        if k == 0:
            left_bound = max(0, int(apex) - int(config.max_flank_hours))
        else:
            left_bound = (int(apexes[k - 1]) + int(apex)) // 2

        if k == len(apexes) - 1:
            right_bound = min(n - 1, int(apex) + int(config.max_flank_hours))
        else:
            right_bound = (int(apex) + int(apexes[k + 1])) // 2

        onset = left_bound + int(np.argmin(values[left_bound : int(apex) + 1]))
        offset = int(apex) + int(np.argmin(values[int(apex) : right_bound + 1]))

        event = {
            "onset_idx": int(onset),
            "end_idx": int(offset),
            "duration": int(offset - onset + 1),
            "apex_idx": int(apex),
            "apex_intensity": float(values[int(apex)]),
        }
        if event["duration"] >= config.min_duration:
            events.append(event)

    # Merge strictly-overlapped ranges only while preserving anchor apex semantics.
    # NOTE: midpoint mode intentionally does NOT merge shared-boundary cases
    # (prev.end_idx == next.onset_idx), which are common at partition midpoints.
    if not events:
        return events
    merged = [events[0]]
    for event in events[1:]:
        prev = merged[-1]
        if event["onset_idx"] < prev["end_idx"]:
            apex_prev = prev if prev["apex_intensity"] >= event["apex_intensity"] else event
            merged[-1] = {
                "onset_idx": int(prev["onset_idx"]),
                "end_idx": int(max(prev["end_idx"], event["end_idx"])),
                "duration": int(max(prev["end_idx"], event["end_idx"]) - prev["onset_idx"] + 1),
                "apex_idx": int(apex_prev["apex_idx"]),
                "apex_intensity": float(apex_prev["apex_intensity"]),
            }
        else:
            merged.append(event)
    return merged


def detect_events(
    values: np.ndarray,
    enter_threshold: np.ndarray,
    exit_threshold: np.ndarray,
    config: PipelineConfig,
    warmup_steps: int = 0,
) -> List[Dict[str, float]]:
    events: List[Dict[str, float]] = []
    in_event = False
    onset_idx = 0
    warmup_steps = max(0, int(warmup_steps))

    for i, (value, enter_th, exit_th) in enumerate(zip(values, enter_threshold, exit_threshold)):
        if i < warmup_steps:
            continue

        if not in_event:
            if value >= enter_th:
                in_event = True
                onset_idx = i
        else:
            if value <= exit_th:
                end_idx = max(onset_idx, i - 1)
                candidate = _event_from_range(values, onset_idx, end_idx)
                if candidate["duration"] >= config.min_duration:
                    events.append(candidate)
                in_event = False

    if in_event:
        candidate = _event_from_range(values, onset_idx, len(values) - 1)
        if candidate["duration"] >= config.min_duration:
            events.append(candidate)

    return merge_close_events(events, values, config.merge_gap)


def _merge_overlapping_events_preserve_apex(
    events: List[Dict[str, float]],
    values: np.ndarray,
) -> List[Dict[str, float]]:
    if not events:
        return events

    sorted_events = sorted(events, key=lambda e: (int(e["onset_idx"]), int(e["end_idx"])))
    merged: List[Dict[str, float]] = [sorted_events[0]]

    for event in sorted_events[1:]:
        prev = merged[-1]
        if int(event["onset_idx"]) < int(prev["end_idx"]):
            apex_prev = int(prev["apex_idx"])
            apex_event = int(event["apex_idx"])
            apex_idx = apex_prev if values[apex_prev] >= values[apex_event] else apex_event

            merged_event = {
                "onset_idx": int(prev["onset_idx"]),
                "end_idx": int(max(int(prev["end_idx"]), int(event["end_idx"]))),
                "duration": int(max(int(prev["end_idx"]), int(event["end_idx"])) - int(prev["onset_idx"]) + 1),
                "apex_idx": int(apex_idx),
                "apex_intensity": float(values[apex_idx]),
                "event_source": "merged",
            }
            merged[-1] = merged_event
        else:
            merged.append(event)
    return merged


def _build_anchor_fallback_event(
    values: np.ndarray,
    anchor_idx: int,
    flank_hours: int,
    min_duration: int,
) -> Dict[str, float] | None:
    n = len(values)
    flank = max(1, int(flank_hours))
    left = max(0, int(anchor_idx) - flank)
    right = min(n - 1, int(anchor_idx) + flank)

    onset = left + int(np.argmin(values[left : int(anchor_idx) + 1]))
    end = int(anchor_idx) + int(np.argmin(values[int(anchor_idx) : right + 1]))
    duration = int(end - onset + 1)

    if duration < int(min_duration):
        return None

    return {
        "onset_idx": int(onset),
        "end_idx": int(end),
        "duration": int(duration),
        "apex_idx": int(anchor_idx),
        "apex_intensity": float(values[int(anchor_idx)]),
        "event_source": "anchor_fallback",
    }


def detect_events_anchor_only(
    values: np.ndarray,
    is_peak: np.ndarray,
    config: PipelineConfig,
    warmup_steps: int = 0,
) -> List[Dict[str, float]]:
    """
    Cleanest approach for 1-2 peaks/day data:
    - Each is_peak=1 anchor = exactly one event (no dynamic threshold).
    - Boundaries: midpoint-cut between neighbouring apexes, capped by max_flank_hours.
    - onset/offset = local valley (argmin) within respective half-window.
    - No event merging across different anchors.
    """
    apexes = np.where(is_peak.astype(int) == 1)[0]
    apexes = apexes[apexes >= int(warmup_steps)]
    if len(apexes) == 0:
        return []

    n = len(values)
    events: List[Dict[str, float]] = []

    for k, apex in enumerate(apexes):
        apex = int(apex)
        flank = int(config.max_flank_hours)

        # --- left boundary: midpoint with previous apex, ALSO capped by max_flank_hours ---
        # Taking max() means we pick the boundary CLOSER to the apex (tighter window).
        if k == 0:
            left_cut = max(int(warmup_steps), apex - flank)
        else:
            mid_left = (int(apexes[k - 1]) + apex) // 2
            left_cut = max(mid_left, apex - flank)

        # --- right boundary: midpoint with next apex, ALSO capped by max_flank_hours ---
        # Taking min() means we pick the boundary CLOSER to the apex (tighter window).
        if k == len(apexes) - 1:
            right_cut = min(n - 1, apex + flank)
        else:
            mid_right = (apex + int(apexes[k + 1])) // 2
            right_cut = min(mid_right, apex + flank)

        # Local valley as event onset/offset (deepest point in the capped half-window)
        onset = left_cut + int(np.argmin(values[left_cut : apex + 1]))
        offset = apex + int(np.argmin(values[apex : right_cut + 1]))

        duration = offset - onset + 1
        if duration >= int(config.min_duration):
            events.append({
                "onset_idx": int(onset),
                "end_idx": int(offset),
                "duration": int(duration),
                "apex_idx": int(apex),
                "apex_intensity": float(values[apex]),
                "event_source": "anchor_only",
            })

    # Post-processing: guarantee gap >= 1 between adjacent events.
    # When both events claim the same midpoint valley (gap=-1) or are touching
    # (gap=0), split at the inter-apex midpoint so the valley itself is a gap.
    for k in range(len(events) - 1):
        gap = int(events[k + 1]["onset_idx"]) - int(events[k]["end_idx"]) - 1
        if gap < 1:
            mid = (int(events[k]["apex_idx"]) + int(events[k + 1]["apex_idx"])) // 2
            events[k]["end_idx"] = int(mid) - 1
            events[k]["duration"] = int(events[k]["end_idx"]) - int(events[k]["onset_idx"]) + 1
            events[k + 1]["onset_idx"] = int(mid) + 1
            events[k + 1]["duration"] = int(events[k + 1]["end_idx"]) - int(events[k + 1]["onset_idx"]) + 1
    # Drop events that became degenerate (duration < min_duration) after the adjustment.
    events = [e for e in events if int(e["duration"]) >= int(config.min_duration)]

    return events


def detect_events_gradient_width(
    values: np.ndarray,
    is_peak: np.ndarray,
    config: PipelineConfig,
    warmup_steps: int = 0,
    grad_values: np.ndarray | None = None,
) -> List[Dict[str, float]]:
    """
    变宽度峰值区间检测（2–6 h 总宽，左右不对称）。

    算法：高度阈值法（类 FWHM）+ 单调性检查
    ──────────────────────────────────────────────
    1. 以 ±max_h 搜索窗内 value_60min 的局部最大值为「有效峰高」（gv_peak），
       左/右搜索窗内最小值为「基线」（baseline）。
    2. 「停止水平」= baseline + rate_frac × (gv_peak - baseline)
       - rate_frac=0.3 → 包含峰高 30% 以上的范围 → 宽区间
       - rate_frac=0.6 → 只保留顶部 40% → 窄区间
       - 默认 0.40，对应 2-6 h 目标范围
    3. 向左/右逐步扩展：若 gv[i] >= 停止水平 且未触及硬边界 → 继续纳入
       额外：若 gv[i] > gv[i+1]（向左走时负荷反而升高 = 越过了谷底） → 立即停止
    4. 硬约束：每侧最多 max_half_hours，至少 min_half_hours
    5. 后处理：相邻事件 gap >= 1

    参数:
      values      : 事件强度标注用（通常是 value_max）
      grad_values : 停止水平计算用（应为 value_60min，真实逐小时负荷）；None 则回退到 values
      rate_frac   : 高度截止分位 [0,1]；越小 → 包含越宽，越大 → 越窄

    效果：
      尖峰（高而窄）→ 小区间(2-3h)，宽峰（矮而宽）→ 大区间(5-6h)，左右天然不对称。
    """
    gv = grad_values if grad_values is not None else values

    apexes = np.where(is_peak.astype(int) == 1)[0]
    apexes = apexes[apexes >= int(warmup_steps)]
    if len(apexes) == 0:
        return []

    n = len(values)
    max_h = int(getattr(config, "max_half_hours", 3))
    min_h = int(getattr(config, "min_half_hours", 1))
    height_frac = float(getattr(config, "rate_frac", 0.40))  # rate_frac 在此模式下即高度分位

    events: List[Dict[str, float]] = []

    for k, apex in enumerate(apexes):
        apex = int(apex)
        apex_val = float(values[apex])

        # 中点边界（不跨越相邻峰）
        left_mid_cap = int(apexes[k - 1] + apex) // 2 if k > 0 else int(warmup_steps)
        right_mid_cap = int(apex + apexes[k + 1]) // 2 if k < len(apexes) - 1 else n - 1

        # 综合左/右硬边界
        left_hard = max(left_mid_cap + 1, apex - max_h, int(warmup_steps))
        right_hard = min(right_mid_cap - 1, apex + max_h, n - 1)

        # ── 局部有效峰高（搜索窗内 gv 最大值）────────────────────
        gv_peak = float(np.max(gv[left_hard : right_hard + 1]))

        # 分侧基线（各侧搜索窗内 gv 最小值）
        left_baseline = float(np.min(gv[left_hard : apex + 1])) if left_hard <= apex else float(gv[apex])
        right_baseline = float(np.min(gv[apex : right_hard + 1])) if apex <= right_hard else float(gv[apex])

        # 停止水平 = baseline + height_frac × (峰高 - 基线)
        # → 包含负荷高于「基线以上 height_frac 处」的范围
        left_stop = left_baseline + height_frac * (gv_peak - left_baseline)
        right_stop = right_baseline + height_frac * (gv_peak - right_baseline)

        # ── 左侧扩展 ────────────────────────────────────────────
        onset = apex
        for step in range(1, max_h + 1):
            i = apex - step
            if i < left_hard:
                break
            if float(gv[i]) < left_stop:
                break
            onset = i

        onset = min(onset, max(apex - min_h, left_hard))  # 最小宽度保证

        # ── 右侧扩展 ────────────────────────────────────────────
        offset = apex
        for step in range(1, max_h + 1):
            i = apex + step
            if i > right_hard:
                break
            if float(gv[i]) < right_stop:
                break
            offset = i

        offset = max(offset, min(apex + min_h, right_hard))  # 最小宽度保证

        duration = offset - onset + 1
        events.append({
            "onset_idx": int(onset),
            "end_idx": int(offset),
            "duration": int(duration),
            "apex_idx": int(apex),
            "apex_intensity": float(apex_val),
            "event_source": "gradient_width",
        })

    # ── 后处理：保证相邻事件之间 gap >= 1 ──────────────────────
    for k in range(len(events) - 1):
        gap = int(events[k + 1]["onset_idx"]) - int(events[k]["end_idx"]) - 1
        if gap < 1:
            mid = (int(events[k]["apex_idx"]) + int(events[k + 1]["apex_idx"])) // 2
            events[k]["end_idx"] = int(mid) - 1
            events[k + 1]["onset_idx"] = int(mid) + 1
            events[k]["duration"] = int(events[k]["end_idx"]) - int(events[k]["onset_idx"]) + 1
            events[k + 1]["duration"] = int(events[k + 1]["end_idx"]) - int(events[k + 1]["onset_idx"]) + 1

    events = [e for e in events if int(e["duration"]) >= 1]
    return events


def detect_events_hybrid(
    values: np.ndarray,
    is_peak: np.ndarray,
    enter_threshold: np.ndarray,
    exit_threshold: np.ndarray,
    config: PipelineConfig,
    warmup_steps: int = 0,
) -> List[Dict[str, float]]:
    # Hybrid GT: threshold-crossing boundaries + peak-label apex refinement + anchor fallback.
    raw_events = detect_events(
        values=values,
        enter_threshold=enter_threshold,
        exit_threshold=exit_threshold,
        config=config,
        warmup_steps=warmup_steps,
    )

    peak_arr = is_peak.astype(int)
    refined: List[Dict[str, float]] = []
    covered = np.zeros(len(values), dtype=bool)

    for event in raw_events:
        s = int(event["onset_idx"])
        e = int(event["end_idx"])

        local_anchors = np.where(peak_arr[s : e + 1] == 1)[0]
        if len(local_anchors) > 0:
            candidate_idx = s + local_anchors
            apex_idx = int(candidate_idx[int(np.argmax(values[candidate_idx]))])
        else:
            apex_idx = int(event["apex_idx"])

        refined_event = {
            "onset_idx": s,
            "end_idx": e,
            "duration": int(e - s + 1),
            "apex_idx": apex_idx,
            "apex_intensity": float(values[apex_idx]),
            "event_source": "threshold",
        }
        refined.append(refined_event)
        covered[s : e + 1] = True

    anchor_indices = np.where(peak_arr == 1)[0]
    for anchor in anchor_indices:
        if int(anchor) < warmup_steps:
            continue
        if covered[int(anchor)]:
            continue

        fallback_event = _build_anchor_fallback_event(
            values=values,
            anchor_idx=int(anchor),
            flank_hours=config.anchor_fallback_window_hours,
            min_duration=config.min_duration,
        )
        if fallback_event is None:
            continue

        s = int(fallback_event["onset_idx"])
        e = int(fallback_event["end_idx"])
        refined.append(fallback_event)
        covered[s : e + 1] = True

    return _merge_overlapping_events_preserve_apex(refined, values)
