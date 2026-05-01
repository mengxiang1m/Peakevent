"""Shared point-level and patch-level labeling helpers."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def attach_point_level_event_fields(
    df: pd.DataFrame,
    events: List[Dict[str, float]],
    timestamp_col: str,
) -> pd.DataFrame:
    """
    为每个时间点添加以下字段：

    phase       : 相位标签（background / rising / apex / falling）
                  - rising   : [onset_idx, apex_idx - 1]
                  - apex     : apex_idx 本身（is_peak=1 的锚点）
                  - falling  : [apex_idx + 1, end_idx]
                  - background: 不属于任何事件的时间点

    d_to_apex   : 到「最近峰顶」的时间步距离（小时），始终 >= 0
                  - 语义：当前点距前后峰顶中最近那个的绝对距离
                  - 在峰顶本身 = 0；向两侧单调递增，呈 V 形谷；
                    信号连续无跳变，适合作为序列模型的近峰度特征。

    is_peak_event: 该点是否属于任意事件窗口（0/1）；事件窗口 = [onset_idx, end_idx] 闭区间，
                   涵盖 rising / apex / falling 三个阶段，background 点为 0。
    """
    n = len(df)
    phase = np.full(n, "background", dtype=object)
    is_peak_event = np.zeros(n, dtype=int)

    # ── Phase 标注 ──────────────────────────────────────────
    for event in events:
        s = int(event["onset_idx"])
        e = int(event["end_idx"])
        a = int(event["apex_idx"])

        if s < a:
            phase[s:a] = "rising"
        phase[a] = "apex"
        is_peak_event[s : e + 1] = 1  # 标记整个事件窗口（onset→end 闭区间）
        if a < e:
            phase[a + 1 : e + 1] = "falling"

    # ── d_to_apex：nearest-apex 距离（所有点，始终 >= 0）──────────────────
    if events:
        apex_arr = np.array(sorted(int(e["apex_idx"]) for e in events), dtype=int)
        i_arr = np.arange(n, dtype=int)
        # pos = 第一个 apex_arr[pos] >= i 的位置（右侧峰顶）
        pos = np.searchsorted(apex_arr, i_arr, side="left")  # shape (n,)

        _INF = np.iinfo(np.int64).max // 2
        # 右侧最近峰顶距离
        dist_right = np.where(
            pos < len(apex_arr),
            apex_arr[np.minimum(pos, len(apex_arr) - 1)] - i_arr,
            _INF,
        )
        # 左侧最近峰顶距离
        dist_left = np.where(
            pos > 0,
            i_arr - apex_arr[np.maximum(pos - 1, 0)],
            _INF,
        )
        d_to_apex = np.minimum(dist_right, dist_left).astype(int)
    else:
        d_to_apex = np.zeros(n, dtype=int)

    df_out = df.copy()
    df_out["d_to_apex"] = d_to_apex
    df_out["is_peak_event"] = is_peak_event
    df_out["phase"] = phase
    return df_out


def build_patch_label_table(
    df: pd.DataFrame,
    patch_len: int = 8,
    stride: int = 4,
) -> pd.DataFrame:
    """
    生成 patch 级监督标签表，用于编码器（CPP 模块 A）的训练。

    设计原则（方向 B）：
      patch_len=8h > 事件最大宽度(6h)，保证一个完整峰值事件能被单个 patch 覆盖，
      实现「1 event ≈ 1 patch ≈ 1 LLM token」的自然对齐。
      监督信号采用软标签（相位比例 + apex 偏移 + 强度），而非单点硬分类。

    输出列：
      patch_start_idx     : patch 起始行在序列 CSV 中的行号（0-based）
      patch_start_time    : patch 第一个点的时间戳
      has_apex            : patch 内是否含 apex（0/1），严格二分类信号
      apex_offset         : apex 在 patch 内的相对位置（0 ~ patch_len-1），无 apex 时为 -1
      min_d_to_apex       : patch 内所有点 d_to_apex 的最小值（连续回归目标）
      phase_dist_bg       : background 点占比（0~1，软标签回归目标）
      phase_dist_rising   : rising 点占比
      phase_dist_apex     : apex 点占比
      phase_dist_falling  : falling 点占比
      apex_intensity      : apex 时刻的负荷值（MW），无 apex 时为 -1.0
      split               : 继承 patch 中心点的 train/val/test 划分
    """
    n = len(df)
    records = []
    for start in range(0, n - patch_len + 1, stride):
        patch = df.iloc[start : start + patch_len]
        ph = patch["phase"].values

        # has_apex & apex_offset
        apex_mask = patch["is_peak"].values == 1
        has_apex = int(apex_mask.any())
        apex_pos = int(np.where(apex_mask)[0][0]) if has_apex else -1
        apex_intensity = float(patch.iloc[apex_pos]["value"]) if has_apex else -1.0

        # min d_to_apex
        min_d = int(patch["d_to_apex"].min())

        # phase distribution（软标签）
        bg_frac = float((ph == "background").sum()) / patch_len
        rising_frac = float((ph == "rising").sum()) / patch_len
        apex_frac = float((ph == "apex").sum()) / patch_len
        falling_frac = float((ph == "falling").sum()) / patch_len

        # split：使用 patch 中心点的划分
        split = str(df.iloc[start + patch_len // 2]["split"])

        records.append({
            "patch_start_idx": start,
            "patch_start_time": str(patch.iloc[0]["timestamp"]),
            "has_apex": has_apex,
            "apex_offset": apex_pos,
            "min_d_to_apex": min_d,
            "phase_dist_bg": round(bg_frac, 4),
            "phase_dist_rising": round(rising_frac, 4),
            "phase_dist_apex": round(apex_frac, 4),
            "phase_dist_falling": round(falling_frac, 4),
            "apex_intensity": round(apex_intensity, 3),
            "split": split,
        })
    return pd.DataFrame(records)
