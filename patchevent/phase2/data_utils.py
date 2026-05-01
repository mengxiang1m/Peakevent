"""
data_utils.py
=============
Phase-2 共享工具：JSON 事件构建、解析、collate function。

主要内容
--------
- build_event_json(events, pred_start)     : 从事件列表 + 预测窗口起点构建 JSON 字符串
- parse_event_json(json_str)               : 鲁棒解析 LLM 输出的 JSON，返回事件列表
- collate_fn(batch)                        : DataLoader collate，处理变长 token padding
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

import warnings

import torch
import numpy as np

INSTRUCTION_PROMPT = (
    'Output only JSON, no explanation. '
    'Format: {"peak_events":[[onset,duration,apex_index,intensity],...]}. '
    'Output: '
)
JSON_START_PREFIX = '{"peak_events":[['


# ─────────────────────────────────────────────────────────────────────────────
# JSON 构建
# ─────────────────────────────────────────────────────────────────────────────

def build_event_json(
    events: list[dict],
    pred_start: int,
    norm_mean: float | None = None,
    pred_len: int | None = None,
    event_schema: str = 'quad',
) -> str:
    """
    将绝对索引事件列表转换为相对于预测窗口起点的紧凑 JSON 字符串。

    Parameters
    ----------
    events     : 筛选后的事件列表，每个元素含 onset_idx, duration, apex_idx, apex_intensity
    pred_start : 预测窗口的全局起始索引（= sample_start + seq_len）
    norm_mean  : 若提供，则将 apex_intensity 除以该值归一化
    pred_len   : 预测窗口长度；若提供，apex_rel 超出窗口时截断到 pred_len - 1

    Returns
    -------
    紧凑数组格式 JSON 字符串，示例：
        {"peak_events":[[12,5,15,1.43],[30,3,31,1.32]]}
    每个子数组：[onset, duration, apex_index, apex_intensity(2位小数)]
    无事件时返回：{"peak_events":[]}
    """
    if event_schema not in ('quad', 'triplet'):
        raise ValueError(f'Unsupported event_schema: {event_schema}')
    peak_events = []
    for ev in events:
        onset_rel = int(ev['onset_idx']) - pred_start
        apex_rel  = int(ev['apex_idx'])  - pred_start
        # 过滤负数相对索引（边界情况）
        if onset_rel < 0 or apex_rel < 0:
            continue
        # onset 超出窗口的事件不应出现（由 _get_events_in_window 保证），双重校验
        if pred_len is not None and onset_rel >= pred_len:
            continue
        # apex 可能略微超出窗口，截断到窗口末尾，避免训练时出现越界索引
        if pred_len is not None and apex_rel >= pred_len:
            apex_rel = pred_len - 1
        intensity = float(ev['apex_intensity'])
        if norm_mean is not None and norm_mean > 0:
            intensity = intensity / norm_mean
        item = [
            onset_rel,
            int(ev['duration']),
            apex_rel,
        ]
        if event_schema == 'quad':
            item.append(round(intensity, 2))
        peak_events.append(item)
    return json.dumps({'peak_events': peak_events}, separators=(',', ':'))


def fill_event_intensity_from_values(
    events: list[dict],
    future_values,
    source: str = 'apex',
) -> list[dict]:
    if source not in ('apex', 'span_max'):
        raise ValueError(f'Unsupported intensity source: {source}')
    future_arr = np.asarray(future_values, dtype=np.float32).reshape(-1)
    n = int(future_arr.shape[0])
    results = []
    for ev in events:
        new_ev = dict(ev)
        intensity = float('nan')
        onset = int(ev.get('onset', 0))
        duration = max(int(ev.get('duration', 1)), 1)
        apex = int(ev.get('apex_index', onset))
        if n > 0:
            if source == 'span_max':
                lo = max(0, onset)
                hi = min(n, onset + duration)
                if hi <= lo:
                    idx = min(max(apex, 0), n - 1)
                    intensity = float(future_arr[idx])
                else:
                    intensity = float(np.max(future_arr[lo:hi]))
            else:
                idx = min(max(apex, 0), n - 1)
                intensity = float(future_arr[idx])
        # Guard against NaN/Inf from dense value head outputs
        if not np.isfinite(intensity):
            intensity = 1.0  # neutral default (ratio ≈ 1 means apex ≈ local mean)
        new_ev['apex_intensity'] = intensity
        results.append(new_ev)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# JSON 解析（鲁棒版）
# ─────────────────────────────────────────────────────────────────────────────

def parse_event_json(text: str) -> tuple[list[dict] | None, str]:
    """
    从 LLM 输出文本中鲁棒地解析峰值事件列表。

    解析策略
    --------
    1. 直接 json.loads
    2. 正则提取最大 {...} 块后 json.loads
    3. 正则提取 peak_events: [...] 模式（缺外层 {} 的情况），含截断修复
    4. 提取裸数组 [...] — 带 ] 或截断版本
    5. Python-style dict/list（ast.literal_eval）
    6. 截断修复最后 {...} 块
    7. 正则逐行提取事件（numbered 格式 fallback）
    8. 全部失败 → 返回 (None, 'parse_fail')

    Returns
    -------
    (events, status)
      events : List[dict] 或 None（解析失败时）
      status : 'ok' | 'recovered' | 'parse_fail'
    """
    if not isinstance(text, str) or not text.strip():
        return None, 'parse_fail'
    stripped = text.strip()

    # 前处理：去掉开头的编号前缀，如 '1.' / '2.' / '1.peak events:'
    stripped = re.sub(r'^\d+\.\s*', '', stripped)

    # ── 策略 1: 直接解析 ─────────────────────────────────────────
    try:
        obj = json.loads(stripped)
        events = _extract_events(obj)
        if events is not None:
            return events, 'ok'
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # ── 策略 2a: 括号计数提取第一个完整 JSON 对象（避免贪婪匹配吃掉后续幻觉文本）────
    _brace_start = stripped.find('{')
    if _brace_start >= 0:
        _depth = 0
        for _i, _ch in enumerate(stripped[_brace_start:], _brace_start):
            if _ch == '{':
                _depth += 1
            elif _ch == '}':
                _depth -= 1
                if _depth == 0:
                    try:
                        obj = json.loads(stripped[_brace_start:_i + 1])
                        events = _extract_events(obj)
                        if events is not None:
                            return events, 'ok'
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass
                    break

    # ── 策略 2b: 正则提取最大 {...} 块（兜底）────────────────────
    big_match = re.search(r'\{.*\}', stripped, re.DOTALL)
    if big_match:
        try:
            obj = json.loads(big_match.group())
            events = _extract_events(obj)
            if events is not None:
                return events, 'recovered'
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # ── 策略 3: peak_events: [...] 格式（缺外层 {}，支持下划线/空格）────────
    m3 = re.search(r'peak[_\s]?events\s*[:\s]\s*(\[.*)', stripped, re.DOTALL | re.IGNORECASE)
    if m3:
        arr_text = m3.group(1).strip()
        for arr_candidate in _fix_truncated_array(arr_text):
            try:
                obj = json.loads(arr_candidate)
                events = _extract_events(obj)
                if events is not None:
                    return events, 'recovered'
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    # ── 策略 4a: 完整裸数组 [...] ─────────────────────────────────
    m4 = re.search(r'\[.*\]', stripped, re.DOTALL)
    if m4:
        for arr_candidate in _fix_truncated_array(m4.group(0)):
            try:
                obj = json.loads(arr_candidate)
                events = _extract_events(obj)
                if events is not None:
                    return events, 'recovered'
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    # ── 策略 4b: 截断裸数组（以 [ 开头，无 ]）────────────────────
    arr_start = stripped.find('[')
    if arr_start >= 0:
        arr_text = stripped[arr_start:]
        for arr_candidate in _fix_truncated_array(arr_text):
            try:
                obj = json.loads(arr_candidate)
                events = _extract_events(obj)
                if events is not None:
                    return events, 'recovered'
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    # ── 策略 5: Python-style dict/list（单引号），含截断修复 ─────
    for py_text in [stripped, re.sub(r"^\s*\w+\s*=\s*", '', stripped)]:
        for candidate in _fix_truncated_pytext(py_text):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', SyntaxWarning)
                    obj = ast.literal_eval(candidate)
                events = _extract_events(obj)
                if events is not None:
                    return events, 'recovered'
            except (ValueError, SyntaxError):
                pass

    # ── 策略 6: 截断修复（在 {...} 中间被切断）───────────────────
    for suffix in [']}}', ']}', '}', '"}]}', '"]}']:
        try:
            obj = json.loads(stripped + suffix)
            events = _extract_events(obj)
            if events is not None:
                return events, 'recovered'
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    for cut_marker in ['}, {', '},{"onset', '}, {"onset']:
        idx = stripped.rfind(cut_marker)
        if idx > 0:
            for suffix in [']}}', ']}', '}']:
                try:
                    obj = json.loads(stripped[:idx + 1] + suffix)
                    events = _extract_events(obj)
                    if events is not None:
                        return events, 'recovered'
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

    # ── 策略 7: 正则逐条提取（应对 numbered/Python-str/任意格式）──
    events = _regex_extract_events(stripped)
    if events:
        return events, 'recovered'

    return None, 'parse_fail'


def _fix_truncated_pytext(py_text: str) -> list[str]:
    """
    尝试修复被截断的 Python-style dict/list 字符串，返回候选字符串列表。
    主要处理 {'key': [...]} 或 [...] 被截断的情况。
    """
    candidates = [py_text]
    # 找最后一个完整的 dict 项（以 }, 结尾）
    for marker in ["}, {", "},\n{", "}, '", "},\n'"]:
        idx = py_text.rfind(marker)
        if idx > 0:
            truncated = py_text[:idx + 1]
            for suffix in [']', "']}", '}', "]}",  "']}"]:
                candidates.append(truncated + suffix)
    # 直接追加各种结尾
    for suffix in [']', "']}", '}', "]}",  "']}", "}]}"]:
        candidates.append(py_text + suffix)
    return candidates


def _regex_extract_events(text: str) -> list[dict]:
    """
    最后手段：正则直接从文本中提取 onset/duration/apex_index 数字。
    适用于 numbered 格式如 `0: {peak_index: 4, ...}1: {...}` 等。
    """
    events = []
    onset_pat = re.compile(
        r'(?:onset|start)\s*[:\s=]+\s*(\d+)',
        re.IGNORECASE
    )
    dur_pat  = re.compile(r'(?:duration|dur|length)\s*[:\s=]+\s*(\d+)', re.IGNORECASE)
    apex_pat = re.compile(r'(?:apex[_\s]?index|apex[_\s]?idx|peak[_\s]?index)\s*[:\s=]+\s*(\d+)', re.IGNORECASE)
    int_pat  = re.compile(r'(?:intensity|value|peak[_\s]?value|amplitude)\s*[:\s=]+\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE)

    onsets = [int(m.group(1)) for m in onset_pat.finditer(text)]
    durs   = [int(m.group(1)) for m in dur_pat.finditer(text)]
    apexes = [int(m.group(1)) for m in apex_pat.finditer(text)]
    ints   = [float(m.group(1)) for m in int_pat.finditer(text)]

    if not onsets:
        # fallback 1: peak_index 作为 onset
        peak_idx_pat = re.compile(r'peak[_\s]?index\s*[:\s=]+\s*(\d+)', re.IGNORECASE)
        onsets = [int(m.group(1)) for m in peak_idx_pat.finditer(text)]

    if not onsets:
        # fallback 2: `index:` 单独出现（格式如 `0: {index: 1, time: ...}`）
        # 要排除已经被 apex_pat 匹配的位置
        apex_spans = [m.span() for m in apex_pat.finditer(text)]
        for m in re.finditer(r'\bindex\s*[:\s=]+\s*(\d+)', text, re.IGNORECASE):
            # 确保不在 apex_pat 的匹配范围内
            in_apex = any(s <= m.start() < e for s, e in apex_spans)
            if not in_apex:
                onsets.append(int(m.group(1)))

    for i, onset in enumerate(onsets):
        dur      = durs[i]   if i < len(durs)   else 1
        apex_idx = apexes[i] if i < len(apexes) else onset
        apex_int = ints[i]   if i < len(ints)   else float('nan')
        if onset >= 0 and apex_idx >= 0:
            events.append({
                'onset':          onset,
                'duration':       dur,
                'apex_index':     apex_idx,
                'apex_intensity': apex_int,
            })
    return events


def _fix_truncated_array(arr_text: str) -> list[str]:
    """尝试修复被截断的 JSON 数组，返回候选字符串列表（从严到宽）。"""
    candidates = [arr_text]
    # 截断修复：在最后一个完整对象后切断
    last_obj_end = arr_text.rfind('},')
    if last_obj_end > 0:
        candidates.append(arr_text[:last_obj_end + 1] + ']')
    last_close = arr_text.rfind('}')
    if last_close > 0:
        candidates.append(arr_text[:last_close + 1] + ']')
    # 补充不同结尾
    for suffix in [']}', ']', '"}]}', '"}]']:
        candidates.append(arr_text + suffix)
    return candidates


def _extract_events(obj: Any) -> list[dict] | None:
    """
    从解析成功的 JSON 对象中提取 peak_events 列表，并做字段校验。
    支持多种 LLM 输出格式变体。
    返回 None 表示提取失败（区别于返回空列表 []）。
    """
    # 模型可能直接生成裸列表 [...] 而非 {"peak_events": [...]}
    if isinstance(obj, list):
        raw = obj
    elif isinstance(obj, dict):
        # 优先取 peak_events，也兼容其他常见键
        for key in ('peak_events', 'events', 'peaks', 'results'):
            if key in obj and isinstance(obj[key], list):
                raw = obj[key]
                break
        else:
            # 如果顶层 dict 本身就是单个事件的结构
            raw = [obj]
    else:
        return None

    if not isinstance(raw, list):
        return None

    events = []
    for item in raw:
        if isinstance(item, dict):
            ev = _parse_single_event(item)
        elif isinstance(item, list):
            ev = _parse_list_event(item)
        else:
            continue
        if ev is not None:
            events.append(ev)
    return events


def _parse_list_event(item: list) -> dict | None:
    """
    处理 LLM 输出的位置数组格式事件：
      [onset, apex_index]                         (2 元素)
      [onset, duration, apex_index]               (3 元素)
      [onset, duration, apex_index, intensity]    (4 元素，标准紧凑格式)
      [onset, duration, ["peak", int]]            (3 元素, 第三元素为列表)
    """
    try:
        if len(item) < 2:
            return None
        onset    = int(item[0])
        duration = 1
        apex_index = onset
        apex_intensity = float('nan')
        if len(item) == 2:
            # [onset, apex_index]
            apex_index = int(item[1])
        elif len(item) == 3:
            # [onset, duration, apex_index]
            duration  = int(item[1])
            apex_val  = item[2]
            if isinstance(apex_val, (int, float)):
                apex_index = int(apex_val)
            elif isinstance(apex_val, list) and len(apex_val) >= 2:
                apex_index = onset + duration // 2
                try:
                    apex_intensity = float(apex_val[-1])
                except (TypeError, ValueError):
                    pass
        elif len(item) >= 4:
            # [onset, duration, apex_index, intensity]  ← 标准紧凑格式
            duration       = int(item[1])
            apex_index     = int(item[2])
            apex_intensity = float(item[3])
        if onset < 0 or apex_index < 0:
            return None
        return {
            'onset':          onset,
            'duration':       duration,
            'apex_index':     apex_index,
            'apex_intensity': apex_intensity,
        }
    except (TypeError, ValueError, IndexError):
        return None


def _parse_single_event(item: dict) -> dict | None:
    """
    从单个事件 dict 中提取字段，兼容多种键名变体：
    - 标准:    onset, duration, apex.index, apex.intensity
    - 变体1:   peak_index → onset (无 duration 时默认 1)
    - 变体2:   apex 为 int，index 为兄弟键
    - 变体3:   apex_index / apex_idx 直接在顶层
    """
    try:
        # ─── onset ────────────────────────────────────────────────
        onset = None
        for k in ('onset', 'onset_index', 'start', 'peak_index', 'index'):
            if k in item:
                onset = int(item[k])
                break
        if onset is None:
            return None

        # ─── duration ─────────────────────────────────────────────
        duration = 1
        for k in ('duration', 'dur', 'length', 'width'):
            if k in item:
                duration = int(item[k])
                break

        # ─── apex_index ───────────────────────────────────────────
        apex_index = None
        apex_intensity = float('nan')
        apex_val = item.get('apex')

        if isinstance(apex_val, dict):
            # 标准嵌套结构: {"index": N, "intensity": V}
            apex_index = int(apex_val.get('index', apex_val.get('idx', onset)))
            apex_intensity = float(apex_val.get('intensity', apex_val.get('value', float('nan'))))
        elif isinstance(apex_val, (int, float)):
            # apex 是裸数值：当作 apex_index，强度可能在兄弟键中
            apex_index = int(apex_val)
            for k in ('intensity', 'value', 'peak_value', 'amplitude'):
                if k in item:
                    apex_intensity = float(item[k])
                    break
        else:
            # apex 键不存在，尝试顶层键
            for k in ('apex_index', 'apex_idx', 'peak_apex'):
                if k in item:
                    apex_index = int(item[k])
                    break
            if apex_index is None:
                apex_index = onset  # fallback
            for k in ('apex_intensity', 'intensity', 'value', 'peak_value', 'amplitude'):
                if k in item:
                    apex_intensity = float(item[k])
                    break

        if apex_index is None:
            apex_index = onset

        # ─── 有效性检查 ───────────────────────────────────────────
        if onset < 0 or apex_index < 0 or duration < 0:
            return None

        return {
            'onset':          onset,
            'duration':       duration,
            'apex_index':     apex_index,
            'apex_intensity': apex_intensity,
        }
    except (KeyError, TypeError, ValueError):
        return None


## LLM collate_fn已删除（SmallDecoder使用dataset.py中自定义的collate_fn）
