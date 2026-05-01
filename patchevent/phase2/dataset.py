"""
dataset.py
==========
SmallDecoder 数据集：复用 phase2_llm_finetune 的数据加载，用自定义 EventTokenizer 编码 JSON。
"""

from __future__ import annotations

import json
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # TODO(R-future): migrate to patchevent package import

from patchevent.phase2.data_utils import build_event_json
from patchevent.phase2.model import EventTokenizer

# patch label columns (same as phase1)
_LBL_COLS = [
    'has_apex', 'apex_offset', 'min_d_to_apex',
    'phase_dist_bg', 'phase_dist_rising', 'phase_dist_apex', 'phase_dist_falling',
    'apex_intensity',
]
_DEFAULT_LBL = np.array([0., -1., 0., 1., 0., 0., 0., -1.], dtype=np.float32)


class SmallDecoderDataset(Dataset):
    """
    直接复用 phase2_llm_finetune 的数据加载逻辑，用 EventTokenizer 编码 JSON。

    返回字段：
      x            : (96,) 原始负荷时序
      target_ids   : (T,) 自定义 token ids（含 <bos>/<eos>）
      json_str     : 原始 JSON 字符串（evaluate 用）
      pred_start   : 预测窗口起始全局索引
    """

    def __init__(
        self,
        series_path: str,
        events_path: str,
        split: str = 'train',
        seq_len: int = 96,
        pred_len: int = 96,
        event_schema: str = 'quad',
        window_stride: int = 4,
        mean: float | None = None,
        std: float | None = None,
        patch_labels_path: str | None = None,
        patch_len: int = 8,
        patch_stride: int = 4,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.event_schema = event_schema
        self.tokenizer = EventTokenizer(pred_len=pred_len, event_schema=event_schema)
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.n_patches = (seq_len - patch_len) // patch_stride + 1

        # ── 加载时序 ─────────────────────────────────────────────────────────
        series = pd.read_csv(series_path, parse_dates=['timestamp'])
        self.all_values = series['value'].values.astype(np.float32)
        all_splits = series['split'].values

        # ── 时间特征（小时 / 星期几）─────────────────────────────────────────
        self.hours = series['timestamp'].dt.hour.values.astype(np.int64)
        self.weekdays = series['timestamp'].dt.weekday.values.astype(np.int64)

        train_vals = self.all_values[all_splits == 'train']
        if mean is None or std is None:
            self.mean = float(train_vals.mean())
            self.std = float(train_vals.std()) + 1e-8
        else:
            self.mean = float(mean)
            self.std = float(std)

        # ── 窗口位置 ─────────────────────────────────────────────────────────
        split_idx = np.where(all_splits == split)[0]
        g_start = int(split_idx[0])
        g_end = int(split_idx[-1])
        window_end = g_end - seq_len - pred_len + 1

        offset = (window_stride - g_start % window_stride) % window_stride
        first_aligned = g_start + offset

        if window_end < first_aligned:
            self.starts = np.array([], dtype=np.int64)
        else:
            self.starts = np.arange(first_aligned, window_end + 1, window_stride, dtype=np.int64)

        # ── 加载事件 ─────────────────────────────────────────────────────────
        self.events: list[dict] = []
        with open(events_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.events.append(json.loads(line))
        self.events.sort(key=lambda e: e['onset_idx'])
        self._onset_arr = np.array([e['onset_idx'] for e in self.events], dtype=np.int64)

        # ── 可选: 加载 patch 标签 (joint 训练模式需要) ──────────────────
        self._patch_lbl = None
        if patch_labels_path is not None:
            pl = pd.read_csv(patch_labels_path)
            lbl_arr = pl[_LBL_COLS].values.astype(np.float32)
            psi_arr = pl['patch_start_idx'].values.astype(np.int64)
            self._patch_lbl = {int(psi_arr[i]): lbl_arr[i] for i in range(len(psi_arr))}

    def _get_events_in_window(self, pred_start: int, pred_end: int) -> list[dict]:
        lo = int(np.searchsorted(self._onset_arr, pred_start, side='left'))
        hi = int(np.searchsorted(self._onset_arr, pred_end - 1, side='right'))
        return self.events[lo:hi]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict:
        S = int(self.starts[idx])
        pred_start = S + self.seq_len
        pred_end = pred_start + self.pred_len

        x = self.all_values[S: S + self.seq_len].copy()
        x_future = self.all_values[pred_start: pred_end].copy()
        window_events = self._get_events_in_window(pred_start, pred_end)
        json_str = build_event_json(
            window_events, pred_start,
            norm_mean=self.mean,
            pred_len=self.pred_len,
        )
        target_json_str = build_event_json(
            window_events, pred_start,
            norm_mean=self.mean,
            pred_len=self.pred_len,
            event_schema=self.event_schema,
        )

        target_ids = self.tokenizer.encode(target_json_str, add_bos=True, add_eos=True)

        return {
            'x': torch.from_numpy(x),
            'x_future': torch.from_numpy(x_future),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'json_str': json_str,
            'pred_start': pred_start,
            'hours': torch.from_numpy(self.hours[S: S + self.seq_len].copy()),
            'weekdays': torch.from_numpy(self.weekdays[S: S + self.seq_len].copy()),
            'hours_future': torch.from_numpy(self.hours[pred_start: pred_end].copy()),
            'weekdays_future': torch.from_numpy(self.weekdays[pred_start: pred_end].copy()),
            **self._get_patch_labels(S),
        }

    def _get_patch_labels(self, window_start: int) -> dict:
        """joint 模式返回 patch 级监督标签。"""
        if self._patch_lbl is None:
            return {}
        has_apex = np.empty(self.n_patches, dtype=np.float32)
        apex_offset = np.empty(self.n_patches, dtype=np.float32)
        min_d = np.empty(self.n_patches, dtype=np.float32)
        phase_dist = np.empty((self.n_patches, 4), dtype=np.float32)
        for k in range(self.n_patches):
            lbl = self._patch_lbl.get(window_start + k * self.patch_stride, _DEFAULT_LBL)
            has_apex[k] = lbl[0]
            apex_offset[k] = lbl[1]
            min_d[k] = lbl[2]
            phase_dist[k] = lbl[3:7]
        return {
            'has_apex': torch.from_numpy(has_apex),
            'apex_offset': torch.from_numpy(apex_offset),
            'min_d': torch.from_numpy(min_d),
            'phase_dist': torch.from_numpy(phase_dist),
        }


def collate_fn(batch: list[dict]) -> dict:
    """变长 target_ids padding，可选 patch labels。"""
    x = torch.stack([b['x'] for b in batch])
    json_strs = [b['json_str'] for b in batch]
    pred_starts = [b['pred_start'] for b in batch]

    target_ids = [b['target_ids'] for b in batch]
    max_len = max(t.size(0) for t in target_ids)
    padded = torch.full((len(batch), max_len), EventTokenizer.PAD_ID, dtype=torch.long)
    for i, t in enumerate(target_ids):
        padded[i, :t.size(0)] = t

    result = {
        'x': x,
        'x_future': torch.stack([b['x_future'] for b in batch]),
        'target_ids': padded,
        'json_str': json_strs,
        'pred_start': pred_starts,
        'hours': torch.stack([b['hours'] for b in batch]),
        'weekdays': torch.stack([b['weekdays'] for b in batch]),
    }
    # patch labels (joint mode)
    if 'has_apex' in batch[0]:
        result['has_apex'] = torch.stack([b['has_apex'] for b in batch])
        result['apex_offset'] = torch.stack([b['apex_offset'] for b in batch])
        result['min_d'] = torch.stack([b['min_d'] for b in batch])
        result['phase_dist'] = torch.stack([b['phase_dist'] for b in batch])
    return result


def build_dataloaders(
    series_path: str,
    events_path: str,
    seq_len: int = 96,
    pred_len: int = 96,
    event_schema: str = 'quad',
    window_stride: int = 4,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    patch_labels_path: str | None = None,
) -> tuple:
    train_ds = SmallDecoderDataset(
        series_path, events_path, split='train',
        seq_len=seq_len, pred_len=pred_len, event_schema=event_schema, window_stride=window_stride,
        patch_labels_path=patch_labels_path,
    )
    val_ds = SmallDecoderDataset(
        series_path, events_path, split='val',
        seq_len=seq_len, pred_len=pred_len, event_schema=event_schema, window_stride=window_stride,
        mean=train_ds.mean, std=train_ds.std,
        patch_labels_path=patch_labels_path,
    )
    test_ds = SmallDecoderDataset(
        series_path, events_path, split='test',
        seq_len=seq_len, pred_len=pred_len, event_schema=event_schema, window_stride=window_stride,
        mean=train_ds.mean, std=train_ds.std,
        patch_labels_path=patch_labels_path,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False,
        drop_last=False, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False,
        drop_last=False, collate_fn=collate_fn,
    )

    meta = {
        'mean': train_ds.mean,
        'std': train_ds.std,
        'train_size': len(train_ds),
        'val_size': len(val_ds),
        'test_size': len(test_ds),
    }
    print(f'[SmallDecoder Dataset] train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)}')
    print(f'[SmallDecoder Dataset] norm — mean={train_ds.mean:.2f} std={train_ds.std:.2f}')

    return train_loader, val_loader, test_loader, meta
