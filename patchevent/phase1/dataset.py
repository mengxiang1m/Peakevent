"""
WlelPatchDataset
================
Phase-1 Patch Encoder 预训练数据集。

每个样本（sample）是一个长度 seq_len 的连续时序窗口，
配套返回其中 N = (seq_len - patch_len) // stride + 1 个 patch 的监督标签。

数据来源
--------
- wlel_event_series_v1.csv  ：点级时序（value, split, …）
- wlel_patch_labels_v1.csv  ：patch 级软标签（has_apex, apex_offset, …）

Usage
-----
    from patchevent.phase1.dataset import build_dataloaders

    train_loader, val_loader, test_loader, mean, std = build_dataloaders(
        series_path='dataset/wlel/event_v1/data/wlel_event_series_v1.csv',
        patch_labels_path='dataset/wlel/event_v1/data/wlel_patch_labels_v1.csv',
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Tuple, Optional

# patch-label vector column order (must match _LBL_COLS in __init__ below)
_LBL_COLS = [
    'has_apex',          # [0] 0/1 float
    'apex_offset',       # [1] 0-7 or -1
    'min_d_to_apex',     # [2] ≥ 0
    'phase_dist_bg',     # [3]
    'phase_dist_rising', # [4]
    'phase_dist_apex',   # [5]
    'phase_dist_falling',# [6]
    'apex_intensity',    # [7] -1 if no apex
]

# default label for missing patches (background, no apex)
_DEFAULT_LBL = np.array([0., -1., 0., 1., 0., 0., 0., -1.], dtype=np.float32)


class WlelPatchDataset(Dataset):
    """
    Phase-1 pretraining dataset.

    Parameters
    ----------
    series_path       : path to wlel_event_series_v1.csv
    patch_labels_path : path to wlel_patch_labels_v1.csv
    split             : 'train' | 'val' | 'test'
    seq_len           : sliding-window length in hours (default 96)
    patch_len         : patch length in hours         (default 8)
    stride            : patch stride in hours         (default 4)
    mean / std        : normalisation constants; if None, computed from train split

    Returns (per sample)
    --------------------
    x           : (seq_len,)   float32 — normalised load values
    has_apex    : (N,)         float32 — 0/1 binary
    apex_offset : (N,)         float32 — apex position in patch [0,7], or -1
    min_d       : (N,)         float32 — min distance to nearest apex
    phase_dist  : (N, 4)       float32 — [bg, rising, apex, falling] soft distribution
    apex_int    : (N,)         float32 — apex intensity, -1 if no apex
    """

    def __init__(
        self,
        series_path: str,
        patch_labels_path: str,
        split: str = 'train',
        seq_len: int = 96,
        patch_len: int = 8,
        stride: int = 4,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        log1p_d: bool = False,
    ):
        self.seq_len   = seq_len
        self.patch_len = patch_len
        self.stride    = stride
        self.n_patches = (seq_len - patch_len) // stride + 1  # e.g. 23
        self.log1p_d   = log1p_d

        # ── 1. Load full point-level series ──────────────────────────────────
        series = pd.read_csv(series_path, parse_dates=['timestamp'])
        self.all_values: np.ndarray = series['value'].values.astype(np.float32)
        all_splits = series['split'].values

        # ── 2. Normalisation (always computed from training data) ─────────────
        if mean is None or std is None:
            train_vals = self.all_values[all_splits == 'train']
            self.mean = float(train_vals.mean())
            self.std  = float(train_vals.std()) + 1e-8
        else:
            self.mean = float(mean)
            self.std  = float(std)

        # ── 3. Identify global start/end for requested split ──────────────────
        split_mask = (all_splits == split)
        split_idx  = np.where(split_mask)[0]
        if len(split_idx) == 0:
            raise ValueError(f"No rows found for split='{split}' in {series_path}")
        g_start = int(split_idx[0])
        g_end   = int(split_idx[-1])

        # ── 4. Build valid window start positions ─────────────────────────────
        # Align to global patch grid (multiples of stride) so every patch in a
        # window has a pre-computed label in the patch-label table.
        offset        = (stride - g_start % stride) % stride
        first_aligned = g_start + offset
        last_start    = g_end - seq_len + 1
        if last_start < first_aligned:
            self.starts = np.array([], dtype=np.int64)
        else:
            self.starts = np.arange(first_aligned, last_start + 1, stride, dtype=np.int64)

        # ── 5. Load patch-label lookup dict {patch_start_idx → np.array(8)} ──
        pl = pd.read_csv(patch_labels_path)
        lbl_arr = pl[_LBL_COLS].values.astype(np.float32)
        psi_arr = pl['patch_start_idx'].values.astype(np.int64)
        self._lbl: dict[int, np.ndarray] = {
            int(psi_arr[i]): lbl_arr[i] for i in range(len(psi_arr))
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_lbl(self, psi: int) -> np.ndarray:
        return self._lbl.get(psi, _DEFAULT_LBL)

    # ── Dataset API ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict:
        S = int(self.starts[idx])

        # normalised value sequence
        x = (self.all_values[S: S + self.seq_len] - self.mean) / self.std

        # patch labels
        has_apex    = np.empty(self.n_patches, dtype=np.float32)
        apex_offset = np.empty(self.n_patches, dtype=np.float32)
        min_d       = np.empty(self.n_patches, dtype=np.float32)
        phase_dist  = np.empty((self.n_patches, 4), dtype=np.float32)
        apex_int    = np.empty(self.n_patches, dtype=np.float32)

        for k in range(self.n_patches):
            lbl = self._get_lbl(S + k * self.stride)
            has_apex[k]    = lbl[0]
            apex_offset[k] = lbl[1]
            min_d[k]       = np.log1p(lbl[2]) if self.log1p_d else lbl[2]
            phase_dist[k]  = lbl[3:7]
            apex_int[k]    = lbl[7]

        return {
            'x':           torch.from_numpy(x.copy()),
            'has_apex':    torch.from_numpy(has_apex),
            'apex_offset': torch.from_numpy(apex_offset),
            'min_d':       torch.from_numpy(min_d),
            'phase_dist':  torch.from_numpy(phase_dist),
            'apex_int':    torch.from_numpy(apex_int),
        }


# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    series_path: str,
    patch_labels_path: str,
    seq_len: int = 96,
    patch_len: int = 8,
    stride: int = 4,
    batch_size: int = 64,
    num_workers: int = 0,
    log1p_d: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, float, float]:
    """
    Build train / val / test DataLoaders with shared z-score normalisation.

    Returns
    -------
    train_loader, val_loader, test_loader, mean, std
    """
    train_ds = WlelPatchDataset(series_path, patch_labels_path, 'train',
                                seq_len, patch_len, stride, log1p_d=log1p_d)
    val_ds   = WlelPatchDataset(series_path, patch_labels_path, 'val',
                                seq_len, patch_len, stride,
                                mean=train_ds.mean, std=train_ds.std, log1p_d=log1p_d)
    test_ds  = WlelPatchDataset(series_path, patch_labels_path, 'test',
                                seq_len, patch_len, stride,
                                mean=train_ds.mean, std=train_ds.std, log1p_d=log1p_d)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True,
                              pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False,
                              pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False,
                              pin_memory=False)

    print(f'[Dataset] split sizes — train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)}')
    print(f'[Dataset] norm  — mean={train_ds.mean:.2f} std={train_ds.std:.2f}')

    return train_loader, val_loader, test_loader, train_ds.mean, train_ds.std


def build_multi_domain_dataloaders(
    domain_configs: list[dict],
    seq_len: int = 96,
    patch_len: int = 8,
    stride: int = 4,
    batch_size: int = 64,
    num_workers: int = 0,
    log1p_d: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, list[Tuple[float, float]]]:
    """
    Build joint dataloaders from multiple domains.

    Parameters
    ----------
    domain_configs : list of dicts, each with keys 'series_path' and 'labels_path'
    
    Returns
    -------
    train_loader, val_loader, test_loader, domain_norms (list of (mean, std) per domain)
    """
    train_datasets, val_datasets, test_datasets = [], [], []
    domain_norms = []

    for i, cfg in enumerate(domain_configs):
        sp = cfg['series_path']
        lp = cfg['labels_path']
        name = cfg.get('name', f'domain_{i}')

        train_ds = WlelPatchDataset(sp, lp, 'train', seq_len, patch_len, stride, log1p_d=log1p_d)
        val_ds = WlelPatchDataset(sp, lp, 'val', seq_len, patch_len, stride,
                                   mean=train_ds.mean, std=train_ds.std, log1p_d=log1p_d)
        test_ds = WlelPatchDataset(sp, lp, 'test', seq_len, patch_len, stride,
                                    mean=train_ds.mean, std=train_ds.std, log1p_d=log1p_d)

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        test_datasets.append(test_ds)
        domain_norms.append((train_ds.mean, train_ds.std))

        print(f'[MultiDomain] {name}: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} '
              f'mean={train_ds.mean:.2f} std={train_ds.std:.2f}')

    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)
    combined_test = ConcatDataset(test_datasets)

    print(f'[MultiDomain] Combined: train={len(combined_train)} val={len(combined_val)} test={len(combined_test)}')

    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True, pin_memory=False)
    val_loader = DataLoader(combined_val, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False, pin_memory=False)
    test_loader = DataLoader(combined_test, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False, pin_memory=False)

    return train_loader, val_loader, test_loader, domain_norms
