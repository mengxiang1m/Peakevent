"""Shared split helpers for event dataset builders."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import PipelineConfig


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, but got {total:.8f}")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("All split ratios must be positive.")


def build_split_indices(
    n_rows: int,
    config: PipelineConfig,
) -> Tuple[Dict[str, List[int]], np.ndarray]:
    train_end = int(n_rows * config.train_ratio)
    val_end = int(n_rows * (config.train_ratio + config.val_ratio))

    train_ids = list(range(0, train_end))
    val_ids = list(range(train_end, val_end))
    test_ids = list(range(val_end, n_rows))

    split = np.empty(n_rows, dtype=object)
    split[train_ids] = "train"
    split[val_ids] = "val"
    split[test_ids] = "test"

    return {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}, split
