"""Learning rate schedulers for training."""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR

def build_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Warmup + Cosine Annealing学习率调度器。"""
    warmup_steps = max(int(warmup_steps), 0)
    total_steps = max(int(total_steps), 1)

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


__all__ = ["build_warmup_cosine_scheduler"]
