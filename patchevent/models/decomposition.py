"""Input series decomposition modules."""

from __future__ import annotations

import torch
import torch.nn as nn

class InputSeriesDecomposition(nn.Module):
    """
    输入级时序分解：直接在原始1维时间序列上做指数平滑趋势分解，
    将分解后的趋势和残差分别切patch并投影为d_model维memory tokens，
    通过门控融合与encoder路径的detail memory合并。

    数据流:
      x_norm (B, L=96)
        ├── trend = exp_smooth(x_norm)           (B, L)
        ├── residual = x_norm - trend            (B, L)
        ├── trend → unfold → trend_proj          (B, N, d)
        └── resid → unfold → resid_proj          (B, N, d)
      内部gate: structure = g₁⊙trend + (1-g₁)⊙resid
      外部gate: output = g₂⊙detail + (1-g₂)⊙structure

    支持mode: 'full' | 'trend_only' | 'resid_only' | 'no_inner_gate' | 'no_fusion_gate'
    """

    def __init__(self, seq_len: int = 96, patch_len: int = 8, stride: int = 4,
                 d_model: int = 128, alpha: float = 0.3, mode: str = 'full'):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.alpha = alpha
        self.mode = mode
        self.n_patches = (seq_len - patch_len) // stride + 1

        # 趋势分支投影: patch_len → d_model
        self.trend_proj = nn.Sequential(
            nn.Linear(patch_len, d_model),
            nn.LayerNorm(d_model),
        )
        # 残差分支投影: patch_len → d_model
        self.resid_proj = nn.Sequential(
            nn.Linear(patch_len, d_model),
            nn.LayerNorm(d_model),
        )
        # 内部门控: 自适应合并趋势和残差
        self.inner_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        # 外部门控: 自适应合并detail_memory和structure_memory
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def _exp_smooth_1d(self, x: torch.Tensor) -> torch.Tensor:
        """指数平滑: x (B, L) → smoothed (B, L)"""
        B, L = x.shape
        smoothed = torch.zeros_like(x)
        smoothed[:, 0] = x[:, 0]
        for t in range(1, L):
            smoothed[:, t] = self.alpha * x[:, t] + (1 - self.alpha) * smoothed[:, t - 1]
        return smoothed

    def forward(self, x_norm: torch.Tensor, detail_memory: torch.Tensor) -> torch.Tensor:
        """
        x_norm: (B, L) 归一化后的原始时间序列
        detail_memory: (B, N, d) encoder路径产生的memory
        returns: (B, N, d) 融合后的memory
        """
        # 1. 指数平滑趋势分解
        trend = self._exp_smooth_1d(x_norm)
        residual = x_norm - trend

        # 2. 切patch并投影
        trend_patches = trend.unfold(1, self.patch_len, self.stride)
        resid_patches = residual.unfold(1, self.patch_len, self.stride)
        trend_mem = self.trend_proj(trend_patches)
        resid_mem = self.resid_proj(resid_patches)

        # 3. 内部门控: 合并趋势和残差
        if self.mode == 'trend_only':
            structure_memory = trend_mem
        elif self.mode == 'resid_only':
            structure_memory = resid_mem
        elif self.mode == 'no_inner_gate':
            structure_memory = trend_mem + resid_mem
        else:  # 'full'
            inner_g = self.inner_gate(torch.cat([trend_mem, resid_mem], dim=-1))
            structure_memory = inner_g * trend_mem + (1 - inner_g) * resid_mem

        # 4. 外部门控: 合并detail_memory和structure_memory
        if self.mode == 'no_fusion_gate':
            return detail_memory + structure_memory
        else:
            fusion_g = self.fusion_gate(torch.cat([detail_memory, structure_memory], dim=-1))
            return fusion_g * detail_memory + (1 - fusion_g) * structure_memory


__all__ = ["InputSeriesDecomposition"]
