"""
Naive Baselines for Electricity Load Peak Forecasting (ELPF).

Three simple baselines that require no training:
1. Persistence (Last-Value): repeats the last known value H times
2. Seasonal Naive: uses values from the same time period one week (168h) ago
3. Moving Average: repeats the mean of the last k steps H times

These baselines are used to demonstrate the non-triviality of the ELPF task
and the necessity of learned models.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Naive Baseline wrapper that follows the same interface as other models.
    Controlled by configs.naive_type: 'persistence', 'seasonal', 'movavg'
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.naive_type = getattr(configs, 'naive_type', 'persistence')
        self.movavg_window = getattr(configs, 'movavg_window', 24)
        self.seasonal_period = getattr(configs, 'seasonal_period', 168)  # 7 days * 24 hours
        self.c_out = configs.c_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Args:
            x_enc: [B, seq_len, C] - encoder input (historical series)
            x_mark_enc: [B, seq_len, D] - encoder time features (unused)
            x_dec: [B, label_len + pred_len, C] - decoder input (unused)
            x_mark_dec: [B, label_len + pred_len, D] - decoder time features (unused)
        Returns:
            outputs: [B, pred_len, C] - forecasted values
            For peak_detect_ltf task: (outputs, peak_outputs) where peak_outputs is None
        """
        B, L, C = x_enc.shape

        if self.naive_type == 'persistence':
            # Repeat the last value H times
            last_val = x_enc[:, -1:, :]  # [B, 1, C]
            outputs = last_val.repeat(1, self.pred_len, 1)  # [B, H, C]

        elif self.naive_type == 'seasonal':
            # Use values from one seasonal_period ago
            period = self.seasonal_period
            if L >= period:
                # Take the last `pred_len` values starting from `period` steps back
                start = L - period
                end = start + self.pred_len
                if end <= L:
                    outputs = x_enc[:, start:end, :]
                else:
                    # If pred_len > period, tile and truncate
                    segment = x_enc[:, start:, :]  # [B, period, C]
                    n_repeats = (self.pred_len // segment.shape[1]) + 1
                    outputs = segment.repeat(1, n_repeats, 1)[:, :self.pred_len, :]
            else:
                # Fallback to persistence if seq_len < period
                last_val = x_enc[:, -1:, :]
                outputs = last_val.repeat(1, self.pred_len, 1)

        elif self.naive_type == 'movavg':
            # Moving average of the last k steps, repeated H times
            k = min(self.movavg_window, L)
            avg_val = x_enc[:, -k:, :].mean(dim=1, keepdim=True)  # [B, 1, C]
            outputs = avg_val.repeat(1, self.pred_len, 1)  # [B, H, C]

        else:
            raise ValueError(f"Unknown naive_type: {self.naive_type}")

        if self.task_name == 'peak_detect_ltf':
            # Return (intensity_pred, peak_pred) tuple; peak_pred is None for naive
            return outputs, None
        
        return outputs
