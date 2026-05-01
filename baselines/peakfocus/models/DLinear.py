import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.task_name == 'classification':
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)
        
        # Peak分类头：仅用于peak_detect_ltf任务
        if self.task_name == 'peak_detect_ltf':
            # Peak分类头：输出每个时间步是否为peak (0 or 1)
            if self.individual:
                self.Peak_Linear_Seasonal = nn.ModuleList()
                self.Peak_Linear_Trend = nn.ModuleList()

                for i in range(self.channels):
                    self.Peak_Linear_Seasonal.append(
                        nn.Linear(self.seq_len, self.pred_len))
                    self.Peak_Linear_Trend.append(
                        nn.Linear(self.seq_len, self.pred_len))

                    self.Peak_Linear_Seasonal[i].weight = nn.Parameter(
                        (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                    self.Peak_Linear_Trend[i].weight = nn.Parameter(
                        (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            else:
                self.Peak_Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
                self.Peak_Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

                self.Peak_Linear_Seasonal.weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Peak_Linear_Trend.weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def encoder(self, x, return_peak=False):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        
        # Peak分类预测（如果需要）
        if return_peak and hasattr(self, 'Peak_Linear_Seasonal'):
            if self.individual:
                peak_seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                              dtype=seasonal_init.dtype).to(seasonal_init.device)
                peak_trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                           dtype=trend_init.dtype).to(trend_init.device)
                for i in range(self.channels):
                    peak_seasonal_output[:, i, :] = self.Peak_Linear_Seasonal[i](
                        seasonal_init[:, i, :])
                    peak_trend_output[:, i, :] = self.Peak_Linear_Trend[i](
                        trend_init[:, i, :])
            else:
                peak_seasonal_output = self.Peak_Linear_Seasonal(seasonal_init)
                peak_trend_output = self.Peak_Linear_Trend(trend_init)
            peak_x = peak_seasonal_output + peak_trend_output
            return x.permute(0, 2, 1), peak_x.permute(0, 2, 1)
        
        return x.permute(0, 2, 1)

    def forecast(self, x_enc, return_peak=False):
        # Encoder
        return self.encoder(x_enc, return_peak=return_peak)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, return_peak=False)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'peak_detect_ltf':
            # Peak检测任务：返回值预测和peak分类
            dec_out, peak_out = self.forecast(x_enc, return_peak=True)
            dec_out = dec_out[:, -self.pred_len:, :]  # [B, L, D] 值预测
            peak_out = peak_out[:, -self.pred_len:, :]  # [B, L, D] peak logits
            return dec_out, peak_out
        if self.task_name == 'peak_detect_ltf_basic':
            # Peak检测基础任务：只返回值预测
            dec_out = self.forecast(x_enc, return_peak=False)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
