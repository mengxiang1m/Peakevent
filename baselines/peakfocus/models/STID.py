import torch
from torch import nn

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links using Linear layers."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, N, D]

        Returns:
            torch.Tensor: latent repr with shape [B, N, D]
        """
        # input_data: [B, N, D]
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        hidden = hidden + input_data  # residual connection
        return hidden


class Model(nn.Module):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    Venue: CIKM 2022
    Task: Spatial-Temporal Forecasting
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        
        # attributes
        self.task_name = configs.task_name
        self.num_nodes = configs.num_nodes
        self.node_dim = configs.node_dim
        self.input_len = configs.seq_len
        self.input_dim = configs.enc_in
        self.embed_dim = configs.embed_dim
        self.output_len = configs.pred_len
        self.num_layer = configs.num_layer
        self.temp_dim_tid = configs.temp_dim_tid
        self.temp_dim_diw = configs.temp_dim_diw
        self.time_of_day_size = configs.time_of_day_size
        self.day_of_week_size = configs.day_of_week_size

        self.if_time_in_day = configs.if_T_i_D
        self.if_day_in_week = configs.if_D_i_W
        self.if_spatial = configs.if_node

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer - 替换Conv2d为Linear
        # 原来: [B, input_dim * input_len, N, 1] -> [B, embed_dim, N, 1]
        # 现在: [B, N, input_dim * input_len] -> [B, N, embed_dim]
        self.time_series_emb_layer = nn.Linear(
            self.input_dim * self.input_len, self.embed_dim, bias=True)

        # encoding
        self.hidden_dim = self.embed_dim + self.node_dim * \
            int(self.if_spatial) + self.temp_dim_tid * int(self.if_time_in_day) + \
            self.temp_dim_diw * int(self.if_day_in_week)
        
        # 替换Conv2d为Linear的MLP层
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression layer - 替换Conv2d为Linear
        # 原来: [B, hidden_dim, N, 1] -> [B, output_len, N, 1]
        # 现在: [B, N, hidden_dim] -> [B, N, output_len]
        self.regression_layer = nn.Linear(
            self.hidden_dim, self.output_len, bias=True)
        
        # Peak分类头：仅用于peak_detect_ltf任务
        if self.task_name == 'peak_detect_ltf':
            # Peak分类头：输出每个时间步是否为peak (0 or 1)
            self.peak_regression_layer = nn.Linear(
                self.hidden_dim, self.output_len, bias=True)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """Feed forward of STID.

        Args:
            x_enc (torch.Tensor): input data with shape [B, L, C] for univariate or [B, L, N, C] for multivariate

        Returns:
            torch.Tensor: prediction with shape [B, pred_len, C]
            For peak_detect_ltf: returns (prediction, peak_out) tuple
        """
        
        # Normalization from Non-stationary Transformer
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        # Reshape input to match STID's expected format [B, L, N, C]
        if len(x_enc.shape) == 3:  # [B, L, C] -> [B, L, N, C]
            batch_size, seq_len, channels = x_enc.shape
            x_enc = x_enc.unsqueeze(2)  # [B, L, 1, C]
            
        # prepare data
        input_data = x_enc[..., :self.input_dim]  # [B, L, N, input_dim]

        if self.if_time_in_day and x_mark_enc is not None:
            # x_mark_enc contains time features, need to check the format
            if x_mark_enc.shape[-1] >= 1:  # At least one time feature
                # Check if using timeenc=0 (raw values) or timeenc=1 (normalized values)
                time_feature = x_mark_enc[..., 0:1]  # First feature (HourOfDay or hour)
                
                if torch.min(time_feature) >= -0.6 and torch.max(time_feature) <= 0.6:
                    # timeenc=1: HourOfDay normalized to [-0.5, 0.5]
                    hour_data = (time_feature + 0.5) * 23.0
                else:
                    # timeenc=0: raw hour values [0-23] in position 3
                    if x_mark_enc.shape[-1] >= 4:
                        hour_data = x_mark_enc[..., 3:4]  # hour feature
                    else:
                        hour_data = time_feature  # fallback to first feature
                
                # Clamp to valid range and convert to indices
                hour_indices = torch.clamp(hour_data, 0, self.time_of_day_size - 1).long().to(x_enc.device)
                # Get embeddings for the last time step
                time_in_day_emb = self.time_in_day_emb[hour_indices[:, -1, 0]]  # [B, temp_dim_tid]
            else:
                time_in_day_emb = None
        else:
            time_in_day_emb = None
            
        if self.if_day_in_week and x_mark_enc is not None:
            if x_mark_enc.shape[-1] >= 2:  # At least two time features
                time_feature = x_mark_enc[..., 1:2] if x_mark_enc.shape[-1] >= 2 else x_mark_enc[..., 0:1]
                
                if torch.min(time_feature) >= -0.6 and torch.max(time_feature) <= 0.6:
                    # timeenc=1: DayOfWeek normalized to [-0.5, 0.5]
                    weekday_data = (time_feature + 0.5) * 6.0
                else:
                    # timeenc=0: raw weekday values [0-6] in position 2
                    if x_mark_enc.shape[-1] >= 3:
                        weekday_data = x_mark_enc[..., 2:3]  # weekday feature
                    else:
                        weekday_data = time_feature  # fallback
                
                # Clamp to valid range and convert to indices
                weekday_indices = torch.clamp(weekday_data, 0, self.day_of_week_size - 1).long().to(x_enc.device)
                # Get embeddings for the last time step
                day_in_week_emb = self.day_in_week_emb[weekday_indices[:, -1, 0]]  # [B, temp_dim_diw]
            else:
                day_in_week_emb = None
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, seq_len, num_nodes, _ = input_data.shape
        
        # 重新组织数据用于Linear层
        # [B, L, N, input_dim] -> [B, N, L * input_dim]
        input_data = input_data.permute(0, 2, 1, 3).contiguous()  # [B, N, L, input_dim]
        input_data = input_data.view(batch_size, num_nodes, -1)  # [B, N, L * input_dim]
        
        # 应用时间序列嵌入 [B, N, L * input_dim] -> [B, N, embed_dim]
        time_series_emb = self.time_series_emb_layer(input_data)

        # 构建所有嵌入
        embeddings = [time_series_emb]  # [B, N, embed_dim]
        
        # spatial embeddings
        if self.if_spatial:
            # 扩展节点嵌入 [num_nodes, node_dim] -> [B, N, node_dim]
            node_emb_expanded = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1)
            embeddings.append(node_emb_expanded)
        
        # temporal embeddings
        if time_in_day_emb is not None:
            # 扩展时间嵌入 [B, temp_dim_tid] -> [B, N, temp_dim_tid]
            time_in_day_emb_expanded = time_in_day_emb.unsqueeze(1).expand(
                batch_size, num_nodes, self.temp_dim_tid)
            embeddings.append(time_in_day_emb_expanded)
            
        if day_in_week_emb is not None:
            # 扩展星期嵌入 [B, temp_dim_diw] -> [B, N, temp_dim_diw]
            day_in_week_emb_expanded = day_in_week_emb.unsqueeze(1).expand(
                batch_size, num_nodes, self.temp_dim_diw)
            embeddings.append(day_in_week_emb_expanded)

        # 连接所有嵌入 -> [B, N, hidden_dim]
        hidden = torch.cat(embeddings, dim=-1)

        # encoding - 应用MLP层
        hidden = self.encoder(hidden)  # [B, N, hidden_dim]

        # regression - 生成预测
        prediction = self.regression_layer(hidden)  # [B, N, output_len]
        
        # 重新整理输出维度 [B, N, output_len] -> [B, output_len, N]
        prediction = prediction.transpose(1, 2)  # [B, output_len, N]

        # De-Normalization from Non-stationary Transformer
        # prediction[:, -self.output_len:, :] = prediction[:, -self.output_len:, :] * \
        #           (stdev[:, 0, :].unsqueeze(1).repeat(1, self.output_len, 1))
        # prediction[:, -self.output_len:, :] = prediction[:, -self.output_len:, :] + \
        #           (means[:, 0, :].unsqueeze(1).repeat(1, self.output_len, 1))

        # 根据任务类型返回不同的输出
        if self.task_name == 'peak_detect_ltf':
            # Peak检测任务：返回值预测和peak分类
            peak_out = self.peak_regression_layer(hidden)  # [B, N, output_len]
            # 注意: 不要在这里sigmoid! BCEWithLogitsLoss会自动处理
            # 输出原始logits，让损失函数内部进行sigmoid
            peak_out = peak_out.transpose(1, 2)  # [B, output_len, N]
            return prediction, peak_out
        elif self.task_name == 'peak_detect_ltf_basic':
            # Peak检测基础任务：只返回值预测
            return prediction
        else:
            # 其他任务：默认返回值预测
            return prediction