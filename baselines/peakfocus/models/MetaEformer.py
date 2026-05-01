"""
MetaEformer - 整合版本
所有组件已整合到单个文件中
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================================================
# 工具函数 (来自 utils.py)
# ============================================================================

def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2, 1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
    return torch.softmax(m, -1)


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


def attention(Q, K, V):
    # Attention(Q, K, V) = norm(QK)V
    a = a_norm(Q, K)  # (batch_size, dim_attn, seq_length)
    return torch.matmul(a, V)  # (batch_size, seq_length, seq_length)


class AttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)

    def forward(self, x, kv=None):
        if kv is None:
            # Attention with x connected to Q,K and V (For encoder)
            return attention(self.query(x), self.key(x), self.value(x))
        # Attention with x as Q, external vector kv as K an V (For decoder)
        return attention(self.query(x), self.key(kv), self.value(kv))


class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn))
        self.heads = nn.ModuleList(self.heads)
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias=False)

    def forward(self, x, kv=None):
        a = []
        for h in self.heads:
            a.append(h(x, kv=kv))
        a = torch.stack(a, dim=-1)  # combine heads
        a = a.flatten(start_dim=2)  # flatten all head outputs
        x = self.fc(a)
        return x


class Value(torch.nn.Module):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.dim_val = dim_val
        self.fc1 = nn.Linear(dim_input, dim_val, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x


class Key(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn
        self.fc1 = nn.Linear(dim_input, dim_attn, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x


class Query(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn
        self.fc1 = nn.Linear(dim_input, dim_attn, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


# ============================================================================
# Series Decomposition (来自 series_decomp.py)
# ============================================================================

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        if self.kernel_size & 1 == 0:
            front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2)
            end = x[:, -1:].repeat(1, (self.kernel_size) // 2)
        else:
            front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2)
            end = x[:, -1:].repeat(1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=1)
        x = torch.unsqueeze(x, dim=1)
        x = self.avg(x)
        x = torch.squeeze(x, dim=1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ============================================================================
# Embedding Layers (来自 Embed.py)
# ============================================================================

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.d_inp = d_inp
        self.d_model = d_model
        self.embed = nn.Linear(d_inp, d_model, bias=False)
        self.embed_initialized = False

    def forward(self, x):
        # 动态适应输入维度（处理可能添加的 is_peak 列）
        if x.shape[-1] != self.d_inp and not self.embed_initialized:
            # 重新初始化 Linear 层以匹配实际输入维度
            actual_d_inp = x.shape[-1]
            self.embed = nn.Linear(actual_d_inp, self.d_model, bias=False).to(x.device)
            self.embed_initialized = True
            print(f"TimeFeatureEmbedding: Adjusted input dimension from {self.d_inp} to {actual_d_inp}")
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # 加了时间戳的信息
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class StaticContextEmbedding(nn.Module):
    def __init__(self, dim_val, dim_w, dim_static, dropout):
        super().__init__()
        self.dim_val = dim_val
        half = int(dim_val / 2)
        self.fuse_src = nn.Linear(
            in_features=dim_val,
            out_features=dim_w
        )
        self.static_embedding = nn.Linear(dim_static, dim_w)
        self.norm = nn.LayerNorm(dim_val)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, static_context):
        q = self.fuse_src(src)
        v = self.static_embedding(static_context)
        # 计算权重
        key = torch.softmax(q, -1)
        fuse = key * v.unsqueeze(1)
        return self.norm(src + self.dropout(fuse))


# ============================================================================
# Self Attention Family (来自 SelfAttention_Family.py)
# ============================================================================

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


# ============================================================================
# Meta Pattern Pool (来自 MetaPatternPool.py)
# ============================================================================

class MetaPatternPool():
    def __init__(self, mpp_size, mp_len, threshold, device):
        self.seasonal_pool = torch.empty(mpp_size, mp_len, requires_grad=False).to(device)
        self.mp_len = mp_len
        self.threshold = threshold
        self.count = 0
        self.update_count = 0

        self.s_init = False
        self.t_init = False

        self.step = 0
        self.pool_update_step = torch.zeros(mpp_size, dtype=torch.long)

    def build_pool_seasonal(self, series):
        processed_indices = set()

        patch_series = series.reshape(-1, self.mp_len)
        product_matrix = patch_series.unsqueeze(1) * patch_series.unsqueeze(0)
        product_sum = product_matrix.sum(dim=2)
        product_matrix.view(product_matrix.size(0), -1)[:, ::product_matrix.size(-1) + 1] = 0
        new_seasonal_pool = self.seasonal_pool.clone()
        origin_seasonal_pool = self.seasonal_pool.clone()

        for i in range(product_sum.size(0)):
            if i in processed_indices:
                continue
            if self.count >= self.seasonal_pool.size(0):
                break

            max_value, max_index = product_sum[i].max(dim=0)

            # If the maximum value of a certain column is significantly different from
            # other series in the novel, save it as a new pattern
            if max_value.item() <= self.threshold:
                new_seasonal_pool[self.count] = patch_series[i]
                self.count += 1
            # If there is a large value, it indicates that there is a similar pattern,
            # and similar merging should be performed
            else:
                max_sequence_index = max_index.item()
                other_max_indices = (product_sum[i] > self.threshold).nonzero().squeeze(1)

                processed_indices.update(other_max_indices.tolist())

                if torch.all(other_max_indices == max_sequence_index):
                    weights = product_sum[i, other_max_indices] / product_sum[i, other_max_indices].sum()
                    mean_sequence = (patch_series[other_max_indices].T @ weights.unsqueeze(1)).squeeze(1)
                    new_seasonal_pool[self.count] = mean_sequence
                    self.count += 1

        self.seasonal_pool = new_seasonal_pool
        self.s_init = True
        mpp = self.seasonal_pool.clone()

        return mpp

    def update_pool(self, series, alpha=0.1):
        patch_series = series.reshape(-1, self.mp_len)
        product_matrix = patch_series.unsqueeze(1) * self.seasonal_pool.unsqueeze(0)
        product_sum = product_matrix.sum(dim=2)

        new_seasonal_pool = self.seasonal_pool.clone()
        original_seasonal_pool = self.seasonal_pool.clone()

        max_values, max_indices = product_sum.max(dim=1)
        update_mask = max_values > self.threshold

        updated_indices = []

        if update_mask.any():
            updated_indices.extend(max_indices[update_mask].tolist())
            new_seasonal_pool[max_indices[update_mask]] = (
                    alpha * patch_series[update_mask] + (1 - alpha) * new_seasonal_pool[max_indices[update_mask]]
            )

        no_update_indices = torch.where(~update_mask)[0]

        for idx in no_update_indices:
            if (new_seasonal_pool == 0).all(dim=1).any():
                zero_row_index = (new_seasonal_pool == 0).all(dim=1).nonzero(as_tuple=True)[0][0]
                new_seasonal_pool[zero_row_index] = patch_series[idx]
                updated_indices.append(zero_row_index.item())
            else:
                max_sim_index = product_sum[idx].argmax()
                new_seasonal_pool[max_sim_index] = (
                        alpha * patch_series[idx] + (1 - alpha) * new_seasonal_pool[max_sim_index]
                )
                updated_indices.append(max_sim_index.item())

        self.seasonal_pool = new_seasonal_pool

        return new_seasonal_pool.clone()


# ============================================================================
# Encoder & Decoder (来自 MetaEformer_EncDec.py)
# ============================================================================

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        bias = True
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.ff(y)
        y = self.dropout(y)

        return self.norm2(x + y), attn


class EchoLayer(nn.Module):
    def __init__(self, d_model, pattern_num, pattern_len, enc_seq_len, device, low_layer, sim_num):
        super(EchoLayer, self).__init__()
        self.d_model = d_model
        self.half_dim = int(d_model / 2)
        self.patch_seq_len = pattern_len
        self.enc_seq_len = enc_seq_len
        self.device = device
        self.mpp_size = pattern_num
        self.low_layer = low_layer
        self.sim_num = sim_num
        self.update_count = 0

        self.fusion_layer = nn.Linear(
            in_features=self.patch_seq_len * self.half_dim,
            out_features=self.sim_num
        )

        self.recovery_layer = nn.Linear(
            in_features=self.sim_num,
            out_features=self.half_dim
        )

        self.dimension_reduction_layer = nn.Linear(
            in_features=self.half_dim,
            out_features=10
        )

    def forward(self, src, meta_pattern_pool):

        half_dim = int(self.d_model / 2)  # 1/2D
        src_half = src[:, :, half_dim:].flatten(1).clone()  # B * (T*1/2D)
        patches = src_half.reshape(-1, int(self.enc_seq_len / self.patch_seq_len), self.patch_seq_len, half_dim)

        patches_low_dim = self.dimension_reduction_layer(patches)  # [batch, num_patches, pattern_length, 10]
        patches_low_dim_mean = torch.mean(patches_low_dim, dim=-1)  # [batch, num_patches, pattern_length]

        output_features = torch.empty((src.shape[0], 0, half_dim)).to(self.device)
        padding_features = torch.empty((src.shape[0], 0)).to(self.device)

        for patch_idx in range(patches.shape[1]):
            current_patch = patches[:, patch_idx, :, :]

            current_patch_low = patches_low_dim_mean[:, patch_idx, :]

            similarity_matrix = current_patch_low.unsqueeze(1) * meta_pattern_pool.unsqueeze(0)
            similarity_scores = similarity_matrix.sum(dim=2)
            topk_scores, topk_indices = torch.topk(similarity_scores, self.sim_num, dim=1)

            selected_patterns = meta_pattern_pool[topk_indices]

            fusion_input = current_patch.reshape(-1, self.patch_seq_len * half_dim)
            fusion_weights_raw = self.fusion_layer(fusion_input)
            self.update_count += 1
            fusion_weights = torch.softmax(fusion_weights_raw, dim=1)

            patterns_for_fusion = selected_patterns.transpose(-2, -1).to(self.device)

            fused_features = torch.einsum('bk,bpk->bpk', fusion_weights, patterns_for_fusion)

            padding_features_current = torch.einsum('bk,bkp->bp', fusion_weights, selected_patterns)

            padding_features = torch.cat((padding_features, padding_features_current), dim=-1)

            recovered_features = self.recovery_layer(fused_features)
            output_features = torch.cat((output_features, recovered_features), dim=1)

        final_output = torch.cat((src[:, :, :half_dim], output_features), dim=-1)  # [B*T*D]

        return final_output, padding_features


class MPPBuilder(nn.Module):
    def __init__(self, low_layer):
        super(MPPBuilder, self).__init__()
        self.low_layer = low_layer

    def forward(self, enc_out, mpp, decomp):
        low_enc = self.low_layer(enc_out)  # B*T*D  -> B*T*d
        ser_re = torch.mean(low_enc, dim=-1)  # B*T

        season, _ = decomp(ser_re)
        season_no_grad = season.detach()

        if not mpp.s_init:
            mpp = mpp.build_pool_seasonal(season_no_grad)
        else:
            mpp = mpp.update_pool(season_no_grad)

        return mpp


class EchoEncoder(nn.Module):
    def __init__(self, attn_layers, mpp_builders, echo_layers, conv_layers=None, norm_layer=None):
        super(EchoEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.mpp_builders = nn.ModuleList(mpp_builders)
        self.echo_layers = nn.ModuleList(echo_layers)
        self.norm = norm_layer

    def forward(self, x, MPP, stl, MPP_update_flag, attn_mask=None):
        # x [B, L, D]
        mpp = []
        attns = []

        for attn_layer, mpp_builder, echo_layer in zip(self.attn_layers, self.mpp_builders, self.echo_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask)
            if MPP_update_flag:
                mpp = mpp_builder(x, MPP, stl)
            if MPP != None:
                x, padding = echo_layer(x, MPP.seasonal_pool)

            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        if MPP != None:
            return x, padding, attns, mpp
        else:
            return x, None, attns, mpp


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


# ============================================================================
# 主模型 MetaEformer (来自 MetaEformer/MetaEformer.py)
# ============================================================================

class MetaEformer(nn.Module):
    """
    MetaEformer
    """
    def __init__(self, configs):
        super(MetaEformer, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.device = configs.device

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # static
        self.static_layer = StaticContextEmbedding(configs.d_model, configs.d_model, configs.dim_static, configs.dropout)
        self.if_padding = configs.if_padding

        # MetaPatternpool
        self.MPP = MetaPatternPool(configs.mpp_size, configs.mp_len, configs.threshold, self.device)
        self.decomp = series_decomp(configs.kernel_size)
        self.low_layer = nn.Linear(
            in_features=configs.d_model,
            out_features=configs.d_low
        )

        # Encoder - 修正：使用 seq_len 替代 enc_len
        self.echoencoder = EchoEncoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],

            [MPPBuilder(self.low_layer) for l in range(configs.e_layers)],
            [EchoLayer(configs.d_model, configs.mpp_size, configs.mp_len, configs.seq_len, self.device, self.low_layer, configs.sim_num) for l in range(configs.e_layers)],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        
        # Peak detection decoder (for peak_detect_ltf task)
        if self.task_name == 'peak_detect_ltf':
            # 独立的 peak decoder，与 value decoder 结构相同
            self.peak_decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )

        # MPP 更新标志
        self.mpp_update_step = getattr(configs, 'mpp_update', 50)
        self.current_step = 0

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_static=None, MPP_update_flag=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, return_peak_pred=False):
        
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # 如果未指定 MPP_update_flag，根据步数自动判断
        if MPP_update_flag is None:
            MPP_update_flag = (self.current_step % self.mpp_update_step == 0)
            self.current_step += 1

        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        enc_out, Echo_padding, attns, mpp = self.echoencoder(enc_out, self.MPP, self.decomp, MPP_update_flag, attn_mask=enc_self_mask)

        if self.MPP != None:
            if self.if_padding:  # padding
                if Echo_padding.shape[-1] >= self.pred_len:

                    x_dec[:, -self.pred_len:, 0] = Echo_padding[:, -self.pred_len:]
                else:
                    repeat_factor = (self.pred_len // Echo_padding.shape[-1]) + 1
                    extended_global_padding = Echo_padding.repeat(1, repeat_factor)
                    x_dec[:, -self.pred_len:, 0] = extended_global_padding[:, -self.pred_len:]
            else:
                x_dec[:, -self.pred_len:, 0] = 0

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        if x_static != None:
            dec_out = self.static_layer(dec_out, x_static)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        
        # De-Normalization from Non-stationary Transformer
        dec_out[:, -self.pred_len:, :] = dec_out[:, -self.pred_len:, :] * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out[:, -self.pred_len:, :] = dec_out[:, -self.pred_len:, :] + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # Peak分类预测（如果需要）
        if return_peak_pred and hasattr(self, 'peak_decoder'):
            # 使用独立的 peak decoder 进行预测
            peak_dec_input = self.dec_embedding(x_dec, x_mark_dec)
            if x_static != None:
                peak_dec_input = self.static_layer(peak_dec_input, x_static)
            peak_out = self.peak_decoder(peak_dec_input, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            peak_out = peak_out[:, -self.pred_len:, :]  # 只取预测部分
            
            if self.output_attention:
                return dec_out[:, -self.pred_len:, :], peak_out, attns, mpp
            else:
                return dec_out[:, -self.pred_len:, :], peak_out, mpp

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns, mpp
        else:
            return dec_out[:, -self.pred_len:, :], mpp  # [B, L, D]


# ============================================================================
# Model 包装器 (兼容 Time-Series-Library 框架)
# ============================================================================

class Model(nn.Module):
    """
    Model wrapper for MetaEformer to be compatible with Time-Series-Library framework.
    Similar to PatchTST wrapper.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # 初始化 MetaEformer 核心模型
        self.metaeformer = MetaEformer(configs)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        统一的 forward 接口，兼容不同任务
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            result = self.metaeformer(x_enc, x_mark_enc, x_dec, x_mark_dec, return_peak_pred=False)
            if len(result) == 2:  # (dec_out, mpp)
                dec_out, mpp = result
            else:  # (dec_out, attns, mpp) when output_attention=True
                dec_out, attns, mpp = result
            return dec_out  # [B, L, D]
        
        elif self.task_name == 'peak_detect_ltf_basic':
            # 峰值检测基础任务：只返回值预测
            result = self.metaeformer(x_enc, x_mark_enc, x_dec, x_mark_dec, return_peak_pred=False)
            if len(result) == 2:
                dec_out, mpp = result
            else:
                dec_out, attns, mpp = result
            return dec_out  # [B, L, D]
        
        elif self.task_name == 'peak_detect_ltf':
            # 峰值检测任务：返回值预测和peak分类
            result = self.metaeformer(x_enc, x_mark_enc, x_dec, x_mark_dec, return_peak_pred=True)
            if len(result) == 3:  # (dec_out, peak_out, mpp)
                dec_out, peak_out, mpp = result
            else:  # (dec_out, peak_out, attns, mpp) when output_attention=True
                dec_out, peak_out, attns, mpp = result
            # 注意: peak_out 是原始 logits，让损失函数内部进行 sigmoid
            return dec_out, peak_out  # [B, L, D], [B, L, D]
        
        elif self.task_name == 'imputation':
            # 插值任务暂不支持
            raise NotImplementedError("Imputation task is not implemented for MetaEformer")
        
        elif self.task_name == 'anomaly_detection':
            # 异常检测任务暂不支持
            raise NotImplementedError("Anomaly detection task is not implemented for MetaEformer")
        
        elif self.task_name == 'classification':
            # 分类任务暂不支持
            raise NotImplementedError("Classification task is not implemented for MetaEformer")
        
        else:
            raise ValueError(f"Unknown task: {self.task_name}")
        
        return None
