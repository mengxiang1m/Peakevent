import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.Embed import DataEmbedding, TemporalEmbedding, TimeFeatureEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.covariate_bridge import TimeXerCovariateBridge

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""
    def __init__(self, input_dim, hidden_dim, dropout=0.15):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, input_data):
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        hidden = hidden + input_data
        return hidden

class Simple_Head(nn.Module):
    """Simple Linear Head without decoder embedding"""
    def __init__(self, d_model, c_out):
        super(Simple_Head, self).__init__()
        self.projection = nn.Linear(d_model, c_out)
    
    def forward(self, x, x_mark_dec=None):
        return self.projection(x), None

class GatedWeatherFusion(nn.Module):
    def __init__(self, d_model, dropout):
        super(GatedWeatherFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.candidate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, signal_embed, weather_embed):
        fusion_input = torch.cat([signal_embed, weather_embed], dim=-1)
        gate = self.gate(fusion_input)
        candidate = self.candidate(fusion_input)
        return self.norm(signal_embed + gate * candidate)

class Multi_Scale_Mixing_Peak_Locator(nn.Module):
    """MSM-PL: Multi-Scale Mixing Peak Locator
    
    Args:
        n_scales: Number of pyramid scales for ablation.
            0 = no pyramid (linear projection only)
            1 = single scale (conv on original resolution only)
            2 = dual scale (default, original paper)
            3 = triple scale (adds a third downsampling level)
    """
    def __init__(self, d_model, c_out, dec_te, pred_len, if_te=True, dropout=0.1, n_scales=2):
        super(Multi_Scale_Mixing_Peak_Locator, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout_te = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)
        self.dec_te = dec_te
        self.pred_len = pred_len
        self.if_te = if_te  # if use temporal embedding
        self.n_scales = n_scales
        
        if n_scales == 0:
            # No pyramid: simple linear projection (ablation baseline)
            self.linear_proj = nn.Linear(d_model, d_model)
        else:
            # 特征提取 - scale 0 always exists
            self.process_s0 = nn.Sequential(nn.Conv1d(d_model, d_model, 3, padding=1), nn.GELU())
            
            if n_scales >= 2:
                # 多尺度下采样
                self.pool1 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
                self.process_s1 = nn.Sequential(nn.Conv1d(d_model, d_model, 3, padding=1), nn.GELU())
            
            if n_scales >= 3:
                self.pool2 = nn.AvgPool1d(kernel_size=5, stride=2, padding=2)
                self.process_s2 = nn.Sequential(nn.Conv1d(d_model, d_model, 3, padding=1), nn.GELU())
        
        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x, x_mark_dec):
        # Decoder Embedding - 根据if_te决定是否使用时间嵌入
        if self.if_te and x_mark_dec is not None:
            dec_te = self.dec_te(x_mark_dec)
            dec_te = dec_te[:, -self.pred_len:, :]
            # Feature Fusion
            x = self.dropout_te(x + dec_te)
        
        # x: [B, Seq_Len, D]
        x = self.norm(x)
        
        if self.n_scales == 0:
            # No pyramid: simple linear
            out = self.linear_proj(x)
            return self.projection(self.dropout_out(out)), out
        
        x_in = x.permute(0, 2, 1) # [B, D, L]
        
        if self.n_scales == 1:
            # Single scale: conv only on original resolution
            f0 = self.process_s0(x_in)
            out = f0.permute(0, 2, 1)
            return self.projection(self.dropout_out(out)), out
        
        if self.n_scales == 2:
            # Dual scale (default): original paper behavior
            s0 = x_in
            s1 = self.pool1(s0)
            f1 = self.process_s1(s1)
            f0 = self.process_s0(s0)
            f1_up = F.interpolate(f1, size=f0.shape[2], mode='linear', align_corners=True)
            f0_fused = f0 + f1_up
            out = f0_fused.permute(0, 2, 1)
            return self.projection(self.dropout_out(out)), out
        
        if self.n_scales >= 3:
            # Triple scale
            s0 = x_in
            s1 = self.pool1(s0)
            s2 = self.pool2(s1)
            f2 = self.process_s2(s2)
            f1 = self.process_s1(s1)
            f0 = self.process_s0(s0)
            # Cascade Injection
            f2_up = F.interpolate(f2, size=f1.shape[2], mode='linear', align_corners=True)
            f1_fused = f1 + f2_up
            f1_up = F.interpolate(f1_fused, size=f0.shape[2], mode='linear', align_corners=True)
            f0_fused = f0 + f1_up
            out = f0_fused.permute(0, 2, 1)
            return self.projection(self.dropout_out(out)), out

class Location_Aware_Decoder_Layer(nn.Module):
    """
    Location-Aware Decoder Layer
    """
    def __init__(self, attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", use_peak_guidance=True):
        super(Location_Aware_Decoder_Layer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention  # 只需要一个 attention
        self.use_peak_guidance = use_peak_guidance  # 控制使用哪个输入
        
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, peak, x_mask=None, cross_mask=None, peak_mask=None):
        """
        Args:
            x: decoder input
            cross: encoder output
            peak: peak prediction features (from peak_head)
        """
        # 二选一：根据 use_peak_guidance 决定使用哪个作为 attention 的 key 和 value
        if self.use_peak_guidance: # self.use_peak_guidance
            # Peak-Guided (proposed): 使用 peak features
            attn_input = torch.tanh(peak) * torch.sigmoid(cross)
            # attn_input = torch.tanh(cross) * torch.sigmoid(peak)
            attn_mask = peak_mask
        else:
            # Basic (baseline): 使用 encoder output
            attn_input = cross
            attn_mask = cross_mask
        
        # Cross-Attention
        x = x + self.dropout(self.attention(x, attn_input, attn_input, attn_mask=attn_mask)[0])
        x = self.norm1(x)
        
        # Feedforward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = self.norm2(x + y)
        
        return x


class Location_Aware_Decoder(nn.Module):
    """
    LAD: Location-Aware Decoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Location_Aware_Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, peak, x_mask=None, cross_mask=None, peak_mask=None):
        """
        Args:
            x: decoder input [B, L, D]
            cross: encoder output [B, L, D]
            peak: peak features [B, L, D]
        """
        for layer in self.layers:
            x = layer(x, cross, peak, x_mask=x_mask, cross_mask=cross_mask, peak_mask=peak_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        self.c_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.mlp_layers = configs.mlp_layers
        self.use_external_features = bool(getattr(configs, 'use_external_features', 0))
        self.external_feature_mode = getattr(configs, 'external_feature_mode', 'concat')
        self.endogenous_in = getattr(configs, 'endogenous_in', self.c_out)
        self.num_external_features = getattr(configs, 'num_external_features', max(self.c_in - self.endogenous_in, 0))
        self.use_fusion_external = self.use_external_features and self.external_feature_mode == 'fusion'

        # 1. ============================Encoder============================
        # 1.1 Encoder Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.signal_enc_embedding = None
        self.weather_enc_embedding = None
        self.encoder_weather_fusion = None
        self.endogenous_enc_embedding = None
        self.covariate_bridge = None
        self.covariate_fusion = None
        if self.use_fusion_external:
            self.signal_enc_embedding = DataEmbedding(self.endogenous_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.weather_enc_embedding = DataEmbedding(self.num_external_features, configs.d_model, configs.embed, configs.freq,
                                                       configs.dropout)
            self.encoder_weather_fusion = GatedWeatherFusion(configs.d_model, configs.dropout)
        if self.use_external_features and self.external_feature_mode == 'timexer':
            self.endogenous_enc_embedding = DataEmbedding(self.endogenous_in, configs.d_model, configs.embed, configs.freq,
                                                          configs.dropout)
            self.covariate_bridge = TimeXerCovariateBridge(
                endogenous_channels=self.endogenous_in,
                exogenous_channels=self.num_external_features,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                patch_len=getattr(configs, 'patch_len', 16),
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                d_ff=configs.d_ff,
                dropout=configs.dropout,
                factor=configs.factor,
                activation=configs.activation,
                e_layers=configs.e_layers,
                embed=configs.embed,
                freq=configs.freq,
            )
            self.covariate_fusion = nn.Sequential(
                nn.Linear(configs.d_model * 2, configs.d_model),
                nn.GELU(),
                nn.Dropout(configs.dropout),
            )
        
        # 1.2 Encoder Backbone (Res-MLP Stack)
        self.encoder_backbone = nn.Sequential(
            *[MultiLayerPerceptron(self.d_model, self.d_model, dropout=self.dropout) 
              for _ in range(self.mlp_layers)]
        )

        # 1.2 Encoder Backbone (Transformer Encoder)
        # self.encoder_backbone = Encoder(
        #     [
        #         EncoderLayer(
        #             AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=False), configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation
        #         ) for l in range(configs.e_layers)
        #     ],
        #     norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        # )

        # 1.3 Temporal Projection
        # 将过去的时间长度映射到未来的时间长度
        self.temporal_proj = nn.Linear(self.seq_len, self.pred_len)

        # 2. ============================Dual Heads Decoder============================
        # 2.1 Decoder Embedding (Future Knowledge Injection)
        self.dec_te = TemporalEmbedding(d_model=configs.d_model, embed_type=configs.embed,
                                               freq=configs.freq) if configs.embed != 'timeF' else TimeFeatureEmbedding(
            d_model=configs.d_model, embed_type=configs.embed, freq=configs.freq)

        # 2.2 Peak Head
        self.if_msm_pl = configs.if_msm_pl # if use HPR-Head for peak detection
        if self.if_msm_pl:
            self.if_lad = configs.if_lad # 控制 PeakGuidedDecoder 是否使用 peak guidance
            n_scales = getattr(configs, 'n_scales', 2)
            self.peak_head = Multi_Scale_Mixing_Peak_Locator(self.d_model, self.c_out, self.dec_te, self.pred_len, dropout=self.dropout, n_scales=n_scales)
        else:
            self.if_lad = 0 # 无法使用 peak guidance
            self.peak_head = Simple_Head(self.d_model, self.c_out)

        # 2.3 Value Head
        self.signal_dec_embedding = None
        self.weather_dec_embedding = None
        self.decoder_weather_fusion = None
        if self.use_fusion_external:
            self.signal_dec_embedding = DataEmbedding(self.endogenous_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.weather_dec_embedding = DataEmbedding(self.num_external_features, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.decoder_weather_fusion = GatedWeatherFusion(configs.d_model, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.value_head = Location_Aware_Decoder(
            [
                Location_Aware_Decoder_Layer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    use_peak_guidance=self.if_lad  # 控制是否使用 use_lad
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=None, # torch.nn.LayerNorm(configs.d_model)
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
    
    def _split_covariate_inputs(self, x_enc):
        if not (self.use_external_features and self.external_feature_mode in ['fusion', 'timexer']):
            return x_enc, None
        if x_enc.shape[-1] <= self.endogenous_in:
            raise ValueError(f'{self.external_feature_mode} mode requires inputs to contain both endogenous and exogenous channels')
        return x_enc[:, :, :self.endogenous_in], x_enc[:, :, self.endogenous_in:]

    def _prepare_encoder_embedding(self, x_enc, x_mark_enc):
        endogenous_x, exogenous_x = self._split_covariate_inputs(x_enc)
        if self.use_fusion_external:
            signal_embed = self.signal_enc_embedding(endogenous_x, x_mark_enc)
            weather_embed = self.weather_enc_embedding(exogenous_x, x_mark_enc)
            return self.encoder_weather_fusion(signal_embed, weather_embed), endogenous_x, exogenous_x
        if self.use_external_features and self.external_feature_mode == 'timexer':
            return self.endogenous_enc_embedding(endogenous_x, x_mark_enc), endogenous_x, exogenous_x
        return self.enc_embedding(x_enc, x_mark_enc), x_enc, None

    def _prepare_decoder_embedding(self, x_dec, x_mark_dec):
        if self.use_fusion_external:
            endogenous_dec, exogenous_dec = self._split_covariate_inputs(x_dec)
            signal_embed = self.signal_dec_embedding(endogenous_dec, x_mark_dec)
            weather_embed = self.weather_dec_embedding(exogenous_dec, x_mark_dec)
            return self.decoder_weather_fusion(signal_embed, weather_embed)
        return self.dec_embedding(x_dec, x_mark_dec)
    
    def forward_backbone(self, x_enc, x_mark_enc):
        """
        核心逻辑：MLP 提取特征 -> 时间维投影
        """
        enc_embed, endogenous_x, exogenous_x = self._prepare_encoder_embedding(x_enc, x_mark_enc)
        # --- A. Encoder Processing ---
        # 1. Embedding: [B, L, C] -> [B, L, D] (包含历史的时间信息)
        # 注意:这里 x_enc 已经是归一化后的数据
        enc_out = enc_embed
        
        # 2. Res-MLP: [B, L, D] -> [B, L, D]
        enc_out = self.encoder_backbone(enc_out)
        if len(enc_out) == 2:
            enc_out = enc_out[0]  # Transformer Encoder 返回 tuple (output, attns)，取 output 部分 
        
        # 3. Temporal Projection:    [B, L, D] -> [B, P, D]
        enc_out = enc_out.permute(0, 2, 1)      # [B, D, L]
        enc_out = self.temporal_proj(enc_out)   # [B, D, P]
        enc_out = enc_out.permute(0, 2, 1)      # [B, P, D] (这是基于历史推演出的未来特征)
        if self.use_external_features and self.external_feature_mode == 'timexer' and exogenous_x is not None:
            covariate_context = self.covariate_bridge(endogenous_x, exogenous_x, x_mark_enc)
            enc_out = self.covariate_fusion(torch.cat([enc_out, covariate_context], dim=-1))
        
        return enc_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # ============================Step 1: RevIn Normalization============================
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # ============================Step 2: Encoder Forward============================
        enc_out = self.forward_backbone(x_enc, x_mark_enc)
        
        # ============================Step 3: Peak Head Decoder============================
        dec_input = self._prepare_decoder_embedding(x_dec, x_mark_dec)
        dec_out_val = self.value_head(dec_input, enc_out, enc_out, x_mask=None, cross_mask=None, peak_mask=None)
        
        # ============================Step 4: De-Normalization============================
        dec_out_val = dec_out_val.clone()
        dec_out_val[:, -self.pred_len:, :] = dec_out_val[:, -self.pred_len:, :] * \
                  (stdev[:, 0, :self.c_out].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out_val[:, -self.pred_len:, :] = dec_out_val[:, -self.pred_len:, :] + \
                  (means[:, 0, :self.c_out].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out_val

    def peak_detect_ltf(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # ============================Step 1: RevIn Normalization============================
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # ============================Step 2: Encoder============================
        enc_out = self.forward_backbone(x_enc, x_mark_enc)
        
        # ============================Step 3: Dual Heads Decoder============================
        # ----------------------------3.1 Peak Head----------------------------
        peak_out, peak_features = self.peak_head(enc_out, x_mark_dec)
        # ----------------------------3.2 Value Head----------------------------
        dec_input = self._prepare_decoder_embedding(x_dec, None)
        dec_out_val = self.value_head(dec_input, enc_out, peak_features, x_mask=None, cross_mask=None, peak_mask=None)

        # ============================Step 4: De-Normalization for Value Head============================
        dec_out_val = dec_out_val.clone()
        dec_out_val[:, -self.pred_len:, :] = dec_out_val[:, -self.pred_len:, :] * \
                  (stdev[:, 0, :self.c_out].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out_val[:, -self.pred_len:, :] = dec_out_val[:, -self.pred_len:, :] + \
                  (means[:, 0, :self.c_out].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out_val, peak_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'peak_detect_ltf_basic':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
            
        if self.task_name == 'peak_detect_ltf':
            dec_out, peak_out = self.peak_detect_ltf(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], peak_out[:, -self.pred_len:, :]
            
        return None