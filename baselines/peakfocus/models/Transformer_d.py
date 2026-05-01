import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, PositionalEmbedding
import numpy as np

# -------------------------------------------------------
# 新增模块 1: Patch Embedding (借鉴 PatchTST/iTransformer)
# -------------------------------------------------------
class PatchEmbedding(nn.Module):
    """
    将连续的时间步切分为 Patch，并增加位置编码
    Input: [Batch, Seq_Len, Channels]
    Output: [Batch, Num_Patches, D_Model]
    """
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        # 投影层：将 (patch_len * channels) 投影到 d_model
        # 注意：这里我们将在 forward 中动态计算输入维度，因为可能会拼接导数特征
        self.projection = None 
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
    
    def lazy_init_projection(self, input_dim, device):
        # 延迟初始化，因为输入的 channel 可能会因为导数特征的加入而变化
        if self.projection is None:
            self.projection = nn.Linear(input_dim * self.patch_len, self.d_model).to(device)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        B, L, C = x.shape
        
        # 1. Padding to allow precise patching
        res = (L - self.patch_len) % self.stride
        if res != 0:
            pad_len = self.stride - res
            x = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
        
        # 2. Unfold (Patching)
        # Output: [Batch, Num_Patches, Channels, Patch_Len]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        
        # 3. Reshape for projection
        # [Batch, Num_Patches, Channels * Patch_Len]
        x = x.reshape(B, -1, C * self.patch_len)
        
        # Initialize projection layer if first run
        self.lazy_init_projection(C, x.device)
        
        # 4. Project + Positional Embed (Learnable or Sinusoidal is handled in layers usually, 
        # here we add a simple learnable position embedding logic or rely on the Encoder's position awareness)
        # For simplicity in this structure, we project to d_model. 
        # Ideally, you should add PositionalEmbedding(d_model) here.
        x = self.projection(x)
        
        return self.dropout(x)

# -------------------------------------------------------
# 新增模块 2: 物理特征提取 (导数)
# -------------------------------------------------------
def compute_derivatives(x):
    """
    计算一阶导数(速度)和二阶导数(加速度)
    x: [Batch, Len, Channel]
    """
    # 1. Velocity (First difference): x(t) - x(t-1)
    # Use padding to keep length consistent
    v = torch.zeros_like(x)
    v[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
    v[:, 0, :] = v[:, 1, :] # Fill first element
    
    # 2. Acceleration (Second difference): v(t) - v(t-1)
    a = torch.zeros_like(x)
    a[:, 1:, :] = v[:, 1:, :] - v[:, :-1, :]
    a[:, 0, :] = a[:, 1, :] # Fill first element
    
    return v, a

# -------------------------------------------------------
# 新增模块 3: 改进的卷积预测头
# -------------------------------------------------------
class PeakPredictionHead(nn.Module):
    """
    Hierarchical Pyramid Refinement Head (HPR-Head)
    Inspired by MDM: Utilizes a coarse-to-fine cascade to inject multi-scale 
    contextual information into the final peak prediction.
    """
    def __init__(self, d_model, c_out, dropout=0.1):
        super(PeakPredictionHead, self).__init__()
        
        # 1. 基础特征变换
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 2. 多尺度处理模块 (借鉴 MDM 的结构)
        # Scale 0: Original Resolution (No pooling)
        # Scale 1: Coarse Resolution (Pool size 3)
        # Scale 2: Coarser Resolution (Pool size 5)
        self.pool1 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.AvgPool1d(kernel_size=5, stride=2, padding=2)

        # 处理每一层的轻量化卷积 (替代 MDM 中的 Linear，更适合捕捉局部 Peak)
        self.process_s0 = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1), 
            nn.GELU()
        )
        self.process_s1 = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1), 
            nn.GELU()
        )
        self.process_s2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1), 
            nn.GELU()
        )

        # 3. 最终投影
        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x):
        # x: [Batch, Seq_Len, D_Model]
        B, L, D = x.shape
        x = self.norm(x)
        x_in = x.permute(0, 2, 1) # [B, D, L]

        # --- Step 1: Downsampling (构建金字塔) ---
        s0 = x_in               # Original size: L
        s1 = self.pool1(s0)     # Size: L/2
        s2 = self.pool2(s1)     # Size: L/4

        # --- Step 2: Processing (特征提取) ---
        # 在不同尺度下提取特征，Scale 2 看得更广，Scale 0 看得更细
        f2 = self.process_s2(s2)
        f1 = self.process_s1(s1)
        f0 = self.process_s0(s0)

        # --- Step 3: Cascade Injection (级联注入 - MDM 的核心逻辑) ---
        # 将最粗层的特征上采样，加到中间层
        f2_up = F.interpolate(f2, size=f1.shape[2], mode='linear', align_corners=True)
        f1_fused = f1 + f2_up # Add (Residual injection)

        # 将中间层的特征上采样，加到最细层
        f1_up = F.interpolate(f1_fused, size=f0.shape[2], mode='linear', align_corners=True)
        f0_fused = f0 + f1_up # Add

        # --- Step 4: Final Projection ---
        out = f0_fused.permute(0, 2, 1) # [B, L, D]
        return self.projection(self.dropout(out))

class Model(nn.Module):
    """
    Transformer with Physics-Aware Inputs and Patching
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # --- Configs for Patching ---
        # 默认 Patch 长度 16 (约半天), Stride 8. 
        # 如果输入是 168，Patch 后长度约为 (168-16)/8 + 1 = 20 个 token，计算量大幅下降
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)
        
        # Use simple embedding for decoder (point-wise), Patch for encoder
        self.enc_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, configs.dropout)
        
        # Add Positional Embedding specifically for Patches (Sequence length shrinks)
        # Maximum patches roughly: 168 / 8 = 21. Setting max_len=100 is safe.
        self.patch_pos_embedding = PositionalEmbedding(configs.d_model, max_len=500) 

        # Decoder embedding remains standard (Point-wise) because we need full resolution output
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Encoder
        self.encoder = Encoder(
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
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Decoder
        if self.task_name in ['long_term_forecast', 'short_term_forecast', 'peak_detect_ltf', 'peak_detect_ltf_basic']:
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
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
                projection=PeakPredictionHead(configs.d_model, configs.c_out, configs.dropout)
            )
            
            if self.task_name == 'peak_detect_ltf':
                # Fusion Layer: Map Value features to Peak features
                self.value_fusion = nn.Linear(configs.c_out, configs.d_model)
                
                # Peak Decoder with Specialized Head
                self.peak_decoder = Decoder(
                    [
                        DecoderLayer(
                            AttentionLayer(
                                FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
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

        # Other tasks...
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1. Physics Augmentation (Derivatives)
        # x_enc shape: [B, 168, C]
        v_enc, a_enc = compute_derivatives(x_enc)
        # Concatenate: [B, 168, C*3]
        x_enc_augmented = torch.cat([x_enc, v_enc, a_enc], dim=-1)
        
        # 2. Patch Embedding (Encoder Only)
        # Output: [B, Num_Patches, D_Model]
        enc_out = self.enc_embedding(x_enc_augmented)
        # Add Positional Embedding manually since PatchEmbedding output is raw projection
        enc_out = enc_out + self.patch_pos_embedding(enc_out)
        
        # 3. Encoder (Processes Patches)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 4. Decoder (Standard Point-wise)
        # Note: Cross-Attention will handle [Query: 720, Key: Num_Patches]
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def peak_detect_ltf(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # -------------------------------------------
        # Step 1: Normalization
        # -------------------------------------------
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # -------------------------------------------
        # Step 3: Patch Encoder
        # -------------------------------------------
        # Embed patches. Since channels increased (*3), projection layer adjusts dynamically
        enc_out = self.enc_embedding(x_enc)
        enc_out = enc_out + self.patch_pos_embedding(enc_out)
        
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # -------------------------------------------
        # Step 4: Value Decoding (Standard)
        # -------------------------------------------
        dec_input = self.dec_embedding(x_dec, x_mark_dec)
        dec_out_val = self.decoder(dec_input, enc_out, x_mask=None, cross_mask=None)

        # -------------------------------------------
        # Step 5: Peak Decoding (Cascade)
        # -------------------------------------------
        
        peak_out = self.peak_decoder(dec_input, enc_out, x_mask=None, cross_mask=None)

        # -------------------------------------------
        # Step 6: De-Normalization
        # -------------------------------------------
        dec_out_val = dec_out_val.clone()
        dec_out_val[:, -self.pred_len:, :] = dec_out_val[:, -self.pred_len:, :] * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out_val[:, -self.pred_len:, :] = dec_out_val[:, -self.pred_len:, :] + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out_val, peak_out

    # ... (Other methods remain largely same, adapt usage of enc_embedding if needed) ...

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'peak_detect_ltf':
            dec_out, peak_out = self.peak_detect_ltf(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], peak_out[:, -self.pred_len:, :]
        # Keep other tasks compatible if necessary (omitted for brevity)
        return None