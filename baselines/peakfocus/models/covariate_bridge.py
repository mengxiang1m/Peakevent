import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class EndogenousPatchEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        if x.shape[-1] < self.patch_len:
            raise ValueError(f"seq_len {x.shape[-1]} must be >= patch_len {self.patch_len}")
        glb = self.glb_token[:, :n_vars].repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class CovariateBridgeEncoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class CovariateBridgeEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, factor, activation):
        super().__init__()
        self.self_attention = AttentionLayer(
            FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
            d_model,
            n_heads,
        )
        self.cross_attention = AttentionLayer(
            FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
            d_model,
            n_heads,
        )
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        batch_size, _, d_model = cross.shape
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)

        x_glb = x[:, -1:, :]
        x_glb_reshaped = torch.reshape(x_glb, (batch_size, -1, d_model))
        x_glb_attn = self.dropout(self.cross_attention(x_glb_reshaped, cross, cross, attn_mask=cross_mask)[0])
        x_glb_attn = torch.reshape(
            x_glb_attn,
            (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2]),
        ).unsqueeze(1)
        x_glb = self.norm2(x_glb + x_glb_attn)

        x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class TimeXerCovariateBridge(nn.Module):
    def __init__(
        self,
        endogenous_channels,
        exogenous_channels,
        seq_len,
        pred_len,
        patch_len,
        d_model,
        n_heads,
        d_ff,
        dropout,
        factor,
        activation,
        e_layers,
        embed,
        freq,
    ):
        super().__init__()
        self.endogenous_channels = endogenous_channels
        self.exogenous_channels = exogenous_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.patch_num = seq_len // patch_len
        if self.patch_num < 1:
            raise ValueError(f"patch_len {patch_len} is too large for seq_len {seq_len}")

        self.endogenous_embedding = EndogenousPatchEmbedding(endogenous_channels, d_model, patch_len, dropout)
        self.exogenous_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)
        self.encoder = CovariateBridgeEncoder(
            [
                CovariateBridgeEncoderLayer(d_model, n_heads, d_ff, dropout, factor, activation)
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.temporal_projection = nn.Linear(self.patch_num + 1, pred_len)
        self.time_feature_dims = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        self._freq = freq

    def _select_time_marks(self, x_mark):
        if x_mark is None:
            return None
        keep_dims = self.time_feature_dims.get(self.freq, x_mark.shape[-1])
        return x_mark[:, :, :keep_dims]

    @property
    def freq(self):
        return getattr(self, '_freq', 'h')

    @freq.setter
    def freq(self, value):
        self._freq = value

    def forward(self, x_endogenous, x_exogenous, x_mark_enc):
        if x_exogenous is None or x_exogenous.shape[-1] == 0:
            raise ValueError('TimeXerCovariateBridge requires non-empty exogenous inputs')
        en_embed, n_vars = self.endogenous_embedding(x_endogenous.permute(0, 2, 1))
        ex_mark = self._select_time_marks(x_mark_enc)
        ex_embed = self.exogenous_embedding(x_exogenous, ex_mark)
        bridge_out = self.encoder(en_embed, ex_embed)
        bridge_out = torch.reshape(bridge_out, (-1, n_vars, bridge_out.shape[-2], bridge_out.shape[-1]))
        bridge_out = bridge_out.mean(dim=1).permute(0, 2, 1)
        bridge_out = self.temporal_projection(bridge_out)
        return bridge_out.permute(0, 2, 1)
