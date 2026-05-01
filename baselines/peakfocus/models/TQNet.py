import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin

        self.use_tq = True  # ablation parameter, default: True
        self.channel_aggre = True   # ablation parameter, default: True

        if self.use_tq:
            self.temporalQuery = torch.nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)

        if self.channel_aggre:
            self.channelAggregator = nn.MultiheadAttention(embed_dim=self.seq_len, num_heads=4, batch_first=True, dropout=0.5)

        self.input_proj = nn.Linear(self.seq_len, self.d_model)

        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )

        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )


    def forecast(self, x_enc, x_mark_enc=None):
        return self._forward_impl(x_enc, x_mark_enc)

    def _forward_impl(self, x, x_mark_enc=None):
        # Calculate cycle_index from time marks if available
        # For hourly data with weekly cycle (168 hours), we use hour of week
        if x_mark_enc is not None and x_mark_enc.shape[-1] >= 4:
            # Assuming time features include [month, day, weekday, hour, minute]
            # For 60-min data, hour of week = weekday * 24 + hour
            weekday = x_mark_enc[:, -1, 2]  # weekday (0-6)
            hour = x_mark_enc[:, -1, 3]     # hour (0-23)
            cycle_index = (weekday * 24 + hour) % self.cycle_len
            cycle_index = cycle_index.long()
        else:
            # Fallback: use sequence position modulo cycle_len
            batch_size = x.shape[0]
            cycle_index = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # b,s,c -> b,c,s
        x_input = x.permute(0, 2, 1)

        if self.use_tq:
            gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
            query_input = self.temporalQuery[gather_index].permute(0, 2, 1)  # (b, c, s)
            if self.channel_aggre:
                channel_information = self.channelAggregator(query=query_input, key=x_input, value=x_input)[0]
            else:
                channel_information = query_input
        else:
            if self.channel_aggre:
                channel_information = self.channelAggregator(query=x_input, key=x_input, value=x_input)[0]
            else:
                channel_information = 0

        input = self.input_proj(x_input+channel_information)

        hidden = self.model(input)

        output = self.output_proj(hidden+input).permute(0, 2, 1)

        # instance denorm
        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: (batch_size, seq_len, enc_in)
        return self._forward_impl(x_enc, x_mark_enc)


