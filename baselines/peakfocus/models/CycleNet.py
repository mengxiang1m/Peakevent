import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

class RecurrentCycle(torch.nn.Module):
    # Thanks for the contribution of wayhoww.
    # The new implementation uses index arithmetic with modulo to directly gather cyclic data in a single operation,
    # while the original implementation manually rolls and repeats the data through looping.
    # It achieves a significant speed improvement (2x ~ 3x acceleration).
    # See https://github.com/ACAT-SCUT/CycleNet/pull/4 for more details.
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len    
        return self.data[gather_index]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        self.if_vis = 0
        # 可视化计数器
        self.forward_count = 0
        self.viz_interval = 1000  # 每1000次forward生成一次可视化
        self.viz_dir = './visualization/output/predictions'
        os.makedirs(self.viz_dir, exist_ok=True)

        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )
        
        # Peak分类头：仅用于peak_detect_ltf任务
        if self.task_name == 'peak_detect_ltf':
            # Peak分类头：输出每个时间步是否为peak (0 or 1)
            if self.model_type == 'linear':
                self.peak_model = nn.Linear(self.seq_len, self.pred_len)
            elif self.model_type == 'mlp':
                self.peak_model = nn.Sequential(
                    nn.Linear(self.seq_len, self.d_model),
                    nn.ReLU(),
                    nn.Linear(self.d_model, self.pred_len)
                )

    def forecast(self, x_enc, cycle_index, x_mark_enc=None):
        return self.forward(x_enc, cycle_index, x_mark_enc)

    def forward(self, x_enc, cycle_index, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)
        x = x_enc
        
        # 增加前向传播计数
        self.forward_count += 1
        
        # 保存原始输入用于可视化
        x_original = x.clone() if self.forward_count % self.viz_interval == 0 else None
        
        # Ensure cycle_index is on the correct device and has the right dtype
        cycle_index = cycle_index.long().to(x.device)

        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # 保存RevIN后的数据用于可视化
        x_revin = x.clone() if self.forward_count % self.viz_interval == 0 else None

        # remove the cycle of the input data
        cycle_component = self.cycleQueue(cycle_index, self.seq_len)
        x = x - cycle_component

        # 保存残差项用于可视化或peak检测任务
        # 对于peak_detect_ltf任务，始终需要保存残差分量
        if self.task_name == 'peak_detect_ltf' or self.forward_count % self.viz_interval == 0:
            residual_component = x.clone()
        else:
            residual_component = None

        # forecasting with channel independence (parameters-sharing)
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        # add back the cycle of the output data
        pred_cycle_component = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
        y = y + pred_cycle_component

        # instance denorm
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        if self.if_vis:
            # 每1000次生成一次可视化
            if self.forward_count % self.viz_interval == 0 and self.training:
                self._visualize_decomposition(
                    x_original, x_revin, cycle_component, residual_component, 
                    y, pred_cycle_component, cycle_index, self.forward_count
                )

        # 根据任务类型返回不同的输出
        if self.task_name == 'peak_detect_ltf':
            # Peak检测任务：返回值预测和peak分类
            # 对残差项应用peak分类模型
            peak_y = self.peak_model(residual_component.permute(0, 2, 1)).permute(0, 2, 1)
            # 注意: 不要在这里sigmoid! BCEWithLogitsLoss会自动处理
            # 输出原始logits，让损失函数内部进行sigmoid
            return y, peak_y
        elif self.task_name == 'peak_detect_ltf_basic':
            # Peak检测基础任务：只返回值预测
            return y
        else:
            # 其他任务：默认返回值预测
            return y

    def _visualize_decomposition(self, x_original, x_revin, cycle_component, residual_component, 
                               y_pred, pred_cycle_component, cycle_index, forward_count):
        """
        Visualize CycleNet decomposition process
        """
        try:
            # Set matplotlib backend to avoid display issues
            plt.switch_backend('Agg')
            
            # Convert to numpy (visualize first batch and first feature)
            batch_idx = 0
            feature_idx = 0
            
            x_orig = x_original[batch_idx, :, feature_idx].detach().cpu().numpy()
            x_rev = x_revin[batch_idx, :, feature_idx].detach().cpu().numpy()
            cycle_comp = cycle_component[batch_idx, :, feature_idx].detach().cpu().numpy()
            residual_comp = residual_component[batch_idx, :, feature_idx].detach().cpu().numpy()
            y_pred_comp = y_pred[batch_idx, :, feature_idx].detach().cpu().numpy()
            pred_cycle_comp = pred_cycle_component[batch_idx, :, feature_idx].detach().cpu().numpy()
            
            # Create time axis
            seq_len = len(x_orig)
            pred_len = len(y_pred_comp)
            time_input = np.arange(seq_len)
            time_pred = np.arange(seq_len, seq_len + pred_len)
            
            # Create subplots
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle(f'CycleNet Decomposition - Forward #{forward_count}', fontsize=16, fontweight='bold')
            
            # 1. Original input vs RevIN processed
            ax = axes[0, 0]
            ax.plot(time_input, x_orig, 'b-', linewidth=2, label='Original Input', alpha=0.8)
            ax.plot(time_input, x_rev, 'g--', linewidth=2, label='After RevIN', alpha=0.7)
            ax.set_title('Original Input vs RevIN Processed')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. Cycle components
            ax = axes[0, 1]
            ax.plot(time_input, cycle_comp, 'r-', linewidth=2, label='Input Cycle', alpha=0.8)
            ax.plot(time_pred, pred_cycle_comp, 'orange', linewidth=2, label='Pred Cycle', alpha=0.8)
            ax.set_title('Cycle Components Comparison')
            ax.set_ylabel('Cycle Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 3. Residual components
            ax = axes[1, 0]
            ax.plot(time_input, residual_comp, 'purple', linewidth=2, label='Input Residual', alpha=0.8)
            ax.plot(time_pred, y_pred_comp, 'brown', linewidth=2, label='Pred Residual', alpha=0.8)
            ax.set_title('Residual Components Comparison')
            ax.set_ylabel('Residual Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 4. Decomposition process
            ax = axes[1, 1]
            ax.plot(time_input, x_rev, 'g-', linewidth=1.5, label='RevIN Input', alpha=0.7)
            ax.plot(time_input, cycle_comp, 'r:', linewidth=2, label='Cycle Component', alpha=0.8)
            ax.plot(time_input, residual_comp, 'purple', linewidth=1.5, label='Residual Component', alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_title('Decomposition: RevIN = Cycle + Residual')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 5. Final prediction results
            ax = axes[2, 0]
            # Reconstruct final prediction (add back RevIN parameters)
            if self.use_revin:
                seq_mean = torch.mean(x_original, dim=1, keepdim=True)
                seq_var = torch.var(x_original, dim=1, keepdim=True) + 1e-5
                final_pred = (y_pred + pred_cycle_component) * torch.sqrt(seq_var) + seq_mean
                final_pred_np = final_pred[batch_idx, :, feature_idx].detach().cpu().numpy()
            else:
                final_pred_np = y_pred_comp + pred_cycle_comp
            
            ax.plot(time_input, x_orig, 'b-', linewidth=2, label='Historical Data', alpha=0.8)
            ax.plot(time_pred, final_pred_np, 'red', linewidth=3, label='Final Prediction', alpha=0.9)
            ax.axvline(x=seq_len-1, color='black', linestyle='--', alpha=0.5, label='Prediction Start')
            ax.set_title('Historical Data vs Prediction Results')
            ax.set_ylabel('Value')
            ax.set_xlabel('Time Step')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 6. Learned cycle pattern
            ax = axes[2, 1]
            cycle_data = self.cycleQueue.data.detach().cpu().numpy()
            cycle_start = cycle_index[batch_idx].item()
            cycle_positions = [(cycle_start + i) % self.cycle_len for i in range(min(seq_len, self.cycle_len))]
            cycle_values = cycle_data[cycle_positions, feature_idx]
            
            ax.plot(cycle_positions, cycle_values, 'ro-', linewidth=2, markersize=4, alpha=0.8)
            ax.set_title(f'Learned Cycle Pattern (Length={self.cycle_len})')
            ax.set_xlabel('Position in Cycle')
            ax.set_ylabel('Cycle Value')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'cyclenet_decomposition_forward_{forward_count}_{timestamp}.png'
            save_path = os.path.join(self.viz_dir, filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization saved: {save_path}")
            
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            plt.close('all')  # Ensure all figures are closed
