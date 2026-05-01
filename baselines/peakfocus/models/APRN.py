import torch
import torch.nn as nn
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import os

# RecurrentCycle 保持不变
class RecurrentCycle(torch.nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]

# 核心创新：Sin-Cos组合细节生成器
class SinCosDetailGenerator(nn.Module):
    """
    [新思路] 基于Sin-Cos组合的细节生成器
    核心思想：
    1. 直接生成幅度(amplitude)和相位(phase)参数
    2. 使用固定的sin/cos基函数组合
    3. 简单而稳定的设计，避免复杂的傅里叶系数计算
    """
    def __init__(self, time_context_dim, total_len, enc_in, num_harmonics, detail_scale):
        super(SinCosDetailGenerator, self).__init__()
        self.total_len = total_len
        self.enc_in = enc_in
        self.num_harmonics = num_harmonics
        self.detail_scale = detail_scale
        
        # 简单的特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(time_context_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 为每个通道和每个谐波分量生成幅度和相位
        # 每个谐波需要：amplitude_sin, amplitude_cos, phase_sin, phase_cos
        self.amplitude_generator = nn.Sequential(
            nn.Linear(128, enc_in * num_harmonics * 2),  # *2 for sin and cos
            nn.Sigmoid()  # 幅度限制在[0,1]
        )
        
        self.phase_generator = nn.Sequential(
            nn.Linear(128, enc_in * num_harmonics * 2),  # *2 for sin and cos
            nn.Tanh()  # 相位在[-1,1]，对应[-π,π]
        )
        
        # 预计算时间轴
        t = torch.linspace(0, 1, total_len)  # 归一化时间轴 [0,1]
        self.register_buffer('t', t)
        
        # 预计算基频率 (不同谐波的频率)
        base_freqs = torch.arange(1, num_harmonics + 1, dtype=torch.float32)
        self.register_buffer('base_freqs', base_freqs)

    def forward(self, time_context_emb):
        # 1. 提取特征
        features = self.feature_extractor(time_context_emb)  # [B, 128]
        
        # 2. 生成幅度和相位参数
        amplitudes = self.amplitude_generator(features)  # [B, N*K*2]
        phases = self.phase_generator(features) * np.pi  # [B, N*K*2], 转换到[-π,π]
        
        # 3. 重塑为 [B, N, K, 2] (2 for sin and cos)
        batch_size = time_context_emb.size(0)
        amplitudes = amplitudes.view(batch_size, self.enc_in, self.num_harmonics, 2)
        phases = phases.view(batch_size, self.enc_in, self.num_harmonics, 2)
        
        # 4. 生成细节波形
        detail_waveform = torch.zeros(batch_size, self.total_len, self.enc_in, device=self.t.device)
        
        # 对每个谐波分量计算sin/cos贡献
        for k in range(self.num_harmonics):
            freq = self.base_freqs[k]  # 当前谐波频率
            
            # 提取当前谐波的参数 [B, N]
            amp_sin = amplitudes[:, :, k, 0]  # sin分量幅度
            amp_cos = amplitudes[:, :, k, 1]  # cos分量幅度
            phase_sin = phases[:, :, k, 0]    # sin分量相位
            phase_cos = phases[:, :, k, 1]    # cos分量相位
            
            # 计算时间项 [T]
            time_term = 2 * np.pi * freq * self.t
            
            # 计算sin和cos波形 [B, T, N]
            # sin分量: A_sin * sin(2πft + φ_sin)
            sin_component = (amp_sin.unsqueeze(1) * 
                           torch.sin(time_term.unsqueeze(0).unsqueeze(-1) + 
                                   phase_sin.unsqueeze(1)))
            
            # cos分量: A_cos * cos(2πft + φ_cos)  
            cos_component = (amp_cos.unsqueeze(1) * 
                           torch.cos(time_term.unsqueeze(0).unsqueeze(-1) + 
                                   phase_cos.unsqueeze(1)))
            
            # 累加到总波形
            detail_waveform += sin_component + cos_component
        
        # 5. 应用幅度衰减 (高频分量自动衰减)
        freq_decay = 1.0 / torch.sqrt(self.base_freqs)  # [K]
        for k in range(self.num_harmonics):
            detail_waveform *= freq_decay[k] if k == 0 else 1.0  # 简化处理
        
        # 6. 最终缩放
        detail_waveform = detail_waveform * self.detail_scale
        
        return detail_waveform

class Model(nn.Module):
    """
    FourierCycleNet: 具有傅里叶动态细节的最终网络
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        # --- 基础参数 ---
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = getattr(configs, 'cycle_len', configs.cycle)
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        self.detail_scale = getattr(configs, 'detail_scale', 0.05)  # 降低默认值从0.1到0.05
        self.num_fourier_terms = getattr(configs, 'num_fourier_terms', 8)  # 降低默认项数从10到8

        # 基础周期模式生成器
        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        # --- STID 时间嵌入 ---
        self.if_time_in_day = configs.if_T_i_D
        self.if_day_in_week = configs.if_D_i_W
        self.time_of_day_size = configs.time_of_day_size
        self.day_of_week_size = configs.day_of_week_size
        self.temp_dim_tid = configs.temp_dim_tid
        self.temp_dim_diw = configs.temp_dim_diw
        
        time_context_dim = 0
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
            time_context_dim += self.temp_dim_tid
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)
            time_context_dim += self.temp_dim_diw
        
        # --- 核心创新：实例化Sin-Cos细节生成器 ---
        if time_context_dim > 0:
            total_len = self.seq_len + self.pred_len
            self.detail_generator = SinCosDetailGenerator(
                time_context_dim, total_len, self.enc_in, self.num_fourier_terms, self.detail_scale
            )
        
        # --- 骨干网络 ---
        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear': self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp': self.model = nn.Sequential(nn.Linear(self.seq_len, self.d_model), nn.ReLU(), nn.Linear(self.d_model, self.pred_len))

        self.forward_count = 0

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x = x_enc
        batch_size = x.shape[0]
        
        detail_waveform, input_details = None, None
        cycle_index = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        if x_mark_enc is not None and hasattr(self, 'detail_generator'):
            # 参考STID的时间特征处理方式
            time_embeds = []
            
            # 处理时间特征
            if self.if_time_in_day and x_mark_enc.shape[-1] >= 1:
                # 获取小时特征
                if x_mark_enc.shape[-1] >= 4:
                    hour_data = x_mark_enc[:, -1, 3]  # hour feature at position 3
                else:
                    hour_data = x_mark_enc[:, -1, 0]  # fallback to first feature
                
                # 确保在有效范围内并转换为索引
                hour_indices = torch.clamp(hour_data, 0, self.time_of_day_size - 1).long()
                time_in_day_emb = self.time_in_day_emb[hour_indices]  # [B, temp_dim_tid]
                time_embeds.append(time_in_day_emb)
                
                # 计算周期索引
                cycle_index = hour_indices % self.cycle_len
            
            if self.if_day_in_week and x_mark_enc.shape[-1] >= 3:
                # 获取星期特征
                weekday_data = x_mark_enc[:, -1, 2]  # weekday feature at position 2
                
                # 确保在有效范围内并转换为索引
                weekday_indices = torch.clamp(weekday_data, 0, self.day_of_week_size - 1).long()
                day_in_week_emb = self.day_in_week_emb[weekday_indices]  # [B, temp_dim_diw]
                time_embeds.append(day_in_week_emb)
                
                # 更新周期索引（如果有星期信息）
                if self.if_time_in_day:
                    cycle_index = (weekday_indices * 24 + hour_indices) % self.cycle_len
                else:
                    cycle_index = weekday_indices % self.cycle_len
            
            # 合并时间嵌入
            if time_embeds:
                time_context_emb = torch.cat(time_embeds, dim=-1)
                detail_waveform = self.detail_generator(time_context_emb)
            else:
                # 如果没有时间特征，使用零索引
                cycle_index = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        base_cycle_seq = self.cycleQueue(cycle_index, self.seq_len)
        if detail_waveform is not None:
            input_details = detail_waveform[:, :self.seq_len, :]
            detailed_cycle_seq = base_cycle_seq + input_details
        else:
            detailed_cycle_seq = base_cycle_seq
        
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True); seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        residual = x - detailed_cycle_seq
        
        self.forward_count += 1
        if self.forward_count % 1000 == 0:
            save_dir = "./cycle_visualizations/"; os.makedirs(save_dir, exist_ok=True)
            self._save_cycle_plots(x[0, :, 0], residual[0, :, 0], base_cycle_seq[0, :, 0], input_details[0, :, 0] if input_details is not None else None, detailed_cycle_seq[0, :, 0], self.forward_count, save_dir)

        residual_permuted = residual.permute(0, 2, 1)
        predicted_residual = self.model(residual_permuted).permute(0, 2, 1)
        
        future_base_cycle = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
        if detail_waveform is not None:
            future_details = detail_waveform[:, self.seq_len:, :]
            detailed_future_cycle = future_base_cycle + future_details
        else:
            detailed_future_cycle = future_base_cycle

        y = predicted_residual + detailed_future_cycle
        if self.use_revin: y = y * torch.sqrt(seq_var) + seq_mean
        return y

    def _save_cycle_plots(self, input_seq, residual_seq, base_cycle_seq, detail_seq, detailed_cycle_seq, count, save_dir):
        """[改进版可视化] 包含所有重要信息的综合可视化"""
        try:
            input_data = input_seq.detach().cpu().numpy()
            res_data = residual_seq.detach().cpu().numpy()
            base_data = base_cycle_seq.detach().cpu().numpy()
            detailed_data = detailed_cycle_seq.detach().cpu().numpy()
            
            # 创建更大的图形以容纳更多信息
            plt.figure(figsize=(16, 20))
            
            # 子图1: Input Sequence (原始输入数据)
            plt.subplot(6, 1, 1)
            plt.plot(input_data, label='Input Sequence', color='black', linewidth=2)
            plt.title(f'1. Input Sequence (Original Data, Forward #{count})')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            # 添加统计信息
            plt.text(0.02, 0.95, f'Mean: {np.mean(input_data):.4f}, Std: {np.std(input_data):.4f}', 
                    transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            # 子图2: Base Cycle (稳定的周期模式)
            plt.subplot(6, 1, 2)
            plt.plot(base_data, label='Base Cycle', color='blue', linestyle='--', linewidth=2)
            plt.title('2. Base Cycle (Stable Repeating Pattern)')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.text(0.02, 0.95, f'Mean: {np.mean(base_data):.4f}, Std: {np.std(base_data):.4f}', 
                    transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            # 子图3: Detail Waveform (动态细节)
            plt.subplot(6, 1, 3)
            if detail_seq is not None:
                detail_data = detail_seq.detach().cpu().numpy()
                plt.plot(detail_data, label='Detail Waveform', color='red', linewidth=1.5)
                plt.title(f'3. Dynamic Detail Waveform (Scale: {self.detail_scale})')
                # 添加零线
                plt.axhline(0, color='gray', linestyle=':', alpha=0.7)
                # 设置y轴范围
                detail_range = max(abs(np.min(detail_data)), abs(np.max(detail_data)))
                if detail_range > 0:
                    plt.ylim(-detail_range * 1.2, detail_range * 1.2)
                plt.text(0.02, 0.95, f'Range: [{np.min(detail_data):.4f}, {np.max(detail_data):.4f}]', 
                        transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            else:
                plt.plot(np.zeros_like(base_data), label='No Details', color='gray', linestyle=':')
                plt.title('3. Dynamic Detail Waveform (No Details Generated)')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 子图4: Detailed Cycle (合成后的完整模式)
            plt.subplot(6, 1, 4)
            plt.plot(detailed_data, label='Detailed Cycle (Base + Detail)', color='green', linewidth=2)
            plt.title('4. Detailed Cycle (Complete Pattern = Base + Detail)')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.text(0.02, 0.95, f'Mean: {np.mean(detailed_data):.4f}, Std: {np.std(detailed_data):.4f}', 
                    transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            # 子图5: Residual (需要预测的残差)
            plt.subplot(6, 1, 5)
            plt.plot(res_data, label='Residual (Input - Detailed Cycle)', color='purple', linewidth=1.5)
            plt.title('5. Residual Sequence (What the Backbone Network Needs to Predict)')
            plt.ylabel('Value')
            plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
            plt.legend()
            plt.grid(True, alpha=0.3)
            # 添加残差统计信息
            residual_energy = np.mean(res_data ** 2)
            plt.text(0.02, 0.95, f'Energy: {residual_energy:.6f}, Range: [{np.min(res_data):.4f}, {np.max(res_data):.4f}]', 
                    transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            # 子图6: 对比图 (输入 vs 模式)
            plt.subplot(6, 1, 6)
            plt.plot(input_data, label='Input Sequence', color='black', linewidth=2, alpha=0.8)
            plt.plot(detailed_data, label='Detailed Cycle', color='green', linewidth=2, alpha=0.8)
            plt.fill_between(range(len(res_data)), detailed_data, input_data, 
                           color='red', alpha=0.3, label='Residual Area')
            plt.title('6. Comparison: Input vs Model Pattern')
            plt.xlabel('Time Steps')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 计算拟合质量
            mse = np.mean((input_data - detailed_data) ** 2)
            correlation = np.corrcoef(input_data, detailed_data)[0, 1]
            plt.text(0.02, 0.95, f'MSE: {mse:.6f}, Correlation: {correlation:.4f}', 
                    transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.tight_layout()
            filepath = os.path.join(save_dir, f"sincos_cycle_forward_{count}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 打印详细信息到控制台
            print(f"\n=== Cycle Visualization #{count} ===")
            print(f"Input stats: mean={np.mean(input_data):.4f}, std={np.std(input_data):.4f}")
            print(f"Base cycle stats: mean={np.mean(base_data):.4f}, std={np.std(base_data):.4f}")
            if detail_seq is not None:
                detail_data = detail_seq.detach().cpu().numpy()
                print(f"Detail stats: range=[{np.min(detail_data):.4f}, {np.max(detail_data):.4f}]")
            print(f"Residual energy: {np.mean(res_data ** 2):.6f}")
            print(f"Pattern fit quality: MSE={mse:.6f}, Correlation={correlation:.4f}")
            print(f"Saved visualization to: {filepath}")
            print("=" * 50)
            
        except Exception as e:
            print(f"Error saving cycle plots: {e}")
            plt.close()