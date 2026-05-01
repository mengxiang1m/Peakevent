"""
时间序列预测中使用 Wasserstein 损失的示例
适用场景：
1. 预测未来时间窗口的分布而非单点值
2. 负荷预测（电力、交通等）
3. 价格分布预测
4. 不确定性建模
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_stats_loss as stats_loss
import matplotlib.pyplot as plt

# 配置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

#######################################################
#           简单的时间序列预测模型                        #
#######################################################

class TimeSeriesLSTM(nn.Module):
    """
    简单的 LSTM 模型用于时间序列预测
    输入：历史时间序列窗口
    输出：未来时间窗口的预测值分布
    """
    def __init__(self, input_size=1, hidden_size=64, output_size=24, num_layers=2):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softplus = nn.Softplus()  # 确保输出非负（概率分布要求）
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])
        # 使用 softplus 确保输出为正值（可以被归一化为概率分布）
        out = self.softplus(out)
        return out


#######################################################
#              生成模拟时间序列数据                        #
#######################################################

def generate_time_series_data(n_samples=1000, seq_length=48, pred_length=24):
    """
    生成模拟的时间序列数据
    模拟日负荷曲线：包含趋势、周期性和噪声
    
    Args:
        n_samples: 样本数量
        seq_length: 输入序列长度（历史窗口）
        pred_length: 预测序列长度（未来窗口）
    """
    X_list = []
    y_list = []
    
    for i in range(n_samples):
        # 生成带有周期性的时间序列（模拟24小时周期）
        t = np.arange(seq_length + pred_length)
        
        # 基础周期信号（日周期）
        base_signal = 50 + 30 * np.sin(2 * np.pi * t / 24)
        
        # 添加趋势
        trend = 0.1 * t
        
        # 添加随机噪声
        noise = np.random.normal(0, 5, seq_length + pred_length)
        
        # 组合信号
        signal = base_signal + trend + noise
        signal = np.maximum(signal, 0)  # 确保非负
        
        # 分割为输入和输出
        X_list.append(signal[:seq_length])
        y_list.append(signal[seq_length:seq_length + pred_length])
    
    X = np.array(X_list).reshape(-1, seq_length, 1)  # (n_samples, seq_length, 1)
    y = np.array(y_list)  # (n_samples, pred_length)
    
    return X, y


#######################################################
#              训练函数                                #
#######################################################

def train_model_with_wasserstein_loss(model, X_train, y_train, epochs=50, lr=0.001, 
                                     loss_type='wasserstein', lambda_mse=0.1):
    """
    使用 Wasserstein 损失训练模型
    
    Args:
        model: PyTorch 模型
        X_train, y_train: 训练数据
        epochs: 训练轮数
        lr: 学习率
        loss_type: 损失类型 ('wasserstein', 'energy', 'mse', 'combined')
        lambda_mse: MSE 损失的权重（用于组合损失）
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 转换为 PyTorch tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    loss_history = []
    
    print(f"\n--- Training model with {loss_type.upper()} loss ---")
    print(f"Starting training with {loss_type} loss function...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        predictions = model(X_train_tensor)
        
        # 计算损失
        if loss_type == 'wasserstein':
            loss = stats_loss.torch_wasserstein_loss(predictions, y_train_tensor)
        elif loss_type == 'energy':
            loss = stats_loss.torch_energy_loss(predictions, y_train_tensor)
        elif loss_type == 'mse':
            loss = nn.MSELoss()(predictions, y_train_tensor)
        elif loss_type == 'combined':
            # 组合损失：Wasserstein + MSE
            wass_loss = stats_loss.torch_wasserstein_loss(predictions, y_train_tensor)
            mse_loss = nn.MSELoss()(predictions, y_train_tensor)
            loss = wass_loss + lambda_mse * mse_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    return loss_history


#######################################################
#              评估和可视化                             #
#######################################################

def evaluate_and_visualize(model, X_test, y_test, num_samples=3):
    """
    评估模型并可视化预测结果
    """
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    with torch.no_grad():
        predictions = model(X_test_tensor)
    
    # 计算各种评估指标
    mse = nn.MSELoss()(predictions, y_test_tensor).item()
    mae = torch.mean(torch.abs(predictions - y_test_tensor)).item()
    wass_dist = stats_loss.torch_wasserstein_loss(predictions, y_test_tensor).item()
    energy_dist = stats_loss.torch_energy_loss(predictions, y_test_tensor).item()
    
    print("\n=== Evaluation Metrics ===")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Wasserstein Distance: {wass_dist:.4f}")
    print(f"Energy Distance: {energy_dist:.4f}")
    
    # 可视化几个预测样本
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        # 归一化为概率分布用于可视化
        pred_normalized = predictions[i].numpy()
        pred_normalized = pred_normalized / pred_normalized.sum()
        
        true_normalized = y_test[i]
        true_normalized = true_normalized / true_normalized.sum()
        
        axes[i].plot(true_normalized, label='Ground Truth', marker='o', linewidth=2)
        axes[i].plot(pred_normalized, label='Prediction', marker='s', linewidth=2, alpha=0.7)
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Normalized Value (Distribution)')
        axes[i].set_title(f'Sample {i+1}: Predicted vs True Distribution')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_series_prediction_results.png', dpi=150)
    print("\nVisualization saved to: time_series_prediction_results.png")
    plt.show()


#######################################################
#              比较不同损失函数                          #
#######################################################

def compare_loss_functions(X_train, y_train, X_test, y_test):
    """
    比较使用不同损失函数训练的模型性能
    """
    loss_types = ['wasserstein', 'energy', 'mse', 'combined']
    results = {}
    
    print("\n" + "="*60)
    print("Comparing Different Loss Functions")
    print("="*60)
    
    for loss_type in loss_types:
        print(f"\n--- Training model with {loss_type.upper()} loss ---")
        
        # 创建新模型
        model = TimeSeriesLSTM(input_size=1, hidden_size=64, output_size=24, num_layers=2)
        
        # 训练
        loss_history = train_model_with_wasserstein_loss(
            model, X_train, y_train, 
            epochs=30, 
            lr=0.001,
            loss_type=loss_type
        )
        
        # 评估
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        with torch.no_grad():
            predictions = model(X_test_tensor)
        
        mse = nn.MSELoss()(predictions, y_test_tensor).item()
        wass_dist = stats_loss.torch_wasserstein_loss(predictions, y_test_tensor).item()
        
        results[loss_type] = {
            'mse': mse,
            'wasserstein': wass_dist,
            'loss_history': loss_history
        }
        
        print(f"Test MSE: {mse:.4f}")
        print(f"Test Wasserstein Distance: {wass_dist:.4f}")
    
    # 可视化比较结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    for loss_type, result in results.items():
        ax1.plot(result['loss_history'], label=loss_type, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Curves with Different Loss Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 测试集性能对比
    loss_names = list(results.keys())
    mse_values = [results[lt]['mse'] for lt in loss_names]
    wass_values = [results[lt]['wasserstein'] for lt in loss_names]
    
    x = np.arange(len(loss_names))
    width = 0.35
    
    ax2.bar(x - width/2, mse_values, width, label='MSE', alpha=0.8)
    ax2.bar(x + width/2, wass_values, width, label='Wasserstein', alpha=0.8)
    ax2.set_xlabel('Loss Function Type')
    ax2.set_ylabel('Test Set Error')
    ax2.set_title('Test Performance with Different Loss Functions')
    ax2.set_xticks(x)
    ax2.set_xticklabels(loss_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('loss_function_comparison.png', dpi=150)
    print("\nComparison results saved to: loss_function_comparison.png")
    plt.show()
    
    return results


#######################################################
#              主函数                                  #
#######################################################

def main():
    print("="*60)
    print("Time Series Prediction with Wasserstein Loss Demo")
    print("="*60)
    
    # 生成数据
    print("\n1. Generating simulated time series data...")
    X, y = generate_time_series_data(n_samples=500, seq_length=48, pred_length=24)
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")
    
    # 方式1: 单独训练一个模型
    print("\n2. Training model (with Wasserstein loss)...")
    model = TimeSeriesLSTM(input_size=1, hidden_size=64, output_size=24, num_layers=2)
    
    loss_history = train_model_with_wasserstein_loss(
        model, X_train, y_train, 
        epochs=50, 
        lr=0.001,
        loss_type='wasserstein'  # Options: 'wasserstein', 'energy', 'mse', 'combined'
    )
    
    # 评估和可视化
    print("\n3. Evaluating model performance...")
    evaluate_and_visualize(model, X_test, y_test, num_samples=3)
    
    # 方式2: 比较不同损失函数（可选，较耗时）
    compare = input("\nCompare different loss functions? (y/n): ")
    if compare.lower() == 'y':
        print("\n4. Comparing different loss functions...")
        results = compare_loss_functions(X_train, y_train, X_test, y_test)
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
