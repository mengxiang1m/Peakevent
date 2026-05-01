"""
Statistical Loss Functions for PyTorch
包含 Wasserstein Distance、Energy Distance 和 CDF Loss 的实现
"""

import torch
import numpy as np


def torch_validate_distibution(tensor_a, tensor_b):
    """
    验证两个张量的维度是否匹配
    
    Parameters
    ---
    tensor_a : torch.Tensor
        第一个张量，形状为 (batch_size, dim)
    tensor_b : torch.Tensor
        第二个张量，形状为 (batch_size, dim)
    
    Returns
    ---
    None or raises ValueError
    """
    if tensor_a.shape != tensor_b.shape:
        raise ValueError(f"Shape mismatch: tensor_a {tensor_a.shape} vs tensor_b {tensor_b.shape}")
    return None


def torch_wasserstein_loss(tensor_a, tensor_b, p=1):
    """
    计算 Wasserstein Distance (Earth Mover's Distance)
    
    基于累积分布函数(CDF)的差异来计算
    对于一维分布，Wasserstein距离等于其CDF差的L^p范数
    
    Parameters
    ---
    tensor_a : torch.Tensor
        预测分布，形状为 (batch_size, seq_len) 或 (batch_size, seq_len, features)
    tensor_b : torch.Tensor
        真实分布，形状与 tensor_a 相同
    p : int or float
        范数阶数，默认为1 (推荐使用1-Wasserstein距离)
    
    Returns
    ---
    loss : torch.Tensor
        Wasserstein距离损失（标量）
    
    Reference
    ---
    - Wasserstein Distance: https://en.wikipedia.org/wiki/Wasserstein_metric
    - 对于一维分布: W_p(μ,ν) = (∫|F^{-1}_μ(t) - F^{-1}_ν(t)|^p dt)^{1/p}
    """
    # 确保张量维度一致
    torch_validate_distibution(tensor_a, tensor_b)
    
    # 保存原始形状
    original_shape = tensor_a.shape
    
    # 如果是3D张量 (batch, seq_len, features)，展平为 (batch, seq_len*features)
    if len(original_shape) == 3:
        batch_size = original_shape[0]
        tensor_a = tensor_a.reshape(batch_size, -1)
        tensor_b = tensor_b.reshape(batch_size, -1)
    
    # 对每个batch维度的样本进行排序
    # 排序后的值可以看作是累积分布函数的逆函数
    tensor_a_sorted, _ = torch.sort(tensor_a, dim=1)
    tensor_b_sorted, _ = torch.sort(tensor_b, dim=1)
    
    # 计算排序后张量的差异（L^p 范数）
    # 这等价于计算两个分布的CDF差的积分
    if p == 1:
        # L1 范数（1-Wasserstein距离，也称为Earth Mover's Distance）
        loss = torch.mean(torch.abs(tensor_a_sorted - tensor_b_sorted))
    else:
        # L^p 范数（p-Wasserstein距离）
        loss = torch.mean(torch.abs(tensor_a_sorted - tensor_b_sorted) ** p) ** (1.0 / p)
    
    return loss


def torch_energy_loss(tensor_a, tensor_b):
    """
    计算 Energy Distance
    
    Energy Distance 是一种基于能量统计的距离度量
    相比Wasserstein距离，它对分布的差异更敏感
    
    Parameters
    ---
    tensor_a : torch.Tensor
        预测分布，形状为 (batch_size, seq_len) 或 (batch_size, seq_len, features)
    tensor_b : torch.Tensor
        真实分布，形状与 tensor_a 相同
    
    Returns
    ---
    loss : torch.Tensor
        Energy距离损失（标量）
    
    Reference
    ---
    - Energy Distance: https://en.wikipedia.org/wiki/Energy_distance
    - E(X,Y) = 2*E[|X-Y|] - E[|X-X'|] - E[|Y-Y'|]
      其中 X, X' 是独立同分布的样本，Y, Y' 也是
    """
    # 确保张量维度一致
    torch_validate_distibution(tensor_a, tensor_b)
    
    # 保存原始形状
    original_shape = tensor_a.shape
    
    # 如果是3D张量，展平
    if len(original_shape) == 3:
        batch_size = original_shape[0]
        tensor_a = tensor_a.reshape(batch_size, -1)
        tensor_b = tensor_b.reshape(batch_size, -1)
    
    # 计算 Energy Distance 的三个项
    # 项1: 2 * E[|X - Y|]
    term1 = 2.0 * torch.mean(torch.abs(tensor_a - tensor_b))
    
    # 项2: E[|X - X'|] - 使用同一batch内不同样本之间的距离
    # 这里简化计算：使用相邻样本的距离来近似
    if tensor_a.shape[0] > 1:
        term2 = torch.mean(torch.abs(tensor_a[:-1] - tensor_a[1:]))
    else:
        term2 = 0.0
    
    # 项3: E[|Y - Y'|]
    if tensor_b.shape[0] > 1:
        term3 = torch.mean(torch.abs(tensor_b[:-1] - tensor_b[1:]))
    else:
        term3 = 0.0
    
    # Energy Distance = 2*E[|X-Y|] - E[|X-X'|] - E[|Y-Y'|]
    loss = term1 - term2 - term3
    
    return loss


def torch_cdf_loss(tensor_a, tensor_b, p=1):
    """
    计算基于累积分布函数(CDF)的损失
    
    直接计算两个分布的经验CDF之间的差异
    这是 Wasserstein 距离的一种替代实现
    
    Parameters
    ---
    tensor_a : torch.Tensor
        预测分布，形状为 (batch_size, seq_len) 或 (batch_size, seq_len, features)
    tensor_b : torch.Tensor
        真实分布，形状与 tensor_a 相同
    p : int or float
        范数阶数，默认为1
    
    Returns
    ---
    loss : torch.Tensor
        CDF距离损失（标量）
    """
    # 确保张量维度一致
    torch_validate_distibution(tensor_a, tensor_b)
    
    # 保存原始形状
    original_shape = tensor_a.shape
    
    # 如果是3D张量，展平
    if len(original_shape) == 3:
        batch_size = original_shape[0]
        tensor_a = tensor_a.reshape(batch_size, -1)
        tensor_b = tensor_b.reshape(batch_size, -1)
    
    # 排序以获得经验CDF
    tensor_a_sorted, _ = torch.sort(tensor_a, dim=1)
    tensor_b_sorted, _ = torch.sort(tensor_b, dim=1)
    
    # 计算CDF差异（L^p 范数）
    cdf_diff = torch.abs(tensor_a_sorted - tensor_b_sorted)
    
    if p == 1:
        loss = torch.mean(cdf_diff)
    elif p == 2:
        loss = torch.sqrt(torch.mean(cdf_diff ** 2))
    else:
        loss = torch.mean(cdf_diff ** p) ** (1.0 / p)
    
    return loss


def torch_combined_loss(tensor_a, tensor_b, alpha=0.5, loss_type='mse+wasserstein'):
    """
    组合损失函数：结合传统损失（MSE/MAE）和统计距离损失
    
    Parameters
    ---
    tensor_a : torch.Tensor
        预测值
    tensor_b : torch.Tensor
        真实值
    alpha : float
        统计距离损失的权重，范围 [0, 1]
        loss = (1-alpha) * MSE + alpha * Wasserstein
    loss_type : str
        组合类型: 'mse+wasserstein', 'mae+wasserstein', 'mse+energy'
    
    Returns
    ---
    loss : torch.Tensor
        组合损失（标量）
    """
    if loss_type == 'mse+wasserstein':
        mse_loss = torch.mean((tensor_a - tensor_b) ** 2)
        wass_loss = torch_wasserstein_loss(tensor_a, tensor_b)
        loss = (1 - alpha) * mse_loss + alpha * wass_loss
    elif loss_type == 'mae+wasserstein':
        mae_loss = torch.mean(torch.abs(tensor_a - tensor_b))
        wass_loss = torch_wasserstein_loss(tensor_a, tensor_b)
        loss = (1 - alpha) * mae_loss + alpha * wass_loss
    elif loss_type == 'mse+energy':
        mse_loss = torch.mean((tensor_a - tensor_b) ** 2)
        energy_loss = torch_energy_loss(tensor_a, tensor_b)
        loss = (1 - alpha) * mse_loss + alpha * energy_loss
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return loss


# ==================== PyTorch Module 包装器 ====================

class WassersteinLoss(torch.nn.Module):
    """
    Wasserstein Distance 损失的 PyTorch Module 包装器
    可以直接用作 criterion
    """
    def __init__(self, p=1):
        super(WassersteinLoss, self).__init__()
        self.p = p
    
    def forward(self, pred, target):
        return torch_wasserstein_loss(pred, target, p=self.p)


class EnergyLoss(torch.nn.Module):
    """
    Energy Distance 损失的 PyTorch Module 包装器
    """
    def __init__(self):
        super(EnergyLoss, self).__init__()
    
    def forward(self, pred, target):
        return torch_energy_loss(pred, target)


class CombinedLoss(torch.nn.Module):
    """
    组合损失的 PyTorch Module 包装器
    """
    def __init__(self, alpha=0.5, loss_type='mse+wasserstein'):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.loss_type = loss_type
    
    def forward(self, pred, target):
        return torch_combined_loss(pred, target, alpha=self.alpha, loss_type=self.loss_type)
