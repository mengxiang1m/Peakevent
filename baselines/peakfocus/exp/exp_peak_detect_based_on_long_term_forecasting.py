# ==================== 导入库 ====================
import os
import time
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import datetime
from typing import Optional, Any, Dict, List, Tuple, Union

# 内部依赖
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.dtw_metric import dtw, accelerated_dtw
from findpeaks import findpeaks

# 导入统计损失函数
from utils.pytorch_stats_loss import (
    WassersteinLoss, 
    EnergyLoss, 
    CombinedLoss
)

# ==================== 配置 ====================
warnings.filterwarnings('ignore')
matplotlib.use('Agg')  # 使用非交互式后端

# ==================== Matplotlib 全局字体配置 ====================
plt.rcParams['font.family'] = 'serif'  # 字体系列
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']  # 衬线字体
plt.rcParams['font.size'] = 18  # 默认字体大小
plt.rcParams['axes.labelsize'] = 24  # 坐标轴标签字体大小
plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
plt.rcParams['axes.titlesize'] = 24  # 图表标题字体大小
plt.rcParams['xtick.labelsize'] = 24  # x轴刻度标签字体大小（增大2号）
plt.rcParams['ytick.labelsize'] = 24  # y轴刻度标签字体大小（增大2号）
plt.rcParams['legend.fontsize'] = 22  # 图例字体大小（增大2号）
plt.rcParams['figure.dpi'] = 100  # 图形显示分辨率
plt.rcParams['savefig.dpi'] = 300  # 保存图形分辨率
plt.rcParams['savefig.bbox'] = 'tight'  # 保存时自动裁剪空白
plt.rcParams['axes.grid'] = True  # 显示网格
plt.rcParams['grid.alpha'] = 0.25  # 网格透明度
plt.rcParams['grid.linestyle'] = '--'  # 网格线样式
plt.rcParams['axes.axisbelow'] = True  # 网格在图层下方
plt.rcParams['xtick.direction'] = 'in'  # x轴刻度线方向向内
plt.rcParams['ytick.direction'] = 'in'  # y轴刻度线方向向内
plt.rcParams['xtick.major.size'] = 5  # x轴主刻度线长度
plt.rcParams['ytick.major.size'] = 5  # y轴主刻度线长度


# Focal Loss for handling class imbalance in peak detection
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Focal Loss for binary classification (expects logits).
        
        Args:
            alpha: Weight factor for balancing positive/negative samples (0.25 means 25% positive)
            gamma: Focusing parameter, >=0. Higher values focus more on hard samples
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        p = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, p, 1 - p)
        
        focal_weight = (1 - pt).pow(self.gamma)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_t * focal_weight * BCE_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def detect_peaks_findpeaks(data: Union[np.ndarray, torch.Tensor], 
                           method: str = 'peakdetect', 
                           lookahead: int = 10, 
                           limit: Optional[float] = None) -> np.ndarray:
    """
    使用findpeaks库检测波峰
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if len(data.shape) > 1:
        data = data.flatten()
    
    fp = findpeaks(method=method, lookahead=lookahead, limit=limit)
    results = fp.fit(data)
    
    if results and 'df' in results and results['df'] is not None:
        return results['df'][results['df']['peak'] == True].index.values
    
    return np.array([])


def match_peaks_with_tolerance(true_peaks: np.ndarray, 
                               pred_peaks: np.ndarray, 
                               tolerance: int = 5) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    使用容忍窗口匹配真实峰值和预测峰值
    
    容错范围：对于真实峰值 t_idx，预测峰值需要在 [t_idx - tolerance, t_idx + tolerance] 范围内
    (注意：您的原始代码实现是对称窗口，我们遵循代码实现)
    """
    tp_pairs = []
    matched_true = set()
    matched_pred = set()

    # <--- 修改 (语法错误修复, 添加 #)
    # 为了获得最佳匹配, 最好先对 true_peaks 排序
    true_peaks_sorted = np.sort(true_peaks)
    
    for t_idx in true_peaks_sorted:
        if t_idx in matched_true:
            continue
            
        candidates = pred_peaks[np.abs(pred_peaks - t_idx) <= tolerance]
        
        if len(candidates) > 0:
            unmatched_candidates = [p for p in candidates if p not in matched_pred]
            
            if len(unmatched_candidates) > 0:
                best_pred = unmatched_candidates[np.argmin(np.abs(unmatched_candidates - t_idx))]
                
                tp_pairs.append((int(t_idx), int(best_pred)))
                matched_true.add(t_idx)
                matched_pred.add(best_pred)
    
    fp = [int(p) for p in pred_peaks if p not in matched_pred]
    fn = [int(t) for t in true_peaks if t not in matched_true]
    
    return tp_pairs, fp, fn


def calculate_peak_metrics(tp_pairs: List[Tuple[int, int]], 
                           fp: List[int], 
                           fn: List[int], 
                           true_values: np.ndarray, 
                           pred_values: np.ndarray) -> Dict[str, Any]:
    """
    计算峰值检测的评估指标
    """
    tp_count = len(tp_pairs)
    fp_count = len(fp)
    fn_count = len(fn)
    
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    mae, mse = 0.0, 0.0
    if tp_count > 0:
        true_peak_values = true_values[[t for t, p in tp_pairs]]
        pred_peak_values = pred_values[[p for t, p in tp_pairs]]
        errors = true_peak_values - pred_peak_values
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
    
    # 计算整体 MAE 和 MSE (所有点，不只是TP)
    overall_mae = np.mean(np.abs(pred_values - true_values))
    overall_mse = np.mean((pred_values - true_values) ** 2)
    
    # 计算 R² (coefficient of determination)
    ss_res = np.sum((true_values - pred_values) ** 2)
    ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
    
    # 计算混合指标 BPE (Balanced Peak Error)
    alpha = 0.5
    e_class = 1.0 - f1_score
    e_reg = 1.0 - (1.0 / (1.0 + mse))
    balanced_peak_error = (alpha * e_class) + ((1.0 - alpha) * e_reg)
    
    # PIM (Peak Integrated Metric)
    peak_pim = (1.0 + mse) / (f1_score + 0.01)
    
    return {
        'TP': tp_count, 'FP': fp_count, 'FN': fn_count,
        'Precision': precision, 'Recall': recall, 'F1_Score': f1_score,
        'Peak_MAE': mae, 'Peak_MSE': mse,
        'Overall_MAE': overall_mae, 'Overall_MSE': overall_mse,
        'R2': r2,
        'Balanced_Peak_Error': balanced_peak_error, 'Peak_PIM': peak_pim
    }


def _condense_peak_indices(pred_binary: np.ndarray, pred_probs: np.ndarray) -> np.ndarray: # <--- 新增辅助函数
    """
    将连续的 [0, 1, 1, 1, 0] 块转换为该块中概率最高的单个索引.
            [0, 0.6, 0.7, 0.5, 0]
            [0, 0, 1, 0, 0]
    这解决了将一个 "峰值块" 误报为多个 FP 的问题.
    """
    condensed_indices = []
    i = 0
    n = len(pred_binary)
    while i < n:
        if pred_binary[i] == 1:
            start_idx = i
            # 寻找这个 [1, 1, 1] 块的末尾
            while i < n and pred_binary[i] == 1:
                i += 1
            end_idx = i # end_idx 是块之后的第一个 0
            
            # 在 [start_idx, end_idx) 范围内找到概率最高的索引
            if start_idx < end_idx:
                block_probs = pred_probs[start_idx:end_idx]
                best_relative_idx = np.argmax(block_probs)
                absolute_idx = start_idx + best_relative_idx
                condensed_indices.append(absolute_idx)
        else:
            i += 1
    return np.array(condensed_indices)


def calculate_peak_classification_metrics(peak_preds: np.ndarray, 
                                          peak_trues: np.ndarray, 
                                          value_preds: np.ndarray, # <--- 修改 (新增)
                                          value_trues: np.ndarray, # <--- 修改 (新增)
                                          threshold: float = 0.5,
                                          tolerance: int = 5,
                                          fn_penalty_weight: float = 2.0,
                                          fp_penalty_weight: float = 1.0,
                                          alpha: float = 0.5) -> Dict[str, Any]: # <--- 新增 alpha
    """
    (重构) 计算peak二分类的评估指标
    - 使用 _condense_peak_indices 来合并预测的 peak“块”
    - 使用 match_peaks_with_tolerance 来计算 TP/FP/FN
    - (新增) 计算 TP_MSE 和 TP_MAE
    """
    
    # 不再 flatten, 逐个样本处理
    n_samples = peak_preds.shape[0]
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_tp_errors = []  # 仅用于 TP
    all_fn_errors = []  # 仅用于 FN
    all_fp_errors = []  # (新增) 仅用于 FP
    all_true_peak_errors = []  # (新增) 所有真实峰值的误差 (TP + FN)
    
    for i in range(n_samples):
        true_seq = peak_trues[i, :, 0].astype(int)
        pred_seq_prob = peak_preds[i, :, 0]
        value_true_seq = value_trues[i, :, 0]
        value_pred_seq = value_preds[i, :, 0]
        
        # 1. 二值化
        pred_seq_binary = (pred_seq_prob >= threshold).astype(int)
        
        # 2. 获取真实 peak 索引
        true_indices = np.where(true_seq == 1)[0]
        
        # 3. 获取 "压缩" 后的预测 peak 索引
        pred_indices_condensed = _condense_peak_indices(pred_seq_binary, pred_seq_prob)
        
        # 4. 使用您现有的容忍度匹配函数
        tp_pairs, fp_list, fn_list = match_peaks_with_tolerance(
            true_indices, 
            pred_indices_condensed, 
            tolerance=tolerance
        )
        
        total_tp += len(tp_pairs)
        total_fp += len(fp_list)
        total_fn += len(fn_list)

        # 5. (现有) 计算 TP MSE/MAE
        if len(tp_pairs) > 0:
            # Get indices from pairs: (true_idx, pred_idx)
            true_tp_indices = [t for t, p in tp_pairs]
            pred_tp_indices = [p for t, p in tp_pairs]
            
            # Get values at these indices
            true_tp_values = value_true_seq[true_tp_indices]
            pred_tp_values = value_pred_seq[pred_tp_indices]  # Get value at the *predicted* point
            
            # Calculate errors
            errors = true_tp_values - pred_tp_values
            all_tp_errors.extend(errors)
            
            # (新增) 也加入到所有真实峰值误差中
            all_true_peak_errors.extend(errors)
            
        # 6. (新增) 计算 FN MSE/MAE
        if len(fn_list) > 0:
            # fn_list 包含的是真实峰值的索引 (模型没有检测到的)
            fn_true_values = value_true_seq[fn_list]
            # 获取在 *相同索引* 处的预测值 (模型在该位置的预测)
            fn_pred_values = value_pred_seq[fn_list]
            
            # 计算 FN 误差 (真实峰值的值 - 对应位置的预测值)
            fn_errors = fn_true_values - fn_pred_values
            all_fn_errors.extend(fn_errors)
            
            # (新增) 也加入到所有真实峰值误差中
            all_true_peak_errors.extend(fn_errors)
            
        # 7. (新增) 计算 FP MSE/MAE
        if len(fp_list) > 0:
            # fp_list 包含的是"误报"的预测峰值索引 (真实没有峰值)
            fp_pred_values = value_pred_seq[fp_list]
            # 获取在 *相同索引* 处的真实值 (应该是非峰值/波谷)
            fp_true_values = value_true_seq[fp_list]
            
            # 计算 FP 误差 (预测的高值 - 真实的低值，会很大)
            fp_errors = fp_pred_values - fp_true_values
            all_fp_errors.extend(fp_errors)
    
    # --- 扩展的指标计算 ---
    
    # 计算总指标 (F1, P, R)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 计算平均值 (TP, FP, FN)
    avg_tp = total_tp / n_samples if n_samples > 0 else 0.0
    avg_fp = total_fp / n_samples if n_samples > 0 else 0.0
    avg_fn = total_fn / n_samples if n_samples > 0 else 0.0
    
    # (现有) 计算 仅TP 的 MSE/MAE
    tp_count = len(all_tp_errors)
    if tp_count > 0:
        all_tp_errors_np = np.array(all_tp_errors)
        tp_mse = np.mean(all_tp_errors_np ** 2)
        tp_mae = np.mean(np.abs(all_tp_errors_np))
    else:
        all_tp_errors_np = np.array([])  # 初始化为空数组
        tp_mse = 0.0
        tp_mae = 0.0
    
    # (新增) 计算 仅FN 的 MSE/MAE
    fn_count = len(all_fn_errors)
    if fn_count > 0:
        all_fn_errors_np = np.array(all_fn_errors)
        fn_mse = np.mean(all_fn_errors_np ** 2)
        fn_mae = np.mean(np.abs(all_fn_errors_np))
    else:
        all_fn_errors_np = np.array([])  # 初始化为空数组
        fn_mse = 0.0
        fn_mae = 0.0

    # (新增) 计算 仅FP 的 MSE/MAE
    fp_count = len(all_fp_errors)
    if fp_count > 0:
        all_fp_errors_np = np.array(all_fp_errors)
        fp_mse = np.mean(all_fp_errors_np ** 2)
        fp_mae = np.mean(np.abs(all_fp_errors_np))
    else:
        all_fp_errors_np = np.array([])  # 初始化为空数组
        fp_mse = 0.0
        fp_mae = 0.0
    
    # (新增) 计算所有真实峰值的 MSE/MAE (包含TP和FN)
    # 这个指标衡量的是：对于所有真实存在的峰值，模型的预测误差
    true_peak_count = len(all_true_peak_errors)
    if true_peak_count > 0:
        all_true_peak_errors_np = np.array(all_true_peak_errors)
        all_true_peaks_mse = np.mean(all_true_peak_errors_np ** 2)
        all_true_peaks_mae = np.mean(np.abs(all_true_peak_errors_np))
    else:
        all_true_peaks_mse = 0.0
        all_true_peaks_mae = 0.0

    # (新增) 计算综合性混合指标 (Comprehensive: TP + FN + FP)
    # 公式: Comprehensive = (TP*w_tp + FN*w_fn + FP*w_fp) / (w_tp + w_fn + w_fp)
    # 注意: 分母是权重总和，因为这是加权平均
    total_comprehensive_mse_sum = (tp_mse * tp_count) + \
                                  (fn_mse * fn_count * fn_penalty_weight) + \
                                  (fp_mse * fp_count * fp_penalty_weight)
                                  
    total_comprehensive_weight = tp_count + \
                                 (fn_count * fn_penalty_weight) + \
                                 (fp_count * fp_penalty_weight)
                                 
    if total_comprehensive_weight > 0:
        comprehensive_mse = total_comprehensive_mse_sum / total_comprehensive_weight
        comprehensive_mae = ((tp_mae * tp_count) + \
                            (fn_mae * fn_count * fn_penalty_weight) + \
                            (fp_mae * fp_count * fp_penalty_weight)) / total_comprehensive_weight
    else:
        comprehensive_mse = 0.0
        comprehensive_mae = 0.0
    
    # (新增) 计算考虑样本数量的加权总误差
    # 1. 首先获取总误差 (Sum of Squared Errors)
    if tp_count > 0:
        tp_total_sq_error = np.sum(all_tp_errors_np ** 2)  # TP_MSE * tp_count
    else:
        tp_total_sq_error = 0.0

    if fn_count > 0:
        fn_total_sq_error = np.sum(all_fn_errors_np ** 2)  # FN_MSE * fn_count
    else:
        fn_total_sq_error = 0.0

    if fp_count > 0:
        fp_total_sq_error = np.sum(all_fp_errors_np ** 2)  # FP_MSE * fp_count
    else:
        fp_total_sq_error = 0.0
    
    # 2. 新指标: Weighted_Total_MSE (加权总误差)
    weighted_total_mse = (1.0 * tp_total_sq_error) + \
                         (fn_penalty_weight * fn_total_sq_error) + \
                         (fp_penalty_weight * fp_total_sq_error)
    
    # MAE 版本: Weighted_Total_MAE
    if tp_count > 0:
        tp_total_abs_error = np.sum(np.abs(all_tp_errors_np))  # TP_MAE * tp_count
    else:
        tp_total_abs_error = 0.0

    if fn_count > 0:
        fn_total_abs_error = np.sum(np.abs(all_fn_errors_np))  # FN_MAE * fn_count
    else:
        fn_total_abs_error = 0.0

    if fp_count > 0:
        fp_total_abs_error = np.sum(np.abs(all_fp_errors_np))  # FP_MAE * fp_count
    else:
        fp_total_abs_error = 0.0
        
    weighted_total_mae = (1.0 * tp_total_abs_error) + \
                         (fn_penalty_weight * fn_total_abs_error) + \
                         (fp_penalty_weight * fp_total_abs_error)

    # 3. 新指标: Proportional_Weighted_MSE (比例加权平均误差)
    total_peak_events = tp_count + fn_count + fp_count
    
    if total_peak_events > 0:
        proportional_weighted_mse = weighted_total_mse / total_peak_events
        proportional_weighted_mae = weighted_total_mae / total_peak_events
    else:
        # 没有峰值事件 (没有 TP, FN, FP), 误差为 0
        proportional_weighted_mse = 0.0
        proportional_weighted_mae = 0.0
        
    # (现有) 计算整体 MAE 和 MSE (所有点，不只是TP)
    overall_mae = np.mean(np.abs(value_preds - value_trues))
    overall_mse = np.mean((value_preds - value_trues) ** 2)

    # (现有) 我们仍然可以计算 BCE 用于监控
    preds_flat = peak_preds.flatten()
    trues_flat = peak_trues.flatten()
    epsilon = 1e-7
    bce = -np.mean(trues_flat * np.log(preds_flat + epsilon) + 
                   (1 - trues_flat) * np.log(1 - preds_flat + epsilon))
    
    # --- (新增) BPE (Balanced Peak Error) 计算 ---
    # 将 F1 和 TP_MSE 统一到 [0, 1] 误差空间
    e_class = 1.0 - f1  # 分类误差: F1=1.0 (完美) -> e_class=0.0
    e_reg = 1.0 - (1.0 / (1.0 + tp_mse))  # 标准化回归误差: TP_MSE=0.0 -> e_reg=0.0
    
    # 加权平衡: alpha控制F1和TP_MSE的重要性
    balanced_peak_error = (alpha * e_class) + ((1.0 - alpha) * e_reg)
    # --- 结束 BPE 计算 ---
    
    # --- (新增) PIM (Peak Integrated Metric) 计算 ---
    # 逻辑: PIM = (1 + TP_MSE) * (1 + (1 - F1_Score)) - 1.0
    # E_reg = 1.0 + tp_mse (回归误差项)
    # E_cls = 1.0 + (1.0 - f1) (检测误差项)
    # PIM 是一个乘法指标, 最小(完美)值为 1.0, 任何错误都会使其增加
    # peak_pim = (1.0 + tp_mse) * (1.0 + (1.0 - f1)) - 1.0
    peak_pim = (1.0 + tp_mse) / (f1 + 0.01)
    # --- 结束 PIM 计算 ---
    
    return {
        'Peak_Cls_TP': avg_tp,
        'Peak_Cls_FP': avg_fp,
        'Peak_Cls_FN': avg_fn,
        'Peak_Cls_Precision': precision,
        'Peak_Cls_Recall': recall,
        'Peak_Cls_F1': f1,
        'Peak_Cls_TP_MSE': tp_mse,
        'Peak_Cls_TP_MAE': tp_mae,
        'Peak_Cls_FN_MSE': fn_mse,  # (新增) FN的MSE
        'Peak_Cls_FN_MAE': fn_mae,  # (新增) FN的MAE
        'Peak_Cls_All_True_Peaks_MSE': all_true_peaks_mse,  # (新增) 所有真实峰值的MSE
        'Peak_Cls_All_True_Peaks_MAE': all_true_peaks_mae,  # (新增) 所有真实峰值的MAE
        'Peak_Cls_Balanced_Error': balanced_peak_error,
        'Peak_Cls_PIM': peak_pim,
        'Peak_Cls_MAE': overall_mae,
        'Peak_Cls_MSE': overall_mse,
        'Peak_Cls_BCE': bce
    }


def _process_single_sample_peaks(args: Tuple) -> Dict[str, Any]:
    """
    处理单个样本的峰值检测（用于多进程）
    """
    sample_idx, pred_seq, true_seq, tolerance, method, lookahead = args
    
    pred_peaks = detect_peaks_findpeaks(pred_seq, method=method, lookahead=lookahead)
    true_peaks = detect_peaks_findpeaks(true_seq, method=method, lookahead=lookahead)
    
    tp_pairs, fp, fn = match_peaks_with_tolerance(true_peaks, pred_peaks, tolerance=tolerance)
    metrics = calculate_peak_metrics(tp_pairs, fp, fn, true_seq, pred_seq)
    metrics['sample_idx'] = sample_idx
    
    return metrics


def process_peaks_multiprocess(preds: np.ndarray, 
                               trues: np.ndarray, 
                               tolerance: int = 5, 
                               method: str = 'peakdetect', 
                               lookahead: int = 10, 
                               num_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    使用多进程处理峰值检测
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    
    n_samples = preds.shape[0]
    args_list = [
        (i, preds[i, :, 0], trues[i, :, 0], tolerance, method, lookahead) 
        for i in range(n_samples)
    ]
    
    with Pool(processes=num_workers) as pool:
        peak_metrics_list = pool.map(_process_single_sample_peaks, args_list)
    
    return peak_metrics_list


def _process_single_visualization(args: Tuple) -> None:
    """
    处理单个样本的可视化（用于多进程）- 单图版本
    
    Args:
        args: (sample_idx, value_true_seq, value_pred_seq, peak_true_seq, peak_pred_seq,
               save_folder, epoch, data_flag, threshold, tolerance)
    """
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    
    (sample_idx, value_true_seq, value_pred_seq, peak_true_seq, peak_pred_seq,
     save_folder, epoch, data_flag, threshold, tolerance) = args
    
    # 创建单个图形 - 高度增加到1.3倍
    fig, ax = plt.subplots(1, 1, figsize=(18, 7.8))  # 6 * 1.3 = 7.8
    time_steps = np.arange(len(value_true_seq))
    
    # 绘制值预测曲线
    ax.plot(time_steps, value_true_seq, 'b-', label='True Value', linewidth=2, alpha=0.8)
    ax.plot(time_steps, value_pred_seq, 'r-', label='Predicted Value', linewidth=2, alpha=0.8)
    
    # 获取峰值匹配结果
    true_peak_indices = np.where(peak_true_seq > 0.5)[0]
    pred_binary = (peak_pred_seq > threshold).astype(float)
    pred_peak_indices_condensed = _condense_peak_indices(pred_binary, peak_pred_seq)
    tp_pairs, fp_indices, fn_indices = match_peaks_with_tolerance(
        true_peak_indices, pred_peak_indices_condensed, tolerance=tolerance
    )
    
    # 绘制True Peaks
    if len(true_peak_indices) > 0:
        ax.scatter(true_peak_indices, value_true_seq[true_peak_indices], 
                   c='green', s=120, marker='o', edgecolors='darkgreen', 
                   linewidths=2.5, label=f'True Peaks (n={len(true_peak_indices)})', 
                   zorder=5, alpha=0.8)
    
    # 绘制TP（不显示Predicted Peaks）
    tp_plot_indices_pred = [p for t, p in tp_pairs]
    if len(tp_plot_indices_pred) > 0:
        ax.scatter(tp_plot_indices_pred, value_pred_seq[tp_plot_indices_pred], 
                   c='lime', s=180, marker='*', edgecolors='green', 
                   linewidths=3, label=f'TP (n={len(tp_pairs)})', 
                   zorder=6, alpha=0.9)
    
    # 绘制FP
    if len(fp_indices) > 0:
        ax.scatter(fp_indices, value_pred_seq[fp_indices], 
                   c='red', s=150, marker='x', linewidths=3, 
                   label=f'FP (n={len(fp_indices)})', zorder=6)
    
    # 绘制FN
    if len(fn_indices) > 0:
        ax.scatter(fn_indices, value_true_seq[fn_indices], 
                   c='blue', s=150, marker='x', linewidths=3, 
                   label=f'FN (n={len(fn_indices)})', zorder=6)
    
    # 设置图例 - 紧凑布局，不显示坐标轴标签
    ax.legend(loc='best', ncol=3, prop={'weight': 'bold'}, 
             columnspacing=1.0, handletextpad=0.5, borderpad=0.4)
    ax.grid(True, alpha=0.3)
    
    # 刻度标签加粗
    plt.setp(ax.xaxis.get_majorticklabels(), weight='bold')
    plt.setp(ax.yaxis.get_majorticklabels(), weight='bold')
    
    # Tick marks inside the frame
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    
    plt.tight_layout()
    
    # 保存图像
    filename = f'{data_flag}_peak_cls_epoch{epoch}_sample{sample_idx}.png' if epoch else f'{data_flag}_peak_cls_sample{sample_idx}.png'
    save_path = os.path.join(save_folder, filename)
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    except Exception as e:
        print(f"Warning: Failed to save visualization to {save_path}: {e}")
    finally:
        plt.close()


# ==================== 实验类 ====================

class Exp_Peak_Detect_LTF(Exp_Basic):
    def __init__(self, args):
        super(Exp_Peak_Detect_LTF, self).__init__(args)
        
        # 初始化可视化文件夹 - 使用 args.setting
        if getattr(self.args, 'enable_peak_eval', False) and getattr(self.args, 'vis_test_peaks', False):
            self.vis_folder = f'./visualization/output/predictions/{self.args.setting}/'
            if not os.path.exists(self.vis_folder):
                os.makedirs(self.vis_folder, exist_ok=True)
                print(f"Created visualization folder: {self.vis_folder}")
        else:
            self.vis_folder = None
        
        # 峰值检测参数
        self.peak_tolerance = getattr(self.args, 'peak_tolerance', 3) # <--- 这个参数现在会被用于F1计算        
        # Peak分类参数
        self.peak_threshold = getattr(self.args, 'peak_threshold', 0.4)  # 分类阈值
        self.use_soft_labels = getattr(self.args, 'use_soft_labels', True)  # 是否使用软标签
        self.soft_label_sigma = getattr(self.args, 'soft_label_sigma', 2.0)  # 高斯平滑的sigma
        self.soft_label_tolerance = getattr(self.args, 'soft_label_tolerance', 3)  # 平滑范围
        self.mask_type = getattr(self.args, 'mask_type', 'soft')  # 'soft' (Gaussian) or 'hard' (binary)
        
        # <--- 修改 (新增) ---
        # 损失权重
        self.value_loss_weight = getattr(self.args, 'value_loss_weight', 1.0)  # 原始MSE loss权重
        self.peak_loss_weight = getattr(self.args, 'peak_loss_weight', 1.0)    # Peak分类loss权重
        self.tp_mse_loss_weight = getattr(self.args, 'tp_mse_loss_weight', 0.0)  # TP MSE loss权重（索引loss）
        # --- 结束新增 ---

        print(f"Peak classification threshold: {self.peak_threshold}")
        print(f"Peak classification F1 tolerance: {self.peak_tolerance}") 
        print(f"Value Loss Weight: {self.value_loss_weight}")  # <--- 修改 (新增)
        print(f"Peak Cls Loss Weight: {self.peak_loss_weight}") # <--- 修改 (新增)
        print(f"TP MSE Loss Weight: {self.tp_mse_loss_weight}") # <--- 修改 (新增)
        
        # (新增) 为新的 BPE 指标添加 alpha 权重
        self.metric_alpha = getattr(self.args, 'metric_alpha', 0.5) 
        print(f"Metric Balance Alpha (F1 vs TP_MSE): {self.metric_alpha}")
        
        if self.use_soft_labels:
            print(f"Using soft labels: sigma={self.soft_label_sigma}, tolerance={self.soft_label_tolerance}")
        else:
            print(f"Using hard labels (FocalLoss or BCEWithLogitsLoss)")


    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag: str):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        """
        (润色) 选择损失函数
        """
        loss_type = getattr(self.args, 'loss', 'MSE').upper()
        loss_map = {
            'MSE': nn.MSELoss(),
            'MAE': nn.L1Loss(),
            'WASSERSTEIN': WassersteinLoss(p=1),
            'ENERGY': EnergyLoss(),
            'COMBINED_MSE_WASS': CombinedLoss(alpha=getattr(self.args, 'loss_alpha', 0.5), loss_type='mse+wasserstein'),
            'COMBINED_MAE_WASS': CombinedLoss(alpha=getattr(self.args, 'loss_alpha', 0.5), loss_type='mae+wasserstein'),
            'COMBINED_MSE_ENERGY': CombinedLoss(alpha=getattr(self.args, 'loss_alpha', 0.5), loss_type='mse+energy'),
        }
        
        if loss_type in loss_map:
            print(f"Using Loss: {loss_type}")
            return loss_map[loss_type]
        else:
            print(f"Warning: Unknown loss type '{loss_type}', using MSE as default")
            return nn.MSELoss()
    
    def _select_peak_criterion(self):
        """
        (修改) Peak分类任务的损失函数
        - 软标签模式: BCELoss (概率回归, 优于MSE)
        - 硬标签模式: FocalLoss (处理类别不平衡)
        """
        if self.use_soft_labels:
            return nn.BCELoss()
        else:
            # 硬标签：使用 Focal Loss
            # alpha 和 gamma 是可调超参数, 可以在 self.args 中设置
            alpha = getattr(self.args, 'focal_alpha', 0.25)
            gamma = getattr(self.args, 'focal_gamma', 2.0)
            print(f"Peak Loss: FocalLoss (for hard labels) with alpha={alpha}, gamma={gamma}")
            # FocalLoss 内部自动处理 logits, 无需 sigmoid
            return FocalLoss(alpha=alpha, gamma=gamma)
    
    # @staticmethod
    # def create_soft_peak_labels(hard_labels: np.ndarray, 
    #                             sigma: float = 1.0, 
    #                             tolerance: int = 3) -> np.ndarray:
    #     n_samples, seq_len, n_features = hard_labels.shape
    #     soft_labels = np.zeros_like(hard_labels, dtype=np.float32)
        
    #     for i in range(n_samples):
    #         for f in range(n_features):
    #             hard_label_seq = hard_labels[i, :, f]
    #             soft_label_seq = np.zeros_like(hard_label_seq, dtype=np.float32)
                
    #             peak_indices = np.where(hard_label_seq > 0.5)[0]
                
    #             for peak_idx in peak_indices:
    #                 start = max(0, peak_idx - tolerance)
    #                 end = min(seq_len, peak_idx + tolerance + 1)
                    
    #                 for t in range(start, end):
    #                     distance = abs(t - peak_idx)
    #                     weight = np.exp(-(distance ** 2) / (2 * sigma ** 2))
    #                     soft_label_seq[t] = max(soft_label_seq[t], weight)
                
    #             soft_labels[i, :, f] = soft_label_seq
        
    #     return soft_labels

    @staticmethod
    def create_soft_peak_labels(hard_labels: torch.Tensor, 
                                      sigma: float = 1.0, 
                                      tolerance: int = 3) -> torch.Tensor:
        """
        将硬peak标签转换为软标签（高斯平滑，使用 PyTorch 向量化在 GPU 上运行）

        Args:
            hard_labels: 硬标签 [B, T, C]，在 self.device 上的 PyTorch 张量
            sigma: 高斯核的标准差
            tolerance: 平滑的范围（±tolerance 个时间点）

        Returns:
            soft_labels: 软标签 [B, T, C]，在 self.device 上的 PyTorch 张量
        """
        n_samples, seq_len, n_features = hard_labels.shape
        device = hard_labels.device
        
        # 1. 创建位置网格 [1, T]
        positions = torch.arange(seq_len, device=device, dtype=torch.float32).view(1, seq_len)
        
        soft_labels = torch.zeros_like(hard_labels, dtype=torch.float32)

        # 2. 遍历 Batch 和 Channel (这部分循环开销很小)
        for i in range(n_samples):
            for f in range(n_features):
                hard_seq = hard_labels[i, :, f] # Shape: [T]
                
                # 3. 找到所有峰值索引 [P]
                # .nonzero(as_tuple=False) 返回 [[idx1], [idx2], ...]
                # .view(-1) 将其展平为 [idx1, idx2, ...]
                peak_indices = (hard_seq > 0.5).nonzero(as_tuple=False).view(-1) # Shape: [P]
                
                if peak_indices.numel() == 0:
                    continue # 如果没有峰值，跳过
                    
                # 4. 准备广播 [P, 1]
                peaks = peak_indices.view(-1, 1) 
                
                # 5. 计算距离矩阵 (广播: [1, T] - [P, 1] -> [P, T])
                dist = torch.abs(positions - peaks)
                
                # 6. 计算高斯权重 [P, T]
                weights = torch.exp(-(dist.pow(2)) / (2 * sigma ** 2))
                
                # 7. 应用容忍度窗口，窗口外的权重清零
                weights[dist > tolerance] = 0.0
                
                # 8. 沿峰值维度(dim=0)取最大值，得到每个时间点 t 的最终权重 [T]
                soft_seq, _ = torch.max(weights, dim=0)
                
                soft_labels[i, :, f] = soft_seq
                
        return soft_labels

    def _compute_peak_loss(self, peak_outputs: torch.Tensor, peak_targets_for_loss: torch.Tensor, 
                          peak_criterion: nn.Module) -> torch.Tensor:
        """
        (修改) 统一计算Peak分类损失
        
        Args:
            peak_outputs: Peak预测logits [B, T, 1]
            peak_targets_for_loss: Peak标签 [B, T, 1] (必须是已处理好的软标签或硬标签)
            peak_criterion: Peak损失函数
        
        Returns:
            peak_loss: 计算得到的损失值
        """
        if self.use_soft_labels:
            # 软标签模式: BCELoss需要概率而非logits
            peak_outputs_prob = torch.sigmoid(peak_outputs)
            return peak_criterion(peak_outputs_prob, peak_targets_for_loss)
        else:
            # 硬标签模式: FocalLoss直接使用logits
            return peak_criterion(peak_outputs, peak_targets_for_loss)
    
    def _compute_tp_mse_loss(self, outputs: torch.Tensor, targets: torch.Tensor, 
                            peak_mask: torch.Tensor) -> torch.Tensor:
        """
        (新增) 计算TP点的MSE损失
        
        Args:
            outputs: 预测值 [B, T, 1]
            targets: 真实值 [B, T, 1]
            peak_mask: Peak掩码 [B, T, 1]，值为0或1 (硬标签) 或连续值 [0, 1] (软标签)
                      软标签模式下，峰值中心权重为1.0，容忍度内的点权重根据高斯函数衰减
        
        Returns:
            tp_mse_loss: TP点的加权平均MSE (硬标签模式) 或加权MSE (软标签模式)
        """
        all_squared_errors = F.mse_loss(outputs, targets, reduction='none')
        tp_squared_errors = all_squared_errors * peak_mask
        num_tp = torch.sum(peak_mask) + 1e-7
        return torch.sum(tp_squared_errors) / num_tp
    
    def _compute_weighted_loss(self, value_loss: Optional[float] = None, 
                              peak_loss: Optional[float] = None,
                              tp_mse_loss: Optional[float] = None) -> Optional[float]:
        """
        (新增) 计算加权总损失
        
        Args:
            value_loss: 值预测损失
            peak_loss: Peak分类损失
            tp_mse_loss: TP MSE损失
        
        Returns:
            weighted_loss: 加权后的总损失
        """
        if value_loss is None:
            return None
        
        weighted_loss = self.value_loss_weight * value_loss
        if peak_loss is not None:
            weighted_loss += self.peak_loss_weight * peak_loss
        if tp_mse_loss is not None:
            weighted_loss += self.tp_mse_loss_weight * tp_mse_loss
        return weighted_loss

    def _process_inference_batch(self, data_batch, criterion, peak_criterion):
        """处理单个推理批次"""
        batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, dec_inp = self._prepare_batch_data(data_batch)
        batch_y = batch_y.float()
        
        if self.args.use_amp:
            with autocast():
                model_out = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
        else:
            model_out = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
        
        if isinstance(model_out, tuple) and len(model_out) == 2:
            outputs, peak_outputs = model_out
        else:
            outputs = model_out
            peak_outputs = None

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y_targets = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        pred = outputs.detach()
        true = batch_y_targets.detach()

        value_loss = None
        if criterion:
            value_loss = criterion(pred, true)
        
        peak_loss = None
        tp_mse_loss = None
        peak_pred = None
        peak_true = None
        
        if peak_outputs is not None:
            hard_peak_targets = batch_y_mark[:, -self.args.pred_len:, -1:].to(self.device)
            peak_outputs = peak_outputs[:, :, f_dim:]
            
            if self.mask_type == 'hard':
                loss_mask = hard_peak_targets
            elif self.use_soft_labels:
                loss_mask = self.create_soft_peak_labels(
                    hard_peak_targets, 
                    self.soft_label_sigma, 
                    self.soft_label_tolerance
                )
            else:
                loss_mask = hard_peak_targets
            
            peak_loss = self._compute_peak_loss(peak_outputs, loss_mask, peak_criterion)
            peak_pred = torch.sigmoid(peak_outputs).detach().cpu().numpy()
            peak_true = hard_peak_targets.detach().cpu().numpy()
            
            if self.tp_mse_loss_weight > 0:
                tp_mse_loss = self._compute_tp_mse_loss(outputs, batch_y_targets, loss_mask)
        
        return pred, true, value_loss, peak_loss, tp_mse_loss, peak_pred, peak_true

    def _run_inference(self, data_loader, criterion: Optional[nn.Module] = None, 
                       inverse_transform: bool = False, data_flag: str = 'val') -> Tuple[Optional[float], np.ndarray, np.ndarray]:
        """
        统一的模型推理函数 (用于 vali 和 test)
        返回: (total_loss, value_loss, peak_loss, tp_mse_loss, preds, trues, peak_preds, peak_trues)
        """
        self.model.eval()
        total_weighted_loss = []
        total_value_loss = []
        total_peak_loss = []
        total_tp_mse_loss = []
        all_preds, all_trues = [], []
        all_peak_preds, all_peak_trues = [], []
        
        data_set = None
        if inverse_transform:
            data_set, _ = self._get_data(flag=data_flag)
        
        peak_criterion = self._select_peak_criterion()

        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                pred, true, value_loss, peak_loss, tp_mse_loss, peak_pred, peak_true = self._process_inference_batch(
                    data_batch, criterion, peak_criterion
                )
                
                if value_loss is not None:
                    total_value_loss.append(value_loss.item())
                if peak_loss is not None:
                    total_peak_loss.append(peak_loss.item())
                if tp_mse_loss is not None:
                    total_tp_mse_loss.append(tp_mse_loss.item())
                if peak_pred is not None:
                    all_peak_preds.append(peak_pred)
                    all_peak_trues.append(peak_true)
                
                if value_loss is not None:
                    weighted_loss = self._compute_weighted_loss(
                        value_loss.item(), 
                        peak_loss.item() if peak_loss is not None else None,
                        tp_mse_loss.item() if tp_mse_loss is not None else None
                    )
                    total_weighted_loss.append(weighted_loss)
                
                pred_np = pred.cpu().numpy()
                true_np = true.cpu().numpy()
                
                if inverse_transform and data_set.scale and self.args.inverse:
                    shape = true_np.shape
                    if pred_np.shape[-1] != true_np.shape[-1]:
                        pred_np = np.tile(pred_np, [1, 1, int(true_np.shape[-1] / pred_np.shape[-1])])
                    
                    pred_np = data_set.inverse_transform(pred_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    true_np = data_set.inverse_transform(true_np.reshape(shape[0] * shape[1], -1)).reshape(shape)

                all_preds.append(pred_np)
                all_trues.append(true_np)

        avg_weighted_loss = np.average(total_weighted_loss) if total_weighted_loss else None
        avg_value_loss = np.average(total_value_loss) if total_value_loss else None
        avg_peak_loss = np.average(total_peak_loss) if total_peak_loss else None
        avg_tp_mse_loss = np.average(total_tp_mse_loss) if total_tp_mse_loss else None
        
        preds_all = np.concatenate(all_preds, axis=0)
        trues_all = np.concatenate(all_trues, axis=0)
        
        peak_preds_all = np.concatenate(all_peak_preds, axis=0) if all_peak_preds else None
        peak_trues_all = np.concatenate(all_peak_trues, axis=0) if all_peak_trues else None
        
        return avg_weighted_loss, avg_value_loss, avg_peak_loss, avg_tp_mse_loss, preds_all, trues_all, peak_preds_all, peak_trues_all

    def _match_peaks_for_visualization(self, peak_true_seq: np.ndarray, 
                                       peak_pred_seq: np.ndarray,
                                       threshold: float) -> Tuple[np.ndarray, np.ndarray, List, List, List]:
        """
        (新增) 为可视化准备peak匹配结果
        
        Returns:
            true_peak_indices, pred_peak_indices_condensed, tp_pairs, fp_indices, fn_indices
        """
        # 获取真实 peak 索引
        true_peak_indices = np.where(peak_true_seq > 0.5)[0]
        
        # 获取压缩后的预测 peak 索引
        pred_binary = (peak_pred_seq > threshold).astype(float)
        pred_peak_indices_condensed = _condense_peak_indices(pred_binary, peak_pred_seq)
        
        # 使用容忍度匹配
        tp_pairs, fp_indices, fn_indices = match_peaks_with_tolerance(
            true_peak_indices,
            pred_peak_indices_condensed,
            tolerance=self.peak_tolerance
        )
        
        return true_peak_indices, pred_peak_indices_condensed, tp_pairs, fp_indices, fn_indices
    
    def _compute_peak_metrics_from_matches(self, tp_pairs: List, fp_indices: List, 
                                          fn_indices: List) -> Tuple[float, float, float]:
        """
        (新增) 从匹配结果计算Precision, Recall, F1
        
        Returns:
            precision, recall, f1
        """
        tp = len(tp_pairs)
        fp = len(fp_indices)
        fn = len(fn_indices)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1

    def _evaluate_peaks(self, preds: np.ndarray, trues: np.ndarray, 
                        peak_preds: Optional[np.ndarray] = None,
                        peak_trues: Optional[np.ndarray] = None,
                        visualize: bool = False, epoch: Optional[int] = None, 
                        data_flag: str = 'test', num_workers: Optional[int] = 8) -> Optional[Dict[str, float]]:
        """
        (修改) 统一的峰值评估函数 - 支持peak分类可视化，使用多进程加速
        
        Args:
            preds: 值预测结果 [N, T, 1]
            trues: 真实值 [N, T, 1]
            peak_preds: Peak分类预测概率 [N, T, 1] (可选)
            peak_trues: Peak分类真实标签 [N, T, 1] (可选)
            visualize: 是否可视化
            epoch: 当前epoch
            data_flag: 数据集标志 (train/val/test)
            num_workers: 进程数，默认为8
        """
        if not (hasattr(self.args, 'enable_peak_eval') and self.args.enable_peak_eval):
            return None
        
        # 如果提供了peak分类预测,使用新的可视化逻辑
        if peak_preds is not None and peak_trues is not None and visualize and self.vis_folder:
            
            # 可视化部分样本的peak分类结果
            n_samples = min(preds.shape[0], 10000)  # 最多可视化10000个样本
            
            # 准备需要可视化的样本索引（每168个样本可视化一次）
            sample_indices = [i for i in range(n_samples) if i % 168 == 0]
            
            if len(sample_indices) > 0:
                print(f"        Save folder: {self.vis_folder}")
                
                # 准备多进程参数列表
                args_list = [
                    (
                        sample_idx,
                        trues[sample_idx, :, 0],
                        preds[sample_idx, :, 0],
                        peak_trues[sample_idx, :, 0],
                        peak_preds[sample_idx, :, 0],
                        self.vis_folder,
                        epoch,
                        data_flag,
                        self.peak_threshold,
                        self.peak_tolerance
                    )
                    for sample_idx in sample_indices
                ]
                
                # 使用多进程处理可视化
                with Pool(processes=num_workers) as pool:
                    pool.map(_process_single_visualization, args_list)
        
        return None

    def _visualize_value_and_peaks(self, ax1, time_steps, value_true_seq, value_pred_seq, 
                                    peak_true_seq, peak_pred_seq, threshold, data_flag, sample_idx, epoch):
        """可视化值预测和峰值标注"""
        ax1.plot(time_steps, value_true_seq, 'b-', label='True Value', linewidth=1.5, alpha=0.7)
        ax1.plot(time_steps, value_pred_seq, 'r-', label='Predicted Value', linewidth=1.5, alpha=0.7)
        
        true_peak_indices, pred_peak_indices_condensed, tp_pairs, fp_indices, fn_indices = self._match_peaks_for_visualization(
            peak_true_seq, peak_pred_seq, threshold
        )
        
        if len(true_peak_indices) > 0:
            ax1.scatter(true_peak_indices, value_true_seq[true_peak_indices], 
                       c='green', s=120, marker='o', edgecolors='darkgreen', 
                       linewidths=2.5, label=f'True Peaks (n={len(true_peak_indices)})', 
                       zorder=5, alpha=0.8)
        
        if len(pred_peak_indices_condensed) > 0:
            ax1.scatter(pred_peak_indices_condensed, value_pred_seq[pred_peak_indices_condensed], 
                       c='orange', s=120, marker='^', edgecolors='darkorange', 
                       linewidths=2.5, label=f'Predicted Peaks (n={len(pred_peak_indices_condensed)})', 
                       zorder=5, alpha=0.8)

        tp_plot_indices_pred = [p for t, p in tp_pairs]
        if len(tp_plot_indices_pred) > 0:
            ax1.scatter(tp_plot_indices_pred, value_pred_seq[tp_plot_indices_pred], 
                       c='lime', s=180, marker='*', edgecolors='green', 
                       linewidths=3, label=f'TP (n={len(tp_pairs)})', 
                       zorder=6, alpha=0.9)
        if len(fp_indices) > 0:
            ax1.scatter(fp_indices, value_pred_seq[fp_indices], 
                       c='red', s=150, marker='x', linewidths=3, 
                       label=f'FP (n={len(fp_indices)})', zorder=6)
        if len(fn_indices) > 0:
            ax1.scatter(fn_indices, value_true_seq[fn_indices], 
                       c='blue', s=150, marker='x', linewidths=3, 
                       label=f'FN (n={len(fn_indices)})', zorder=6)
        
        ax1.set_ylabel('Value', fontsize=12)
        ax1.legend(loc='best', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Peak Classification [{data_flag.upper()}] - Sample {sample_idx} (Epoch {epoch})', 
                     fontsize=14, fontweight='bold')
        
        return tp_pairs, fp_indices, fn_indices

    def _visualize_peak_probability(self, ax2, time_steps, peak_pred_seq, peak_true_seq, 
                                     threshold, tp_pairs, fp_indices, fn_indices):
        """可视化峰值概率曲线"""
        ax2.plot(time_steps, peak_pred_seq, 'purple', label='Peak Probability (Predicted)', 
                linewidth=2, alpha=0.7)
        ax2.plot(time_steps, peak_true_seq, 'g--', label='Peak Label (Ground Truth)', 
                linewidth=1.5, alpha=0.6)
        ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Threshold ({threshold})', alpha=0.6)
        
        ax2.fill_between(time_steps, 0, peak_true_seq, 
                        color='green', alpha=0.2, label='True Peak Regions')
        pred_binary = (peak_pred_seq > threshold).astype(float)
        ax2.fill_between(time_steps, 0, pred_binary, 
                        color='orange', alpha=0.2, label='Predicted Peak Regions (Raw)')
        
        ax2.set_xlabel('Time Steps', fontsize=12)
        ax2.set_ylabel('Peak Probability', fontsize=12)
        ax2.set_ylim([-0.1, 1.1])
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        precision, recall, f1 = self._compute_peak_metrics_from_matches(tp_pairs, fp_indices, fn_indices)
        metrics_text = f'Tolerance={self.peak_tolerance} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}'
        ax2.text(0.5, -0.05, metrics_text, transform=ax2.transAxes, 
                fontsize=11, ha='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))

    def _visualize_peak_classification(self, value_true_seq: np.ndarray, value_pred_seq: np.ndarray,
                                       peak_true_seq: np.ndarray, peak_pred_seq: np.ndarray,
                                       save_folder: str, sample_idx: int, 
                                       epoch: Optional[int] = None, data_flag: str = 'val',
                                       threshold: float = 0.5) -> None:
        """可视化peak分类结果"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        time_steps = np.arange(len(value_true_seq))
        
        tp_pairs, fp_indices, fn_indices = self._visualize_value_and_peaks(
            ax1, time_steps, value_true_seq, value_pred_seq, 
            peak_true_seq, peak_pred_seq, threshold, data_flag, sample_idx, epoch
        )
        
        self._visualize_peak_probability(
            ax2, time_steps, peak_pred_seq, peak_true_seq, 
            threshold, tp_pairs, fp_indices, fn_indices
        )
        
        plt.tight_layout()
        
        filename = f'{data_flag}_peak_cls_epoch{epoch}_sample{sample_idx}.png' if epoch else f'{data_flag}_peak_cls_sample{sample_idx}.png'
        save_path = os.path.join(save_folder, filename)
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self._print_and_log(f"\tSaved: {save_path}")
        except Exception as e:
            print(f"Warning: Failed to save visualization to {save_path}: {e}")
        finally:
            plt.close()

    def _visualize_peak_detection(self, true_seq: np.ndarray, pred_seq: np.ndarray, 
                                   true_peaks: np.ndarray, pred_peaks: np.ndarray, 
                                   tp_pairs: List, fp: List, fn: List, 
                                   metrics: Dict, save_folder: str, 
                                   sample_idx: int, epoch: Optional[int] = None, 
                                   data_flag: str = 'val') -> None:
        """
        (保留) 可视化峰值检测结果 (findpeaks方法)
        """
        plt.figure(figsize=(14, 6))
        time_steps = np.arange(len(true_seq))
        
        plt.plot(time_steps, true_seq, 'b-', label='True', linewidth=1.5, alpha=0.7)
        plt.plot(time_steps, pred_seq, 'r-', label='Predicted', linewidth=1.5, alpha=0.7)
        
        if len(true_peaks) > 0:
            plt.scatter(true_peaks, true_seq[true_peaks.astype(int)], 
                       c='green', s=100, marker='o', edgecolors='darkgreen', 
                       linewidths=2, label='True Peaks', zorder=5)
        if len(pred_peaks) > 0:
            plt.scatter(pred_peaks, pred_seq[pred_peaks.astype(int)], 
                       c='orange', s=100, marker='^', edgecolors='darkorange', 
                       linewidths=2, label='Pred Peaks', zorder=5)
        
        if len(tp_pairs) > 0:
            tp_true_idx = [t for t, p in tp_pairs]
            plt.scatter(tp_true_idx, true_seq[tp_true_idx], 
                       c='lime', s=150, marker='o', edgecolors='green', 
                       linewidths=3, label=f'TP (n={len(tp_pairs)})', zorder=6, alpha=0.6)
        if len(fp) > 0:
            plt.scatter(fp, pred_seq[fp], 
                       c='red', s=150, marker='x', linewidths=3, 
                       label=f'FP (n={len(fp)})', zorder=6)
        if len(fn) > 0:
            plt.scatter(fn, true_seq[fn], 
                       c='blue', s=150, marker='x', linewidths=3, 
                       label=f'FN (n={len(fn)})', zorder=6)
        
        title = (f'Peak Detection [{data_flag.upper()}] - Sample {sample_idx} (Epoch {epoch})\n'
                 f'F1={metrics["F1_Score"]:.3f}, P={metrics["Precision"]:.3f}, R={metrics["Recall"]:.3f}, Peak_MAE={metrics["Peak_MAE"]:.3f}')
        plt.title(title, fontsize=13)
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'{data_flag}_epoch{epoch}_sample{sample_idx}.png' if epoch else f'{data_flag}_sample{sample_idx}.png'
        save_path = os.path.join(save_folder, filename)
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Failed to save visualization to {save_path}: {e}")
        finally:
            plt.close()

    def vali(self, vali_data, vali_loader, criterion: nn.Module, 
             evaluate_peaks: bool = False, visualize_peaks: bool = False, 
             epoch: Optional[int] = None, setting: Optional[str] = None, 
             data_flag: str = 'val') -> Tuple[float, Optional[float], Optional[float], Optional[float], Optional[Dict[str, float]]]:
        """
        验证/测试方法
        返回: (weighted_loss, value_loss, peak_loss, tp_mse_loss, peak_classification_metrics)
        """
        weighted_loss, value_loss, peak_loss, tp_mse_loss, all_preds, all_trues, peak_preds_all, peak_trues_all = self._run_inference(
            vali_loader, criterion, 
            inverse_transform=False,
            data_flag=data_flag
        )
        
        peak_cls_metrics = None
        if peak_preds_all is not None and peak_trues_all is not None:
            peak_cls_metrics = calculate_peak_classification_metrics(
                peak_preds_all, peak_trues_all, 
                all_preds, all_trues,
                threshold=self.peak_threshold,
                tolerance=self.peak_tolerance,
                fn_penalty_weight=2.0,
                fp_penalty_weight=1.0,
                alpha=self.metric_alpha
            )
        
        self.model.train()
        return weighted_loss, value_loss, peak_loss, tp_mse_loss, peak_cls_metrics, all_preds, all_trues, peak_preds_all, peak_trues_all

    def _prepare_batch_data(self, data_batch) -> Tuple:
        """准备批次数据"""
        batch_cycle = None
        batch_dec_ext = None
        use_external_features = getattr(self.args, 'use_external_features', 0)
        external_feature_mode = getattr(self.args, 'external_feature_mode', 'concat')
        use_decoder_external = use_external_features and external_feature_mode in ['concat', 'fusion', 'future_only']
        use_future_only_external = use_external_features and external_feature_mode in ['fusion', 'future_only']
        if len(data_batch) == 5:
            batch_x, batch_y, batch_x_mark, batch_y_mark, extra = data_batch
            if use_decoder_external:
                batch_dec_ext = extra.float().to(self.device)
            elif not use_external_features:
                batch_cycle = extra.int().to(self.device)
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = data_batch
        
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # Construct decoder input: load channel = [label_values, zeros], external features = full window
        dec_inp_load = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp_load = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp_load], dim=1).float().to(self.device)
        
        if batch_dec_ext is not None:
            if use_future_only_external:
                dec_ext = torch.zeros_like(batch_dec_ext)
                dec_ext[:, self.args.label_len:, :] = batch_dec_ext[:, self.args.label_len:, :]
            else:
                dec_ext = batch_dec_ext
            dec_inp = torch.cat([dec_inp_load, dec_ext], dim=-1)
        else:
            dec_inp = dec_inp_load
        
        return batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, dec_inp

    def _forward_model(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle):
        """模型前向传播"""
        if 'CycleNet' in self.args.model and batch_cycle is not None:
            return self.model(batch_x, batch_cycle)
        else:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def _compute_train_loss(self, model_out, batch_y, batch_y_mark, criterion, peak_criterion):
        """计算训练损失"""
        if isinstance(model_out, tuple) and len(model_out) == 2:
            outputs, peak_outputs = model_out
        else:
            outputs = model_out
            peak_outputs = None
        
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y_targets = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        
        value_loss = criterion(outputs, batch_y_targets)
        loss = self.value_loss_weight * value_loss
        
        cls_loss = None
        tp_mse_loss = None
        if peak_outputs is not None:
            peak_outputs = peak_outputs[:, :, f_dim:]
            hard_peak_targets = batch_y_mark[:, -self.args.pred_len:, -1:].to(self.device)
            
            if self.mask_type == 'hard':
                loss_mask = hard_peak_targets
            elif self.use_soft_labels:
                loss_mask = self.create_soft_peak_labels(
                    hard_peak_targets, 
                    self.soft_label_sigma, 
                    self.soft_label_tolerance
                )
            else:
                loss_mask = hard_peak_targets
            
            cls_loss = self._compute_peak_loss(peak_outputs, loss_mask, peak_criterion)
            loss = loss + self.peak_loss_weight * cls_loss
            
            if self.tp_mse_loss_weight > 0:
                tp_mse_loss = self._compute_tp_mse_loss(outputs, batch_y_targets, loss_mask)
                loss = loss + self.tp_mse_loss_weight * tp_mse_loss
        
        return loss, value_loss, cls_loss, tp_mse_loss, peak_outputs

    def _train_one_batch(self, data_batch, model_optim, criterion, peak_criterion, scaler):
        """训练单个批次"""
        model_optim.zero_grad()
        
        batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, dec_inp = self._prepare_batch_data(data_batch)
        
        if self.args.use_amp:
            with autocast():
                model_out = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                loss, value_loss, cls_loss, tp_mse_loss, peak_outputs = self._compute_train_loss(
                    model_out, batch_y, batch_y_mark, criterion, peak_criterion
                )
            scaler.scale(loss).backward()
            scaler.step(model_optim)
            scaler.update()
        else:
            model_out = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
            loss, value_loss, cls_loss, tp_mse_loss, peak_outputs = self._compute_train_loss(
                model_out, batch_y, batch_y_mark, criterion, peak_criterion
            )
            loss.backward()
            model_optim.step()
        
        return loss, value_loss, cls_loss, tp_mse_loss, peak_outputs

    def _train_one_epoch(self, train_loader, model_optim, criterion, peak_criterion, scaler, epoch):
        """训练一个epoch"""
        iter_count = 0
        train_loss = []
        train_value_loss = []
        train_peak_loss = []
        train_tp_mse_loss = []

        self.model.train()
        epoch_time = time.time()
        
        for i, data_batch in enumerate(train_loader):
            iter_count += 1
            
            loss, value_loss, cls_loss, tp_mse_loss, peak_outputs = self._train_one_batch(
                data_batch, model_optim, criterion, peak_criterion, scaler
            )
            
            train_loss.append(loss.item())
            train_value_loss.append(value_loss.item())
            if cls_loss is not None:
                train_peak_loss.append(cls_loss.item())
            if tp_mse_loss is not None:
                train_tp_mse_loss.append(tp_mse_loss.item())

            if (i + 1) % 1000 == 0:
                speed = (time.time() - epoch_time) / iter_count
                loss_str = f"loss: {loss.item():.7f} (val: {value_loss.item():.7f}×{self.value_loss_weight}"
                if peak_outputs is not None and cls_loss is not None:
                    loss_str += f", peak: {cls_loss.item():.7f}×{self.peak_loss_weight}"
                if self.tp_mse_loss_weight > 0 and tp_mse_loss is not None:
                    loss_str += f", tpmse: {tp_mse_loss.item():.7f}×{self.tp_mse_loss_weight}"
                loss_str += ")"
                print(f"\titers: {i + 1}, epoch: {epoch + 1}")
                print(f"\t{loss_str} | speed: {speed:.4f}s/iter")
                iter_count = 0

        epoch_cost_time = time.time() - epoch_time
        train_loss_avg = np.average(train_loss)
        train_value_loss_avg = np.average(train_value_loss)
        train_peak_loss_avg = np.average(train_peak_loss) if train_peak_loss else None
        train_tp_mse_loss_avg = np.average(train_tp_mse_loss) if train_tp_mse_loss else None
        
        return epoch_cost_time, train_loss_avg, train_value_loss_avg, train_peak_loss_avg, train_tp_mse_loss_avg

    def _evaluate_all_splits(self, train_data, train_loader, vali_data, vali_loader, 
                             test_data, test_loader, criterion, setting, epoch):
        """评估所有数据集"""
        eval_peak_args = getattr(self.args, 'enable_peak_eval', False)
        eval_val_peaks = getattr(self.args, 'eval_val_peaks', True) and eval_peak_args
        eval_test_peaks = getattr(self.args, 'eval_test_peaks', True) and eval_peak_args
        
        vis_val_peaks = getattr(self.args, 'vis_val_peaks', False)
        vis_test_peaks = getattr(self.args, 'vis_test_peaks', True)

        vali_results = self.vali(
            vali_data, vali_loader, criterion,
            evaluate_peaks=eval_val_peaks,
            visualize_peaks=False,
            epoch=epoch+1, setting=setting, data_flag='val'
        )
        vali_weighted_loss, vali_value_loss, vali_peak_loss, vali_tp_mse_loss, vali_peak_cls_metrics, \
        vali_preds, vali_trues, vali_peak_preds, vali_peak_trues = vali_results
        
        test_results = self.vali(
            test_data, test_loader, criterion,
            evaluate_peaks=eval_test_peaks,
            visualize_peaks=False,
            epoch=epoch+1, setting=setting, data_flag='test'
        )
        test_weighted_loss, test_value_loss, test_peak_loss, test_tp_mse_loss, test_peak_cls_metrics, \
        test_preds, test_trues, test_peak_preds, test_peak_trues = test_results
        
        return (vali_weighted_loss, vali_value_loss, vali_peak_loss, vali_tp_mse_loss, vali_peak_cls_metrics,
                test_weighted_loss, test_value_loss, test_peak_loss, test_tp_mse_loss, test_peak_cls_metrics,
                vali_preds, vali_trues, vali_peak_preds, vali_peak_trues,
                test_preds, test_trues, test_peak_preds, test_peak_trues)

    def _select_early_stopping_metric(self, vali_peak_cls_metrics, vali_peak_loss, vali_tp_mse_loss):
        """选择early stopping指标"""
        if vali_peak_cls_metrics is not None and 'Peak_Cls_Balanced_Error' in vali_peak_cls_metrics:
            return vali_peak_cls_metrics['Peak_Cls_Balanced_Error']
        else:
            vali_peak_based_loss = 0.0
            if vali_peak_loss is not None:
                vali_peak_based_loss += self.peak_loss_weight * vali_peak_loss
            if vali_tp_mse_loss is not None:
                vali_peak_based_loss += self.tp_mse_loss_weight * vali_tp_mse_loss
            return vali_peak_based_loss

    def train(self, setting: str):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # [SCALING INSTRUMENTATION] — param count + GPU peak memory reset
        try:
            _num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            _param_mb = _num_params * 4 / 1024 / 1024
            self.write_log(f"[PARAMS] Trainable: {_num_params:,}  ({_param_mb:.2f} MB fp32)")
            print(f"[PARAMS] Trainable: {_num_params:,}  ({_param_mb:.2f} MB fp32)")
        except Exception as _e:
            self.write_log(f"[PARAMS] failed to count: {_e}")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=False)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        peak_criterion = self._select_peak_criterion()
        scaler = GradScaler() if self.args.use_amp else None
        
        print("\n" + "="*80)
        print("MODEL SELECTION STRATEGY:")
        print("  Primary metric: BPE (Balanced Peak Error) - Lower is better")
        print(f"  BPE formula: {self.metric_alpha:.1f}×(1-F1) + {1-self.metric_alpha:.1f}×(1-1/(1+TP_MSE))")
        print(f"  Peak Loss: {'BCELoss (soft labels)' if self.use_soft_labels else 'FocalLoss (hard labels)'}")
        print("  Fallback: Peak-based Loss (if BPE unavailable)")
        print("="*80 + "\n")

        for epoch in range(self.args.train_epochs):
            epoch_cost_time, train_loss_avg, train_value_loss_avg, train_peak_loss_avg, train_tp_mse_loss_avg = self._train_one_epoch(
                train_loader, model_optim, criterion, peak_criterion, scaler, epoch
            )
            
            eval_results = self._evaluate_all_splits(
                train_data, train_loader, vali_data, vali_loader, 
                test_data, test_loader, criterion, setting, epoch
            )
            
            vali_weighted_loss, vali_value_loss, vali_peak_loss, vali_tp_mse_loss, vali_peak_cls_metrics, \
            test_weighted_loss, test_value_loss, test_peak_loss, test_tp_mse_loss, test_peak_cls_metrics, \
            vali_preds, vali_trues, vali_peak_preds, vali_peak_trues, \
            test_preds, test_trues, test_peak_preds, test_peak_trues = eval_results

            self._log_epoch_results(
                epoch, epoch_cost_time,
                train_metrics=(train_loss_avg, train_value_loss_avg, train_peak_loss_avg, train_tp_mse_loss_avg),
                vali_metrics=(vali_weighted_loss, vali_value_loss, vali_peak_loss, vali_tp_mse_loss, vali_peak_cls_metrics),
                test_metrics=(test_weighted_loss, test_value_loss, test_peak_loss, test_tp_mse_loss, test_peak_cls_metrics)
            )

            vali_metric = self._select_early_stopping_metric(vali_peak_cls_metrics, vali_peak_loss, vali_tp_mse_loss)
            
            prev_best = early_stopping.val_loss_min
            early_stopping(vali_metric, self.model, path)
            
            self._print_and_log("[OTHERS]")
            self._print_and_log(" [MODEL SELECTION]")
            metric_name = "BPE" if vali_peak_cls_metrics is not None else "Peak-based Loss"
            
            if early_stopping.early_stop:
                self._print_and_log(f"\t{metric_name}: {vali_metric:.6f}")
                self._print_and_log("\tEarly stopping triggered!")
            elif early_stopping.val_loss_min == vali_metric:
                self._print_and_log(f"\t{metric_name} improved ({prev_best:.6f} --> {vali_metric:.6f}). Saving model...")
            else:
                self._print_and_log(f"\t{metric_name}: {vali_metric:.6f} (best: {prev_best:.6f})")
                self._print_and_log(f"\tEarlyStopping counter: {early_stopping.counter}/{early_stopping.patience}")
            
            vis_val_peaks = getattr(self.args, 'vis_val_peaks', False)
            vis_test_peaks = getattr(self.args, 'vis_test_peaks', True)
            if vis_val_peaks or vis_test_peaks:
                self._print_and_log(" [VISUALIZATION]")
                
                if vis_val_peaks and vali_peak_preds is not None:
                    self._evaluate_peaks(
                        vali_preds, vali_trues,
                        peak_preds=vali_peak_preds,
                        peak_trues=vali_peak_trues,
                        visualize=True,
                        epoch=epoch+1,
                        data_flag='val'
                    )
                
                if vis_test_peaks and test_peak_preds is not None:
                    self._evaluate_peaks(
                        test_preds, test_trues,
                        peak_preds=test_peak_preds,
                        peak_trues=test_peak_trues,
                        visualize=True,
                        epoch=epoch+1,
                        data_flag='test'
                    )
            
            self._print_and_log("=" * 56)
            
            if early_stopping.early_stop:
                break
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # [SCALING INSTRUMENTATION] — log peak GPU memory
        if torch.cuda.is_available():
            _peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            self.write_log(f"[MEMORY] Peak GPU memory during train: {_peak_mb:.1f} MB")
            print(f"[MEMORY] Peak GPU memory during train: {_peak_mb:.1f} MB")

        best_model_path = path + '/' + 'checkpoint.pth'
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda(torch.cuda.current_device())
        else:
            map_location = 'cpu'
        self.model.load_state_dict(torch.load(best_model_path, map_location=map_location))
        return self.model

    SEPARATOR_LENGTH = 100
    
    def _print_and_log(self, message):
        """统一的打印和日志记录"""
        print(message)
        self.write_log(message)
    
    def _log_train_metrics(self, train_loss, train_value_loss, train_peak_loss, train_tp_mse_loss):
        """记录训练指标"""
        self._print_and_log("[TRAIN]")
        loss_str = f" <LOSS> {train_loss:.7f} (Val×{self.value_loss_weight}: {train_value_loss:.7f}"
        if train_peak_loss is not None:
            loss_str += f", Peak×{self.peak_loss_weight}: {train_peak_loss:.7f}"
        if train_tp_mse_loss is not None:
            loss_str += f", TP_MSE×{self.tp_mse_loss_weight}: {train_tp_mse_loss:.7f}"
        loss_str += ")"
        self._print_and_log(loss_str)

    def _log_split_metrics(self, prefix, weighted_loss, value_loss, peak_loss, tp_mse_loss, peak_cls_metrics):
        """记录单个数据集的指标"""
        self._print_and_log(f"[{prefix}]")
        loss_str = f" <LOSS> {weighted_loss:.7f} (Val×{self.value_loss_weight}: {value_loss:.7f}"
        if peak_loss is not None:
            loss_str += f", Peak×{self.peak_loss_weight}: {peak_loss:.7f}"
        if tp_mse_loss is not None:
            loss_str += f", TP_MSE×{self.tp_mse_loss_weight}: {tp_mse_loss:.7f}"
        loss_str += ")"
        self._print_and_log(loss_str)
        
        if peak_cls_metrics:
            self._print_and_log(" <METRICS>")
            metrics = [
                f"\tPrediction | MSE:    {peak_cls_metrics['Peak_Cls_MSE']:.4f}, MAE:    {peak_cls_metrics['Peak_Cls_MAE']:.4f}",
                f"\tTP Error   | TP_MSE: {peak_cls_metrics['Peak_Cls_TP_MSE']:.4f}, TP_MAE: {peak_cls_metrics['Peak_Cls_TP_MAE']:.4f}",
                f"\tFN Error   | FN_MSE: {peak_cls_metrics['Peak_Cls_FN_MSE']:.4f}, FN_MAE: {peak_cls_metrics['Peak_Cls_FN_MAE']:.4f}",
                f"\tAll Peaks  | A_MSE:  {peak_cls_metrics['Peak_Cls_All_True_Peaks_MSE']:.4f}, A_MSE:  {peak_cls_metrics['Peak_Cls_All_True_Peaks_MAE']:.4f}",
                f"\tDetection  | F1:     {peak_cls_metrics['Peak_Cls_F1']:.4f}, P:      {peak_cls_metrics['Peak_Cls_Precision']:.4f}, R: {peak_cls_metrics['Peak_Cls_Recall']:.4f}",
                f"\tMixed      | BPE:    {peak_cls_metrics['Peak_Cls_Balanced_Error']:.4f}, PIM:    {peak_cls_metrics['Peak_Cls_PIM']:.4f}"
            ]
            for metric_line in metrics:
                self._print_and_log(metric_line)

    def _log_epoch_results(self, epoch, cost_time, train_metrics, vali_metrics, test_metrics):
        """统一打印和记录Epoch结果"""
        train_loss, train_value_loss, train_peak_loss, train_tp_mse_loss = train_metrics
        vali_weighted_loss, vali_value_loss, vali_peak_loss, vali_tp_mse_loss, vali_peak_cls_metrics = vali_metrics
        test_weighted_loss, test_value_loss, test_peak_loss, test_tp_mse_loss, test_peak_cls_metrics = test_metrics
        
        self._print_and_log(f"=================EPOCH {epoch + 1} | Tolerance={self.peak_tolerance}=================")
        
        self._log_train_metrics(train_loss, train_value_loss, train_peak_loss, train_tp_mse_loss)
        self._print_and_log(f" <TIME> Epoch cost: {cost_time:.2f}s")
        
        self._log_split_metrics("VAL", vali_weighted_loss, vali_value_loss, vali_peak_loss, vali_tp_mse_loss, vali_peak_cls_metrics)
        self._log_split_metrics("TEST", test_weighted_loss, test_value_loss, test_peak_loss, test_tp_mse_loss, test_peak_cls_metrics)

    def _calculate_test_metrics(self, preds: np.ndarray, trues: np.ndarray) -> Dict[str, Any]:
        """
        (新增) 统一计算测试集指标 (MAE, MSE, DTW)
        """
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        dtw_val = 'Not calculated'
        
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                if i % 200 == 0:
                    print(f"calculating dtw iter: {i}/{preds.shape[0]}")
                x = preds[i, :, 0].reshape(-1, 1)
                y = trues[i, :, 0].reshape(-1, 1)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw_val = np.array(dtw_list).mean()
        
        return {
            'MAE': mae, 'MSE': mse, 'RMSE': rmse, 
            'MAPE': mape, 'MSPE': mspe, 'DTW': dtw_val
        }

    def _print_and_log_metrics(self, metrics_dict: Dict, title: str):
        """打印和记录指标"""
        print(f"\n{title}")
        print("=" * 50)
        self.write_log(f"\n{title}\n" + "=" * 50)
        
        for key, value in metrics_dict.items():
            line = f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
            print(line)
            self.write_log(line)

    def _save_test_results(self, setting: str, std_metrics: Dict, 
                           peak_metrics: Optional[Dict], 
                           preds: np.ndarray, trues: np.ndarray) -> None:
        """统一保存所有测试结果 (npy, txt, log)"""
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        np.save(folder_path + 'metrics.npy', np.array([
            std_metrics['MAE'], std_metrics['MSE'], std_metrics['RMSE'], 
            std_metrics['MAPE'], std_metrics['MSPE']
        ]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        if peak_metrics:
            np.save(folder_path + 'peak_metrics.npy', np.array(peak_metrics, dtype=object))

        self._print_and_log_metrics(std_metrics, "EVALUATION RESULTS")

        if peak_metrics:
            peak_display_metrics = {
                'Balanced_Peak_Error (BPE)': peak_metrics['Peak_Cls_Balanced_Error'],
                'Peak_Integrated_Metric (PIM)': peak_metrics['Peak_Cls_PIM'],
                'F1-Score': peak_metrics['Peak_Cls_F1'],
                'Precision': peak_metrics['Peak_Cls_Precision'],
                'Recall': peak_metrics['Peak_Cls_Recall'],
                'TP_MSE': peak_metrics['Peak_Cls_TP_MSE'],
                'TP_MAE': peak_metrics['Peak_Cls_TP_MAE'],
                'FN_MSE': peak_metrics['Peak_Cls_FN_MSE'],
                'FN_MAE': peak_metrics['Peak_Cls_FN_MAE'],
                'All_True_Peaks_MSE': peak_metrics['Peak_Cls_All_True_Peaks_MSE'],
                'All_True_Peaks_MAE': peak_metrics['Peak_Cls_All_True_Peaks_MAE'],
                'Average TP': peak_metrics['Peak_Cls_TP'],
                'Average FP': peak_metrics['Peak_Cls_FP'],
                'Average FN': peak_metrics['Peak_Cls_FN'],
            }
            self._print_and_log_metrics(peak_display_metrics, f"PEAK CLASSIFICATION METRICS (Tolerance={self.peak_tolerance})")

        try:
            with open("result_peak_detect_based_on_long_term_forecasting.txt", 'a') as f:
                f.write(setting + "  \n")
                f.write(f"mse:{std_metrics['MSE']:.6f}, mae:{std_metrics['MAE']:.6f}, dtw:{std_metrics['DTW']}\n")
                if peak_metrics:
                    f.write('BPE:{:.4f}, PIM:{:.4f}, F1:{:.4f}, P:{:.4f}, R:{:.4f}, TP_MSE:{:.4f}, TP_MAE:{:.4f}, FN_MSE:{:.4f}, All_Peaks_MSE:{:.4f} (Tol={})\n'.format(
                        peak_metrics['Peak_Cls_Balanced_Error'],
                        peak_metrics['Peak_Cls_PIM'],
                        peak_metrics['Peak_Cls_F1'], 
                        peak_metrics['Peak_Cls_Precision'], 
                        peak_metrics['Peak_Cls_Recall'], 
                        peak_metrics['Peak_Cls_TP_MSE'],
                        peak_metrics['Peak_Cls_TP_MAE'],
                        peak_metrics['Peak_Cls_FN_MSE'],
                        peak_metrics['Peak_Cls_All_True_Peaks_MSE'],
                        self.peak_tolerance))
                f.write('\n')
        except Exception as e:
            print(f"Warning: Failed to write to global result file: {e}")

    def _compute_self_consistency(self, preds: np.ndarray, peak_preds: np.ndarray,
                                    method: str = 'peakdetect', lookahead: int = 10) -> float:
        """
        Self-consistency metric: compare MSM-PL predicted peaks vs peaks 
        derived from the forecast intensity via findpeaks.
        Returns averaged F1 across all samples.
        """
        n_samples = preds.shape[0]
        f1_scores = []
        threshold = self.peak_threshold
        tolerance = self.peak_tolerance
        
        for i in range(n_samples):
            pred_seq = preds[i, :, 0]
            peak_prob = peak_preds[i, :, 0]
            
            # MSM-PL peaks (binarized from probabilities)
            peak_binary = (peak_prob >= threshold).astype(int)
            msm_peaks = _condense_peak_indices(peak_binary, peak_prob)
            
            # Forecast-derived peaks via findpeaks
            forecast_peaks = detect_peaks_findpeaks(pred_seq, method=method, lookahead=lookahead)
            
            if len(msm_peaks) == 0 and len(forecast_peaks) == 0:
                f1_scores.append(1.0)
                continue
            if len(msm_peaks) == 0 or len(forecast_peaks) == 0:
                f1_scores.append(0.0)
                continue
            
            tp_pairs, fp, fn = match_peaks_with_tolerance(msm_peaks, forecast_peaks, tolerance=tolerance)
            tp_count = len(tp_pairs)
            fp_count = len(fp)
            fn_count = len(fn)
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
        
        return np.mean(f1_scores) if f1_scores else 0.0

    def _visualize_test_samples(self, preds: np.ndarray, trues: np.ndarray, setting: str, max_samples: int = 20):
        """可视化测试样本 — 已禁用，避免生成大量 PDF 文件"""
        return

    def test(self, setting: str, test: int = 0):
        """最终测试函数"""
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        criterion = self._select_criterion()

        # [SCALING INSTRUMENTATION] — wall-clock timer for inference
        import time as _time_mod
        _infer_t0 = _time_mod.perf_counter()
        test_weighted_loss, test_value_loss, test_peak_loss, test_tp_mse_loss, preds, trues, peak_preds, peak_trues = self._run_inference(
            test_loader,
            criterion=criterion,
            inverse_transform=True,
            data_flag='test'
        )
        _infer_total_s = _time_mod.perf_counter() - _infer_t0
        try:
            _n_samples = preds.shape[0] if hasattr(preds, 'shape') else len(preds)
            _per_ms = (_infer_total_s / max(1, _n_samples)) * 1000.0
            self.write_log(
                f"[INFER] Test inference: total={_infer_total_s:.2f}s, "
                f"n_samples={_n_samples}, per_sample={_per_ms:.4f}ms"
            )
            print(
                f"[INFER] Test inference: total={_infer_total_s:.2f}s, "
                f"n_samples={_n_samples}, per_sample={_per_ms:.4f}ms"
            )
        except Exception as _e:
            self.write_log(f"[INFER] failed to measure: {_e}")
        
        print("\n" + "=" * 80)
        print("TEST SET LOSSES (After Inverse Transform)")
        print("=" * 80)
        
        test_loss_info = f"Test Weighted Loss: {test_weighted_loss:.7f}"
        if test_value_loss is not None:
            test_loss_info += f" (Value×{self.value_loss_weight}: {test_value_loss:.7f}"
            if test_peak_loss is not None:
                test_loss_info += f", Peak×{self.peak_loss_weight}: {test_peak_loss:.7f}"
            if test_tp_mse_loss is not None:
                test_loss_info += f", TP_MSE×{self.tp_mse_loss_weight}: {test_tp_mse_loss:.7f}"
            test_loss_info += ")"
        
        print(test_loss_info)
        self.write_log("\n" + "=" * 80)
        self.write_log("TEST SET LOSSES (After Inverse Transform)")
        self.write_log("=" * 80)
        self.write_log(test_loss_info)
        print("=" * 80 + "\n")
        self.write_log("=" * 80 + "\n")
        
        f_dim = -1 if self.args.features == 'MS' else 0
        preds = preds[:, :, f_dim:]
        trues = trues[:, :, f_dim:]
        
        print('test shape:', preds.shape, trues.shape)

        std_metrics = self._calculate_test_metrics(preds, trues)
        
        peak_cls_metrics = None
        if peak_preds is not None and peak_trues is not None:
            peak_cls_metrics = calculate_peak_classification_metrics(
                peak_preds, peak_trues, 
                preds, trues,
                threshold=self.peak_threshold,
                tolerance=self.peak_tolerance,
                fn_penalty_weight=1.0,
                fp_penalty_weight=1.0,
                alpha=self.metric_alpha
            )
            print('Peak classification shape:', peak_preds.shape, peak_trues.shape)
            
            vis_test_peaks = getattr(self.args, 'vis_test_peaks', True)
            if vis_test_peaks:
                print('Visualizing peak classification results...')
                self._evaluate_peaks(
                    preds, trues,
                    peak_preds=peak_preds,
                    peak_trues=peak_trues,
                    visualize=True,
                    epoch=None,
                    data_flag='test'
                )
        
        # Self-consistency metric: MSM-PL peaks vs forecast-derived peaks
        self_consistency = None
        if peak_preds is not None:
            self_consistency = self._compute_self_consistency(preds, peak_preds)
            print(f"\n[Self-Consistency] MSM-PL vs Forecast-derived peaks F1: {self_consistency:.4f}")
            self.write_log(f"[Self-Consistency] MSM-PL vs Forecast-derived peaks F1: {self_consistency:.4f}")
        
        self._save_test_results(setting, std_metrics, peak_cls_metrics, preds, trues)
        self._visualize_test_samples(preds, trues, setting, max_samples=20)
        
        print("\n" + "=" * 80)
        print("TRAINING AND TESTING COMPLETED!")
        print("=" * 80)
        self.write_log("\n" + "=" * 80)
        self.write_log("TRAINING AND TESTING COMPLETED!")
        self.write_log("=" * 80)
        
        return