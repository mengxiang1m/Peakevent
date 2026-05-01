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


# ==================== 峰值检测与评估 (辅助函数) ====================

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
    
    容错范围：当前时间点和往前 tolerance 个时间点
    即：对于真实峰值 t_idx，预测峰值需要在 [t_idx - tolerance, t_idx] 范围内
    """
    tp_pairs = []
    matched_true = set()
    matched_pred = set()
    
    for t_idx in true_peaks:
        # 只考虑当前时间点和往前 tolerance 个时间点
        # 预测峰值必须在 [t_idx - tolerance, t_idx] 范围内
        candidates = pred_peaks[np.abs(pred_peaks - t_idx) <= tolerance]
        if len(candidates) > 0:
            # 选择最接近真实峰值的预测峰值
            best_pred = candidates[np.argmin(np.abs(candidates - t_idx))]
            if best_pred not in matched_pred:
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
    
    # 计算 TP MAE 和 MSE
    tp_mae, tp_mse = 0.0, 0.0
    tp_errors = []
    if tp_count > 0:
        true_peak_values = true_values[[t for t, p in tp_pairs]]
        pred_peak_values = pred_values[[p for t, p in tp_pairs]]
        tp_errors = true_peak_values - pred_peak_values
        tp_mae = np.mean(np.abs(tp_errors))
        tp_mse = np.mean(tp_errors ** 2)
    
    # (新增) 计算 FN MAE 和 MSE
    fn_mae, fn_mse = 0.0, 0.0
    fn_errors = []
    if fn_count > 0:
        fn_true_values = true_values[fn]
        fn_pred_values = pred_values[fn]
        fn_errors = fn_true_values - fn_pred_values
        fn_mae = np.mean(np.abs(fn_errors))
        fn_mse = np.mean(fn_errors ** 2)
    
    # (新增) 计算所有真实峰值的 MSE/MAE (包含TP和FN)
    all_true_peak_errors = []
    if tp_count > 0:
        all_true_peak_errors.extend(tp_errors)
    if fn_count > 0:
        all_true_peak_errors.extend(fn_errors)
    
    all_true_peaks_mse = 0.0
    all_true_peaks_mae = 0.0
    if len(all_true_peak_errors) > 0:
        all_true_peak_errors_np = np.array(all_true_peak_errors)
        all_true_peaks_mse = np.mean(all_true_peak_errors_np ** 2)
        all_true_peaks_mae = np.mean(np.abs(all_true_peak_errors_np))
    
    # 计算整体 MAE 和 MSE (所有点，不只是TP)
    overall_mae = np.mean(np.abs(pred_values - true_values))
    overall_mse = np.mean((pred_values - true_values) ** 2)
    
    # 计算混合指标 - 与主版本保持一致
    # BPE (Balanced Peak Error): 综合检测和回归性能
    alpha = 0.5  # 默认平衡F1和TP_MSE
    e_class = 1.0 - f1_score  # 分类误差: F1=1.0 -> e_class=0.0
    e_reg = 1.0 - (1.0 / (1.0 + tp_mse))  # 标准化回归误差
    balanced_peak_error = (alpha * e_class) + ((1.0 - alpha) * e_reg)
    
    # PIM (Peak Integrated Metric): 检测精度与值误差的比值
    peak_pim = (1.0 + tp_mse) / (f1_score + 0.01)
    
    return {
        'TP': tp_count, 'FP': fp_count, 'FN': fn_count,
        'Precision': precision, 'Recall': recall, 'F1_Score': f1_score,
        'Peak_MAE': tp_mae,  # TP点的MAE
        'Peak_MSE': tp_mse,  # TP点的MSE
        'FN_MAE': fn_mae,  # FN点的MAE
        'FN_MSE': fn_mse,  # FN点的MSE
        'All_True_Peaks_MAE': all_true_peaks_mae,  # 所有真实峰值的MAE
        'All_True_Peaks_MSE': all_true_peaks_mse,  # 所有真实峰值的MSE
        'Overall_MAE': overall_mae,  # 整体MAE
        'Overall_MSE': overall_mse,   # 整体MSE
        'Balanced_Peak_Error': balanced_peak_error,  # BPE混合指标
        'Peak_PIM': peak_pim  # PIM混合指标
    }


def _process_single_sample_peaks(args: Tuple) -> Dict[str, Any]:
    """
    处理单个样本的峰值检测（用于多进程）
    修改：支持直接使用传入的 peak 标签，而不是通过 findpeaks 算法计算
    """
    sample_idx, pred_seq, true_seq, tolerance, method, lookahead, true_peak_seq = args
    
    # 检测预测峰值（使用 findpeaks 算法）
    pred_peaks = detect_peaks_findpeaks(pred_seq, method=method, lookahead=lookahead)
    
    # 获取真实峰值：
    # 如果提供了 true_peak_seq (peak 标签)，则直接使用
    # 否则使用 findpeaks 算法检测
    if true_peak_seq is not None:
        # 从 peak 标签中提取峰值索引（标签值 > 0.5 的位置）
        true_peaks = np.where(true_peak_seq > 0.5)[0]
    else:
        # 使用 findpeaks 算法检测
        true_peaks = detect_peaks_findpeaks(true_seq, method=method, lookahead=lookahead)
    
    tp_pairs, fp, fn = match_peaks_with_tolerance(true_peaks, pred_peaks, tolerance=tolerance)
    metrics = calculate_peak_metrics(tp_pairs, fp, fn, true_seq, pred_seq)
    metrics['sample_idx'] = sample_idx
    
    return metrics


def _process_single_visualization_basic(args: Tuple) -> None:
    """
    处理单个样本的可视化（用于多进程）
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    (sample_idx, true_seq, pred_seq, true_peaks, pred_peaks,
     tp_pairs, fp, fn, metrics, save_folder, epoch, data_flag) = args
    
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


def process_peaks_multiprocess(preds: np.ndarray, 
                               trues: np.ndarray, 
                               tolerance: int = 5, 
                               method: str = 'peakdetect', 
                               lookahead: int = 10, 
                               num_workers: Optional[int] = None,
                               peak_trues: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """
    使用多进程处理峰值检测
    修改：支持传入 peak_trues (峰值标签)
    
    Args:
        preds: 预测值 [N, T, 1]
        trues: 真实值 [N, T, 1]
        tolerance: 容忍度
        method: findpeaks 方法
        lookahead: findpeaks 参数
        num_workers: 进程数
        peak_trues: 可选的峰值标签 [N, T, 1]，如果提供则直接使用，否则用 findpeaks 检测
    
    Returns:
        peak_metrics_list: 每个样本的峰值指标列表
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    
    n_samples = preds.shape[0]
    args_list = []
    for i in range(n_samples):
        true_peak_seq = peak_trues[i, :, 0] if peak_trues is not None else None
        args_list.append((i, preds[i, :, 0], trues[i, :, 0], tolerance, method, lookahead, true_peak_seq))
    
    with Pool(processes=num_workers) as pool:
        peak_metrics_list = pool.map(_process_single_sample_peaks, args_list)
    
    return peak_metrics_list


# ==================== 实验类 ====================

class Exp_Peak_Detect_LTF_basic(Exp_Basic):
    def __init__(self, args):
        super(Exp_Peak_Detect_LTF_basic, self).__init__(args)

        # 初始化可视化文件夹 - 使用 args.setting
        if getattr(self.args, 'enable_peak_eval', False) and getattr(self.args, 'vis_test_peaks', False):
            self.vis_folder = f'./visualization/output/predictions/{self.args.setting}/'
            if not os.path.exists(self.vis_folder):
                os.makedirs(self.vis_folder, exist_ok=True)
                print(f"Created visualization folder: {self.vis_folder}")
        else:
            self.vis_folder = None
        
        # 峰值检测参数
        self.peak_tolerance = getattr(self.args, 'peak_tolerance', 2)
        self.peak_lookahead = getattr(self.args, 'peak_lookahead', 10)
        self.peak_method = getattr(self.args, 'peak_method', 'peakdetect')
        self.peak_workers = getattr(self.args, 'peak_workers', 20)

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

    def _run_inference(self, data_loader, criterion: Optional[nn.Module] = None, 
                       inverse_transform: bool = False, data_flag: str = 'val') -> Tuple[Optional[float], np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        (修改) 统一的模型推理函数 (用于 vali 和 test)
        返回: (total_loss, preds, trues, peak_trues)
        """
        self.model.eval()
        total_loss = []
        all_preds, all_trues = [], []
        all_peak_trues = []  # 新增：收集 peak 标签
        
        data_set = None
        if inverse_transform:
            data_set, _ = self._get_data(flag=data_flag)

        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                # 统一处理 batch 格式
                batch_cycle = None
                batch_peak_y = None  # 新增：peak 标签
                batch_dec_ext = None
                use_external_features = getattr(self.args, 'use_external_features', 0)
                external_feature_mode = getattr(self.args, 'external_feature_mode', 'concat')
                use_decoder_external = use_external_features and external_feature_mode in ['concat', 'fusion', 'future_only']
                use_future_only_external = use_external_features and external_feature_mode in ['fusion', 'future_only']
                
                # 根据数据长度判断是否包含 peak 标签
                if len(data_batch) == 6:
                    # 包含 cycle 和 peak 标签
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, batch_peak_y = data_batch
                    batch_cycle = batch_cycle.int().to(self.device)
                    batch_peak_y = batch_peak_y.float()
                elif len(data_batch) == 5:
                    # 只包含 cycle 或只包含 peak 标签 或 external features
                    batch_x, batch_y, batch_x_mark, batch_y_mark, extra = data_batch
                    if use_decoder_external:
                        batch_dec_ext = extra.float().to(self.device)
                    elif not use_external_features and (extra.dtype == torch.int64 or extra.dtype == torch.int32):
                        batch_cycle = extra.int().to(self.device)
                    elif not use_external_features:
                        batch_peak_y = extra.float()
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = data_batch
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

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

                # 模型推理
                if self.args.use_amp:
                    with autocast():
                        if 'CycleNet' in self.args.model and batch_cycle is not None:
                            outputs = self.model(batch_x, batch_cycle)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'CycleNet' in self.args.model and batch_cycle is not None:
                        outputs = self.model(batch_x, batch_cycle)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # 截取预测结果
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_targets = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y_targets.detach()

                # 计算损失 (如果提供了 criterion)
                if criterion:
                    loss = criterion(pred, true)
                    total_loss.append(loss.item())
                
                # 反归一化 (仅在 test 且 inverse_transform=True 时)
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
                
                # 收集 peak 标签 (如果有)
                if batch_peak_y is not None:
                    peak_y_targets = batch_peak_y[:, -self.args.pred_len:, f_dim:].cpu().numpy()
                    all_peak_trues.append(peak_y_targets)

        avg_loss = np.average(total_loss) if total_loss else None
        preds_all = np.concatenate(all_preds, axis=0)
        trues_all = np.concatenate(all_trues, axis=0)
        peak_trues_all = np.concatenate(all_peak_trues, axis=0) if all_peak_trues else None
        
        return avg_loss, preds_all, trues_all, peak_trues_all

    def _evaluate_peaks(self, preds: np.ndarray, trues: np.ndarray, 
                        peak_trues: Optional[np.ndarray] = None,
                        visualize: bool = False, epoch: Optional[int] = None, 
                        data_flag: str = 'test') -> Optional[Dict[str, float]]:
        """
        统一的峰值评估函数
        
        Args:
            preds: 预测值 [N, T, 1]
            trues: 真实值 [N, T, 1]
            peak_trues: 可选的峰值标签 [N, T, 1]，如果提供则直接使用，否则用 findpeaks 检测
            visualize: 是否可视化
            epoch: epoch 编号
            data_flag: 数据集标志
        """
        if not (hasattr(self.args, 'enable_peak_eval') and self.args.enable_peak_eval):
            return None
        
        # 1. 使用多进程处理 - 传入 peak_trues
        peak_metrics_list = process_peaks_multiprocess(
            preds, trues, 
            tolerance=self.peak_tolerance,
            method=self.peak_method,
            lookahead=self.peak_lookahead,
            num_workers=self.peak_workers,
            peak_trues=peak_trues
        )
        
        # 2. 可视化 - 每168个样本可视化一次
        if visualize and self.vis_folder:
            n_samples = min(preds.shape[0], 10000)
            sample_indices = [i for i in range(n_samples) if i % 168 == 0]
            
            if len(sample_indices) > 0:
                # 准备多进程参数列表
                args_list = []
                for sample_idx in sample_indices:
                    pred_seq = preds[sample_idx, :, 0]
                    true_seq = trues[sample_idx, :, 0]
                    
                    # 重新检测用于可视化
                    pred_peaks = detect_peaks_findpeaks(pred_seq, method=self.peak_method, lookahead=self.peak_lookahead)
                    
                    # 获取真实峰值：优先使用 peak 标签
                    if peak_trues is not None:
                        true_peak_seq = peak_trues[sample_idx, :, 0]
                        true_peaks = np.where(true_peak_seq > 0.5)[0]
                    else:
                        true_peaks = detect_peaks_findpeaks(true_seq, method=self.peak_method, lookahead=self.peak_lookahead)
                    
                    tp_pairs, fp, fn = match_peaks_with_tolerance(true_peaks, pred_peaks, tolerance=self.peak_tolerance)
                    
                    # 获取对应的metrics
                    metrics = peak_metrics_list[sample_idx] if sample_idx < len(peak_metrics_list) else {}
                    
                    args_list.append((
                        sample_idx, true_seq, pred_seq, true_peaks, pred_peaks,
                        tp_pairs, fp, fn, metrics, self.vis_folder, epoch, data_flag
                    ))
                
                # 使用多进程处理可视化
                with Pool(processes=self.peak_workers) as pool:
                    pool.map(_process_single_visualization_basic, args_list)
        
        # 3. 汇总指标
        if peak_metrics_list:
            avg_peak_metrics = {
                key: np.mean([m[key] for m in peak_metrics_list])
                for key in peak_metrics_list[0] if key != 'sample_idx'
            }
            return avg_peak_metrics
        
        return None

    def _visualize_peak_detection(self, true_seq: np.ndarray, pred_seq: np.ndarray, 
                                   true_peaks: np.ndarray, pred_peaks: np.ndarray, 
                                   tp_pairs: List, fp: List, fn: List, 
                                   metrics: Dict, save_folder: str, 
                                   sample_idx: int, epoch: Optional[int] = None, 
                                   data_flag: str = 'val') -> None:
        """
        可视化峰值检测结果
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
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
             data_flag: str = 'val') -> Tuple[float, Optional[Dict[str, float]]]:
        """
        (重构) 验证/测试方法
        """
        # 使用统一的推理函数 - 现在返回 peak_trues
        total_loss, all_preds, all_trues, peak_trues_all = self._run_inference(
            vali_loader, criterion, 
            inverse_transform=False,  # Vali/Test 评估损失时不需要反归一化
            data_flag=data_flag
        )
        
        # 使用统一的峰值评估函数 - 传入 peak_trues
        avg_peak_metrics = None
        if evaluate_peaks:
            avg_peak_metrics = self._evaluate_peaks(
                all_preds, all_trues, 
                peak_trues=peak_trues_all,  # 新增：传入 peak 标签
                visualize=visualize_peaks, 
                epoch=epoch, 
                data_flag=data_flag
            )
        
        self.model.train()
        return total_loss, avg_peak_metrics

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
        scaler = GradScaler() if self.args.use_amp else None
        
        print("\n" + "="*80)
        print("MODEL SELECTION STRATEGY:")
        print("  Primary metric: BPE (Balanced Peak Error) - Lower is better")
        print("  BPE formula: 0.5×(1-F1) + 0.5×(1-1/(1+TP_MSE))")
        print("  Fallback: Validation Loss (if BPE unavailable)")
        print("="*80 + "\n")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, data_batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
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

                def forward_pass():
                    if 'CycleNet' in self.args.model and batch_cycle is not None:
                        return self.model(batch_x, batch_cycle)
                    else:
                        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.use_amp:
                    with autocast():
                        outputs = forward_pass()
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = forward_pass()
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                
                train_loss.append(loss.item())

                if self.args.use_amp and scaler:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if (i + 1) % 1000 == 0:
                    speed = (time.time() - epoch_time) / iter_count
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f} | speed: {speed:.4f}s/iter")
                    iter_count = 0

            epoch_cost_time = time.time() - epoch_time
            train_loss_avg = np.average(train_loss)
            
            eval_peak_args = getattr(self.args, 'enable_peak_eval', False)
            eval_train_peaks = getattr(self.args, 'eval_train_peaks', False) and eval_peak_args
            eval_val_peaks = getattr(self.args, 'eval_val_peaks', True) and eval_peak_args
            eval_test_peaks = getattr(self.args, 'eval_test_peaks', True) and eval_peak_args
            
            vis_train_peaks = getattr(self.args, 'vis_train_peaks', False)
            vis_val_peaks = getattr(self.args, 'vis_val_peaks', False)
            vis_test_peaks = getattr(self.args, 'vis_test_peaks', True)

            vali_loss, vali_peak_metrics = self.vali(
                vali_data, vali_loader, criterion,
                evaluate_peaks=eval_val_peaks,
                visualize_peaks=vis_val_peaks,
                epoch=epoch+1, setting=setting, data_flag='val'
            )
            
            test_loss, test_peak_metrics = self.vali(
                test_data, test_loader, criterion,
                evaluate_peaks=eval_test_peaks,
                visualize_peaks=vis_test_peaks,
                epoch=epoch+1, setting=setting, data_flag='test'
            )

            train_peak_metrics = None
            if eval_train_peaks:
                _, train_peak_metrics = self.vali(
                    train_data, train_loader, criterion, 
                    evaluate_peaks=True, visualize_peaks=vis_train_peaks,
                    epoch=epoch+1, setting=setting, data_flag='train'
                )

            self._log_epoch_results(epoch, train_steps, epoch_cost_time, train_loss_avg, 
                                    vali_loss, test_loss, train_peak_metrics, 
                                    vali_peak_metrics, test_peak_metrics)

            # 选择early stopping指标
            vali_metric = self._select_early_stopping_metric(vali_peak_metrics, vali_loss)
            
            prev_best = early_stopping.val_loss_min
            early_stopping(vali_metric, self.model, path)
            
            self._print_and_log("[OTHERS]")
            self._print_and_log(" [MODEL SELECTION]")
            metric_name = "BPE" if (vali_peak_metrics is not None and 'Balanced_Peak_Error' in vali_peak_metrics) else "Val Loss"
            
            if early_stopping.early_stop:
                self._print_and_log(f"\t{metric_name}: {vali_metric:.6f}")
                self._print_and_log("\tEarly stopping triggered!")
            elif early_stopping.val_loss_min == vali_metric:
                self._print_and_log(f"\t{metric_name} improved ({prev_best:.6f} --> {vali_metric:.6f}). Saving model...")
            else:
                self._print_and_log(f"\t{metric_name}: {vali_metric:.6f} (best: {prev_best:.6f})")
                self._print_and_log(f"\tEarlyStopping counter: {early_stopping.counter}/{early_stopping.patience}")
            
            # Visualization section
            vis_val_peaks = getattr(self.args, 'vis_val_peaks', False)
            vis_test_peaks = getattr(self.args, 'vis_test_peaks', True)
            if vis_val_peaks or vis_test_peaks:
                self._print_and_log(" [VISUALIZATION]")
                
                # 打印保存文件夹
                if self.vis_folder:
                    self._print_and_log(f"        Save folder: {self.vis_folder}")
                
                # 获取 val 和 test 的数据用于可视化
                if vis_val_peaks:
                    vali_loss_vis, vali_preds_vis, vali_trues_vis, vali_peak_trues_vis = self._run_inference(
                        vali_loader, None, inverse_transform=False, data_flag='val'
                    )
                    if vali_peak_trues_vis is not None:
                        self._evaluate_peaks(
                            vali_preds_vis, vali_trues_vis,
                            peak_trues=vali_peak_trues_vis,
                            visualize=True,
                            epoch=epoch+1,
                            data_flag='val'
                        )
                
                if vis_test_peaks:
                    test_loss_vis, test_preds_vis, test_trues_vis, test_peak_trues_vis = self._run_inference(
                        test_loader, None, inverse_transform=False, data_flag='test'
                    )
                    if test_peak_trues_vis is not None:
                        self._evaluate_peaks(
                            test_preds_vis, test_trues_vis,
                            peak_trues=test_peak_trues_vis,
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

    def _select_early_stopping_metric(self, vali_peak_metrics, vali_loss):
        """选择early stopping指标 - 优先使用BPE"""
        if vali_peak_metrics is not None and 'Balanced_Peak_Error' in vali_peak_metrics:
            return vali_peak_metrics['Balanced_Peak_Error']
        else:
            return vali_loss

    SEPARATOR_LENGTH = 100
    
    def _print_and_log(self, message):
        """统一的打印和日志记录"""
        print(message)
        self.write_log(message)
    
    def _log_train_metrics(self, train_loss):
        """记录训练指标"""
        self._print_and_log("[TRAIN]")
        self._print_and_log(f" <LOSS> {train_loss:.7f}")
        self._print_and_log(f" <TIME> Epoch cost: 0.00s")
    
    def _log_split_metrics(self, prefix, loss, peak_metrics):
        """记录单个数据集的指标"""
        self._print_and_log(f"[{prefix}]")
        self._print_and_log(f" <LOSS> {loss:.7f}")
        
        if peak_metrics:
            self._print_and_log(" <METRICS>")
            metrics = [
                f"\tPrediction | MSE:    {peak_metrics['Overall_MSE']:.4f}, MAE:    {peak_metrics['Overall_MAE']:.4f}",
                f"\tTP Error   | TP_MSE: {peak_metrics['Peak_MSE']:.4f}, TP_MAE: {peak_metrics['Peak_MAE']:.4f}",
                f"\tFN Error   | FN_MSE: {peak_metrics['FN_MSE']:.4f}, FN_MAE: {peak_metrics['FN_MAE']:.4f}",
                f"\tAll Peaks  | A_MSE:  {peak_metrics['All_True_Peaks_MSE']:.4f}, A_MSE:  {peak_metrics['All_True_Peaks_MAE']:.4f}",
                f"\tDetection  | F1:     {peak_metrics['F1_Score']:.4f}, P:      {peak_metrics['Precision']:.4f}, R: {peak_metrics['Recall']:.4f}",
                f"\tMixed      | BPE:    {peak_metrics['Balanced_Peak_Error']:.4f}, PIM:    {peak_metrics['Peak_PIM']:.4f}"
            ]
            for metric_line in metrics:
                self._print_and_log(metric_line)

    def _log_epoch_results(self, epoch, train_steps, cost_time, train_loss, 
                           vali_loss, test_loss, train_peaks, val_peaks, test_peaks):
        """
        统一打印和记录Epoch结果 - 与主版本格式一致
        """
        self._print_and_log(f"=================EPOCH {epoch + 1} | Tolerance={self.peak_tolerance}=================")
        
        self._print_and_log("[TRAIN]")
        self._print_and_log(f" <LOSS> {train_loss:.7f}")
        self._print_and_log(f" <TIME> Epoch cost: {cost_time:.2f}s")
        
        self._log_split_metrics("VAL", vali_loss, val_peaks)
        self._log_split_metrics("TEST", test_loss, test_peaks)

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
        """
        统一保存所有测试结果 (npy, txt, log)
        """
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
                'Balanced_Peak_Error (BPE)': peak_metrics['Balanced_Peak_Error'],
                'Peak_Integrated_Metric (PIM)': peak_metrics['Peak_PIM'],
                'F1-Score': peak_metrics['F1_Score'],
                'Precision': peak_metrics['Precision'],
                'Recall': peak_metrics['Recall'],
                'TP_MSE': peak_metrics['Peak_MSE'],
                'TP_MAE': peak_metrics['Peak_MAE'],
                'FN_MSE': peak_metrics['FN_MSE'],
                'FN_MAE': peak_metrics['FN_MAE'],
                'All_True_Peaks_MSE': peak_metrics['All_True_Peaks_MSE'],
                'All_True_Peaks_MAE': peak_metrics['All_True_Peaks_MAE'],
                'Average TP': peak_metrics['TP'],
                'Average FP': peak_metrics['FP'],
                'Average FN': peak_metrics['FN'],
            }
            self._print_and_log_metrics(peak_display_metrics, f"PEAK CLASSIFICATION METRICS (Tolerance={self.peak_tolerance})")

        try:
            with open("result_peak_detect_based_on_long_term_forecasting_basic.txt", 'a') as f:
                f.write(setting + "  \n")
                f.write(f"mse:{std_metrics['MSE']:.6f}, mae:{std_metrics['MAE']:.6f}, dtw:{std_metrics['DTW']}\n")
                if peak_metrics:
                    f.write('BPE:{:.4f}, PIM:{:.4f}, F1:{:.4f}, P:{:.4f}, R:{:.4f}, TP_MSE:{:.4f}, TP_MAE:{:.4f}, FN_MSE:{:.4f}, All_Peaks_MSE:{:.4f} (Tol={})\n'.format(
                        peak_metrics['Balanced_Peak_Error'],
                        peak_metrics['Peak_PIM'],
                        peak_metrics['F1_Score'], 
                        peak_metrics['Precision'], 
                        peak_metrics['Recall'], 
                        peak_metrics['Peak_MSE'],
                        peak_metrics['Peak_MAE'],
                        peak_metrics['FN_MSE'],
                        peak_metrics['All_True_Peaks_MSE'],
                        self.peak_tolerance))
                f.write('\n')
        except Exception as e:
            print(f"Warning: Failed to write to global result file: {e}")

    def test(self, setting: str, test: int = 0):
        """
        最终测试函数
        """
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        criterion = self._select_criterion()

        # [SCALING INSTRUMENTATION] — wall-clock timer for inference
        import time as _time_mod
        _infer_t0 = _time_mod.perf_counter()
        test_loss, preds, trues, peak_trues_all = self._run_inference(
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
        test_loss_info = f"Test Loss: {test_loss:.7f}"
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
        
        peak_metrics = self._evaluate_peaks(
            preds, trues, 
            peak_trues=peak_trues_all,
            visualize=getattr(self.args, 'vis_test_peaks', True),
            data_flag='test'
        )
        
        if peak_metrics and peak_trues_all is not None:
            print('Peak detection shape:', preds.shape, peak_trues_all.shape)
        
        self._save_test_results(setting, std_metrics, peak_metrics, preds, trues)
        
        # test_results PDF 生成已禁用，避免产生大量文件
        
        print("\n" + "=" * 80)
        print("TRAINING AND TESTING COMPLETED!")
        print("=" * 80)
        self.write_log("\n" + "=" * 80)
        self.write_log("TRAINING AND TESTING COMPLETED!")
        self.write_log("=" * 80)
            
        return