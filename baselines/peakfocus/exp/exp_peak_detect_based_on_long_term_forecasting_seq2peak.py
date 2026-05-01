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

# ==================== Matplotlib 全局字体配置 (完全对齐 Reference) ====================
plt.rcParams['font.family'] = 'serif'  # 字体系列
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']  # 衬线字体
plt.rcParams['font.size'] = 18  # 默认字体大小
plt.rcParams['axes.labelsize'] = 24  # 坐标轴标签字体大小
plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
plt.rcParams['axes.titlesize'] = 24  # 图表标题字体大小
plt.rcParams['xtick.labelsize'] = 24  # x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 24  # y轴刻度标签字体大小
plt.rcParams['legend.fontsize'] = 22  # 图例字体大小 (修正: Reference为22)
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


# ==================== 峰值评估辅助函数 ====================

def detect_peaks_findpeaks(data, method='peakdetect', lookahead=10):
    """使用findpeaks检测峰值 (仅用于真实值 Ground Truth)"""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if len(data.shape) > 1:
        data = data.flatten()
    
    fp = findpeaks(method=method, lookahead=lookahead)
    results = fp.fit(data)
    
    if results and 'df' in results and results['df'] is not None:
        return results['df'][results['df']['peak'] == True].index.values
    return np.array([])


def match_peaks_with_tolerance(true_peaks, pred_peaks, tolerance=5):
    """匹配真实峰值和预测峰值"""
    tp_pairs = []
    matched_true = set()
    matched_pred = set()

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


def calculate_peak_metrics(tp_pairs, fp, fn, true_values, pred_values):
    """计算峰值检测指标"""
    tp = len(tp_pairs)
    fp_count = len(fp)
    fn_count = len(fn)
    
    precision = tp / (tp + fp_count) if (tp + fp_count) > 0 else 0.0
    recall = tp / (tp + fn_count) if (tp + fn_count) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    mae, mse = 0.0, 0.0
    if tp > 0:
        true_peak_values = true_values[[t for t, p in tp_pairs]]
        pred_peak_values = pred_values[[p for t, p in tp_pairs]]
        errors = true_peak_values - pred_peak_values
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
    
    return {
        'TP': tp, 'FP': fp_count, 'FN': fn_count,
        'Precision': precision, 'Recall': recall, 'F1': f1,
        'Peak_MAE': mae, 'Peak_MSE': mse
    }


class Exp_Peak_Detect_LTF_Seq2Peak(Exp_Basic):
    """
    Seq2Peak任务实验类 - 专门用于输出(全量序列, 峰值)的模型
    
    模型输出：
    - dec_out: 全量预测序列 [B, pred_len, C]
    - busy_out/peak_out: 峰值预测 [B, pred_len//24, C]
    - busy_indices: 峰值位置索引 (Optional)
    
    Loss: 0.5 * value_loss + 0.5 * peak_loss
    """
    
    def __init__(self, args):
        super(Exp_Peak_Detect_LTF_Seq2Peak, self).__init__(args)
        
        # Seq2Peak损失权重
        self.value_loss_weight = getattr(self.args, 'seq2peak_value_loss_weight', 0.5)
        self.peak_loss_weight = getattr(self.args, 'seq2peak_peak_loss_weight', 0.5)
        
        # 峰值评估参数
        self.peak_tolerance = getattr(self.args, 'peak_tolerance', 3)
        self.peak_lookahead = getattr(self.args, 'peak_lookahead', 3)
        self.enable_peak_eval = getattr(self.args, 'enable_peak_eval', True)
        
        # 初始化可视化文件夹
        if self.enable_peak_eval and getattr(self.args, 'vis_test_peaks', False):
            self.vis_folder = f'./visualization/output/predictions/{self.args.setting}/'
            if not os.path.exists(self.vis_folder):
                os.makedirs(self.vis_folder, exist_ok=True)
                print(f"Created visualization folder: {self.vis_folder}")
        else:
            self.vis_folder = None
        
        print("="*80)
        print("SEQ2PEAK TASK CONFIGURATION")
        print("="*80)
        print(f"Model: {self.args.model}")
        print(f"Value Loss Weight (full sequence): {self.value_loss_weight}")
        print(f"Peak Loss Weight (peak values): {self.peak_loss_weight}")
        print(f"Peak Evaluation Enabled: {self.enable_peak_eval}")
        if self.enable_peak_eval:
            print(f"Peak Tolerance: {self.peak_tolerance}")
            print(f"Peak Lookahead: {self.peak_lookahead}")
            print(f"Visualization Enabled: {self.vis_folder is not None}")
        print("="*80)
    
    def _print_and_log(self, message: str):
        """统一的打印和日志记录"""
        print(message)
        self.write_log(message)
    
    def _log_train_metrics(self, train_loss, train_value_loss, train_peak_loss):
        """记录训练指标"""
        self._print_and_log("[TRAIN]")
        loss_str = f" <LOSS> {train_loss:.7f} (Val×{self.value_loss_weight}: {train_value_loss:.7f}"
        if train_peak_loss is not None:
            loss_str += f", Peak×{self.peak_loss_weight}: {train_peak_loss:.7f}"
        loss_str += ")"
        self._print_and_log(loss_str)
    
    def _log_split_metrics(self, prefix, weighted_loss, value_loss, peak_loss, peak_cls_metrics):
        """记录单个数据集的指标"""
        self._print_and_log(f"[{prefix}]")
        loss_str = f" <LOSS> {weighted_loss:.7f} (Val×{self.value_loss_weight}: {value_loss:.7f}"
        if peak_loss is not None:
            loss_str += f", Peak×{self.peak_loss_weight}: {peak_loss:.7f}"
        loss_str += ")"
        self._print_and_log(loss_str)
        
        if peak_cls_metrics:
            self._print_and_log(" <METRICS>")
            metrics = [
                f"\tPrediction | MSE:    {peak_cls_metrics['Peak_Cls_MSE']:.4f}, MAE:    {peak_cls_metrics['Peak_Cls_MAE']:.4f}",
                f"\tTP Error   | TP_MSE: {peak_cls_metrics['Peak_Cls_TP_MSE']:.4f}, TP_MAE: {peak_cls_metrics['Peak_Cls_TP_MAE']:.4f}",
                f"\tFN Error   | FN_MSE: {peak_cls_metrics['Peak_Cls_FN_MSE']:.4f}, FN_MAE: {peak_cls_metrics['Peak_Cls_FN_MAE']:.4f}",
                f"\tAll Peaks  | A_MSE:  {peak_cls_metrics['Peak_Cls_All_True_Peaks_MSE']:.4f}, A_MAE:  {peak_cls_metrics['Peak_Cls_All_True_Peaks_MAE']:.4f}",
                f"\tDetection  | F1:     {peak_cls_metrics['Peak_Cls_F1']:.4f}, P:      {peak_cls_metrics['Peak_Cls_Precision']:.4f}, R: {peak_cls_metrics['Peak_Cls_Recall']:.4f}",
                f"\tMixed      | BPE:    {peak_cls_metrics['Peak_Cls_Balanced_Error']:.4f}, PIM:    {peak_cls_metrics['Peak_Cls_PIM']:.4f}"
            ]
            for m in metrics:
                self._print_and_log(m)

    def _print_and_log_metrics(self, metrics_dict: Dict, title: str):
        """打印和记录指标"""
        print(f"\n{title}")
        print("=" * 80)
        self.write_log(f"\n{title}\n" + "=" * 80)
        
        for key, value in metrics_dict.items():
            line = f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
            print(line)
            self.write_log(line)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        """验证函数"""
        total_loss = []
        total_value_loss = []
        total_peak_loss = []
        
        self.model.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(vali_loader):
                if len(data_batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = data_batch
                    batch_cycle = batch_cycle.int().to(self.device)
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = data_batch
                    batch_cycle = None
                    
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward
                if self.args.use_amp:
                    with autocast():
                        if 'CycleNet' in self.args.model and batch_cycle is not None:
                            model_out = self.model(batch_x, batch_cycle)
                        else:
                            model_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'CycleNet' in self.args.model and batch_cycle is not None:
                        model_out = self.model(batch_x, batch_cycle)
                    else:
                        model_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                if not isinstance(model_out, tuple) or len(model_out) < 2:
                    raise ValueError(f"Seq2Peak models must return (full_output, peak_output) or (full_output, peak_output, peak_indices)")
                
                if len(model_out) == 3:
                    outputs, peak_outputs, peak_indices = model_out
                else:
                    outputs, peak_outputs = model_out
                    peak_indices = None
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                peak_outputs = peak_outputs[:, :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()
                value_loss = criterion(pred, true)
                
                # Peak Loss
                pool_size = 24
                batch_size = batch_y.shape[0]
                num_peaks = peak_outputs.shape[1]
                
                targets_reshaped = batch_y.reshape(batch_size, num_peaks, pool_size, -1)
                targets_max_values, _ = torch.max(targets_reshaped, dim=2)
                peak_loss = criterion(peak_outputs.detach(), targets_max_values.detach())
                
                weighted_loss = self.value_loss_weight * value_loss + self.peak_loss_weight * peak_loss

                total_loss.append(weighted_loss.item())
                total_value_loss.append(value_loss.item())
                total_peak_loss.append(peak_loss.item())
                
        avg_loss = np.average(total_loss)
        avg_value_loss = np.average(total_value_loss)
        avg_peak_loss = np.average(total_peak_loss)
        
        self.model.train()
        return avg_loss, avg_value_loss, avg_peak_loss

    def _evaluate_peaks_in_epoch(self, data_loader, data_obj, criterion, data_flag='val'):
        """在训练epoch中评估峰值指标（简化版，仅用于指标监控，不画图）"""
        self.model.eval()
        
        preds_list = []
        trues_list = []
        peak_indices_list = []
        
        total_loss = []
        total_value_loss = []
        total_peak_loss = []
        
        with torch.no_grad():
            for data_batch in data_loader:
                if len(data_batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = data_batch
                    batch_cycle = batch_cycle.int().to(self.device)
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = data_batch
                    batch_cycle = None
                    
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.use_amp:
                    with autocast():
                        if 'CycleNet' in self.args.model and batch_cycle is not None:
                            model_out = self.model(batch_x, batch_cycle)
                        else:
                            model_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'CycleNet' in self.args.model and batch_cycle is not None:
                        model_out = self.model(batch_x, batch_cycle)
                    else:
                        model_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                if len(model_out) == 3:
                    outputs, peak_outputs, peak_indices = model_out
                    f_dim = -1 if self.args.features == 'MS' else 0
                    peak_indices_np = peak_indices[:, :, f_dim:].detach().cpu().numpy()
                    peak_indices_list.append(peak_indices_np)
                else:
                    outputs, peak_outputs = model_out
                    
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                peak_outputs = peak_outputs[:, :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                value_loss = criterion(outputs, batch_y)
                
                pool_size = 24
                batch_size = batch_y.shape[0]
                num_peaks = peak_outputs.shape[1]
                targets_reshaped = batch_y.reshape(batch_size, num_peaks, pool_size, -1)
                targets_max_values, _ = torch.max(targets_reshaped, dim=2)
                peak_loss = criterion(peak_outputs, targets_max_values)
                
                weighted_loss = self.value_loss_weight * value_loss + self.peak_loss_weight * peak_loss

                total_loss.append(weighted_loss.item())
                total_value_loss.append(value_loss.item())
                total_peak_loss.append(peak_loss.item())
                
                outputs_np = outputs.detach().cpu().numpy()
                batch_y_np = batch_y.detach().cpu().numpy()
                
                # Inverse Transform if needed
                if data_obj.scale and self.args.inverse:
                    shape = outputs_np.shape
                    outputs_np = data_obj.inverse_transform(outputs_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y_np = data_obj.inverse_transform(batch_y_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                
                preds_list.append(outputs_np)
                trues_list.append(batch_y_np)
        
        preds = np.concatenate(preds_list, axis=0)
        trues = np.concatenate(trues_list, axis=0)
        
        avg_loss = np.average(total_loss)
        avg_value_loss = np.average(total_value_loss)
        avg_peak_loss = np.average(total_peak_loss)
        
        # 计算峰值指标
        peak_cls_metrics = None
        if self.enable_peak_eval:
            has_model_indices = len(peak_indices_list) > 0
            if has_model_indices:
                peak_indices_all = np.concatenate(peak_indices_list, axis=0)
            
            # 简化的指标计算（只取部分样本提速）
            n_eval_samples = min(preds.shape[0], 100)
            total_tp, total_fp, total_fn = 0, 0, 0
            all_tp_errors = []
            all_fn_errors = []
            all_fp_errors = []
            all_true_peak_errors = []
            
            for i in range(n_eval_samples):
                pred_seq = preds[i, :, 0]
                true_seq = trues[i, :, 0]
                
                # GT: findpeaks
                true_peaks = detect_peaks_findpeaks(true_seq, method='peakdetect', lookahead=self.peak_lookahead)
                
                # Pred: Model Indices (强制使用模型输出)
                if has_model_indices and i < peak_indices_all.shape[0]:
                    model_peak_indices = peak_indices_all[i, :, 0].astype(int)
                    # 转换 Absolute Indices (relative to decoder input) -> Relative Indices (relative to pred window)
                    label_len = self.args.label_len
                    pred_peaks = []
                    for idx in model_peak_indices:
                        rel_idx = idx - label_len
                        if 0 <= rel_idx < len(pred_seq):
                            pred_peaks.append(rel_idx)
                    pred_peaks = np.array(pred_peaks)
                else:
                    # Fallback
                    pred_peaks = detect_peaks_findpeaks(pred_seq, method='peakdetect', lookahead=self.peak_lookahead)
                
                tp_pairs, fp, fn = match_peaks_with_tolerance(true_peaks, pred_peaks, tolerance=self.peak_tolerance)
                
                total_tp += len(tp_pairs)
                total_fp += len(fp)
                total_fn += len(fn)
                
                if len(tp_pairs) > 0:
                    true_tp_values = true_seq[[t for t, p in tp_pairs]]
                    pred_tp_values = pred_seq[[p for t, p in tp_pairs]]
                    tp_errors = true_tp_values - pred_tp_values
                    all_tp_errors.extend(tp_errors)
                    all_true_peak_errors.extend(tp_errors)

                if len(fn) > 0:
                    fn_errors = true_seq[fn] - pred_seq[fn]
                    all_fn_errors.extend(fn_errors)
                    all_true_peak_errors.extend(fn_errors)
            
            # 计算 F1
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # 计算各项误差
            # 1. TP MSE/MAE
            if len(all_tp_errors) > 0:
                tp_mse = np.mean(np.array(all_tp_errors) ** 2)
                tp_mae = np.mean(np.abs(all_tp_errors))
            else:
                tp_mse = 0.0
                tp_mae = 0.0
            
            # 2. FN MSE/MAE
            if len(all_fn_errors) > 0:
                fn_mse = np.mean(np.array(all_fn_errors) ** 2)
                fn_mae = np.mean(np.abs(all_fn_errors))
            else:
                fn_mse = 0.0
                fn_mae = 0.0
            
            # 3. All True Peaks MSE/MAE
            if len(all_true_peak_errors) > 0:
                all_peaks_mse = np.mean(np.array(all_true_peak_errors) ** 2)
                all_peaks_mae = np.mean(np.abs(all_true_peak_errors))
            else:
                all_peaks_mse = 0.0
                all_peaks_mae = 0.0
                
            # 4. Overall Prediction MSE/MAE (on sampled data)
            # 为了准确，我们对前 n_eval_samples 计算
            sampled_preds = preds[:n_eval_samples]
            sampled_trues = trues[:n_eval_samples]
            overall_mse = np.mean((sampled_preds - sampled_trues) ** 2)
            overall_mae = np.mean(np.abs(sampled_preds - sampled_trues))
            
            # BPE
            alpha = 0.5
            e_class = 1.0 - f1
            e_reg = 1.0 - (1.0 / (1.0 + tp_mse))
            bpe = (alpha * e_class) + ((1.0 - alpha) * e_reg)
            pim = (1.0 + tp_mse) / (f1 + 0.01)

            peak_cls_metrics = {
                'Peak_Cls_MSE': overall_mse,
                'Peak_Cls_MAE': overall_mae,
                'Peak_Cls_TP_MSE': tp_mse,
                'Peak_Cls_TP_MAE': tp_mae,
                'Peak_Cls_FN_MSE': fn_mse,
                'Peak_Cls_FN_MAE': fn_mae,
                'Peak_Cls_All_True_Peaks_MSE': all_peaks_mse,
                'Peak_Cls_All_True_Peaks_MAE': all_peaks_mae,
                'Peak_Cls_Precision': precision,
                'Peak_Cls_Recall': recall,
                'Peak_Cls_F1': f1,
                'Peak_Cls_Balanced_Error': bpe,
                'Peak_Cls_PIM': pim
            }

        self.model.train()
        return avg_loss, avg_value_loss, avg_peak_loss, peak_cls_metrics

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = GradScaler() if self.args.use_amp else None
        
        print("\n" + "="*80)
        print("SEQ2PEAK TRAINING START")
        print("="*80)
        print(f"Model Selection Strategy: Validation Loss (0.5×Value + 0.5×Peak)")
        if self.enable_peak_eval:
            print(f"Peak Evaluation: Enabled (tolerance={self.peak_tolerance}, lookahead={self.peak_lookahead})")
        print("="*80 + "\n")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_value_loss = []
            train_peak_loss = []

            self.model.train()
            epoch_time = time.time()
            
            for i, data_batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                if len(data_batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = data_batch
                    batch_cycle = batch_cycle.int().to(self.device)
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = data_batch
                    batch_cycle = None

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with autocast():
                        if 'CycleNet' in self.args.model and batch_cycle is not None:
                            model_out = self.model(batch_x, batch_cycle)
                        else:
                            model_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                        if len(model_out) == 3:
                            outputs, peak_outputs, peak_indices = model_out
                        else:
                            outputs, peak_outputs = model_out
                        
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        peak_outputs = peak_outputs[:, :, f_dim:]
                        batch_y_targets = batch_y[:, -self.args.pred_len:, f_dim:]
                        
                        value_loss = criterion(outputs, batch_y_targets)
                        
                        pool_size = 24
                        batch_size = batch_y_targets.shape[0]
                        num_peaks = peak_outputs.shape[1]
                        targets_reshaped = batch_y_targets.reshape(batch_size, num_peaks, pool_size, -1)
                        targets_max_values, _ = torch.max(targets_reshaped, dim=2)
                        peak_loss = criterion(peak_outputs, targets_max_values)
                        
                        loss = self.value_loss_weight * value_loss + self.peak_loss_weight * peak_loss
                    
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    if 'CycleNet' in self.args.model and batch_cycle is not None:
                        model_out = self.model(batch_x, batch_cycle)
                    else:
                        model_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    if len(model_out) == 3:
                        outputs, peak_outputs, peak_indices = model_out
                    else:
                        outputs, peak_outputs = model_out
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    peak_outputs = peak_outputs[:, :, f_dim:]
                    batch_y_targets = batch_y[:, -self.args.pred_len:, f_dim:]
                    
                    value_loss = criterion(outputs, batch_y_targets)
                    
                    pool_size = 24
                    batch_size = batch_y_targets.shape[0]
                    num_peaks = peak_outputs.shape[1]
                    targets_reshaped = batch_y_targets.reshape(batch_size, num_peaks, pool_size, -1)
                    targets_max_values, _ = torch.max(targets_reshaped, dim=2)
                    peak_loss = criterion(peak_outputs, targets_max_values)
                    
                    loss = self.value_loss_weight * value_loss + self.peak_loss_weight * peak_loss
                    
                    loss.backward()
                    model_optim.step()

                train_loss.append(loss.item())
                train_value_loss.append(value_loss.item())
                train_peak_loss.append(peak_loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f} "
                          f"(value: {value_loss.item():.4f}, peak: {peak_loss.item():.4f})")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            epoch_cost_time = time.time() - epoch_time
            train_loss = np.average(train_loss)
            train_value_loss = np.average(train_value_loss)
            train_peak_loss = np.average(train_peak_loss)
            
            vali_loss, vali_value_loss, vali_peak_loss, vali_peak_metrics = self._evaluate_peaks_in_epoch(
                vali_loader, vali_data, criterion, 'val')
            test_loss, test_value_loss, test_peak_loss, test_peak_metrics = self._evaluate_peaks_in_epoch(
                test_loader, test_data, criterion, 'test')

            self._print_and_log(f"=================EPOCH {epoch + 1} | Tolerance={self.peak_tolerance}=================")
            
            self._log_train_metrics(train_loss, train_value_loss, train_peak_loss)
            self._print_and_log(f" <TIME> Epoch cost: {epoch_cost_time:.2f}s")
            
            self._log_split_metrics("VAL", vali_loss, vali_value_loss, vali_peak_loss, vali_peak_metrics)
            self._log_split_metrics("TEST", test_loss, test_value_loss, test_peak_loss, test_peak_metrics)
            
            self._print_and_log("[OTHERS]")
            self._print_and_log(" [MODEL SELECTION]")
            
            prev_best = early_stopping.val_loss_min
            early_stopping(vali_loss, self.model, path)
            
            if early_stopping.early_stop:
                self._print_and_log(f"\tValidation Loss: {vali_loss:.6f}")
                self._print_and_log("\tEarly stopping triggered!")
            elif early_stopping.val_loss_min == vali_loss:
                self._print_and_log(f"\tValidation Loss improved ({prev_best:.6f} --> {vali_loss:.6f}). Saving model...")
            else:
                self._print_and_log(f"\tValidation Loss: {vali_loss:.6f} (best: {prev_best:.6f})")
            
            self._print_and_log("=" * 56)
            
            if early_stopping.early_stop:
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def _visualize_peak_detection(self, true_seq: np.ndarray, pred_seq: np.ndarray, 
                                   true_peaks: np.ndarray, pred_peaks: np.ndarray, 
                                   tp_pairs: List, fp: List, fn: List, 
                                   metrics: Dict, save_folder: str, 
                                   sample_idx: int) -> None:
        """
        可视化峰值检测结果 - 严格对齐 Reference File 的样式
        """
        import matplotlib.pyplot as plt
        
        # 创建单个图形 - 尺寸与 Reference 保持一致
        fig, ax = plt.subplots(1, 1, figsize=(18, 7.8))
        time_steps = np.arange(len(true_seq))
        
        # 1. 绘制值预测曲线 (linewidth=2, alpha=0.8)
        ax.plot(time_steps, true_seq, 'b-', label='True Value', linewidth=2, alpha=0.8)
        ax.plot(time_steps, pred_seq, 'r-', label='Predicted Value', linewidth=2, alpha=0.8)
        
        # 2. 绘制True Peaks (green circle, s=120)
        if len(true_peaks) > 0:
            ax.scatter(true_peaks, true_seq[true_peaks.astype(int)], 
                       c='green', s=120, marker='o', edgecolors='darkgreen', 
                       linewidths=2.5, label=f'True Peaks (n={len(true_peaks)})', 
                       zorder=5, alpha=0.8)
        
        # 3. 绘制Predicted Peaks (orange triangle, s=120) - 仅作为辅助显示
        # if len(pred_peaks) > 0:
        #      ax.scatter(pred_peaks, pred_seq[pred_peaks.astype(int)], 
        #                c='orange', s=120, marker='^', edgecolors='darkorange', 
        #                linewidths=2.5, label=f'Pred Peaks (n={len(pred_peaks)})', 
        #                zorder=5, alpha=0.8)
        
        # 4. 绘制匹配结果
        # TP: Lime Star, s=180, edgecolors='green' (Reference样式)
        if len(tp_pairs) > 0:
            tp_plot_indices_pred = [p for t, p in tp_pairs]
            ax.scatter(tp_plot_indices_pred, pred_seq[tp_plot_indices_pred], 
                       c='lime', s=180, marker='*', edgecolors='green', 
                       linewidths=3, label=f'TP (n={len(tp_pairs)})', 
                       zorder=6, alpha=0.9)
            
        # FP: Red X, s=150
        if len(fp) > 0:
            ax.scatter(fp, pred_seq[fp], 
                       c='red', s=150, marker='x', linewidths=3, 
                       label=f'FP (n={len(fp)})', zorder=6)
            
        # FN: Blue X, s=150
        if len(fn) > 0:
            ax.scatter(fn, true_seq[fn], 
                       c='blue', s=150, marker='x', linewidths=3, 
                       label=f'FN (n={len(fn)})', zorder=6)
        
        # 5. 设置图例 (3列, 粗体, 紧凑布局)
        ax.legend(loc='best', ncol=3, prop={'weight': 'bold'}, 
             columnspacing=1.0, handletextpad=0.5, borderpad=0.4)
        ax.grid(True, alpha=0.3)
        
        # 6. 设置标题
        title = (f'Peak Detection [TEST] - Sample {sample_idx}\n'
                 f'F1={metrics["F1"]:.3f}, P={metrics["Precision"]:.3f}, R={metrics["Recall"]:.3f}, Peak_MAE={metrics["Peak_MAE"]:.3f}')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 7. 坐标轴设置
        ax.set_ylabel('Value', fontsize=12)
        ax.set_xlabel('Time Steps', fontsize=12)
        
        # 刻度标签加粗 & 方向向内
        plt.setp(ax.xaxis.get_majorticklabels(), weight='bold')
        plt.setp(ax.yaxis.get_majorticklabels(), weight='bold')
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        
        plt.tight_layout()
        
        # 保存
        filename = f'test_sample{sample_idx}.png'
        save_path = os.path.join(save_folder, filename)
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        except Exception as e:
            print(f"Warning: Failed to save visualization to {save_path}: {e}")
        finally:
            plt.close()

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        peak_indices_list = []  # 收集模型返回的峰值位置
        
        self.model.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(test_loader):
                if len(data_batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle = data_batch
                    batch_cycle = batch_cycle.int().to(self.device)
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = data_batch
                    batch_cycle = None

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.use_amp:
                    with autocast():
                        if 'CycleNet' in self.args.model and batch_cycle is not None:
                            model_out = self.model(batch_x, batch_cycle)
                        else:
                            model_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'CycleNet' in self.args.model and batch_cycle is not None:
                        model_out = self.model(batch_x, batch_cycle)
                    else:
                        model_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if isinstance(model_out, tuple):
                    if len(model_out) == 3:
                        outputs, peak_outputs, peak_indices = model_out
                        f_dim = -1 if self.args.features == 'MS' else 0
                        peak_indices_np = peak_indices[:, :, f_dim:].detach().cpu().numpy()
                        peak_indices_list.append(peak_indices_np)
                    else:
                        outputs = model_out[0]
                else:
                    outputs = model_out
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                preds.append(outputs)
                trues.append(batch_y)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # Result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        # ==================== 峰值评估 ====================
        peak_cls_metrics = None
        if self.enable_peak_eval:
            has_model_indices = len(peak_indices_list) > 0
            if has_model_indices:
                peak_indices_all = np.concatenate(peak_indices_list, axis=0)
            
            all_metrics = []
            all_tp_errors = []
            all_fn_errors = []
            all_fp_errors = []
            all_true_peak_errors = []
            
            total_tp = 0
            total_fp = 0
            total_fn = 0
            
            n_samples = preds.shape[0]
            # 严格按照168采样进行可视化
            vis_indices = [i for i in range(n_samples) if i % 168 == 0]
            
            for i in range(n_samples):
                pred_seq = preds[i, :, 0]
                true_seq = trues[i, :, 0]
                
                # GT: findpeaks
                true_peaks = detect_peaks_findpeaks(true_seq, method='peakdetect', lookahead=self.peak_lookahead)
                
                # Pred: 必须使用模型的 indices (如果存在)
                if has_model_indices and i < peak_indices_all.shape[0]:
                    model_peak_indices = peak_indices_all[i, :, 0].astype(int)
                    # 转换 Absolute Indices (relative to decoder input) -> Relative Indices (relative to pred window)
                    label_len = self.args.label_len
                    pred_peaks = []
                    for idx in model_peak_indices:
                        rel_idx = idx - label_len
                        if 0 <= rel_idx < len(pred_seq):
                            pred_peaks.append(rel_idx)
                    pred_peaks = np.array(pred_peaks)
                else:
                    pred_peaks = detect_peaks_findpeaks(pred_seq, method='peakdetect', lookahead=self.peak_lookahead)
                
                tp_pairs, fp, fn = match_peaks_with_tolerance(true_peaks, pred_peaks, tolerance=self.peak_tolerance)
                
                total_tp += len(tp_pairs)
                total_fp += len(fp)
                total_fn += len(fn)
                
                # 仅对采样点进行可视化 (严格对齐 Reference)
                if self.vis_folder and i in vis_indices:
                    metrics_for_vis = calculate_peak_metrics(tp_pairs, fp, fn, true_seq, pred_seq)
                    self._visualize_peak_detection(
                        true_seq, pred_seq, true_peaks, pred_peaks,
                        tp_pairs, fp, fn, metrics_for_vis, self.vis_folder, i
                    )
                
                if len(tp_pairs) > 0:
                    true_tp_values = true_seq[[t for t, p in tp_pairs]]
                    pred_tp_values = pred_seq[[p for t, p in tp_pairs]]
                    tp_errors = true_tp_values - pred_tp_values
                    all_tp_errors.extend(tp_errors)
                    all_true_peak_errors.extend(tp_errors)
                
                if len(fn) > 0:
                    fn_errors = true_seq[fn] - pred_seq[fn]
                    all_fn_errors.extend(fn_errors)
                    all_true_peak_errors.extend(fn_errors)
                
                if len(fp) > 0:
                    fp_errors = pred_seq[fp] - true_seq[fp]
                    all_fp_errors.extend(fp_errors)
            
            # 聚合指标
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            avg_tp = total_tp / n_samples
            avg_fp = total_fp / n_samples
            avg_fn = total_fn / n_samples
            
            if len(all_tp_errors) > 0:
                tp_mse = np.mean(np.array(all_tp_errors) ** 2)
                tp_mae = np.mean(np.abs(all_tp_errors))
            else:
                tp_mse = 0.0
                tp_mae = 0.0
            
            if len(all_fn_errors) > 0:
                fn_mse = np.mean(np.array(all_fn_errors) ** 2)
                fn_mae = np.mean(np.abs(all_fn_errors))
            else:
                fn_mse = 0.0
                fn_mae = 0.0
                
            if len(all_true_peak_errors) > 0:
                all_true_peak_mse = np.mean(np.array(all_true_peak_errors) ** 2)
                all_true_peak_mae = np.mean(np.abs(all_true_peak_errors))
            else:
                all_true_peak_mse = 0.0
                all_true_peak_mae = 0.0
            
            alpha = 0.5
            e_class = 1.0 - f1
            e_reg = 1.0 - (1.0 / (1.0 + tp_mse))
            balanced_peak_error = (alpha * e_class) + ((1.0 - alpha) * e_reg)
            peak_pim = (1.0 + tp_mse) / (f1 + 0.01)
            
            peak_cls_metrics = {
                'Peak_Cls_TP': avg_tp,
                'Peak_Cls_FP': avg_fp,
                'Peak_Cls_FN': avg_fn,
                'Peak_Cls_Precision': precision,
                'Peak_Cls_Recall': recall,
                'Peak_Cls_F1': f1,
                'Peak_Cls_TP_MSE': tp_mse,
                'Peak_Cls_TP_MAE': tp_mae,
                'Peak_Cls_FN_MSE': fn_mse,
                'Peak_Cls_FN_MAE': fn_mae,
                'Peak_Cls_All_True_Peaks_MSE': all_true_peak_mse,
                'Peak_Cls_All_True_Peaks_MAE': all_true_peak_mae,
                'Peak_Cls_Balanced_Error': balanced_peak_error,
                'Peak_Cls_PIM': peak_pim,
                'Peak_Cls_MSE': mse,
                'Peak_Cls_MAE': mae
            }
        
        # ==================== 保存结果 ====================
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        if peak_cls_metrics:
            np.save(folder_path + 'peak_metrics.npy', np.array(peak_cls_metrics, dtype=object))
        
        # ==================== 打印和记录评估结果 ====================
        std_metrics = {
            'MAE': mae, 'MSE': mse, 'RMSE': rmse, 
            'MAPE': mape, 'MSPE': mspe
        }
        self._print_and_log_metrics(std_metrics, "EVALUATION RESULTS")
        
        if peak_cls_metrics:
            peak_display_metrics = {
                'Balanced_Peak_Error (BPE)': peak_cls_metrics['Peak_Cls_Balanced_Error'],
                'Peak_Integrated_Metric (PIM)': peak_cls_metrics['Peak_Cls_PIM'],
                'F1-Score': peak_cls_metrics['Peak_Cls_F1'],
                'Precision': peak_cls_metrics['Peak_Cls_Precision'],
                'Recall': peak_cls_metrics['Peak_Cls_Recall'],
                'TP_MSE': peak_cls_metrics['Peak_Cls_TP_MSE'],
                'TP_MAE': peak_cls_metrics['Peak_Cls_TP_MAE'],
                'FN_MSE': peak_cls_metrics['Peak_Cls_FN_MSE'],
                'FN_MAE': peak_cls_metrics['Peak_Cls_FN_MAE'],
                'All_True_Peaks_MSE': peak_cls_metrics['Peak_Cls_All_True_Peaks_MSE'],
                'All_True_Peaks_MAE': peak_cls_metrics['Peak_Cls_All_True_Peaks_MAE'],
                'Average TP': peak_cls_metrics['Peak_Cls_TP'],
                'Average FP': peak_cls_metrics['Peak_Cls_FP'],
                'Average FN': peak_cls_metrics['Peak_Cls_FN'],
            }
            self._print_and_log_metrics(peak_display_metrics, f"PEAK CLASSIFICATION METRICS (Tolerance={self.peak_tolerance})")
        
        # ==================== 写入 result_seq2peak.txt ====================
        try:
            with open("result_seq2peak.txt", 'a') as f:
                f.write(setting + "  \n")
                f.write(f"mse:{mse:.6f}, mae:{mae:.6f}\n")
                if peak_cls_metrics:
                    f.write('BPE:{:.4f}, PIM:{:.4f}, F1:{:.4f}, P:{:.4f}, R:{:.4f}, TP_MSE:{:.4f}, TP_MAE:{:.4f}, FN_MSE:{:.4f}, All_Peaks_MSE:{:.4f} (Tol={})\n'.format(
                        peak_cls_metrics['Peak_Cls_Balanced_Error'],
                        peak_cls_metrics['Peak_Cls_PIM'],
                        peak_cls_metrics['Peak_Cls_F1'], 
                        peak_cls_metrics['Peak_Cls_Precision'], 
                        peak_cls_metrics['Peak_Cls_Recall'], 
                        peak_cls_metrics['Peak_Cls_TP_MSE'],
                        peak_cls_metrics['Peak_Cls_TP_MAE'],
                        peak_cls_metrics['Peak_Cls_FN_MSE'],
                        peak_cls_metrics['Peak_Cls_All_True_Peaks_MSE'],
                        self.peak_tolerance))
                f.write('\n')
        except Exception as e:
            print(f"Warning: Failed to write to global result file: {e}")
        
        if self.vis_folder and self.enable_peak_eval:
            print(f"\n[Visualization] Saved peak detection plots to: {self.vis_folder}")

        self._print_and_log("\n" + "="*80)
        self._print_and_log("TESTING COMPLETED!")
        self._print_and_log("="*80)

        return