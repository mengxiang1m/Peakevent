"""
计算多个实验结果的平均指标
从指定文件夹中的 experiment_log.txt 文件中提取最终评估指标并计算平均值
"""

import os
import re
import argparse
import numpy as np
from pathlib import Path


def parse_experiment_log(log_file_path):
    """
    解析实验日志文件,提取最终的评估指标
    
    Args:
        log_file_path: 日志文件路径
        
    Returns:
        dict: 包含所有指标的字典
    """
    metrics = {}
    
    with open(log_file_path, 'r', encoding='gbk') as f:
        content = f.read()
    
    # 查找 EVALUATION RESULTS 部分
    eval_section = re.search(r'EVALUATION RESULTS\s*=+\s*(.*?)\s*(?:PEAK CLASSIFICATION METRICS|$)', 
                            content, re.DOTALL)
    
    if eval_section:
        eval_text = eval_section.group(1)
        
        # 提取各项指标（支持大写和小写格式）
        mae_match = re.search(r'(?:MAE|mae):\s*([\d.]+)', eval_text, re.IGNORECASE)
        mse_match = re.search(r'(?:MSE|mse):\s*([\d.]+)', eval_text, re.IGNORECASE)
        rmse_match = re.search(r'(?:RMSE|rmse):\s*([\d.]+)', eval_text, re.IGNORECASE)
        mape_match = re.search(r'(?:MAPE|mape):\s*([\d.]+)', eval_text, re.IGNORECASE)
        mspe_match = re.search(r'(?:MSPE|mspe):\s*([\d.]+)', eval_text, re.IGNORECASE)
        dtw_match = re.search(r'(?:DTW|dtw):\s*([\d.]+|Not calculated)', eval_text, re.IGNORECASE)
        
        if mae_match:
            metrics['MAE'] = float(mae_match.group(1))
        if mse_match:
            metrics['MSE'] = float(mse_match.group(1))
        if rmse_match:
            metrics['RMSE'] = float(rmse_match.group(1))
        if mape_match:
            metrics['MAPE'] = float(mape_match.group(1))
        if mspe_match:
            metrics['MSPE'] = float(mspe_match.group(1))
        if dtw_match and dtw_match.group(1) != 'Not calculated':
            metrics['DTW'] = float(dtw_match.group(1))
    
    # 查找 PEAK CLASSIFICATION METRICS 部分
    peak_section = re.search(r'PEAK CLASSIFICATION METRICS.*?=+\s*(.*?)$', 
                            content, re.DOTALL)
    
    if peak_section:
        peak_text = peak_section.group(1)
        
        # 提取峰值分类指标（支持下划线和空格两种格式）
        bpe_match = re.search(r'Balanced[_ ]Peak[_ ]Error[_ ]?\(BPE\):\s*([\d.]+)', peak_text, re.IGNORECASE)
        pim_match = re.search(r'Peak[_ ]Integrated[_ ]Metric[_ ]?\(PIM\):\s*([\d.]+)', peak_text, re.IGNORECASE)
        f1_match = re.search(r'F1-Score:\s*([\d.]+)', peak_text)
        precision_match = re.search(r'Precision:\s*([\d.]+)', peak_text)
        recall_match = re.search(r'Recall:\s*([\d.]+)', peak_text)
        tp_mse_match = re.search(r'TP_MSE:\s*([\d.]+)', peak_text)
        tp_mae_match = re.search(r'TP_MAE:\s*([\d.]+)', peak_text)
        fn_mse_match = re.search(r'FN_MSE:\s*([\d.]+)', peak_text)
        fn_mae_match = re.search(r'FN_MAE:\s*([\d.]+)', peak_text)
        all_peaks_mse_match = re.search(r'All[_ ]True[_ ]Peaks[_ ]MSE:\s*([\d.]+)', peak_text, re.IGNORECASE)
        all_peaks_mae_match = re.search(r'All[_ ]True[_ ]Peaks[_ ]MAE:\s*([\d.]+)', peak_text, re.IGNORECASE)
        fp_mse_match = re.search(r'FP_MSE:\s*([\d.]+)', peak_text)
        fp_mae_match = re.search(r'FP_MAE:\s*([\d.]+)', peak_text)
        comp_mse_match = re.search(r'Comprehensive_MSE:\s*([\d.]+)', peak_text)
        comp_mae_match = re.search(r'Comprehensive_MAE:\s*([\d.]+)', peak_text)
        avg_tp_match = re.search(r'Average TP:\s*([\d.]+)', peak_text)
        avg_fp_match = re.search(r'Average FP:\s*([\d.]+)', peak_text)
        avg_fn_match = re.search(r'Average FN:\s*([\d.]+)', peak_text)
        
        if bpe_match:
            metrics['BPE'] = float(bpe_match.group(1))
        if pim_match:
            metrics['PIM'] = float(pim_match.group(1))
        if f1_match:
            metrics['F1-Score'] = float(f1_match.group(1))
        if precision_match:
            metrics['Precision'] = float(precision_match.group(1))
        if recall_match:
            metrics['Recall'] = float(recall_match.group(1))
        if tp_mse_match:
            metrics['TP_MSE'] = float(tp_mse_match.group(1))
        if tp_mae_match:
            metrics['TP_MAE'] = float(tp_mae_match.group(1))
        if fn_mse_match:
            metrics['FN_MSE'] = float(fn_mse_match.group(1))
        if fn_mae_match:
            metrics['FN_MAE'] = float(fn_mae_match.group(1))
        if all_peaks_mse_match:
            metrics['All_True_Peaks_MSE'] = float(all_peaks_mse_match.group(1))
        if all_peaks_mae_match:
            metrics['All_True_Peaks_MAE'] = float(all_peaks_mae_match.group(1))
        if fp_mse_match:
            metrics['FP_MSE'] = float(fp_mse_match.group(1))
        if fp_mae_match:
            metrics['FP_MAE'] = float(fp_mae_match.group(1))
        if comp_mse_match:
            metrics['Comprehensive_MSE'] = float(comp_mse_match.group(1))
        if comp_mae_match:
            metrics['Comprehensive_MAE'] = float(comp_mae_match.group(1))
        if avg_tp_match:
            metrics['Average TP'] = float(avg_tp_match.group(1))
        if avg_fp_match:
            metrics['Average FP'] = float(avg_fp_match.group(1))
        if avg_fn_match:
            metrics['Average FN'] = float(avg_fn_match.group(1))
    
    return metrics


def calculate_mean_metrics(results_dir, folder_pattern=None, folder_list=None):
    """
    计算多个实验结果的平均指标
    
    Args:
        results_dir: results 文件夹路径
        folder_pattern: 文件夹名称模式 (正则表达式), 如果为 None 则使用 folder_list
        folder_list: 指定的文件夹名称列表, 如果为 None 则使用 folder_pattern
        
    Returns:
        dict: 包含所有指标平均值和标准差的字典
    """
    results_path = Path(results_dir)
    
    # 获取所有符合条件的文件夹
    if folder_list is not None:
        # 使用指定的文件夹列表
        folders = [results_path / folder_name for folder_name in folder_list 
                  if (results_path / folder_name).exists()]
    elif folder_pattern is not None:
        # 使用模式匹配
        pattern = re.compile(folder_pattern)
        folders = [f for f in results_path.iterdir() 
                  if f.is_dir() and pattern.match(f.name)]
    else:
        raise ValueError("必须提供 folder_pattern 或 folder_list 之一")
    
    if not folders:
        print(f"警告: 在 {results_dir} 中没有找到匹配的文件夹")
        return {}
    
    print(f"找到 {len(folders)} 个实验文件夹:")
    for folder in folders:
        print(f"  - {folder.name}")
    print()
    
    # 收集所有实验的指标
    all_metrics = []
    valid_folders = []
    
    for folder in folders:
        log_file = folder / 'experiment_log.txt'
        if log_file.exists():
            try:
                metrics = parse_experiment_log(log_file)
                if metrics:
                    all_metrics.append(metrics)
                    valid_folders.append(folder.name)
                    print(f"✓ 成功解析: {folder.name}")
                else:
                    print(f"✗ 未找到指标: {folder.name}")
            except Exception as e:
                print(f"✗ 解析错误 {folder.name}: {e}")
        else:
            print(f"✗ 文件不存在: {log_file}")
    
    if not all_metrics:
        print("\n错误: 没有成功解析任何实验日志")
        return {}
    
    print(f"\n成功解析 {len(all_metrics)} 个实验")
    
    # 计算平均值和标准差
    metric_names = list(all_metrics[0].keys())
    mean_metrics = {}
    std_metrics = {}
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics if metric_name in m]
        if values:
            mean_metrics[metric_name] = np.mean(values)
            std_metrics[metric_name] = np.std(values)
    
    return {
        'mean': mean_metrics,
        'std': std_metrics,
        'count': len(all_metrics),
        'folders': valid_folders,
        'individual': all_metrics
    }


def print_results(results):
    """
    打印结果
    
    Args:
        results: calculate_mean_metrics 返回的结果字典
    """
    if not results:
        return
    
    print("\n" + "=" * 80)
    print("平均指标结果")
    print("=" * 80)
    print(f"实验数量: {results['count']}")
    print(f"实验文件夹: {', '.join(results['folders'])}")
    print()
    
    print("评估指标 (均值 ± 标准差)")
    print("-" * 80)
    
    mean_metrics = results['mean']
    std_metrics = results['std']
    
    # 按类别打印指标
    basic_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE']
    mixed_metrics = ['BPE', 'PIM']  # 混合指标
    detection_metrics = ['F1-Score', 'Precision', 'Recall']  # 峰值检测指标
    tp_metrics = ['TP_MSE', 'TP_MAE']
    fn_metrics = ['FN_MSE', 'FN_MAE']
    all_peaks_metrics = ['All_True_Peaks_MSE', 'All_True_Peaks_MAE']  # 新增：所有真实峰值指标
    fp_metrics = ['FP_MSE', 'FP_MAE']
    comp_metrics = ['Comprehensive_MSE', 'Comprehensive_MAE']
    count_metrics = ['Average TP', 'Average FP', 'Average FN']
    
    print("\n[基本评估指标]")
    for metric in basic_metrics:
        if metric in mean_metrics:
            print(f"  {metric:10s}: {mean_metrics[metric]:12.6f} ± {std_metrics[metric]:12.6f}")
    
    print("\n[混合指标]")
    for metric in mixed_metrics:
        if metric in mean_metrics:
            print(f"  {metric:10s}: {mean_metrics[metric]:12.6f} ± {std_metrics[metric]:12.6f}")
    
    print("\n[峰值检测指标]")
    for metric in detection_metrics:
        if metric in mean_metrics:
            print(f"  {metric:10s}: {mean_metrics[metric]:12.6f} ± {std_metrics[metric]:12.6f}")
    
    print("\n[TP指标]")
    for metric in tp_metrics:
        if metric in mean_metrics:
            print(f"  {metric:10s}: {mean_metrics[metric]:12.6f} ± {std_metrics[metric]:12.6f}")
    
    if any(metric in mean_metrics for metric in fn_metrics):
        print("\n[FN指标]")
        for metric in fn_metrics:
            if metric in mean_metrics:
                print(f"  {metric:10s}: {mean_metrics[metric]:12.6f} ± {std_metrics[metric]:12.6f}")
    
    if any(metric in mean_metrics for metric in all_peaks_metrics):
        print("\n[所有真实峰值指标 (TP+FN)]")
        for metric in all_peaks_metrics:
            if metric in mean_metrics:
                print(f"  {metric:20s}: {mean_metrics[metric]:12.6f} ± {std_metrics[metric]:12.6f}")
    
    if any(metric in mean_metrics for metric in fp_metrics):
        print("\n[FP指标]")
        for metric in fp_metrics:
            if metric in mean_metrics:
                print(f"  {metric:10s}: {mean_metrics[metric]:12.6f} ± {std_metrics[metric]:12.6f}")
    
    if any(metric in mean_metrics for metric in comp_metrics):
        print("\n[综合指标]")
        for metric in comp_metrics:
            if metric in mean_metrics:
                print(f"  {metric:18s}: {mean_metrics[metric]:12.6f} ± {std_metrics[metric]:12.6f}")
    
    print("\n[计数指标]")
    for metric in count_metrics:
        if metric in mean_metrics:
            print(f"  {metric:10s}: {mean_metrics[metric]:12.6f} ± {std_metrics[metric]:12.6f}")
    
    print("\n" + "=" * 80)
    
    # 打印每个实验的详细结果
    print("\n各实验详细结果:")
    print("-" * 80)
    for i, (folder_name, metrics) in enumerate(zip(results['folders'], results['individual'])):
        print(f"\n实验 {i} ({folder_name}):")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name:10s}: {value:.6f}")
            else:
                print(f"  {metric_name:10s}: {value}")


def save_results(results, output_file):
    """
    保存结果到文件
    
    Args:
        results: calculate_mean_metrics 返回的结果字典
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("平均指标结果\n")
        f.write("=" * 80 + "\n")
        f.write(f"实验数量: {results['count']}\n")
        f.write(f"实验文件夹: {', '.join(results['folders'])}\n\n")
        
        mean_metrics = results['mean']
        std_metrics = results['std']
        
        f.write("评估指标 (均值 ± 标准差)\n")
        f.write("-" * 80 + "\n")
        
        basic_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE']
        mixed_metrics = ['BPE', 'PIM']  # 混合指标
        detection_metrics = ['F1-Score', 'Precision', 'Recall']  # 峰值检测指标
        tp_metrics = ['TP_MSE', 'TP_MAE']
        fn_metrics = ['FN_MSE', 'FN_MAE']
        all_peaks_metrics = ['All_True_Peaks_MSE', 'All_True_Peaks_MAE']  # 新增：所有真实峰值指标
        fp_metrics = ['FP_MSE', 'FP_MAE']
        comp_metrics = ['Comprehensive_MSE', 'Comprehensive_MAE']
        count_metrics = ['Average TP', 'Average FP', 'Average FN']
        
        f.write("\n[基本评估指标]\n")
        for metric in basic_metrics:
            if metric in mean_metrics:
                f.write(f"  {metric:10s}: {mean_metrics[metric]:12.6f} ± {std_metrics[metric]:12.6f}\n")
        
        f.write("\n[混合指标]\n")
        for metric in mixed_metrics:
            if metric in mean_metrics:
                f.write(f"  {metric:10s}: {mean_metrics[metric]:12.6f} ± {std_metrics[metric]:12.6f}\n")
        
        f.write("\n[峰值检测指标]\n")
        for metric in detection_metrics:
            if metric in mean_metrics:
                f.write(f"  {metric:10s}: {mean_metrics[metric]:12.6f} ± {std_metrics[metric]:12.6f}\n")
        
        f.write("\n[TP指标]\n")
        for metric in tp_metrics:
            if metric in mean_metrics:
                f.write(f"  {metric:10s}: {mean_metrics[metric]:12.6f} ± {std_metrics[metric]:12.6f}\n")
        
        if any(metric in mean_metrics for metric in fn_metrics):
            f.write("\n[FN指标]\n")
            for metric in fn_metrics:
                if metric in mean_metrics:
                    f.write(f"  {metric:10s}: {mean_metrics[metric]:12.6f} ± {std_metrics[metric]:12.6f}\n")
        
        if any(metric in mean_metrics for metric in all_peaks_metrics):
            f.write("\n[所有真实峰值指标 (TP+FN)]\n")
            for metric in all_peaks_metrics:
                if metric in mean_metrics:
                    f.write(f"  {metric:20s}: {mean_metrics[metric]:12.6f} ± {std_metrics[metric]:12.6f}\n")
        
        if any(metric in mean_metrics for metric in fp_metrics):
            f.write("\n[FP指标]\n")
            for metric in fp_metrics:
                if metric in mean_metrics:
                    f.write(f"  {metric:10s}: {mean_metrics[metric]:12.6f} ± {std_metrics[metric]:12.6f}\n")
        
        if any(metric in mean_metrics for metric in comp_metrics):
            f.write("\n[综合指标]\n")
            for metric in comp_metrics:
                if metric in mean_metrics:
                    f.write(f"  {metric:18s}: {mean_metrics[metric]:12.6f} ± {std_metrics[metric]:12.6f}\n")
        
        f.write("\n[计数指标]\n")
        for metric in count_metrics:
            if metric in mean_metrics:
                f.write(f"  {metric:12s}: {mean_metrics[metric]:12.2f} ± {std_metrics[metric]:12.2f}\n")
        
        f.write("=" * 80 + "\n")
        
        # 保存详细结果
        f.write("\n各实验详细结果:\n")
        f.write("-" * 80 + "\n")
        for i, (folder_name, metrics) in enumerate(zip(results['folders'], results['individual'])):
            f.write(f"\n实验 {i} ({folder_name}):\n")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  {metric_name:10s}: {value:.6f}\n")
                else:
                    f.write(f"  {metric_name:10s}: {value}\n")
    
    print(f"\n结果已保存到: {output_file}")


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='计算多个实验结果的平均指标',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 旧格式 (peak_detect_ltf/peak_detect_ltf_basic)
  python calculate_mean_metrics.py --task peak_detect_ltf_basic --data load_data_mixed --model PatchTST --input_type zeroIn --output_type zeroOut --loss MSE --seq_len 168 --pred_len 720
  
  # 新格式 (seq2peak)
  python calculate_mean_metrics.py --task seq2peak --data load_data_mixed --model peak_Transformer --model_id maxIn_maxOut_244_23 --seq_len 168 --pred_len 336
        """
    )
    
    parser.add_argument('--task', '-t', type=str, required=True,
                        help='任务名称 (例如: peak_detect_ltf, peak_detect_ltf_basic, seq2peak)')
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='数据集名称 (例如: load_data_mixed, ETTh1, electricity)')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='模型名称 (例如: PatchTST, Autoformer, Transformer, peak_Transformer)')
    parser.add_argument('--model_id', type=str, default=None,
                        help='模型ID (仅用于seq2peak任务，例如: maxIn_maxOut_244_23)')
    parser.add_argument('--input_type', '-i', type=str, default=None,
                        help='输入类型: zeroIn 或 maxIn (仅用于旧格式任务)')
    parser.add_argument('--output_type', '-o', type=str, default=None,
                        help='输出类型: zeroOut 或 maxOut (仅用于旧格式任务)')
    parser.add_argument('--loss', '-l', type=str, default=None,
                        help='损失函数 (例如: MSE, MAE) (仅用于旧格式任务)')
    parser.add_argument('--seq_len', '-s', type=int, required=True,
                        help='输入序列长度 (例如: 96, 168, 336, 512)')
    parser.add_argument('--pred_len', '-p', type=int, required=True,
                        help='预测序列长度 (例如: 96, 192, 336, 720)')
    parser.add_argument('--results_dir', '-r', type=str, default='./results',
                        help='results文件夹路径 (默认: ./results)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出文件路径 (默认: 自动生成在results目录下)')
    
    return parser.parse_args()


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    
    # 构建文件夹名称模式
    # 格式1 (seq2peak): {task}_{data}_{model}_{model_id}_{seq_len}_{pred_len}_\d+
    # 格式2 (旧格式): {task}_{data}_{model}_{input_type}_{output_type}_{loss}_{seq_len}_{pred_len}_\d+
    
    if args.task == 'seq2peak':
        # seq2peak任务使用新格式
        if args.model_id is None:
            raise ValueError("seq2peak任务需要提供 --model_id 参数")
        
        folder_pattern = (f'{args.task}_{args.data}_{args.model}_'
                         f'{args.model_id}_{args.seq_len}_{args.pred_len}_\\d+')
        
        print("=" * 80)
        print("实验配置 (Seq2Peak格式):")
        print("-" * 80)
        print(f"  任务 (task):        {args.task}")
        print(f"  数据集 (data):      {args.data}")
        print(f"  模型 (model):       {args.model}")
        print(f"  模型ID (model_id):  {args.model_id}")
        print(f"  输入长度 (seq_len): {args.seq_len}")
        print(f"  预测长度 (pred_len):{args.pred_len}")
        print(f"  结果目录:           {args.results_dir}")
        print(f"  匹配模式:           {folder_pattern}")
        print("=" * 80)
        print()
        
    else:
        # 旧格式任务
        if args.input_type is None or args.output_type is None or args.loss is None:
            raise ValueError("旧格式任务需要提供 --input_type, --output_type, --loss 参数")
        
        folder_pattern = (f'{args.task}_{args.data}_{args.model}_'
                         f'{args.input_type}_{args.output_type}_{args.loss}_'
                         f'{args.seq_len}_{args.pred_len}_\\d+')
        
        print("=" * 80)
        print("实验配置 (旧格式):")
        print("-" * 80)
        print(f"  任务 (task):        {args.task}")
        print(f"  数据集 (data):      {args.data}")
        print(f"  模型 (model):       {args.model}")
        print(f"  输入类型 (input):   {args.input_type}")
        print(f"  输出类型 (output):  {args.output_type}")
        print(f"  损失函数 (loss):    {args.loss}")
        print(f"  输入长度 (seq_len): {args.seq_len}")
        print(f"  预测长度 (pred_len):{args.pred_len}")
        print(f"  结果目录:           {args.results_dir}")
        print(f"  匹配模式:           {folder_pattern}")
        print("=" * 80)
        print()
    
    # 计算平均指标
    results = calculate_mean_metrics(args.results_dir, folder_pattern=folder_pattern)
    
    # 打印结果
    print_results(results)
    
    # 保存结果到文件
    if results:
        if args.output_file is None:
            # 自动生成输出文件名,保存到 results 文件夹
            if args.task == 'seq2peak':
                output_file = (f'{args.results_dir}/mean_metrics_{args.task}_{args.data}_{args.model}_'
                              f'{args.model_id}_{args.seq_len}_{args.pred_len}.txt')
            else:
                output_file = (f'{args.results_dir}/mean_metrics_{args.task}_{args.data}_{args.model}_'
                              f'{args.input_type}_{args.output_type}_{args.loss}_'
                              f'{args.seq_len}_{args.pred_len}.txt')
        else:
            output_file = args.output_file
        
        save_results(results, output_file)
