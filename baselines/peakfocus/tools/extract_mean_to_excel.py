#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从results文件夹读取所有txt文件，提取mean指标并保存为Excel表格
"""

import os
import re
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side


def parse_txt_file(file_path):
    """
    解析单个txt文件，提取mean指标和标准差
    
    原始日志文件中包含的指标（来自exp代码的输出）：
        - MSE, MAE (全量序列的回归指标)
        - Recall, Precision, F1-Score (峰值分类指标)
        - TP_MSE, TP_MAE (真峰值点的回归指标)
        - FP_MSE, FP_MAE, FN_MSE, FN_MAE (误检和漏检的回归指标)
        - BPE (平衡峰值误差), PIM (峰值不匹配指标)
    
    选择输出到Excel的指标（7个核心指标）：
        1. Recall (召回率)
        2. Precision (精确率)
        3. F1-Score (F1分数)
        4. TP_MSE (真峰值的均方误差)
        5. TP_MAE (真峰值的平均绝对误差)
        6. BPE (平衡峰值误差)
        7. PIM (峰值不匹配指标)
    
    返回: dict 包含各项指标的均值和标准差（格式化为"Mean ± Std"）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取文件名作为实验名称
    filename = Path(file_path).stem
    
    # 提取各项指标的均值和标准差（格式: 指标名 : 均值 ± 标准差）
    metrics = {}
    
    # 使用正则表达式提取指标（支持连字符和下划线）
    pattern = r'([\w-]+)\s*:\s*([\d.]+)\s*±\s*([\d.]+)'
    matches = re.findall(pattern, content)
    
    print(f"  从 {Path(file_path).name} 中解析到的所有指标:")
    for metric_name, mean_val, std_val in matches:
        try:
            # 格式化为 "Mean ± Std"
            metrics[metric_name] = f"{float(mean_val):.4f} ± {float(std_val):.4f}"
            print(f"    - {metric_name}: {metrics[metric_name]}")
        except ValueError:
            continue
    
    return filename, metrics


def extract_all_metrics(results_dir):
    """
    从results目录读取所有txt文件并提取指标
    
    参数:
        results_dir: results文件夹路径
        
    返回:
        list of dict: 每个dict包含文件名和各项指标
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"错误: 目录 {results_dir} 不存在")
        return []
    
    all_data = []
    txt_files = list(results_path.glob('*.txt'))
    
    print(f"找到 {len(txt_files)} 个txt文件")
    
    for txt_file in sorted(txt_files):
        print(f"正在处理: {txt_file.name}")
        filename, metrics = parse_txt_file(txt_file)
        
        if metrics:
            data = {'文件名': filename}
            data.update(metrics)
            all_data.append(data)
    
    return all_data


def save_to_excel(data, output_file):
    """
    将数据保存为Excel文件
    
    Excel表格结构说明：
    - 第一列：实验文件名（例如：seq2peak_load_data_mixed_peak_Transformer_maxIn_maxOut_244_23_168_336）
    - 后续7列：核心峰值检测指标，每个单元格格式为 "Mean ± Std"
    
    列顺序（共8列）：
        1. 文件名
        2. Recall (召回率 - 检测到的真实峰值比例)
        3. Precision (精确率 - 预测峰值中真实峰值的比例)
        4. F1-Score (F1分数 - Recall和Precision的调和平均)
        5. TP_MSE (真峰值的均方误差 - 正确检测到的峰值的数值误差)
        6. TP_MAE (真峰值的平均绝对误差)
        7. BPE (平衡峰值误差 - 综合考虑FP和FN的误差)
        8. PIM (峰值不匹配指标 - 综合峰值位置和数值的匹配度)
    
    参数:
        data: list of dict - 从txt文件解析的所有指标数据
        output_file: 输出文件路径
    """
    if not data:
        print("没有数据可保存")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    print("\n" + "="*80)
    print("Excel表格生成说明:")
    print("="*80)
    
    # 只保留需要的指标列（按照要求的顺序）
    priority_cols = ['文件名', 
                     'Recall', 'Precision', 'F1-Score',
                     'TP_MSE', 'TP_MAE',
                     'BPE', 'PIM']
    
    print(f"\n原始数据包含 {len(df.columns)} 列指标")
    print(f"选择输出的 7 个核心指标:")
    for i, col in enumerate(priority_cols[1:], 1):
        print(f"  {i}. {col}")
    
    # 只保留需要的列
    existing_priority_cols = [col for col in priority_cols if col in df.columns]
    
    missing_cols = [col for col in priority_cols if col not in df.columns]
    if missing_cols:
        print(f"\n警告: 以下指标在数据中不存在: {', '.join(missing_cols)}")
    
    df = df[existing_priority_cols]
    
    print(f"\n最终输出表格包含 {len(df.columns)} 列 (文件名 + {len(df.columns)-1} 个指标)")
    print(f"共 {len(df)} 行数据 (实验配置)")
    
    # 保存为Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Mean Metrics', index=False)
        
        # 获取工作表
        worksheet = writer.sheets['Mean Metrics']
        
        # 设置表头样式
        header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
        header_font = Font(bold=True, color='FFFFFF', size=11)
        
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')
        
        # 自动调整列宽
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            
            for cell in column:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except:
                    pass
            
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
        
        # 冻结首行
        worksheet.freeze_panes = 'A2'
    
    print(f"\n✓ 数据已保存到: {output_file}")
    print(f"✓ 总共处理了 {len(df)} 个实验配置")
    print("="*80)


def main():
    """主函数"""
    # 设置路径
    results_dir = './results'
    output_file = './results_mean_metrics.xlsx'
    
    print("=" * 80)
    print("从results文件夹提取mean指标并保存为Excel")
    print("=" * 80)
    print()
    
    # 提取所有指标
    all_data = extract_all_metrics(results_dir)
    
    if all_data:
        # 保存为Excel
        save_to_excel(all_data, output_file)
        
        # 显示统计信息
        df = pd.DataFrame(all_data)
        print("\n" + "=" * 80)
        print("数据统计:")
        print("=" * 80)
        print(f"文件数量: {len(df)}")
        print(f"指标数量: {len(df.columns) - 1}")  # 减去文件名列
        print(f"指标列表: {', '.join([col for col in df.columns if col != '文件名'])}")
        
        # 显示前几行数据预览
        print("\n数据预览 (前5行):")
        print(df.head().to_string())
    else:
        print("未能提取到任何数据")


if __name__ == "__main__":
    main()
