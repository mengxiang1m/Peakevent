"""
合并三个CSV文件或处理ETT数据集，添加峰值标记

支持两种模式:
1. HF模式: 合并三个CSV文件
   - 60分钟整点数据
   - 每小时最大值及其真实时间戳
   - 峰值数据

2. ETT模式: 处理ETT数据集 (已经是小时粒度的数据)
   - ETTh1.csv / ETTh2.csv (小时粒度数据)

输出格式:
date_60min, value_60min, date_max, value_max, is_peak

说明:
- date_60min: 60分钟整点数据的时间戳 (例如: 2021-01-01 01:00:00)
- value_60min: 60分钟整点数据的值
- date_max: 每小时最大值的真实时间戳 (例如: 2021-01-01 01:35:00)
- value_max: 每小时最大值
- is_peak: 标记是否为峰值位置 (1表示是峰值, 0表示不是)
"""

import pandas as pd
import numpy as np
import os
import sys
import importlib.util
from pathlib import Path

# 默认配置
DEFAULT_CONFIG = {
    'ett': {
        'data_dir': 'd:/ywz_experiment_papers/ywz_7_electric/Time-Series-Library/dataset/ETT-small',
        'lookahead': 5,
        'datasets': {
            'ETTh1': {'file': 'ETTh1.csv', 'value_column': 'OT'},
            'ETTh2': {'file': 'ETTh2.csv', 'value_column': 'OT'},
            'ETTm1': {'file': 'ETTm1.csv', 'value_column': 'OT'},
            'ETTm2': {'file': 'ETTm2.csv', 'value_column': 'OT'},
            'electricity': {'file': 'electricity.csv', 'value_column': 'OT', 'data_dir': 'd:/ywz_experiment_papers/ywz_7_electric/Time-Series-Library/dataset/electricity'}
        }
    },
    'hf': {
        'data_dir': './dataset/load_data/hf_load_data',
        'lookahead': 3,
        'file_60min': 'hf_load_data_20210101-20251127_60min.csv',
        'file_hourly_max': 'hf_load_data_20210101-20251127_hourly_max.csv',
        'file_peaks_template': 'hf_load_data_20210101-20251127_hourly_max_peaks_lookhead_{lookahead}.csv',
        'output_template': 'hf_load_data_20210101-20251127_mixed_with_peaks_lookahead_{lookahead}.csv'
    }
}

def align_hour(timestamp):
    """将时间戳对齐到所属的小时起始点"""
    return timestamp.replace(minute=0, second=0, microsecond=0)

def process_ett_dataset(
    ett_file=None,
    data_dir=None,
    output_file=None,
    value_column=None,
    lookahead=None,
    dataset_name=None
):
    """
    处理ETT数据集 (已经是小时粒度)，添加峰值标记
    
    Parameters:
    -----------
    ett_file : str, optional
        ETT数据集文件名 (例如: ETTh1.csv, ETTh2.csv)
        如果为None且dataset_name提供，则从配置中读取
    data_dir : str, optional
        数据文件夹路径。如果为None，使用默认配置
    output_file : str, optional
        输出文件名。如果为None，自动生成
    value_column : str, optional
        要使用的值列名。如果为None，从配置或数据中自动选择
    lookahead : int, optional
        峰值检测的lookahead参数。如果为None，使用默认值5
    dataset_name : str, optional
        数据集名称 (例如: 'ETTh1', 'ETTh2')，用于从配置中自动加载参数
        
    Returns:
    --------
    df_output : DataFrame
        处理后的数据框
        
    Examples:
    ---------
    # 使用数据集名称（推荐）
    df = process_ett_dataset(dataset_name='ETTh2', lookahead=5)
    df = process_ett_dataset(dataset_name='electricity', lookahead=5)
    
    # 直接指定文件（向后兼容）
    df = process_ett_dataset(ett_file='ETTh2.csv', lookahead=5)
    """
    # 从配置加载默认值
    config = DEFAULT_CONFIG['ett']
    
    # 处理dataset_name参数
    if dataset_name:
        if dataset_name not in config['datasets']:
            raise ValueError(f"未知数据集: {dataset_name}. 可用数据集: {list(config['datasets'].keys())}")
        dataset_config = config['datasets'][dataset_name]
        ett_file = ett_file or dataset_config['file']
        value_column = value_column or dataset_config['value_column']
        # 使用数据集特定的data_dir（如果有），优先级高于参数和默认值
        if 'data_dir' in dataset_config and data_dir is None:
            data_dir = dataset_config['data_dir']
    
    # 设置默认值
    if data_dir is None:
        data_dir = config['data_dir']
    lookahead = lookahead if lookahead is not None else config['lookahead']
    
    # 如果没有提供ett_file，尝试从dataset_name推断或使用默认值
    if ett_file is None:
        if dataset_name:
            ett_file = config['datasets'][dataset_name]['file']
        else:
            raise ValueError("必须提供 ett_file 或 dataset_name 参数")
    
    # 自动生成输出文件名
    if output_file is None:
        base_name = Path(ett_file).stem
        output_file = f"{base_name}_mixed_with_peaks_lookahead_{lookahead}.csv"
    # 动态导入，避免模块路径问题
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    peak_detect_path = os.path.join(project_root, 'peak_detect_example')
    
    if peak_detect_path not in sys.path:
        sys.path.insert(0, project_root)  # TODO(R-future): migrate to patchevent package import
    
    try:
        from peak_detect_example.detect_peak import find_peaks_findpeaks
    except ImportError:
        # 如果还是失败，尝试直接导入文件
        import importlib.util
        detect_peak_file = os.path.join(peak_detect_path, 'detect_peak.py')
        spec = importlib.util.spec_from_file_location("detect_peak", detect_peak_file)
        detect_peak = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(detect_peak)
        find_peaks_findpeaks = detect_peak.find_peaks_findpeaks
    
    # 文件路径
    path_ett = os.path.join(data_dir, ett_file)
    path_output = os.path.join(data_dir, output_file)
    
    print("="*70)
    print(f"处理ETT数据集: {ett_file}")
    print("="*70)
    
    # 1. 读取ETT数据
    print("\n步骤 1: 读取ETT数据...")
    df_ett = pd.read_csv(path_ett)
    df_ett['date'] = pd.to_datetime(df_ett['date'])
    
    # 选择值列
    if value_column is None:
        value_column = df_ett.columns[-1]
        print(f"  自动选择列 '{value_column}' 作为数据列")
    elif value_column not in df_ett.columns:
        raise ValueError(f"指定的列 '{value_column}' 不存在。可用列: {list(df_ett.columns)}")
    
    print(f"  数据记录数: {len(df_ett)}")
    print(f"  时间范围: {df_ett['date'].min()} 至 {df_ett['date'].max()}")
    print(f"  使用列: {value_column}")
    
    # 2. 使用findpeaks检测峰值
    print(f"\n步骤 2: 使用findpeaks检测峰值 (lookahead={lookahead})...")
    data_values = df_ett[value_column].values
    peaks, results = find_peaks_findpeaks(data_values, method='peakdetect', lookahead=lookahead)
    
    print(f"  检测到峰值数: {len(peaks)}")
    print(f"  峰值比例: {len(peaks)/len(df_ett)*100:.2f}%")
    
    # 3. 创建输出数据框
    print("\n步骤 3: 创建输出数据框...")
    df_output = pd.DataFrame()
    df_output['date_60min'] = df_ett['date']
    df_output['value_60min'] = df_ett[value_column]
    df_output['date_max'] = df_ett['date']  # ETT数据已经是小时粒度，date_max = date_60min
    df_output['value_max'] = df_ett[value_column]  # value_max = value_60min
    
    # 4. 添加峰值标记
    df_output['is_peak'] = 0
    df_output.loc[peaks, 'is_peak'] = 1
    
    print(f"  标记了 {df_output['is_peak'].sum()} 个峰值位置")
    
    # 5. 数据验证
    print("\n步骤 4: 数据验证...")
    print(f"  输出记录数: {len(df_output)}")
    print(f"  时间范围: {df_output['date_60min'].min()} 至 {df_output['date_60min'].max()}")
    print(f"  峰值数量: {df_output['is_peak'].sum()}")
    
    # 峰值统计
    peak_rows = df_output[df_output['is_peak'] == 1]
    if len(peak_rows) > 0:
        print(f"\n  峰值统计:")
        print(f"    峰值数量: {len(peak_rows)}")
        print(f"    峰值平均值: {peak_rows['value_max'].mean():.2f}")
        print(f"    峰值最大值: {peak_rows['value_max'].max():.2f}")
        print(f"    峰值最小值: {peak_rows['value_max'].min():.2f}")
    
    # 6. 保存数据
    print("\n步骤 5: 保存数据...")
    df_output.to_csv(path_output, index=False)
    print(f"  输出文件: {path_output}")
    
    # 7. 显示样本数据
    print("\n步骤 6: 数据样本...")
    print("\n前10行 (常规数据):")
    print(df_output.head(10).to_string(index=False))
    
    # 显示一些峰值样本
    if len(peak_rows) > 0:
        print("\n峰值样本 (前10个峰值):")
        print(peak_rows.head(10).to_string(index=False))
    
    print("\n" + "="*70)
    print(f"ETT数据集处理完成: {output_file}")
    print("="*70)
    
    print("\n数据集统计:")
    print(f"  总记录数: {len(df_output)}")
    print(f"  峰值数量: {df_output['is_peak'].sum()}")
    print(f"  峰值比例: {df_output['is_peak'].mean()*100:.2f}%")
    print(f"  非峰值数量: {(df_output['is_peak']==0).sum()}")
    
    return df_output

def merge_mixed_data_with_peaks(
    file_60min=None,
    file_hourly_max=None,
    file_peaks=None,
    data_dir=None,
    output_file=None,
    lookahead=None
):
    """
    合并三个CSV文件,保留各自的时间戳
    
    Parameters:
    -----------
    file_60min : str, optional
        60分钟整点数据文件名。如果为None，使用默认配置
    file_hourly_max : str, optional
        每小时最大值数据文件名。如果为None，使用默认配置
    file_peaks : str, optional
        峰值数据文件名。如果为None，根据lookahead自动生成
    data_dir : str, optional
        数据文件夹路径。如果为None，使用默认配置
    output_file : str, optional
        输出文件名。如果为None，根据lookahead自动生成
    lookahead : int, optional
        峰值检测的lookahead参数。如果为None，使用默认值3
        用于自动生成file_peaks和output_file文件名
        
    Returns:
    --------
    df_output : DataFrame
        合并后的数据框
        
    Examples:
    ---------
    # 使用lookahead参数自动生成文件名（推荐）
    df = merge_mixed_data_with_peaks(lookahead=3)
    df = merge_mixed_data_with_peaks(lookahead=5)
    
    # 直接指定文件名（向后兼容）
    df = merge_mixed_data_with_peaks(
        file_peaks='hf_load_data_20210101-20250925_hourly_max_peaks_lookhead_3.csv',
        output_file='hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv'
    )
    """
    # 从配置加载默认值
    config = DEFAULT_CONFIG['hf']
    
    # 设置默认值
    data_dir = data_dir or config['data_dir']
    lookahead = lookahead if lookahead is not None else config['lookahead']
    file_60min = file_60min or config['file_60min']
    file_hourly_max = file_hourly_max or config['file_hourly_max']
    
    # 根据lookahead自动生成文件名
    if file_peaks is None:
        file_peaks = config['file_peaks_template'].format(lookahead=lookahead)
    
    if output_file is None:
        output_file = config['output_template'].format(lookahead=lookahead)
    
    # 文件路径
    path_60min = os.path.join(data_dir, file_60min)
    path_hourly_max = os.path.join(data_dir, file_hourly_max)
    path_peaks = os.path.join(data_dir, file_peaks)
    path_output = os.path.join(data_dir, output_file)
    
    print("="*70)
    print(f"合并混合数据集(包含峰值标记) - lookahead={lookahead}")
    print("="*70)
    print(f"配置:")
    print(f"  file_60min: {file_60min}")
    print(f"  file_hourly_max: {file_hourly_max}")
    print(f"  file_peaks: {file_peaks}")
    print(f"  output_file: {output_file}")
    
    # 1. 读取60分钟数据
    print("\n步骤 1: 读取60分钟整点数据...")
    df_60min = pd.read_csv(path_60min)
    df_60min['date'] = pd.to_datetime(df_60min['date'])
    print(f"  60分钟数据: {len(df_60min)} 条记录")
    print(f"  时间范围: {df_60min['date'].min()} 至 {df_60min['date'].max()}")
    
    # 2. 读取hourly_max数据
    print("\n步骤 2: 读取每小时最大值数据...")
    df_max = pd.read_csv(path_hourly_max)
    df_max['date'] = pd.to_datetime(df_max['date'])
    print(f"  最大值数据: {len(df_max)} 条记录")
    print(f"  时间范围: {df_max['date'].min()} 至 {df_max['date'].max()}")
    
    # 3. 读取峰值数据
    print("\n步骤 3: 读取峰值数据...")
    df_peaks = pd.read_csv(path_peaks)
    df_peaks['timestamp_Findpeaks_peakdetect'] = pd.to_datetime(df_peaks['timestamp_Findpeaks_peakdetect'])
    print(f"  峰值数据: {len(df_peaks)} 条记录")
    print(f"  列名: {list(df_peaks.columns)}")
    
    # 通过timestamp在hourly_max中查找峰值的索引位置
    print("\n步骤 4: 匹配峰值位置...")
    peak_indices = set()
    for idx, row in df_peaks.iterrows():
        peak_time = row['timestamp_Findpeaks_peakdetect']
        peak_value = row['value_Findpeaks_peakdetect']
        
        # 在df_max中查找匹配的时间戳
        matches = df_max[df_max['date'] == peak_time]
        if len(matches) > 0:
            peak_idx = matches.index[0]
            peak_indices.add(peak_idx)
        else:
            print(f"  警告: 未找到峰值 peak_id={row['peak_id']}, time={peak_time}")
    
    print(f"  成功匹配 {len(peak_indices)} 个峰值位置")
    print(f"  峰值位置(索引)示例: {sorted(list(peak_indices))[:10]}...")
    
    # 5. 为hourly_max数据添加小时对齐列和峰值标记
    print("\n步骤 5: 对齐数据并添加峰值标记...")
    df_60min['hour_key'] = df_60min['date']  # 60分钟数据已经是整点
    df_max['hour_key'] = df_max['date'].apply(align_hour)  # 最大值数据对齐到整点
    
    # 为df_max添加峰值标记 (仅保留 is_peak)
    df_max['is_peak'] = 0
    for idx in peak_indices:
        if idx < len(df_max):
            df_max.loc[idx, 'is_peak'] = 1
    
    print(f"  标记了 {df_max['is_peak'].sum()} 个峰值位置")
    
    # 重命名列以便区分
    df_60min = df_60min.rename(columns={'date': 'date_60min', 'value': 'value_60min'})
    df_max = df_max.rename(columns={'date': 'date_max', 'value': 'value_max'})
    
    # 6. 按小时键合并数据
    print("\n步骤 6: 合并数据...")
    df_merged = pd.merge(
        df_60min[['hour_key', 'date_60min', 'value_60min']],
        df_max[['hour_key', 'date_max', 'value_max', 'is_peak']],
        on='hour_key',
        how='inner'
    )
    
    # 7. 标记每天的最大值作为is_peak_seq2peaks
    print("\n步骤 7: 标记每天的最大值 (is_peak_seq2peaks)...")
    
    # 从date_max提取日期(年-月-日)
    df_max['date_only'] = df_max['date_max'].dt.date
    
    # 找出每天的最大值
    df_max['is_peak_seq2peaks'] = 0
    daily_max_indices = df_max.groupby('date_only')['value_max'].idxmax()
    df_max.loc[daily_max_indices, 'is_peak_seq2peaks'] = 1
    
    print(f"  标记了 {df_max['is_peak_seq2peaks'].sum()} 个每日最大值位置")
    print(f"  总天数: {df_max['date_only'].nunique()}")
    
    # 删除临时列
    df_max = df_max.drop('date_only', axis=1)
    
    # 8. 重新合并数据以包含新列
    print("\n步骤 8: 重新合并数据以包含is_peak_seq2peaks...")
    df_merged = pd.merge(
        df_60min[['hour_key', 'date_60min', 'value_60min']],
        df_max[['hour_key', 'date_max', 'value_max', 'is_peak', 'is_peak_seq2peaks']],
        on='hour_key',
        how='inner'
    )
    
    # 9. 整理输出格式
    df_output = df_merged[['date_60min', 'value_60min', 'date_max', 'value_max', 'is_peak', 'is_peak_seq2peaks']].copy()
    
    # 10. 数据验证
    print("\n步骤 9: 数据验证...")
    print(f"  合并后记录数: {len(df_output)}")
    print(f"  时间范围: {df_output['date_60min'].min()} 至 {df_output['date_60min'].max()}")
    print(f"  峰值数量 (is_peak): {df_output['is_peak'].sum()}")
    print(f"  峰值数量 (is_peak_seq2peaks): {df_output['is_peak_seq2peaks'].sum()}")
    
    # 检查时间戳差异
    df_output['time_diff_minutes'] = (df_output['date_max'] - df_output['date_60min']).dt.total_seconds() / 60
    print(f"\n  时间戳差异统计 (分钟):")
    print(f"    最小差异: {df_output['time_diff_minutes'].min():.2f} 分钟")
    print(f"    最大差异: {df_output['time_diff_minutes'].max():.2f} 分钟")
    print(f"    平均差异: {df_output['time_diff_minutes'].mean():.2f} 分钟")
    
    # 检查值的差异
    df_output['value_diff'] = df_output['value_max'] - df_output['value_60min']
    print(f"\n  数值差异统计:")
    print(f"    最小差异: {df_output['value_diff'].min():.2f}")
    print(f"    最大差异: {df_output['value_diff'].max():.2f}")
    print(f"    平均差异: {df_output['value_diff'].mean():.2f}")
    print(f"    value_max >= value_60min 的比例: {(df_output['value_max'] >= df_output['value_60min']).mean()*100:.2f}%")
    
    # 峰值统计 (is_peak)
    peak_rows = df_output[df_output['is_peak'] == 1]
    if len(peak_rows) > 0:
        print(f"\n  峰值统计 (is_peak):")
        print(f"    峰值数量: {len(peak_rows)}")
        print(f"    峰值平均值: {peak_rows['value_max'].mean():.2f}")
        print(f"    峰值最大值: {peak_rows['value_max'].max():.2f}")
        print(f"    峰值最小值: {peak_rows['value_max'].min():.2f}")
    
    # 峰值统计 (is_peak_seq2peaks)
    peak_rows_seq2peaks = df_output[df_output['is_peak_seq2peaks'] == 1]
    if len(peak_rows_seq2peaks) > 0:
        print(f"\n  峰值统计 (is_peak_seq2peaks):")
        print(f"    峰值数量: {len(peak_rows_seq2peaks)}")
        print(f"    峰值平均值: {peak_rows_seq2peaks['value_max'].mean():.2f}")
        print(f"    峰值最大值: {peak_rows_seq2peaks['value_max'].max():.2f}")
        print(f"    峰值最小值: {peak_rows_seq2peaks['value_max'].min():.2f}")
    
    # 删除临时列
    df_output = df_output.drop(['time_diff_minutes', 'value_diff'], axis=1)
    
    # 11. 保存数据
    print("\n步骤 10: 保存数据...")
    df_output.to_csv(path_output, index=False)
    print(f"  输出文件: {path_output}")
    
    # 12. 显示样本数据
    print("\n步骤 11: 数据样本...")
    print("\n前10行 (常规数据):")
    print(df_output.head(10).to_string(index=False))
    
    # 显示一些峰值样本 (is_peak)
    if len(peak_rows) > 0:
        print("\n峰值样本 is_peak (前10个峰值):")
        print(peak_rows.head(10).to_string(index=False))
    
    # 显示一些峰值样本 (is_peak_seq2peaks)
    if len(peak_rows_seq2peaks) > 0:
        print("\n峰值样本 is_peak_seq2peaks (前10个峰值):")
        print(peak_rows_seq2peaks.head(10).to_string(index=False))
    
    print("\n" + "="*70)
    print("混合数据集(包含峰值)创建完成!")
    print("="*70)
    
    print("\n数据集统计:")
    print(f"  总记录数: {len(df_output)}")
    print(f"  峰值数量 (is_peak): {df_output['is_peak'].sum()}")
    print(f"  峰值比例 (is_peak): {df_output['is_peak'].mean()*100:.2f}%")
    print(f"  峰值数量 (is_peak_seq2peaks): {df_output['is_peak_seq2peaks'].sum()}")
    print(f"  峰值比例 (is_peak_seq2peaks): {df_output['is_peak_seq2peaks'].mean()*100:.2f}%")
    print(f"  非峰值数量 (is_peak): {(df_output['is_peak']==0).sum()}")
    
    return df_output

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='合并数据集并添加峰值标记',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # ETT模式 - 使用数据集名称（推荐）
  python %(prog)s --mode ett --dataset ETTh2 --lookahead 5
  python %(prog)s --mode ett --dataset ETTh1 --lookahead 3
  python %(prog)s --mode ett --dataset electricity --lookahead 5
  
  # ETT模式 - 直接指定文件（向后兼容）
  python %(prog)s --mode ett --ett-file ETTh2.csv --lookahead 5
  
  # HF模式 - 使用 prefix 自动推导所有文件名（推荐）
  python %(prog)s --mode hf --prefix hf_load_data_20210101-20251127 --lookahead 3
  python %(prog)s --mode hf --prefix hf_load_data_20210101-20250925 --lookahead 3
  
  # HF模式 - 仅使用 lookahead（使用 DEFAULT_CONFIG 默认 prefix）
  python %(prog)s --mode hf --lookahead 3
  python %(prog)s --mode hf --lookahead 5
  
  # HF模式 - 直接指定文件（向后兼容）
  python %(prog)s --mode hf --file-peaks hf_load_data_xxx_peaks.csv
        """
    )
    
    # 通用参数
    parser.add_argument('--mode', type=str, default='ett', choices=['hf', 'ett'],
                        help='处理模式: hf (合并三个文件) 或 ett (处理ETT数据集)')
    parser.add_argument('--lookahead', type=int, default=5,
                        help='峰值检测的lookahead参数 (默认: ETT=5, HF=3)')
    parser.add_argument('--output-file', type=str, default=None,
                        help='输出文件名 (如果不指定，自动生成)')
    
    # ETT模式参数
    ett_group = parser.add_argument_group('ETT模式参数')
    ett_group.add_argument('--dataset', type=str, default='ETTh2',
                          help='数据集名称: ETTh1, ETTh2, ETTm1, ETTm2, electricity (推荐使用)')
    ett_group.add_argument('--ett-file', type=str, default=None,
                          help='ETT数据集文件名 (例如: ETTh2.csv, 向后兼容)')
    ett_group.add_argument('--ett-dir', type=str, default=None,
                          help='ETT数据集目录 (默认使用配置)')
    ett_group.add_argument('--value-column', type=str, default=None,
                          help='值列名称 (默认: OT 或最后一列)')
    
    # HF模式参数
    hf_group = parser.add_argument_group('HF模式参数')
    hf_group.add_argument('--hf-dir', type=str, default=None,
                         help='HF数据集目录 (默认使用配置)')
    hf_group.add_argument('--prefix', type=str, default=None,
                         help='数据文件前缀，用于自动推导 60min/hourly_max/peaks/output 文件名，'
                              '例如: hf_load_data_20210101-20251127')
    hf_group.add_argument('--file-60min', type=str, default=None,
                         help='60分钟数据文件名 (默认使用配置或由 --prefix 推导)')
    hf_group.add_argument('--file-hourly-max', type=str, default=None,
                         help='每小时最大值文件名 (默认使用配置或由 --prefix 推导)')
    hf_group.add_argument('--file-peaks', type=str, default=None,
                         help='峰值数据文件名 (默认根据 --prefix / lookahead 生成)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'hf':
            # 合并HF数据集的三个文件
            print("=" * 70)
            print("模式: 合并HF数据集的三个文件")
            print("=" * 70)

            # 若指定了 --prefix，自动推导文件名（优先级高于 DEFAULT_CONFIG）
            file_60min = args.file_60min
            file_hourly_max = args.file_hourly_max
            file_peaks = args.file_peaks
            output_file = args.output_file
            if args.prefix:
                p = args.prefix
                la = args.lookahead
                file_60min = file_60min or f"{p}_60min.csv"
                file_hourly_max = file_hourly_max or f"{p}_hourly_max.csv"
                file_peaks = file_peaks or f"{p}_hourly_max_peaks_lookhead_{la}.csv"
                output_file = output_file or f"{p}_mixed_with_peaks_lookahead_{la}.csv"

            df_result = merge_mixed_data_with_peaks(
                file_60min=file_60min,
                file_hourly_max=file_hourly_max,
                file_peaks=file_peaks,
                data_dir=args.hf_dir,
                output_file=output_file,
                lookahead=args.lookahead
            )
            
        elif args.mode == 'ett':
            # 处理ETT数据集
            print("=" * 70)
            print("模式: 处理ETT数据集")
            print("=" * 70)
            
            # 检查参数
            if not args.dataset and not args.ett_file:
                parser.error("ETT模式需要提供 --dataset 或 --ett-file 参数")
            
            df_result = process_ett_dataset(
                dataset_name=args.dataset,
                ett_file=args.ett_file,
                data_dir=args.ett_dir,
                output_file=args.output_file,
                value_column=args.value_column,
                lookahead=args.lookahead
            )
        
        print("\n" + "=" * 70)
        print("处理完成!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
