"""
数据降采样脚本:将负荷数据按不同粒度和模式进行降采样

支持的输入格式:
1. index, date, value (包含index列的格式)
2. 序号, 时间, 值 (中文列名格式)
3. date, value (标准格式)

支持的降采样模式:
Mode 1: 15分钟间隔降采样 - 从完整自然日00:00开始,每15分钟采样一次,保留原始采样点的值和时间戳
Mode 2: 60分钟间隔降采样 - 从完整自然日00:00开始,每小时采样一次,保留原始采样点的值和时间戳
Mode 3: 逐小时最大值提取 - 每小时提取最大负荷值及其真实出现时间戳(非整点时间)

关键特性:
- 所有模式均从完整自然日(00:00)开始,首日不完整则跳过
- Mode 1/2保留原始采样点数据,不做平均或插值
- Mode 3保留每小时最大值的真实时间戳,不对齐到整点

输入文件:hf_load_data_20210101-20250925.csv (原始粒度)
输出文件:
- hf_load_data_20210101-20250925_15min_mode1.csv (15分钟降采样,mode1)
- hf_load_data_20210101-20250925_60min_mode2.csv (60分钟降采样,mode2)
- hf_load_data_20210101-20250925_hourly_max_mode3.csv (逐小时最大值及真实时间戳,mode3)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_and_preprocess_data(file_path):
    """
    加载原始数据并预处理
    """
    print(f"正在加载数据: {file_path}")
    
    # 读取CSV文件
    try:
        # 尝试不同的编码格式
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin1']
        df = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                used_encoding = encoding
                print(f"成功使用编码 {encoding} 读取文件")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                if encoding == encodings[-1]:  # 最后一个编码
                    raise e
                continue
        
        if df is None:
            raise Exception("无法使用任何编码格式读取文件")
            
        print(f"原始数据形状: {df.shape}")
        print(f"原始列名: {list(df.columns)}")
        
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None
    
    # 1. 处理不同的列格式
    print(f"检测列格式...")
    
    # 删除所有空列
    df = df.dropna(axis=1, how='all')
    
    # 处理不同的列名情况
    if len(df.columns) == 3:
        # 三列的情况，删除第一列（index/序号列）
        if df.columns[0].lower() in ['index', '序号', 'unnamed: 0'] or df.columns[0].startswith('Unnamed'):
            df = df.drop(df.columns[0], axis=1)
            print(f"已删除第一列: {df.columns[0] if len(df.columns) > 0 else '未知'}")
        else:
            # 如果第一列不是index/序号，保留后两列
            df = df.iloc[:, 1:]
            print("保留后两列作为时间和数值列")
    
    elif len(df.columns) == 2:
        # 两列的情况，直接使用
        print("检测到两列数据，直接使用")
    
    else:
        print(f"错误: 不支持的列数 {len(df.columns)}，期望2或3列")
        return None
    
    # 2. 标准化列名
    expected_columns = ['date', 'value']
    if len(df.columns) == 2:
        # 统一列名，不管原来叫什么
        df.columns = expected_columns
        print(f"已标准化列名为: {list(df.columns)}")
    else:
        print(f"错误: 处理后列数不正确，期望2列，实际{len(df.columns)}列")
        return None
    
    # 3. 处理时间格式
    print("\n处理时间格式...")
    try:
        # 尝试多种时间格式
        time_formats = [
            '%Y/%m/%d %H:%M',      # 2021/1/1 0:00
            '%Y-%m-%d %H:%M:%S',   # 2021-01-01 00:00:00  
            '%Y-%m-%d %H:%M',      # 2021-01-01 00:00
            '%Y/%m/%d %H:%M:%S',   # 2021/1/1 0:00:00
        ]
        
        success = False
        for fmt in time_formats:
            try:
                df['date'] = pd.to_datetime(df['date'], format=fmt)
                print(f"时间格式转换成功，使用格式: {fmt}")
                success = True
                break
            except:
                continue
        
        if not success:
            # 如果所有格式都失败，尝试自动解析
            df['date'] = pd.to_datetime(df['date'])
            print("使用自动解析成功转换时间格式")
            
    except Exception as e:
        print(f"时间格式转换失败: {e}")
        return None
    
    # 4. 处理数值数据
    print("\n处理数值数据...")
    try:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        print(f"数值转换完成，缺失值数量: {df['value'].isnull().sum()}")
    except Exception as e:
        print(f"数值转换失败: {e}")
        return None
    
    # 5. 处理缺失值
    if df['value'].isnull().sum() > 0:
        print("检测到缺失值，使用前向填充...")
        df['value'] = df['value'].fillna(method='ffill')
        # 如果第一个值是NaN，使用后向填充
        df['value'] = df['value'].fillna(method='bfill')
        print(f"缺失值处理完成，剩余缺失值: {df['value'].isnull().sum()}")
    
    # 按日期排序
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")
    
    return df

def aggregate_mode1_15min(df):
    """
    Mode 1: 15分钟间隔降采样
    从完整自然日00:00开始,每15分钟采样一次,保留原始采样点的值和时间戳
    不做平均聚合,仅筛选出符合15分钟间隔的原始数据点
    
    参数:
    df: 原始数据DataFrame
    """
    print(f"\n正在执行 Mode 1: 15分钟间隔降采样...")
    
    # 获取数据的起始时间
    start_time = df['date'].min()
    print(f"原始数据起始时间: {start_time}")
    
    # 判断首日是否从00:00开始
    if start_time.hour == 0 and start_time.minute == 0 and start_time.second == 0:
        # 首日是完整的,从当天00:00开始
        first_day_start = start_time
        print(f"首日完整,从 {first_day_start} 开始采样")
    else:
        # 首日不完整,从第二天00:00开始
        first_day_start = pd.Timestamp(start_time.date()) + pd.Timedelta(days=1)
        print(f"首日不完整,跳过首日,从 {first_day_start} 开始采样")
    
    # 过滤数据,只保留从起始点开始的数据
    df_filtered = df[df['date'] >= first_day_start].copy()
    print(f"从 {first_day_start} 开始,共 {len(df_filtered)} 条原始记录")
    
    # 筛选出每15分钟的采样点(分钟数为0,15,30,45)
    df_filtered['minute'] = df_filtered['date'].dt.minute
    df_filtered['second'] = df_filtered['date'].dt.second
    
    # 保留分钟数为0,15,30,45且秒数为0的点
    mask = (df_filtered['minute'] % 15 == 0) & (df_filtered['second'] == 0)
    result = df_filtered[mask][['date', 'value']].copy()
    
    print(f"Mode 1 完成,共 {len(result)} 条记录")
    print(f"时间范围: {result['date'].min()} 到 {result['date'].max()}")
    
    # 显示采样间隔统计
    time_diffs = result['date'].diff().dropna()
    print(f"采样间隔统计: 最小={time_diffs.min()}, 最大={time_diffs.max()}, 均值={time_diffs.mean()}")
    
    return result

def aggregate_mode2_60min(df):
    """
    Mode 2: 60分钟间隔降采样
    从完整自然日00:00开始,每小时采样一次,保留原始采样点的值和时间戳
    不做平均聚合,仅筛选出符合整点的原始数据点
    
    参数:
    df: 原始数据DataFrame
    """
    print(f"\n正在执行 Mode 2: 60分钟间隔降采样...")
    
    # 获取数据的起始时间
    start_time = df['date'].min()
    print(f"原始数据起始时间: {start_time}")
    
    # 判断首日是否从00:00开始
    if start_time.hour == 0 and start_time.minute == 0 and start_time.second == 0:
        # 首日是完整的,从当天00:00开始
        first_day_start = start_time
        print(f"首日完整,从 {first_day_start} 开始采样")
    else:
        # 首日不完整,从第二天00:00开始
        first_day_start = pd.Timestamp(start_time.date()) + pd.Timedelta(days=1)
        print(f"首日不完整,跳过首日,从 {first_day_start} 开始采样")
    
    # 过滤数据,只保留从起始点开始的数据
    df_filtered = df[df['date'] >= first_day_start].copy()
    print(f"从 {first_day_start} 开始,共 {len(df_filtered)} 条原始记录")
    
    # 筛选出整点的采样点(分钟数和秒数都为0)
    df_filtered['minute'] = df_filtered['date'].dt.minute
    df_filtered['second'] = df_filtered['date'].dt.second
    
    # 保留分钟数和秒数都为0的点
    mask = (df_filtered['minute'] == 0) & (df_filtered['second'] == 0)
    result = df_filtered[mask][['date', 'value']].copy()
    
    print(f"Mode 2 完成,共 {len(result)} 条记录")
    print(f"时间范围: {result['date'].min()} 到 {result['date'].max()}")
    
    # 显示采样间隔统计
    time_diffs = result['date'].diff().dropna()
    print(f"采样间隔统计: 最小={time_diffs.min()}, 最大={time_diffs.max()}, 均值={time_diffs.mean()}")
    
    return result

def aggregate_mode3_hourly_max(df):
    """
    Mode 3: 逐小时最大值提取(保留真实时间戳)
    每天每个小时提取最大负荷值及其真实出现的时间戳
    例如:14:00~15:00间最大值1500出现在14:35,则记录(14:35, 1500)
    结果为每天24个点,index为最大值真实时间戳,value为该最大负荷
    
    参数:
    df: 原始数据DataFrame
    """
    print(f"\n正在执行 Mode 3: 逐小时最大值提取(保留真实时间戳)...")
    
    # 获取数据的起始时间
    start_time = df['date'].min()
    print(f"原始数据起始时间: {start_time}")
    
    # 判断首日是否从00:00开始
    if start_time.hour == 0 and start_time.minute == 0 and start_time.second == 0:
        # 首日是完整的,从当天00:00开始
        first_day_start = start_time
        print(f"首日完整,从 {first_day_start} 开始处理")
    else:
        # 首日不完整,从第二天00:00开始
        first_day_start = pd.Timestamp(start_time.date()) + pd.Timedelta(days=1)
        print(f"首日不完整,跳过首日,从 {first_day_start} 开始处理")
    
    # 过滤数据,只保留从起始点开始的数据
    df_filtered = df[df['date'] >= first_day_start].copy()
    print(f"从 {first_day_start} 开始,共 {len(df_filtered)} 条原始记录")
    
    # 添加辅助列:年月日和小时
    df_filtered['date_only'] = df_filtered['date'].dt.date
    df_filtered['hour'] = df_filtered['date'].dt.hour
    
    # 按日期和小时分组,找出每组的最大值及其索引
    result_list = []
    
    for (date_val, hour_val), group in df_filtered.groupby(['date_only', 'hour']):
        # 找到该小时内最大值的索引
        max_idx = group['value'].idxmax()
        max_row = group.loc[max_idx]
        
        result_list.append({
            'date': max_row['date'],
            'value': max_row['value']
        })
    
    # 转换为DataFrame并按时间排序
    result = pd.DataFrame(result_list).sort_values('date').reset_index(drop=True)
    
    print(f"Mode 3 完成,共 {len(result)} 条记录")
    print(f"时间范围: {result['date'].min()} 到 {result['date'].max()}")
    
    # 显示统计信息
    total_days = (result['date'].max() - result['date'].min()).days + 1
    expected_records = total_days * 24
    print(f"预期记录数(天数×24): {expected_records},实际记录数: {len(result)}")
    
    # 显示几个样本,展示非整点时间戳
    print(f"\n样本数据(展示非整点时间戳):")
    for i in range(min(5, len(result))):
        row = result.iloc[i]
        print(f"  {row['date']} -> {row['value']:.2f}")
    
    return result

def save_aggregated_data(df, output_path, freq_name):
    """
    保存聚合后的数据
    """
    print(f"正在保存 {freq_name} 数据到: {output_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换日期格式为标准格式
    df_copy = df.copy()
    df_copy['date'] = df_copy['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 保存为CSV
    df_copy.to_csv(output_path, index=False)
    
    print(f"{freq_name} 数据保存完成")
    
    # 显示保存的数据样本
    print(f"{freq_name} 数据样本 (前5行):")
    print(df_copy.head())

def main():
    """
    主函数
    """
    import argparse

    parser = argparse.ArgumentParser(description="电力负荷数据双模式降采样工具")
    parser.add_argument(
        "--input-file", type=str,
        default="./dataset/load_data/hf_load_data/hf_load_data_20210101-20251127.csv",
        help="预处理后的原始 CSV 文件路径（默认: hf_load_data_20210101-20251127.csv）"
    )
    args = parser.parse_args()

    # 文件路径设置
    input_file = args.input_file
    output_dir = os.path.dirname(os.path.abspath(input_file))
    stem = os.path.splitext(os.path.basename(input_file))[0]

    # 输出文件路径（从输入文件名自动推导）
    output_mode1 = os.path.join(output_dir, f"{stem}_15min.csv")
    output_mode2 = os.path.join(output_dir, f"{stem}_60min.csv")
    output_mode3 = os.path.join(output_dir, f"{stem}_hourly_max.csv")

    try:
        # 1. 加载原始数据
        print("="*70)
        print("步骤 1: 加载原始数据")
        print("="*70)
        df_original = load_and_preprocess_data(input_file)
        
        if df_original is None:
            print("数据加载失败，程序终止")
            return
        
        # 2. Mode 1: 15分钟间隔降采样
        print("\n" + "="*70)
        print("步骤 2: Mode 1 - 15分钟间隔降采样(保留原始采样点)")
        print("="*70)
        df_mode1 = aggregate_mode1_15min(df_original)
        save_aggregated_data(df_mode1, output_mode1, 'Mode 1 (15分钟降采样)')
        
        # 3. Mode 2: 60分钟间隔降采样
        print("\n" + "="*70)
        print("步骤 3: Mode 2 - 60分钟间隔降采样(保留原始采样点)")
        print("="*70)
        df_mode2 = aggregate_mode2_60min(df_original)
        save_aggregated_data(df_mode2, output_mode2, 'Mode 2 (60分钟降采样)')
        
        # 4. Mode 3: 逐小时最大值提取(保留真实时间戳)
        print("\n" + "="*70)
        print("步骤 4: Mode 3 - 逐小时最大值提取(保留真实时间戳)")
        print("="*70)
        df_mode3 = aggregate_mode3_hourly_max(df_original)
        save_aggregated_data(df_mode3, output_mode3, 'Mode 3 (逐小时最大值+真实时间戳)')
        
        # 5. 打印统计信息
        print("\n" + "="*70)
        print("所有模式数据降采样完成！统计信息:")
        print("="*70)
        print(f"原始数据:                    {len(df_original):,} 条记录")
        print(f"Mode 1 (15分钟降采样):       {len(df_mode1):,} 条记录")
        print(f"Mode 2 (60分钟降采样):       {len(df_mode2):,} 条记录")
        print(f"Mode 3 (逐小时最大值):       {len(df_mode3):,} 条记录")
        print(f"\n数据降采样比:")
        print(f"  Mode 1: {len(df_original)/len(df_mode1):.2f}:1")
        print(f"  Mode 2: {len(df_original)/len(df_mode2):.2f}:1")
        print(f"  Mode 3: {len(df_original)/len(df_mode3):.2f}:1")
        
        print("\n输出文件:")
        print(f"  Mode 1: {output_mode1}")
        print(f"  Mode 2: {output_mode2}")
        print(f"  Mode 3: {output_mode3}")
        print("="*70)
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()