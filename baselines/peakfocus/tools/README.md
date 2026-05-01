# tools/ — 数据管线与结果处理工具

## 用途

本目录包含实验结果的后处理、汇总统计、参数统计和数据收集等独立脚本。
这些工具用于实验完成后的结果整理与分析，不参与训练/推理流程。

## 关键文件

| 文件 | 说明 |
|------|------|
| `calculate_mean_metrics.py` | 计算多次运行的均值指标（MSE/MAE 等） |
| `extract_mean_to_excel.py` | 将均值指标导出为 Excel 表格 |
| `build_figure_data.py` | 构建论文图表所需的数据 |
| `parse_epoch_time.py` | 解析实验日志中的 epoch 耗时信息 |
| `extract_scaling_results.py` | 提取参数规模实验的结果 |
| `count_all_params.py` | 统计所有模型的参数量 |
| `count_baseline_params.py` | 统计基线模型的参数量 |
| `profile_baselines_infer.py` | 基线模型推理性能分析 |
| `collect_weather_results.py` | 收集天气数据集实验结果 |
| `prepare_weather_merge.py` | 准备天气数据合并 |
| `organize_dirs.py` | 整理实验结果目录结构 |
| `cal.sh` / `cal_seq2peak.sh` / `cal_single.sh` | 批量计算指标的 Shell 脚本 |

## 使用方式

各脚本独立运行，通常在实验完成后使用：

```bash
python tools/calculate_mean_metrics.py    # 汇总多次实验的均值
python tools/extract_mean_to_excel.py     # 导出 Excel
python tools/count_all_params.py          # 统计参数量
bash tools/cal.sh                         # 批量计算
```
