# 数据集分析工具与结果

本文件夹包含三域数据集（WLEL / ETT / ELC）的数据质量分析脚本和可视化结果。

## 文件说明

| 文件 | 说明 |
|------|------|
| `viz_ett_detailed.py` | ETT (ETTh2, la=3) 数据+消融可视化（全局时序、事件放大、分布统计、消融对比） |
| `viz_elc_detailed.py` | ELC (Electricity, la=5) 数据+消融可视化（同上格式） |
| `diagnose_peaks.py` | 三域峰值检测质量诊断（密度、间距、prominence、局部极大值检验） |
| `regen_peaks_la3.py` | ETT/ELC lookahead=3 峰值重新生成（对比 la=5 的密度差异） |

## 可视化结果

| 图片 | 内容 |
|------|------|
| `viz_ett_detailed.png` | ETT 数据概览 + 5配置消融MAE/F1对比 |
| `viz_elc_detailed.png` | ELC 数据概览 + 5配置消融MAE/F1对比（3-seed mean±std） |
| `diagnose_peaks.png` | 三域峰值检测质量对比 |

## 运行方式

```bash
cd ywz2
python dataset/analysis/viz_ett_detailed.py   # 输出 viz_ett_detailed.png
python dataset/analysis/viz_elc_detailed.py   # 输出 viz_elc_detailed.png
python dataset/analysis/diagnose_peaks.py     # 输出 diagnose_peaks.png
```
