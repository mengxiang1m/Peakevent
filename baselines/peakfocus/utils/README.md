# utils/ — 工具函数目录

## 用途

本目录包含项目中通用的工具函数和辅助模块，共 **15 个文件**。
涵盖评估指标、损失函数、时间特征、数据增强、统计检验等功能。

## 关键文件

| 文件 | 说明 |
|------|------|
| `metrics.py` | **评估指标**（MSE、MAE、RMSE、MAPE 等） |
| `tools.py` | 通用工具函数（EarlyStopping、学习率调整、模型保存等） |
| `losses.py` | 自定义损失函数 |
| `masking.py` | 掩码生成工具（用于缺失填补等任务） |
| `timefeatures.py` | 时间特征提取（小时、星期、月份等周期性特征） |
| `augmentation.py` | 数据增强方法 |
| `dtw.py` / `dtw_metric.py` | DTW（动态时间规整）距离计算 |
| `ADFtest.py` | ADF 平稳性检验 |
| `print_args.py` | 实验参数格式化打印 |
| `m4_summary.py` | M4 数据集评估汇总 |
| `pytorch_stats_loss.py` | 基于 PyTorch 的统计损失函数 |

## 使用方式

在实验类或模型中按需 import：

```python
from utils.metrics import metric  # 计算 MSE/MAE
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.timefeatures import time_features
```
