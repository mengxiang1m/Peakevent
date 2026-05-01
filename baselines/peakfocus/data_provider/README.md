# data_provider/ — 数据加载器目录

## 用途

本目录负责数据集的读取、预处理和 PyTorch DataLoader 的构建。
支持多种时间序列任务（预测、分类、异常检测、缺失填补）的数据管线。

## 关键文件

| 文件 | 说明 |
|------|------|
| `data_factory.py` | **数据工厂**，根据 `--data` 参数返回对应的 Dataset 和 DataLoader |
| `data_loader.py` | 核心数据集类，包括 `Dataset_Custom`、`Dataset_Pred`、电力/ETT 专用数据集等 |
| `m4.py` | M4 竞赛数据集加载器（短期预测任务） |
| `uea.py` | UEA 多变量时间序列分类数据集加载器 |

## 使用方式

在实验类中通过 `data_factory.data_provider()` 获取数据加载器：

```python
from data_provider.data_factory import data_provider

train_data, train_loader = data_provider(args, flag='train')
test_data, test_loader = data_provider(args, flag='test')
```

数据集选择通过命令行参数控制：
- `--data custom` — 通用 CSV 数据集
- `--data ETTh1` / `ETTh2` / `ETTm1` / `ETTm2` — ETT 数据集
- `--data electricity_mixed` — 电力负荷混合数据集（含峰值标注）
