# exp/ — 实验流程类目录

## 用途

本目录包含所有实验流程的实现类，负责模型的训练、验证、测试和评估逻辑。
每个类封装了完整的实验 pipeline，包括数据加载、优化器配置、训练循环和指标计算。

## 关键文件

| 文件 | 说明 |
|------|------|
| `exp_basic.py` | 基类，定义通用接口（`train`/`test`/`predict`） |
| `exp_peak_detect_based_on_long_term_forecasting.py` | **PeakFocus 核心实验类**，峰值检测 + 长期预测 |
| `exp_peak_detect_based_on_long_term_forecasting_basic.py` | 峰值检测基础版本 |
| `exp_peak_detect_based_on_long_term_forecasting_seq2peak.py` | Seq2Peak 模式实验类 |
| `exp_long_term_forecasting.py` | 标准长期预测实验（基线对比用） |
| `exp_short_term_forecasting.py` | 短期预测实验（M4 数据集等） |
| `exp_anomaly_detection.py` | 异常检测实验 |
| `exp_imputation.py` | 缺失值填补实验 |
| `exp_classification.py` | 时间序列分类实验 |

## 使用方式

实验类由主入口脚本（如 `run.py`）根据 `--task_name` 参数自动选择并实例化：

```python
# --task_name long_term_forecast → exp_long_term_forecasting
# --task_name peak_detect_ltf    → exp_peak_detect_based_on_long_term_forecasting
```

典型流程：`Exp.train()` → `Exp.test()` → 输出指标到 `results/` 目录。
