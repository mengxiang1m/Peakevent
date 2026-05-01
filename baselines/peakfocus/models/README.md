# models/ — 模型实现目录

## 用途

本目录包含所有时间序列预测模型的 PyTorch 实现，共 **44 个文件**。
核心模型为 `proposed_model.py`（PeakFocus），其余为基线对比模型。

## 关键文件

| 文件 | 说明 |
|------|------|
| `proposed_model.py` | **PeakFocus**（本文提出的模型） |
| `peak_Transformer.py` | 峰值感知 Transformer 变体 |
| `covariate_bridge.py` | 协变量桥接模块 |
| `naive_baselines.py` | 朴素基线方法（如 Repeat-Last 等） |
| `PatchTST.py` | PatchTST 基线 |
| `iTransformer.py` | iTransformer 基线 |
| `TimeMixer.py` | TimeMixer 基线 |
| `CycleNet.py` | CycleNet 基线 |
| `Informer.py` | Informer 基线 |
| `Autoformer.py` | Autoformer 基线 |

其他基线包括 DLinear、FEDformer、TimesNet、Crossformer、Mamba、TiDE、SegRNN、WPMixer 等。

## 使用方式

模型通过 `exp/` 中的实验类动态加载，无需手动实例化。
所有模型遵循统一接口：接收 `configs` 参数，实现 `forward()` 方法。

```python
# 在实验脚本中通过 --model 参数指定模型名称
# 例如：--model proposed_model 或 --model PatchTST
```
