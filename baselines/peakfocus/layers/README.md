# layers/ — 模型层实现目录

## 用途

本目录包含各模型共用或专用的神经网络层实现，共 **14 个模块**。
提供嵌入层、注意力机制、编码器-解码器结构、频域变换等基础组件。

## 关键文件

| 文件 | 说明 |
|------|------|
| `Embed.py` | 嵌入层（时间特征嵌入、位置编码、Patch 嵌入等） |
| `SelfAttention_Family.py` | 多种自注意力实现（Full/Prob/Flow Attention 等） |
| `Transformer_EncDec.py` | 标准 Transformer 编码器-解码器结构 |
| `Autoformer_EncDec.py` | Autoformer 的序列分解编解码器 |
| `Crossformer_EncDec.py` | Crossformer 的跨维度编解码器 |
| `ETSformer_EncDec.py` | ETSformer 的指数平滑编解码器 |
| `Pyraformer_EncDec.py` | Pyraformer 的金字塔注意力编解码器 |
| `AutoCorrelation.py` | 自相关机制（Autoformer 使用） |
| `FourierCorrelation.py` | 傅里叶域相关性计算（FEDformer 使用） |
| `MultiWaveletCorrelation.py` | 多小波相关性计算 |
| `DWT_Decomposition.py` | 离散小波变换分解 |
| `Conv_Blocks.py` | 卷积模块（Inception Block 等） |
| `StandardNorm.py` | 标准化层（可逆实例归一化等） |

## 使用方式

这些层由 `models/` 中的模型直接 import 使用，无需单独调用：

```python
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import FullAttention
```
