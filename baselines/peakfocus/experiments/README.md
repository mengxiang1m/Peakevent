# experiments/ — 实验脚本目录

## 用途

本目录包含论文中所有实验的运行脚本，按实验编号组织为子目录。
每个子目录对应论文中的一个实验（表/图），包含相应的 Shell 脚本。

## 目录结构

| 目录 | 说明 |
|------|------|
| `01_main_table/` | **主实验表**：全部基线模型对比（WLEL + ELC 数据集） |
| `02_ablation/` | **消融实验**：验证各组件的贡献 |
| `03_generality/` | **通用性实验**：PeakFocus 对不同基线模型的增益 |
| `04_param_sensitivity/` | **参数敏感性**：超参数影响分析 |
| `05_param_scaling/` | **参数规模**：模型大小对性能的影响 |
| `06_efficiency/` | **效率实验**：训练/推理速度与内存开销 |
| `07_weather/` | **天气数据集**：跨领域泛化实验 |
| `08_patchtst_fair/` | **PatchTST 公平对比**：统一设置下的对比 |

## 关键文件

| 文件 | 说明 |
|------|------|
| `_common.sh` | 公共配置（GPU、路径、Python 解释器等） |
| `resume_from_interrupt.sh` | 从中断处恢复实验 |
| `run_missing.sh` 等 | 补跑缺失实验的辅助脚本 |

## 使用方式

```bash
# 1. 先检查并修改 _common.sh 中的环境配置
# 2. 进入对应实验目录运行脚本
cd experiments/01_main_table/
bash run_wlel.sh   # 运行 WLEL 数据集实验（示例）
```

每个子目录内通常按数据集（elc/wlel）分为不同脚本。
