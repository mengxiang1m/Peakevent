# PatchEvent 代码重构计划

> 基于 PeakFocus 代码架构最佳实践 + PatchEvent 现状审计
> 生成日期：2026-04-15 | 状态：待执行

---

## 一、现状诊断

### 1.1 数据概览

| 指标 | PatchEvent (当前) | PeakFocus (参考) |
|------|-------------------|------------------|
| 总 .py 文件数 | ~80 | 129 |
| 最大单文件行数 | model.py 1632 行 | exp_peak_detect.py ~1750 行 |
| 重复函数组数 | 12 组（三域 builder 同构） | 3 组（peak evaluator 重复） |
| 硬编码路径文件数 | 27 个文件 | ~5 个文件 |
| run_*.py 配置脚本 | 21 个，PYTHON 定义 19 次 | 用 shell 脚本 + `_common.sh` |
| tests/ 目录 | 不存在 | 不存在 |
| sys.path hack | 21 处 | 少量 |
| 统一入口 | 无 | `run.py` 单入口 + 任务路由 |
| 配置体系 | 散落在各 run_*.py 中 | argparse 集中在 run.py |
| 数据抽象层 | 无（三个独立 builder） | `data_factory` + `Dataset_Custom_Mixed` |

### 1.2 核心问题分类

| 类别 | 严重度 | 影响范围 | 文件数 |
|------|--------|----------|--------|
| **P0 重复代码** | 🔴 高 | 改一处漏两处 | 3 个 builder + 多个 run_*.py |
| **P0 大文件** | 🔴 高 | 维护困难、合并冲突 | model.py / train.py / build_wlel_events.py |
| **P1 硬编码路径** | 🟡 中 | 换机器就崩 | 27 个文件 |
| **P1 配置散乱** | 🟡 中 | 参数不一致风险 | 21 个 run_*.py |
| **P1 sys.path hack** | 🟡 中 | import 脆弱 | 21 处 |
| **P2 错误处理** | 🟢 低 | 静默失败 | 3 个文件 |
| **P2 缺失测试** | 🟢 低 | 回归风险 | 所有核心模块 |

---

## 二、参考蓝图：从 PeakFocus 学什么

### 2.1 借鉴的设计模式

| PeakFocus 实践 | 怎么做 | 参考文件 |
|---------------|--------|----------|
| **单入口 + 任务路由** | `run.py` 统一 CLI，`if task == 'xxx': Exp = ExpXxx` | `run.py:331` |
| **统一实验命名** | `setting` 字符串自动生成，用于 checkpoint/results 路径 | `run.py:354` |
| **数据工厂** | `data_factory.py` 路由不同数据集，返回统一 DataLoader | `data_provider/data_factory.py:26` |
| **Exp 基类** | `Exp_Basic` 统一模型注册、设备管理、日志接口 | `exp/exp_basic.py:10` |
| **模型组件开关** | `if_msm_pl`, `if_lad` 参数化，一套代码支撑所有消融 | `models/proposed_model.py:322` |
| **日志→论文数据链** | `tools/parse_epoch_time.py → build_figure_data.py → extract_mean_to_excel.py` | `tools/` 目录 |
| **公共工具模块** | `utils/tools.py`（EarlyStopping 等）、`utils/metrics.py` | `utils/` 目录 |

### 2.2 不照搬的部分

| PeakFocus 不足 | PatchEvent 应做的改进 |
|---------------|---------------------|
| 无配置文件体系（纯 argparse+shell） | 引入分层 YAML 配置 + CLI 覆盖 |
| peak evaluator 在 3 个 exp 文件重复 | 从一开始就抽到独立 metrics 模块 |
| 日志解析靠正则 | 用 JSON 结构化日志 |
| shell 脚本硬编码路径 | Python 脚本 + `sys.executable` |

---

## 三、目标架构

### 3.1 目标目录结构

```
ywz2-PatchEvent/
├── configs/                          # [新建] 分层配置
│   ├── base.yaml                     # 共享默认参数
│   ├── datasets/                     # 数据集特定配置
│   │   ├── wlel.yaml
│   │   ├── ett.yaml
│   │   └── elc.yaml
│   └── experiments/                  # 实验特定覆盖
│       ├── arch_ablation.yaml
│       ├── ar_ablation.yaml
│       ├── p1_ablation.yaml
│       ├── pred168.yaml
│       ├── pred336.yaml
│       └── hw_ablation.yaml
│
├── patchevent/                       # [新建] Python 包
│   ├── __init__.py
│   ├── data/                         # 数据抽象层
│   │   ├── __init__.py
│   │   ├── event_pipeline/           # [从 dataset/build_*_events.py 抽取]
│   │   │   ├── __init__.py
│   │   │   ├── config.py             # PipelineConfig dataclass
│   │   │   ├── preprocess.py         # robust_zscore, ensure_hourly_timeline
│   │   │   ├── detectors.py          # detect_events_gradient_width, _anchor_only, _hybrid
│   │   │   ├── labeling.py           # attach_point_level_event_fields, build_patch_label_table
│   │   │   ├── splits.py             # build_split_indices, validate_ratios
│   │   │   ├── reporting.py          # build_quality_report, report_to_markdown
│   │   │   └── io.py                 # save_events, load_events
│   │   ├── dataset_phase1.py         # [从 phase1.../dataset.py 迁移]
│   │   └── dataset_phase2.py         # [从 phase2.../dataset.py 迁移]
│   │
│   ├── models/                       # 模型定义
│   │   ├── __init__.py
│   │   ├── phase1_encoder.py         # [从 phase1.../model.py 迁移]
│   │   ├── encoders.py              # [从 phase2 model.py 拆出] CNNEncoder, LSTMEncoder, MLPEncoder
│   │   ├── event_tokenizer.py        # [从 phase2 model.py 拆出] StructuredEventTokenizer
│   │   ├── small_patch_decoder.py    # [从 phase2 model.py 拆出] SmallPatchDecoder (核心)
│   │   ├── non_ar_decoder.py         # [从 phase2 model.py 拆出] NonAutoRegressiveDecoder
│   │   ├── decomposition.py          # [从 phase2 model.py 拆出] InputSeriesDecomposition
│   │   └── factory.py                # [从 phase2 model.py 拆出] build_model
│   │
│   ├── losses/                       # 损失函数
│   │   ├── __init__.py
│   │   ├── matching.py               # _hungarian_match_loss
│   │   ├── event_localization.py     # compute_event_local_mse_loss_gpu
│   │   ├── value_losses.py           # apex_mape_loss, dense_value_loss, merge_value_loss
│   │   └── intensity.py              # compute_int_cls_loss, compute_intensity_loss
│   │
│   ├── training/                     # 训练逻辑
│   │   ├── __init__.py
│   │   ├── engine.py                 # train_one_epoch
│   │   ├── eval_loop.py              # validate
│   │   └── output_adapter.py         # _unpack_model_outputs
│   │
│   ├── evaluation/                   # 评估指标
│   │   ├── __init__.py
│   │   ├── event_metrics.py          # [从 phase2.../evaluate.py 迁移] F1/Precision/Recall/MAE
│   │   └── posthoc_evaluator.py      # [从 comparison.../posthoc_evaluate.py 迁移]
│   │
│   ├── optim/                        # 优化器 & 调度器
│   │   ├── __init__.py
│   │   └── schedulers.py             # build_warmup_cosine_scheduler
│   │
│   └── utils/                        # 公共工具
│       ├── __init__.py
│       ├── reproducibility.py        # set_seed
│       └── cli.py                    # 共享 argparse 构建器
│
├── scripts/                          # [重组] 实验编排脚本
│   ├── train_phase1.py               # Phase1 统一入口
│   ├── train_phase2.py               # Phase2 统一入口（原 train.py 的 main）
│   ├── run_ablation.py               # [合并 21 个 run_*.py] 统一消融入口
│   ├── run_baselines.py              # 基线实验入口
│   ├── collect_results.py            # 结果汇总
│   └── build_events.py               # [合并 3 个 build_*_events.py] 统一数据构建入口
│
├── tests/                            # [新建] 测试
│   ├── test_event_pipeline.py        # 事件检测/标签/切分
│   ├── test_tokenizer.py             # StructuredEventTokenizer
│   ├── test_model_forward.py         # 模型前向 shape 检查
│   └── test_smoke_train.py           # 1-batch 训练冒烟测试
│
├── phase1_encoder_pretrain_final/    # [保留] 但核心逻辑迁移到 patchevent/
├── phase2_small_decoder_final/       # [保留] 但核心逻辑迁移到 patchevent/
├── comparison_experiments/           # [保留] 基线对比
├── dataset/                          # [保留] 原始数据 + 旧 builder（标记 deprecated）
│
├── CLAUDE.md
├── REFACTOR_PLAN.md                  # 本文件
└── pyproject.toml                    # [新建] 包配置
```

### 3.2 配置体系设计

```yaml
# configs/base.yaml — 所有实验共享的默认值
python: null  # 运行时自动解析为 sys.executable
seeds: [42, 2024, 7]

phase1:
  seq_len: 96
  patch_len: 8
  stride: 4
  d_model: 128
  n_heads: 4
  n_layers: 2
  lr: 1.0e-3
  weight_decay: 1.0e-4
  train_epochs: 50
  batch_size: 64

phase2:
  d_model: 128
  d_ff: 256
  n_heads: 4
  n_layers: 3
  lr: 5.0e-4
  encoder_lr: 1.0e-5
  train_epochs: 100
  batch_size: 32
  dropout: 0.2
  label_smoothing: 0.1
  onset_apex_weight: 2
  dur_int_weight: 0.5
  encoder_mode: frozen
```

```yaml
# configs/datasets/wlel.yaml — WLEL 特定参数
dataset_name: wlel
series_path: dataset/wlel/event_v1/wlel_hourly_mixed_with_peaks.csv
events_path: dataset/wlel/event_v1/events_gradient_width.json
lookahead: 3
eta: 0.40
rate_frac: 0.40
```

```yaml
# configs/experiments/arch_ablation.yaml — 架构消融
experiment_name: arch_ablation_v2
base_config: base
variants:
  G0_full: {}  # 默认配置
  GS1_wo_mempos: {disable_mempos: true}
  GS2_wo_sa: {disable_sa_agg: true}
  GS3_bare: {disable_mempos: true, disable_sa_agg: true}
  GS4_cnn: {encoder_mode: cnn}
  GS4_lstm: {encoder_mode: lstm}
  GS4_mlp: {encoder_mode: mlp}
  GS4_scratch: {encoder_mode: scratch}
  GS5_detr: {decoder_type: detr}
```

---

## 四、分阶段执行计划

### Phase R1：基础设施（无功能变更，纯结构调整）

**目标**：建立包结构和配置体系，不改变任何现有功能行为。

| 步骤 | 任务 | 涉及文件 | 验证方式 | 预估工作量 |
|------|------|----------|----------|-----------|
| R1.1 | 创建 `patchevent/` 包骨架和 `__init__.py` | 新建 ~15 个空文件 | `import patchevent` 成功 | 小 |
| R1.2 | 创建 `configs/` 配置文件 | 新建 base.yaml + 3 个 dataset yaml + 6 个 experiment yaml | YAML 格式正确 | 小 |
| R1.3 | 创建 `pyproject.toml`（可选 pip install -e .） | 新建 1 个文件 | `pip install -e . && python -c "import patchevent"` | 小 |
| R1.4 | 统一 PYTHON 路径为 `sys.executable` | 修改 19 个 run_*.py | 所有 `grep -r "D:/Anaconda"` 返回空 | 中 |

**Commit 节点**：`refactor(R1): establish package skeleton, configs, and remove hardcoded paths`

---

### Phase R2：抽取数据管道公共库（消除最大重复源）

**目标**：将三个 `build_*_events.py` 的 12 组同构函数抽取到 `patchevent/data/event_pipeline/`。

| 步骤 | 任务 | 涉及文件 | 验证方式 |
|------|------|----------|----------|
| R2.1 | 抽取公共函数到 `event_pipeline/` | 从 build_wlel/ett/elc_events.py 提取 12 个函数 | 逐个 import 成功 |
| R2.2 | 瘦身三个 builder 为薄封装 | build_wlel/ett/elc_events.py → 只保留 main + 配置差异 | 各 builder 从 ~500-1200 行降到 ~100 行 |
| R2.3 | 创建统一入口 `scripts/build_events.py` | 新建 1 个文件 | `python scripts/build_events.py --domain wlel` 输出与原始一致 |

**详细函数迁移表**：

| 函数名 | 源文件行号 (wlel/ett/elc) | 目标模块 | 签名差异 |
|--------|--------------------------|----------|----------|
| `robust_zscore` | 130 / 61 / 61 | `preprocess.py` | 无 |
| `ensure_hourly_timeline` | 138 / 69 / 69 | `preprocess.py` | 无 |
| `detect_events_gradient_width` | 519 / 117 / 117 | `detectors.py` | wlel 多 `grad_values` 参数 → 改为可选参数 |
| `detect_events_anchor_only` | 446 / 207 / - | `detectors.py` | elc 缺失 → 无需迁移 elc |
| `detect_events_hybrid` | 636 / - / - | `detectors.py` | 仅 wlel 有 |
| `attach_point_level_event_fields` | 702 / 265 / 202 | `labeling.py` | 无 |
| `build_patch_label_table` | 772 / 313 / 250 | `labeling.py` | 无 |
| `build_split_indices` | 838 / 354 / 291 | `splits.py` | 无 |
| `build_quality_report` | 857 / 401 / 337 | `reporting.py` | elc 缺类型标注 → 统一 |
| `report_to_markdown` | 946 / 459 / 380 | `reporting.py` | elc 缺类型标注 → 统一 |
| `save_events` | 1011 / 373 / 310 | `io.py` | elc 缺类型标注 → 统一 |
| `parse_args` | 51 / 41 / 41 | `config.py` | 无 |

**Commit 节点**：`refactor(R2): extract shared event pipeline, deduplicate dataset builders`

---

### Phase R3：拆分大文件（Phase2 model.py + train.py）

**目标**：将 1632 行 model.py 和 965 行 train.py 拆分为职责清晰的小模块。

| 步骤 | 任务 | 拆出内容 | 行数 |
|------|------|----------|------|
| R3.1 | 拆 `encoders.py` | CNNEncoder (148-168), LSTMEncoder (171-195), MLPEncoder (198-218) | ~70 |
| R3.2 | 拆 `decomposition.py` | InputSeriesDecomposition (51-141) | ~90 |
| R3.3 | 拆 `event_tokenizer.py` | StructuredEventTokenizer (225-355) | ~130 |
| R3.4 | 拆 `non_ar_decoder.py` | NonAutoRegressiveDecoder (1404-1510) | ~110 |
| R3.5 | 拆 `losses/matching.py` | _hungarian_match_loss (1517-1575) | ~60 |
| R3.6 | 拆 `factory.py` | build_model (1582-1632) | ~50 |
| R3.7 | 瘦身 `model.py` | 只保留 SmallPatchDecoder (366-1397) + imports | ~1050 |
| R3.8 | 拆 `losses/` | 从 train.py 拆出 5 个 loss 函数 (225-437) | ~210 |
| R3.9 | 拆 `training/engine.py` | train_one_epoch (440-698) + validate (702-778) | ~340 |
| R3.10 | 拆 `optim/schedulers.py` | build_warmup_cosine_scheduler (209-222) | ~15 |
| R3.11 | 瘦身 `train.py` | 只保留 main + parse_args + imports | ~350 |

**迁移后 model.py 的 import 兼容**：

```python
# phase2_small_decoder_final/model.py — 瘦身后
# 保持向后兼容的 re-export
from patchevent.models.small_patch_decoder import SmallPatchDecoder
from patchevent.models.encoders import CNNEncoder, LSTMEncoder, MLPEncoder
from patchevent.models.event_tokenizer import StructuredEventTokenizer
from patchevent.models.non_ar_decoder import NonAutoRegressiveDecoder
from patchevent.models.decomposition import InputSeriesDecomposition
from patchevent.models.factory import build_model
from patchevent.losses.matching import hungarian_match_loss
```

**Commit 节点**：`refactor(R3): split model.py and train.py into focused modules`

---

### Phase R4：消除 sys.path hack + 包化导入

**目标**：移除 21 处 `sys.path.insert/append` hack，改为 `patchevent` 包导入。

| 文件 | 当前 hack | 改为 |
|------|----------|------|
| `phase1.../model.py:31` | `sys.path.insert(0, ROOT)` | `from patchevent.models import ...` |
| `phase1.../train.py:33,35` | `sys.path.insert(0, ...)` | `from patchevent.data import ...` |
| `phase2.../model.py:36` | `sys.path.insert(0, ROOT)` | `from patchevent.models.phase1_encoder import ...` |
| `phase2.../train.py:28` | `sys.path.insert(0, ROOT)` | `from patchevent.training import ...` |
| `phase2.../dataset.py:21` | `sys.path.insert(0, ROOT)` | `from patchevent.data import ...` |
| `phase2.../evaluate.py:39` | `sys.path.insert(0, ROOT)` | `from patchevent.evaluation import ...` |
| `comparison.../posthoc_evaluate.py:16,18` | `sys.path.insert(0, ...)` | `from patchevent.evaluation import ...` |
| 其余 14 处 | 类似模式 | 类似改法 |

**Commit 节点**：`refactor(R4): replace sys.path hacks with package imports`

---

### Phase R5：合并实验编排脚本

**目标**：将 21 个 `run_*.py` 合并为 1-3 个统一入口，配合 YAML 配置。

| 步骤 | 任务 | 合并源 | 目标 |
|------|------|--------|------|
| R5.1 | 合并 Phase2 消融脚本 | `run_arch_ablation_v2.py`, `run_ar_ablation_v2.py`, `run_p1_ablation_phase2.py`, `run_p1_ablation_phase2_ett_elc.py`, `run_p1_subtask_ablation.py` | `scripts/run_ablation.py --config configs/experiments/xxx.yaml` |
| R5.2 | 合并 Horizon 扩展脚本 | `run_pred168.py`, `run_pred336.py`, `run_pred_hybrid.py` | `scripts/run_ablation.py --config configs/experiments/pred168.yaml` |
| R5.3 | 合并 Grid Search 脚本 | `run_arch_grid_search.py`, `run_full_grid_search.py`, `run_weight_grid_search.py` | `scripts/run_grid_search.py --config configs/experiments/grid_xxx.yaml` |
| R5.4 | 合并 HW 消融脚本 | `run_hw_ablation.py`, `run_hw_p1_ablation.py` | `scripts/run_ablation.py --config configs/experiments/hw_ablation.yaml` |
| R5.5 | 合并 Domain 最优消融 | `run_elc_optimal_ablation.py`, `run_ett_optimal_ablation.py` | `scripts/run_ablation.py --config configs/experiments/domain_optimal.yaml` |
| R5.6 | 保留旧脚本标记 deprecated | 在每个旧 run_*.py 头部加 `# DEPRECATED: use scripts/run_ablation.py` | — |

**统一入口设计**：

```python
# scripts/run_ablation.py
"""统一消融实验入口 — 替代 21 个独立 run_*.py"""
import yaml, sys, subprocess

def main():
    config = load_config(args.config)       # base.yaml + dataset.yaml + experiment.yaml 合并
    for domain in config['domains']:
        for variant_name, overrides in config['variants'].items():
            for seed in config['seeds']:
                cmd = build_command(config, domain, variant_name, overrides, seed)
                run_or_skip(cmd)  # 已完成自动跳过
```

**Commit 节点**：`refactor(R5): consolidate 21 run scripts into unified experiment runner`

---

### Phase R6：错误处理 + 基础测试

**目标**：修复静默失败问题，建立最小测试套件。

| 步骤 | 任务 | 涉及文件 |
|------|------|----------|
| R6.1 | 修复裸 except | `dataset/aggregate_wlel.py:120` |
| R6.2 | 修复 pass 吞错 | `bench_inference.py:74,79`; `data_utils.py:165,183,...` |
| R6.3 | 创建 `tests/test_event_pipeline.py` | 事件检测 + 标签表 + 切分 |
| R6.4 | 创建 `tests/test_tokenizer.py` | StructuredEventTokenizer 编解码一致性 |
| R6.5 | 创建 `tests/test_model_forward.py` | Phase1/Phase2 前向 shape 检查 |
| R6.6 | 创建 `tests/test_smoke_train.py` | 1-batch 训练不报错 |

**Commit 节点**：`refactor(R6): fix error handling and add regression tests`

---

## 五、执行约束

### 5.1 不变量（重构期间必须保持）

- [ ] 所有已有实验结果（checkpoint、JSON、CSV）不受影响
- [ ] `python phase2_small_decoder_final/train.py --help` 参数不变
- [ ] `python phase1_encoder_pretrain_final/train.py --help` 参数不变
- [ ] 已有的 `run_*.py` 脚本在 deprecated 标记后仍可运行
- [ ] 论文中引用的所有实验可被精确复现

### 5.2 提交策略

| Phase | Commit 消息模板 | Push 时机 |
|-------|----------------|----------|
| R1 | `refactor(R1): ...` | 完成后立即 push |
| R2 | `refactor(R2): ...` | 完成后 push |
| R3 | `refactor(R3): ...` | 完成后 push |
| R4 | `refactor(R4): ...` | R4+R5 一起 push |
| R5 | `refactor(R5): ...` | 与 R4 一起 push |
| R6 | `refactor(R6): ...` | 完成后 push + 跑测试 |

### 5.3 风险控制

| 风险 | 缓解措施 |
|------|----------|
| 拆分后 import 环路 | 每个 Phase 完成后跑 `python -c "import patchevent"` |
| 旧脚本 break | 旧文件 re-export + deprecated 标记，不删除 |
| 实验不可复现 | 保留所有 checkpoint，重构前后用相同 seed 对比 1 个实验输出 |

---

## 六、工作量估算

| Phase | 预估时间 | 难度 | 依赖 |
|-------|---------|------|------|
| R1 基础设施 | 30min | 低 | 无 |
| R2 数据管道 | 1-2h | 中 | R1 |
| R3 拆分大文件 | 1-2h | 中 | R1 |
| R4 包化导入 | 30min | 低 | R1+R3 |
| R5 合并脚本 | 1-2h | 中 | R1+R2 |
| R6 测试 | 1h | 低 | R2+R3 |
| **总计** | **5-8h** | | |

---

## 七、Quick Reference：最高优先级改动速查

如果只做一件事，做 **R2**（抽取数据管道公共库）— 消除最多重复代码、降低最大维护风险。

如果做两件事，再做 **R1.4**（消除硬编码路径）— 5 分钟改完 19 个文件，立刻提升可迁移性。

如果做三件事，再做 **R3**（拆分 model.py）— 从 1632 行降到 ~1050 行，后续所有修改都更安全。
