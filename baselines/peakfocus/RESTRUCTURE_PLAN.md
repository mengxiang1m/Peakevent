# 项目重构计划 v3

> **更新时间**：2026-04-13
> **基于**：Codex 全面调研（6 项任务，覆盖目录快照、剩余实验、依赖变化、论文状态、数据完整性、风险评估）
>
> **核心目标**：
> 1. 重组为 release-ready 结构
> 2. 重构后剩余 99 个实验 runs 能立即继续跑
> 3. 实验跑完后能直接刷新论文数据

---

## 一、当前项目状态（2026-04-13 快照）

### 1.1 根目录一级概览

| 目录 | 状态 | 处理方式 |
|---|---|---|
| `run.py` | 框架核心 | 不动 |
| `models/`(43py) `exp/`(10py) `layers/`(14py) `data_provider/`(5py) `utils/`(15py) | 框架核心 | 不动 |
| `dataset/` | 数据 | 不动（gitignore 已排除） |
| `results/`(2732 files) `checkpoints/`(435 files) | 实验产物 | 不动（run.py 硬编码 `./results/`） |
| `visualization_outputs/`(93k files) | 模型 vis 产物 | 不动（31GB，gitignore 已排除） |
| `exp_results/`(15 files) | 聚合数据 | 不动 |
| `scaling_experiments/`(scripts/18sh + tools/6py + viz/7py + 6md) | **要迁移** | → experiments/ + tools/ + visualization/ + docs/ |
| `scripts/`(343sh + 2md) | **要迁移** | 论文脚本 → experiments/，框架脚本 → archive/ |
| `metric_tools/`(2py + 3sh) | **要迁移** | → tools/ |
| `tools/`(5py) | **要迁移** | 合并目标（接收其他工具） |
| `visualization/`(3py) | **要迁移** | → visualization/interpretability/ |
| `analysis/`(3py + 4csv) | **要迁移** | vis → visualization/，其他 → archive/ |
| `archive/`(34 files) | 已归档 | 扩充后保留 |
| `docs/`(350 files) | 文档 | 保留 + 接收 md |
| `.audit_tmp/` `.vscode/` | 临时 | 删除 |

### 1.2 剩余实验

| 脚本 | 已完成 | 总需 | 剩余 |
|---|---|---|---|
| `scaling_dmodel_elc.sh` | 11 | 30 | **19** |
| `scaling_elayers_wlel.sh` | 0 | 20 | **20** |
| `scaling_elayers_elc.sh` | 0 | 20 | **20** |
| `scaling_dff_wlel.sh` | 0 | 20 | **20** |
| `scaling_dff_elc.sh` | 0 | 20 | **20** |
| **合计** | 11 | 110 | **99** |

### 1.3 关键发现

1. **ELC PeakFocus 数据不一致**：论文 Table I 的 ELC PeakFocus 行（MSE=0.602）和 figure_data.csv（MSE=0.443）不匹配。原因可能是论文使用旧的 424 权重数据，figure_data 是新的 244 数据。**实验跑完后需要统一刷新论文所有数值**。
2. **run.py results 路径是硬编码**：`./results/` + setting，不是 CLI 参数。重构不能改变 cwd 和 results 的相对关系。
3. **sys.path 操作**：当前 15 处（比上次少 4 处，部分文件已归档）。
4. **所有 source 语句**：14 处，全在 `scaling_experiments/scripts/`，全 source `_common.sh`。

---

## 二、目标结构

```
Code/
├── README.md
├── run.py                             # 不动（1283 次 shell 调用）
├── environment.yaml
├── LICENSE
│
│  ── 框架核心（不动，294 条 import）──
├── models/   exp/   layers/   data_provider/   utils/   dataset/
│
│  ── 实验产物（不动）──
├── results/   checkpoints/   visualization_outputs/   exp_results/
│
│  ── 重组 ──
├── experiments/                       # ★ 所有实验脚本
│   ├── README.md
│   ├── _common.sh                     # ← scaling_experiments/scripts/_common.sh
│   ├── run_remaining.sh               # ★ 新建：一键跑剩余 99 runs
│   ├── run_all.sh                     # ← run_all_v2.sh 改写
│   ├── logs/                          # 运行日志
│   │
│   ├── 01_main_table/{wlel,elc}/     # Table I（~20 个 sh）
│   ├── 02_ablation/{wlel,elc}/       # 消融（wo_lad/wo_msm_pl/wo_upap/hard_mask）
│   ├── 03_generality/{wlel,elc}/     # UPAP 泛化（vanilla_*.sh）
│   ├── 04_param_sensitivity/          # K sweep + Loss weights
│   ├── 05_param_scaling/              # d_model / e_layers / d_ff（剩余实验在此）
│   ├── 06_efficiency/                 # 参数量 + 推理速度
│   ├── 07_weather/                    # 天气消融
│   └── 08_patchtst_fair/              # PatchTST 公平对比
│
├── tools/                             # ★ 合并：原 tools/ + scaling.../tools/ + metric_tools/
│   ├── README.md
│   ├── parse_epoch_time.py
│   ├── build_figure_data.py
│   ├── count_baseline_params.py
│   ├── profile_baselines_infer.py
│   ├── extract_scaling_results.py
│   ├── organize_dirs.py
│   ├── calculate_mean_metrics.py
│   ├── extract_mean_to_excel.py
│   ├── cal.sh / cal_seq2peak.sh / cal_single.sh
│   ├── count_all_params.py
│   ├── collect_weather_results.py
│   └── prepare_weather_merge.py
│
├── visualization/                     # ★ 合并
│   ├── README.md
│   ├── figures/                       # ← scaling.../viz/（7 个画图脚本）
│   └── interpretability/              # ← visualization/（3 个）+ analysis/vis（1 个）
│
├── docs/                              # 保留 + 接收 md
│   ├── DATA_GUIDE.md
│   ├── EXPERIMENT_CODE_MAP.md
│   └── ...
│
└── archive/                           # 归档
    ├── scripts_framework/             # ← scripts/long_term_forecast/ + legacy_tslib/
    ├── scaling_experiments_legacy/     # ← scaling_experiments/*.md
    ├── analysis_legacy/               # ← analysis/ 非 vis 文件
    └── misc/                          # ← 一次性脚本
```

---

## 三、迁移清单

### 3.1 实验脚本（关键：保证剩余实验可跑）

**运行机制**：所有脚本假设 `cwd = Code/`。

| 原位置 | 目标 | 修改 |
|---|---|---|
| `scaling_experiments/scripts/_common.sh` | `experiments/_common.sh` | `LOG_DIR` 改为 `experiments/logs` |
| `scaling_experiments/scripts/scaling_*.sh`（6 个） | `experiments/05_param_scaling/` | source 路径 `_common.sh` → `../_common.sh` |
| `scaling_experiments/scripts/ablation_wo_upap_*.sh` | `experiments/02_ablation/{wlel,elc}/` | source 路径 → `../../_common.sh` |
| `scaling_experiments/scripts/realign_*.sh` | `experiments/02_ablation/` 或 `04_param_sensitivity/` | source 路径调整 |
| `scaling_experiments/scripts/patchtst_fair_*.sh` | `experiments/08_patchtst_fair/` | source 路径 → `../_common.sh` |
| `scaling_experiments/scripts/run_all_v2.sh` | `experiments/run_all.sh` | 重写路径列表 |
| `scripts/peak_detect_based_on_LTF/load_data/test_*.sh`（~12 个） | `experiments/01_main_table/wlel/` | 头部加 `cd "$(dirname "$0")/../../.."` |
| `scripts/peak_detect_based_on_LTF/ELC/test_*.sh`（~10 个） | `experiments/01_main_table/elc/` | 同上 |
| `scripts/peak_detect_based_on_LTF/load_data/test_proposed_model.sh` | **拆分** → peakfocus.sh + wo_lad.sh + wo_msm_pl.sh + vanilla.sh | 按行号拆分 |
| `scripts/peak_detect_based_on_LTF/ablation_*.sh`（3 个） | `experiments/04_param_sensitivity/` | 头部加 cd |
| `scripts/peak_detect_based_on_LTF/ablation_naive_baselines.sh` | `experiments/01_main_table/wlel/` | 头部加 cd |
| `scripts/peak_detect_based_on_LTF/load_data/test_proposed_model_weather.sh` | `experiments/07_weather/` | 头部加 cd |
| `scripts/long_term_forecast/`（204 个） | `archive/scripts_framework/` | 整体移动 |
| `scripts/legacy_tslib/`（106 个） | `archive/scripts_framework/` | 整体移动 |

**新建 `experiments/run_remaining.sh`**：
```bash
#!/usr/bin/env bash
# 继续跑剩余 99 个 scaling 实验
set -uo pipefail
DIR="$(dirname "$0")"
for s in 05_param_scaling/dmodel_elc.sh \
         05_param_scaling/elayers_wlel.sh 05_param_scaling/elayers_elc.sh \
         05_param_scaling/dff_wlel.sh 05_param_scaling/dff_elc.sh; do
  echo "[$(date +%H:%M:%S)] RUNNING: $s"
  bash "$DIR/$s" && echo "[$(date +%H:%M:%S)] OK: $s" || echo "[$(date +%H:%M:%S)] FAILED: $s"
done
```

### 3.2 工具代码

| 原位置 | 目标 | 代码修改 |
|---|---|---|
| `scaling_experiments/tools/*.py`（6 个） | `tools/` | count_baseline_params + profile_baselines_infer: `REPO_CODE = HERE.parent`；organize_dirs: 去硬编码路径 |
| `metric_tools/*`（5 个） | `tools/` | 无 |
| `tools/collect_baseline_720.py` + `_tmp_*` | `archive/misc/` | 无 |

### 3.3 可视化代码

| 原位置 | 目标 | 代码修改 |
|---|---|---|
| `scaling_experiments/viz/*.py`（7 个） | `visualization/figures/` | `from viz.xxx` → `from figures.xxx`（5 个画图脚本） |
| `visualization/*.py`（3 个） | `visualization/interpretability/` | sys.path 注入改为 Code/（多跳一级） |
| `analysis/visualize_interpretability.py` | `visualization/interpretability/` | sys.path 调整 |

### 3.4 文档 + 归档

| 原位置 | 目标 |
|---|---|
| `scaling_experiments/*.md`（DATA_GUIDE, EXPERIMENT_CODE_MAP, TODO） | `docs/` |
| `scaling_experiments/*.md`（AUDIT_REPORT, COMPAT_REPORT, PAPER_EDITS） | `archive/scaling_experiments_legacy/` |
| `analysis/compute_r2.py` + `load_data_analysis.py` + `irregular/` + `regular/` | `archive/analysis_legacy/` |
| `.audit_tmp/` `.vscode/` | 删除 |
| `result_peak_detect_based_on_long_term_forecasting.txt` | 删除 |

### 3.5 迁移后删除的原目录

`scaling_experiments/` → 内容全部迁出后删除
`scripts/` → 内容全部迁出后删除
`metric_tools/` → 全部迁入 tools/ 后删除
`analysis/` → 内容全部迁出后删除

---

## 四、验证实验可跑

### 4.1 重构后验证命令

```bash
cd Code

# 1. _common.sh 可 source
bash -c 'source experiments/_common.sh && echo "PYTHON=$PYTHON LOG_DIR=$LOG_DIR"'
# 期望：PYTHON=/.../python  LOG_DIR=experiments/logs

# 2. 核心路径可达
test -f run.py && test -d dataset && test -d results && test -d checkpoints && echo "paths OK"

# 3. scaling 脚本可 source
bash -c 'source experiments/05_param_scaling/dmodel_elc.sh <<< ""' 2>&1 | head -3

# 4. 无残留旧路径引用
grep -r "scaling_experiments" experiments/ tools/ visualization/ --include="*.py" --include="*.sh"
# 期望：0 匹配
```

### 4.2 继续实验

```bash
cd Code && nohup bash experiments/run_remaining.sh > experiments/logs/run_remaining.log 2>&1 &
```

---

## 五、实验跑完后的论文更新清单

### 5.1 数据刷新流程

```bash
python3 tools/parse_epoch_time.py --results_root results --out exp_results/parsed_logs.csv
python3 tools/build_figure_data.py --parsed exp_results/parsed_logs.csv --out exp_results/figure_data.csv
python3 visualization/figures/plot_radar.py --real
python3 visualization/figures/plot_ablation_combined.py --real
python3 visualization/figures/plot_generality_bar.py --real
python3 visualization/figures/plot_parameter_panel.py --real
```

### 5.2 论文需要更新的具体部分

| 论文位置 | 要改什么 | 原因 |
|---|---|---|
| **Table I（主表）** | ELC PeakFocus 全部数值 | 当前论文用旧 424 数据，figure_data 是新 244 数据，MSE 差 0.16 |
| **Table I（主表）** | ELC 其他 baseline 核对 | 可能也有偏差 |
| **Fig.4（消融图）** | 用新数据重新生成 | 实验完成后数据更完整 |
| **Fig.5（泛化图）** | 用新数据重新生成 | 同上 |
| **Fig.6（参数面板）** | **必须重新生成** — 目前 ELC d_model 只有 2 个点，e_layers/d_ff 全空 | 99 runs 跑完后才能画完整 2×3 面板 |
| **Table III（K sweep）** | 核对数值 | 已有 244 数据 |
| **Table IV（Loss weights）** | 核对数值 + 默认行标注 | (0.2,0.4,0.4) 应为默认 |
| **Fig.3（雷达图）** | 可能不变 | WLEL 数据已稳定 |
| **正文数值引用** | Observation 2-6 中引用的具体数值 | 跟随表/图数据更新 |
| **红色标记** | 实验完成确认无误后去掉所有 `\textcolor{red}{...}` | 最终提交版本 |

### 5.3 已知数据问题

| 问题 | 详情 | 解决时机 |
|---|---|---|
| ELC PeakFocus Table I vs CSV | 论文 MSE=0.602，CSV MSE=0.443 | 实验跑完后统一用 CSV 数据覆盖论文 |
| ELC H=720 TP-MSE | 论文 1.169，CSV 1.221 | 同上 |
| 消融表已删除但有 stale ref | 可能有 `\ref{tab:ablation}` 残留引用 | 编译检查 LaTeX warnings |

---

## 六、执行步骤（有序）

| Phase | 操作 | 耗时 | 前置条件 |
|---|---|---|---|
| 0 | `git add -A && git commit -m "pre-restructure"` | 1 min | 无 |
| 1 | 创建目录骨架 | 1 min | Phase 0 |
| 2a | 归档不需要的文件（零风险） | 3 min | Phase 1 |
| 2b | 迁移工具代码 + 修改路径 | 5 min | Phase 1 |
| 2c | 迁移可视化代码 + 修改 import | 5 min | Phase 1 |
| 2d | 迁移文档 | 2 min | Phase 1 |
| 2e | 迁移实验脚本 + 改 source 路径 | 10 min | Phase 1 |
| 2f | 拆分 test_proposed_model.sh | 5 min | Phase 2e |
| 3 | 创建 run_remaining.sh + 各 README.md | 5 min | Phase 2 |
| 4 | 验证（Phase 四的命令） | 5 min | Phase 3 |
| 5 | 清理空目录 + 提交 | 2 min | Phase 4 |
| 6 | 启动剩余实验 `nohup bash experiments/run_remaining.sh &` | 1 min | Phase 5 |
| **总计** | | **~45 min** | |

---

## 七、风险总结

| 风险 | 等级 | 缓解 |
|---|---|---|
| scaling 脚本 source 路径断裂 | **中** | 只改 1 行，Phase 4 验证 |
| 画图脚本 import 断裂 | **低** | `viz` → `figures` 全局替换 + grep 验证 |
| run.py results 路径变化 | **零** | 硬编码 `./results/`，不动 |
| 实验被干扰 | **零** | 实验已暂停，重构完才启动 |
| 拆分 test_proposed_model.sh 丢参数 | **中** | 逐行对比原文件 |
