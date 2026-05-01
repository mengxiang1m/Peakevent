# PeakFocus 实验 & 代码整理 — TODO

**Last updated**: 2026-04-12 13:30

---

## ⚠️ 重要决策记录

1. **Loss weights 全部统一为 244** (λ₁=0.2, λ₂=0.4, λ₃=0.4)，两个数据集不再区分。
   - 原因：Table V 证明 PeakFocus 对权重不敏感 (F1 波动 ≤ 0.8pp)。
   - `_common.sh` 已改：ELC_V=0.2, ELC_P=0.4, ELC_T=0.4。

2. **主表 baseline 用 `peak_detect_ltf` 任务数据**（不是 `_basic`），和论文一致。
   - `classify_main` 已修（`build_figure_data.py`）。

3. **Loss config (b) 子图已改为 4 点配置编号**，不再按一维 λ_peak 画。
   - `classify_param_lambda` 改为 config index 1-4 + `lambda_triple` 标签。

4. **模型暂不改名**（proposed_model → PeakFocus 暂缓）。

---

## 🔄 当前实验进度（run_all_v2.sh, PID 54424）

**启动**: 2026-04-12 12:43 | **预计完成**: ~4 月 14 日凌晨

| 脚本 | 状态 | 开始时间 | 说明 |
|---|---|---|---|
| `realign_elc_peakfocus_244.sh` (A1) | ✅ 完成 | 12:43-13:22 | ELC PeakFocus 244 H=336/720, 10 runs |
| `realign_hard_mask_244.sh` (A2+A3) | 🔄 **进行中** | 13:22 | WLEL+ELC Hard mask, 10 runs |
| `realign_k_sweep_244.sh` (A4) | ⏳ | | WLEL K=0/1/2/3, 20 runs |
| `ablation_wo_upap_wlel.sh` (B1) | ⏳ | | WLEL w/o UPAP, 10 runs |
| `ablation_wo_upap_elc.sh` (B2) | ⏳ | | ELC w/o UPAP, 10 runs |
| `scaling_dmodel_wlel.sh` (C1) | ⏳ | | 含 d_model=512 H=720 补种 |
| `scaling_dmodel_elc.sh` (C2) | ⏳ | | ELC d_model sweep, 12 runs |
| `scaling_elayers_wlel.sh` (C3) | ⏳ | | 8 runs |
| `scaling_elayers_elc.sh` (C4) | ⏳ | | 8 runs |
| `scaling_dff_wlel.sh` (C5) | ⏳ | | 8 runs |
| `scaling_dff_elc.sh` (C6) | ⏳ | | 8 runs |

**总计 107 runs，预计 ~38h。**

---

## ✅ 已完成的工作

### 代码修改
- [x] `run.py` 加 `--start_seed` 参数（让 itr 循环从指定 seed 起跑）
- [x] `exp/exp_peak_detect_*.py` 加 `[PARAMS]`/`[MEMORY]`/`[INFER]` instrumentation
- [x] `build_figure_data.py` 修 `classify_main`（从 `peak_detect_ltf_basic` → `peak_detect_ltf`）
- [x] `build_figure_data.py` 修 `classify_param_lambda`（三元组配置编号化）
- [x] `build_figure_data.py` 修 `classify_ablation`（canonical w/o UPAP + proxy fallback）
- [x] `_common.sh` 改 ELC 权重为 244

### 脚本
- [x] `realign_elc_peakfocus_244.sh`、`realign_hard_mask_244.sh`、`realign_k_sweep_244.sh`
- [x] `ablation_wo_upap_wlel.sh`、`ablation_wo_upap_elc.sh`
- [x] `ablation_extra_seeds.sh`（用 --start_seed 重写，已不含 Informer reseed）
- [x] `run_all_v2.sh`（master runner, A→B→C 顺序）
- [x] 全部 `bash -n` 语法检查通过

### 数据 & 可视化
- [x] `figure_data.csv` 重建（主表 36 cells 齐全，CycleNet/WLEL/336 F1=0.706 匹配论文）
- [x] 4 张图全部重画（radar/parameter_panel/ablation/generality），300 DPI
- [x] `docs/experiments_data_manifest.xlsx`（11 sheets）
- [x] `baseline_params.csv` + `baseline_infer.csv`（9 models × 2ds × 2h）

### 审计 & 报告
- [x] `AUDIT_REPORT.md`（454 行，7 sections）
- [x] `COMPAT_REPORT.md`（102 行，checkpoint 加载兼容性）
- [x] `PAPER_EDITS.md`（论文修改建议，红色标注指引）
- [x] `DATA_GUIDE.md`（数据使用指南）

### 代码结构整理（2026-04-12 完成）
- [x] 根目录 11 个散落 .py/.bat/.xlsx/.txt → `archive/`
- [x] `gate_visualizations/` → `visualization_outputs/model_internals/`
- [x] 根目录 `visualize_*.py` (×3) → `visualization/`（输出路径已同步修改）
- [x] `test_results/` (137MB) → `archive/`
- [x] `predict/` → `archive/`
- [x] `dataset_describe/` + `peak_detect_example/` + `pic/` → `docs/`
- [x] `scripts/` 下 anomaly/classification/imputation/exogenous → `scripts/legacy_tslib/`
- [x] `results/pre/` + `checkpoints/pre/` + `visualization_outputs/pre/`（冗余 424 数据归档）
- [x] README.md 路径引用同步更新
- [x] 全局 grep 验证无断路径

---

## ⏳ 实验跑完后要做的事

### 立即执行（~15 分钟）

```bash
cd /mnt/d/ywz_experiment_papers/ywz_7_electric/ywz1-PeakFocus/Code

# 1. 重新解析所有 logs
python3 scaling_experiments/tools/parse_epoch_time.py --results_root results --out exp_results/parsed_logs.csv

# 2. 重建 figure_data
python3 scaling_experiments/tools/build_figure_data.py

# 3. 重画全部图
cd scaling_experiments
python3 -m viz.plot_radar --real
python3 -m viz.plot_parameter_panel --real
python3 -m viz.plot_ablation_combined --real
python3 -m viz.plot_generality_bar --real
```

### 之后需要人工做的

- [ ] 更新 `docs/experiments_data_manifest.xlsx`（重建或重跑 Codex 任务）
- [ ] 按 `PAPER_EDITS.md` 修改论文 tex 文件（红色标注 6 处修改点）：
  - 超参表 λ₁=0.4→0.2, λ₂=0.2→0.4
  - Loss sensitivity 表更新数据
  - w/o UPAP 消融定义 + 数据
  - Hard mask / K sweep 新 244 数据
  - ELC PeakFocus 主表新数据
- [ ] 论文数字 vs figure_data.csv 逐 cell 校对
- [ ] PDF 编译 + 12 页检查

### 代码层面后续（等实验跑完）

- [ ] 合并 `tools/` + `metric_tools/` → `analysis/`（需改 scaling_experiments 里的 import 路径）
- [ ] 可选：模型改名 `proposed_model` → `PeakFocus`（用户暂缓）
- [ ] 可选：给 `torch.load` 加 `map_location` 防 CPU/GPU 反序列化问题
- [ ] 可选：给 checkpoint 旁边保存 `args.json` 防未来 default drift

### 清理（论文录用后）

- [ ] 删除 `results/pre/` + `checkpoints/pre/` + `visualization_outputs/pre/`
- [ ] 删除 `archive/test_results/` (137MB)
- [ ] 删除 `archive/` 里所有旧产物
- [ ] 删除 `scripts/legacy_tslib/`（TSLib 原始脚本）

---

## 📊 论文实验全貌（跑完后）

### 主表 (Table I) — 9 模型 × 2 数据集 × 2 horizon = 36 cells
PeakFocus + CycleNet / Transformer / Informer / PatchTST / SegRNN / STID / TimeMixer / Seq2Peak

### 消融表 — 5 变体 × 2 数据集 × 2 horizon
PeakFocus / w/o MSM-PL / w/o LAD / w/o UPAP (canonical) / Hard mask

### 参数实验 (Fig.11)
| 子图 | sweep | 数据集 |
|---|---|---|
| (a) K sweep | K=0/1/2/3 | WLEL H=336, 5 seeds |
| (b) Loss config | 4 组三元组 | WLEL H=336, 3 seeds |
| (c) d_model | 64/128/256/512 | WLEL+ELC, H=336/720, 5 seeds |

e_layers (1/2/3) 和 d_ff (128/256/512) 作为补充，可选入图或附录。

### 泛化柱状图 — 7 backbone × Vanilla/+UPAP × 2 ds × 2 h
### 效率雷达 — 5 轴: F1↑ / MSE↓ / BCS↓ / Params↓ / Infer↓
### PatchTST fair vs unfair — 附录或 footnote

---

## 📁 代码目录结构（整理后）

```
Code/
├── run.py                          # 唯一主入口
├── models/                          # 模型定义（不动）
├── exp/                             # 实验类（不动）
├── data_provider/ layers/ utils/    # 框架核心（不动）
├── dataset/                         # 数据文件
├── scripts/
│   ├── peak_detect_based_on_LTF/    # PeakFocus 主实验脚本
│   └── legacy_tslib/                # TSLib 遗留脚本（不用）
├── scaling_experiments/             # 参数扩展实验（本文件所在）
│   ├── scripts/                     # 实验启动脚本
│   ├── tools/                       # 数据解析/聚合工具
│   ├── viz/                         # 可视化脚本
│   ├── figs/                        # 生成的图（PDF+PNG）
│   ├── logs/                        # 实验日志
│   ├── TODO.md                      # ← 你在看的这个
│   ├── DATA_GUIDE.md                # 数据使用指南
│   ├── AUDIT_REPORT.md              # 实验审计报告
│   ├── COMPAT_REPORT.md             # 代码兼容性报告
│   ├── PAPER_EDITS.md               # 论文修改建议
│   ├── figure_data.csv              # 所有图的聚合数据源
│   ├── parsed_logs.csv              # 每 seed 级别的长表
│   ├── baseline_params.csv          # baseline 参数量
│   └── baseline_infer.csv           # baseline 推理延迟
├── analysis/                        # 分析工具（待合并 tools/ + metric_tools/）
├── visualization/                   # 可视化脚本（gate/attn 分析）
├── visualization_outputs/
│   ├── {setting}/                   # 预测可视化（exp 自动生成）
│   ├── model_internals/             # 模型内部可视化（gate/attn）
│   └── pre/                         # 冗余归档
├── results/ + checkpoints/          # 实验结果（各有 pre/ 子目录）
├── docs/                            # 文档 + 数据描述 + demo
├── archive/                         # 旧产物备份（.bat/.xlsx/.txt/test_results）
└── test_results/                    # exp 自动创建的 test predictions
```
