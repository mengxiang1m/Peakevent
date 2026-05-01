# PeakFocus → PatchEvent(STSEP) 适配调研与方案

## 0. 目标与范围

本文档基于以下代码做了逐文件调研，并给出 PeakFocus 作为 PatchEvent baseline 的落地适配方案：

- PeakFocus:
  - `baselines/peakfocus/models/proposed_model.py`
  - `baselines/peakfocus/exp/exp_peak_detect_based_on_long_term_forecasting.py`
  - `baselines/peakfocus/data_provider/data_loader.py`
  - `baselines/peakfocus/experiments/01_main_table/`
- PatchEvent:
  - `baselines/posthoc_evaluate.py`
  - `patchevent/phase2/dataset.py`
  - `patchevent/data/event_pipeline/*`

---

## 1. PeakFocus 代码架构调研

### 1.1 模型架构（`proposed_model.py`）

#### 1.1.1 `Model.forward` 与 `peak_detect_ltf` 完整流程

`task_name='peak_detect_ltf'` 时调用链：

1. RevIN 标准化（仅对 `x_enc`）：
   - `means = x_enc.mean(dim=1)`
   - `stdev = sqrt(var + 1e-5)`
   - `x_enc = (x_enc - means) / stdev`
2. Backbone（`forward_backbone`）：
   - `enc_embedding`: `[B, seq_len, enc_in] -> [B, seq_len, d_model]`
   - Res-MLP stack：`[B, seq_len, d_model] -> [B, seq_len, d_model]`
   - Temporal projection（线性投影时间维）：
     - 转置 `[B, d_model, seq_len]`
     - `Linear(seq_len -> pred_len)`
     - 转回 `[B, pred_len, d_model]`
3. Peak Head（MSM-PL）：
   - 输入：`enc_out [B, pred_len, d_model]`
   - 输出：
     - `peak_out [B, pred_len, c_out]`（peak logits）
     - `peak_features [B, pred_len, d_model]`
4. Value Head（LAD）：
   - `dec_input = dec_embedding(x_dec, None)`，shape `[B, label_len+pred_len, d_model]`
   - `value_head(dec_input, cross=enc_out, peak=peak_features)`
   - 输出 `dec_out_val [B, label_len+pred_len, c_out]`
5. Value 分支反归一化（只对最后 `pred_len` 段）
6. `forward` 返回：
   - `value_out = dec_out_val[:, -pred_len:, :]`
   - `peak_out = peak_out[:, -pred_len:, :]`

#### 1.1.2 Peak Head (MSM-PL) 输入/输出与语义

模块：`Multi_Scale_Mixing_Peak_Locator`

- 输入：`x [B, pred_len, d_model]`
- 可选时间嵌入：`x_mark_dec`（内部只取前 `d_inp` 个时间特征，忽略附加 `is_peak` 列）
- 多尺度卷积融合：`n_scales in {0,1,2,3}`
- 输出：
  - `peak_out [B, pred_len, c_out]`：每个未来时刻是否为峰的分类 logits（训练时再 sigmoid）
  - `peak_features [B, pred_len, d_model]`：供 value head 使用的峰位置信息特征

语义上：`peak_out` 是点级 peak 分类；`peak_features` 是“峰值位置感知表示”。

#### 1.1.3 Value Head (LAD) 输入/输出与语义

模块：`Location_Aware_Decoder`

- Query 输入：`dec_input [B, label_len+pred_len, d_model]`
- Cross 输入：`enc_out [B, pred_len, d_model]`
- Peak 输入：`peak_features [B, pred_len, d_model]`
- 输出：`[B, label_len+pred_len, c_out]`，最终取后 `pred_len` 段作为预测值。

在 `Location_Aware_Decoder_Layer` 内部：

- 若 `use_peak_guidance=True`（`if_lad=1`）：
  - `attn_input = tanh(peak) * sigmoid(cross)`
- 否则：
  - `attn_input = cross`

#### 1.1.4 两个 Head 之间的信息流

核心是 `peak_features -> value_head`：

- Peak Head 输出的 `peak_features` 进入 LAD 作为 attention 的 key/value 输入（或与 `cross` 融合后输入）
- 因此 value 回归被 peak 定位特征引导，不是纯 forecast-only 解码

即：`Peak localization feature` 直接指导 `Value regression`。

#### 1.1.5 调研中发现的实现风险

- `forward_backbone` 里有逻辑：
  - `if len(enc_out) == 2: enc_out = enc_out[0]`
- 对 Tensor 来说 `len(enc_out)=batch_size`，当 batch size 恰好为 2 时会误判并破坏维度。
- 该问题会导致某些 batch（特别是末尾 batch）直接报错，建议在适配前修复为：
  - `if isinstance(enc_out, tuple) and len(enc_out) == 2: enc_out = enc_out[0]`

---

### 1.2 训练流程（`exp_peak_detect_based_on_long_term_forecasting.py`）

#### 1.2.1 训练 loss 如何计算与组合

`_compute_train_loss` 中：

- `value_loss = criterion(outputs, batch_y_targets)`
- 若有 `peak_outputs`：
  - `hard_peak_targets = batch_y_mark[:, -pred_len:, -1:]`（最后一列是 `is_peak`）
  - `loss_mask`：
    - `mask_type='hard'` -> 直接 hard mask
    - 否则若 `use_soft_labels=1` -> 高斯软标签 `create_soft_peak_labels`
  - `cls_loss = _compute_peak_loss(peak_outputs, loss_mask, peak_criterion)`
  - 可选 `tp_mse_loss`（若 `tp_mse_loss_weight > 0`）

总损失：

`loss = value_loss_weight * value_loss + peak_loss_weight * cls_loss + tp_mse_loss_weight * tp_mse_loss`

默认权重（`run.py`）：`0.2 / 0.4 / 0.4`。

#### 1.2.2 `peak_out` 的 GT label 格式

是点级 binary mask，而不是 peak index 列表：

- 标签来源：`batch_y_mark` 最后一列 `is_peak`
- shape：`[B, pred_len, 1]`

#### 1.2.3 测试时如何从 `peak_out` 提取 peak

在 `calculate_peak_classification_metrics`：

1. `peak_out logits -> sigmoid -> prob`
2. 用 `peak_threshold` 二值化
3. 用 `_condense_peak_indices` 把连续 1 块压成一个峰（取块内最高概率索引）
4. 用 `match_peaks_with_tolerance`（容差窗口）做 TP/FP/FN 匹配

所以是 `threshold + condense + tolerance-match`。

---

### 1.3 数据格式（`data_loader.py`）

#### 1.3.1 `Dataset_Custom_Mixed.__getitem__` 返回

默认返回：

- `seq_x`: `[seq_len, in_dim]`
- `seq_y`: `[label_len+pred_len, out_dim]`
- `seq_x_mark`: `[seq_len, time_feat_dim(+is_peak)]`
- `seq_y_mark`: `[label_len+pred_len, time_feat_dim(+is_peak)]`

可选返回：

- 外生特征模式：额外 `seq_dec_ext`
- cycle 模式：额外 `cycle_index`

#### 1.3.2 `*_mixed_with_peaks_*.csv` 文件格式

支持两种格式：

- 双时间戳：`date_60min,value_60min,date_max,value_max,...`
- 单时间戳：`date,value_60min,value_max,...`

当前实际样本：

- WLEL: `date_60min,value_60min,date_max,value_max,is_peak,is_peak_seq2peaks`
- ELC: `date_60min,value_60min,date_max,value_max,is_peak`

其中 `is_peak` 会被拼到 `seq_*_mark` 最后一列，作为 peak 监督来源。

#### 1.3.3 与 PatchEvent `series/events` 格式差异

PatchEvent 当前代码（`phase2/dataset.py`）实际使用的是：

- point-level CSV：`timestamp,value,is_peak,d_to_apex,is_peak_event,phase,split`
- event-level JSONL：`onset_idx,end_idx,duration,apex_idx,apex_intensity,...`

与 PeakFocus mixed CSV 的核心差异：

1. PeakFocus 是“单表混合列 + is_peak 点标签”；PatchEvent 是“点级序列表 + 事件 JSONL 双文件”。
2. PeakFocus 训练直接读点级 `is_peak`；PatchEvent 训练/评估大量依赖事件级结构（onset/duration/apex/intensity）。
3. PeakFocus 现有 loader 的 split 默认走日期区间/比例；PatchEvent 对比实验用固定 split 索引（`dataset_configs.json`）。

> 备注：用户描述中的 `series.npy` 在当前仓库实现中对应为 `*_event_series_v1.csv`。

---

### 1.4 评估流程（PeakFocus）

#### 1.4.1 `calculate_peak_classification_metrics` 逻辑

逐样本执行：

1. 从 `peak_preds` 得概率序列
2. threshold 得 binary 序列
3. `_condense_peak_indices` 压缩连续峰块
4. `match_peaks_with_tolerance` 生成 TP/FP/FN
5. 统计分类指标（P/R/F1）
6. 统计回归误差（TP/FN/FP 误差、all true peaks 误差）
7. 计算综合指标：BPE、PIM、整体 MAE/MSE/BCE

#### 1.4.2 `_condense_peak_indices` 作用

把连续 1 区间 `[start,end]` 压为一个预测峰，位置取该区间概率最大点，避免“一个峰块被算多个 FP”。

#### 1.4.3 `match_peaks_with_tolerance` 匹配逻辑

- 对每个真实峰 `t_idx`，在预测峰中找 `|pred - t_idx| <= tolerance` 的候选
- 取最近且未匹配的预测峰
- 一对一贪心匹配
- 未匹配预测为 FP，未匹配真实为 FN

#### 1.4.4 PeakFocus 最终报告指标

测试输出中主要报告：

- 分类：`F1 / Precision / Recall`，`Avg TP/FP/FN`
- 回归：`TP_MSE/MAE`, `FN_MSE/MAE`, `All_True_Peaks_MSE/MAE`
- 综合：`Balanced_Peak_Error (BPE)`, `Peak_Integrated_Metric (PIM)`
- 传统 forecast：`MAE/MSE/RMSE/MAPE/MSPE`

---

### 1.5 实验配置（`experiments/01_main_table/`）

#### 1.5.1 关键参数（主表脚本整体）

- `seq_len=168`
- `label_len=48`
- `pred_len in {336, 720}`（01_main_table 内未使用 96）
- `peak_tolerance=1`
- `peak_lookahead`: WLEL 常用 3，ELC 常用 5

#### 1.5.2 WLEL 与 ELC 配置差异

主要差异：

1. 数据文件不同
   - WLEL: `hf_load_data_*_mixed_with_peaks_lookahead_3.csv`
   - ELC: `electricity_mixed_with_peaks_lookahead_5.csv`
2. `peak_lookahead`
   - WLEL=3
   - ELC=5
3. loss 权重
   - 目前两者都对齐为 244（`V=0.2,P=0.4,T=0.4`）
   - ELC 脚本里常显式写出 3 个权重，WLEL 多数依赖默认值

---

## 2. PatchEvent baseline 评估流程调研

### 2.1 `posthoc_evaluate.py` 的 predict-then-detect pipeline

对每个 run：

1. 读取 `per_window_preds.npy`（每个窗口一条预测曲线）
2. `build_pred_events(pred_curve)`：
   - `find_peaks` 找 apex anchors
   - 生成 `is_peak`
   - 用 `detect_events_gradient_width`（默认）或 `detect_events_from_peak_labels` 提取 onset/duration
   - 形成预测事件 tuple：`{onset,duration,apex_index,apex_intensity}`
3. 从 GT `events.jsonl` 取窗口内事件并转相对索引
4. 用 Hungarian matching + tolerance 评估

### 2.2 现有 baseline 输出如何转 STSEP tuple

当前 baseline（TSLib/Seq2Peak）只输出 value curve，`posthoc_evaluate.py` 通过后处理从曲线提事件：

- apex：`find_peaks`
- onset/duration：`detect_events_gradient_width` 或 midpoint 方式
- intensity：`value[apex]`

### 2.3 Hungarian matching 实现细节

`match_events(pred_events, gt_events, tolerance=3)`：

- cost matrix：`|pred_apex - gt_apex|`
- `linear_sum_assignment` 最小化全局匹配
- 仅保留 `cost <= tolerance` 的配对为 TP
- 剩余预测是 FP，剩余 GT 是 FN

并在 TP 上计算：

- `apex_mae`
- `onset_mae`
- `duration_mae`
- `intensity_mape`

### 2.4 PatchEvent 数据格式（`phase2/dataset.py` + `event_pipeline`）

- point-level series（CSV）：`timestamp,value,...,split`
- events（JSONL）：`onset_idx,end_idx,duration,apex_idx,apex_intensity,...`
- 训练窗口以 `split` + 固定 `window_stride` 划定
- 目标可编码成 STSEP `quad`（`onset,duration,apex,intensity`）序列

---

## 3. PeakFocus 适配 PatchEvent STSEP 评估方案

## 3.1 数据适配

### 3.1.1 方案 A（快速落地）：PatchEvent 数据转 PeakFocus mixed CSV

从 `*_event_series_v1.csv` 转成 PeakFocus 所需列：

- `date_60min = timestamp`
- `value_60min = value`
- `date_max = timestamp`
- `value_max = value`
- `is_peak = is_peak`
- `is_peak_seq2peaks = is_peak`（可选）

优点：改动小，可直接复用 `Dataset_Custom_Mixed`。

风险：默认 split 逻辑是日期区间/比例，不保证与 PatchEvent 固定 split 完全一致。

### 3.1.2 方案 B（推荐）：PeakFocus 直接接 PatchEvent 数据源

新增 `Dataset_STSEP_Mixed`（建议复用 `baselines/tslib`/`seq2peak` 的 `Dataset_STSEP` 思路）：

- 输入：`--dataset_config --dataset_name --window_stride`
- split：严格使用 `dataset_configs.json` 的 index 边界
- 序列：从 `event_series_v1.csv` 读取 `timestamp,value,is_peak`
- 输出保持 PeakFocus 接口：`seq_x, seq_y, seq_x_mark, seq_y_mark`
- 额外保存 `starts_global`（后续写 `per_window_starts.npy`）

这是与 PatchEvent 对齐最干净的路径，推荐用于最终论文对比。

---

## 3.2 模型运行适配

### 3.2.1 PeakFocus 在 `pred_len=96` 是否可运行

结论：架构上可运行。

证据点：

- `run.py` 默认 `pred_len=96`
- `proposed_model` 的关键映射是 `Linear(seq_len -> pred_len)`，无 336/720 硬编码
- 实测前向（`seq_len=96,pred_len=96`）shape 正常

但需注意：

- 01_main_table 脚本未覆盖 96，需要新增脚本配置
- 建议先做单 seed smoke test

### 3.2.2 `pred_len=336` 是否优先

建议优先在 336 对比，理由：

1. PeakFocus 主表已有成熟配置（336）
2. PatchEvent baseline 也支持 336
3. 可先快速建立“可复现实验链路 + 公平评估闭环”

后续再扩展到 96/168（与 PatchEvent 主基线全对齐）。

### 3.2.3 建议的公平配置

若目标是与 PatchEvent baseline 严格可比，建议 PeakFocus 训练改为：

- `seq_len=96`
- `label_len=48`
- `pred_len in {96,168,336}`
- 同一 `window_stride=4`
- 同一 split（`dataset_configs.json`）

---

## 3.3 输出转换适配（双头输出 -> STSEP tuples）

目标：把 PeakFocus `(peak_out, value_out)` 转成 `(onset,duration,apex,intensity)` 序列。

### 3.3.1 Adapter V1（按你的要求：condense + block）

对每个窗口：

1. `peak_prob = sigmoid(peak_out[:, :, 0])`
2. `peak_binary = (peak_prob >= threshold)`
3. `apex_indices = _condense_peak_indices(peak_binary, peak_prob)`
4. 用 `peak_binary` 的连续 1 块定义事件块
5. 对每个块：
   - `onset = block_start`
   - `duration = block_len`
   - `apex = 该块对应的 condensed index`
   - `intensity = value_out[apex]`
6. 产出事件列表（按 onset 排序）

这就是：

`condense -> onset(block start) -> duration(block length) -> apex(condensed idx) -> intensity(value at apex)`

### 3.3.2 Adapter V2（可选，更贴近当前 PatchEvent 后处理）

- 先用 `peak_out` 得 `is_peak`
- 再调用 `detect_events_gradient_width(values=value_out, is_peak=...)`
- 让 onset/duration 的定义与现有 `posthoc_evaluate.py` 一致

建议：

- 主结果用 V1（忠实体现 PeakFocus 双头）
- 附录可报告 V2（与现有 baseline 后处理完全同口径）

---

## 3.4 评估对齐

### 3.4.1 在 PatchEvent STSEP 体系下评估 PeakFocus（主路径）

实现要点：

1. PeakFocus `test()` 额外保存：
   - `per_window_preds.npy`（value）
   - `per_window_peak_probs.npy`（peak prob，新增）
   - `per_window_starts.npy`（窗口起点）
2. 在 `posthoc_evaluate.py` 增加 PeakFocus 模式：
   - 读取 `per_window_peak_probs.npy + per_window_preds.npy`
   - 调 Adapter V1/V2 生成预测事件
   - 复用现有 `match_events`（Hungarian + tolerance=3）

### 3.4.2 可选：在 PeakFocus 体系下评估 PatchEvent

可做辅助分析，不作为主表口径：

- 将 PatchEvent 预测 tuple 映射成点级 apex 标签（one-hot/apex-only）
- 与 GT `is_peak` 用 `calculate_peak_classification_metrics` 比较
- 该方向受“tuple->dense label 反投影规则”影响较大，建议仅附录呈现

---

## 3.5 实验计划（建议执行顺序）

### 3.5.1 阶段 0：工程打通（0.5-1 天）

1. 加 `stsep_mixed` loader（或先做 CSV 转换快速版）
2. 增加 PeakFocus `per_window_*` 文件保存
3. 实现 Adapter V1 并接入 `posthoc_evaluate.py`
4. 单 run smoke test（WLEL, pred_len=336, seed=42）

### 3.5.2 阶段 1：主对比（1-2 天）

- 数据集：WLEL
- horizon：`pred_len=336`
- seeds：`42,123,456`
- 输出：STSEP 主指标（event F1 / onset MAE / apex MAE / duration MAE / intensity MAPE）

### 3.5.3 阶段 2：与 PatchEvent baseline 全对齐（2-4 天）

- 数据集：WLEL / ETT / ELC
- horizon：`96,168,336`
- seeds：`42,123,456`
- 与现有 baseline 汇总同表

### 3.5.4 资源预估（单卡）

粗略估计（20 epochs, batch=128, d_model=256）：

- 单 run：约 `0.8 ~ 2.0` GPU 小时（依数据集和 horizon 变化）
- 阶段 1（3 runs）：约 `3 ~ 6` GPU 小时
- 阶段 2（27 runs）：约 `25 ~ 45` GPU 小时
- 后处理评估 CPU 耗时通常远低于训练（分钟级到数十分钟）

---

## 4. 关键改动清单（实施时）

建议最少改动文件：

1. `baselines/peakfocus/data_provider/data_loader.py`
   - 新增 `Dataset_STSEP_Mixed`（固定 split + starts_global）
2. `baselines/peakfocus/data_provider/data_factory.py`
   - 注册 `stsep_mixed`
3. `baselines/peakfocus/exp/exp_peak_detect_based_on_long_term_forecasting.py`
   - `test()` 保存 `per_window_preds.npy / per_window_peak_probs.npy / per_window_starts.npy`
4. `baselines/posthoc_evaluate.py`
   - 新增 PeakFocus adapter 模式（读 peak probs 直转 STSEP tuple）
5. `baselines/peakfocus/models/proposed_model.py`
   - 修复 `len(enc_out)==2` 的 batch-size 依赖 bug

---

## 5. 结论

- PeakFocus 的双头结构天然可以输出 STSEP 所需信息：
  - `peak_out` 提供 apex 定位
  - `value_out` 提供 intensity
  - onset/duration 可由 block/规则提取
- 最关键的是评估口径与 split 对齐：
  - 必须对齐 `dataset_configs.json` 的窗口与边界
  - 必须在同一 Hungarian+tolerance=3 下比较
- 推荐执行路径：
  - 先跑 WLEL-336 打通，再扩展到 96/168 与跨域（ETT/ELC）。
