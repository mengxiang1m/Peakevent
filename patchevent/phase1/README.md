# Phase 1: 事件感知 Patch Encoder 预训练

## 1. Motivation

准确预测未来事件的前提是对历史时间序列中的**峰值事件模式**建立高质量的语义表征。通用时序 Transformer（如 PatchTST）的预训练目标（masked reconstruction）关注的是数值重建，而非事件结构。Phase 1 通过四个互补的事件感知预训练目标，让 Patch Encoder 在 patch 粒度上学习：

- **峰值存在性**（has_apex）：判断 patch 是否含峰值，建立事件检测的先验
- **距离感知**（d_to_apex）：回归到最近峰值的距离，为 decoder 提供连续距离场
- **事件阶段判别**（phase_dist）：预测 background/rising/apex/falling 四阶段的软分布，编码事件的生命周期结构
- **峰值偏移定位**（apex_offset）：分类峰值在 patch 内的精确位置（0~7），为精细时间定位提供监督

没有这种事件感知的编码，Phase 2 的 decoder 将缺乏结构化的时序先验，被迫从原始信号中同时学习特征提取和事件生成。消融实验证明（v2，3域×3seed）：同一 PatchEncoder 架构，**随机初始化（无预训练）在 WLEL 的 OnsetMAE 为 0.992，而冻结预训练版本为 0.747（差距 33%）**；在 ELC 差距达 7.5%。F1 差异较小（0.838 vs 0.841），但定位精度的差距说明预训练学到了事件位置的结构性先验，而非仅提升了事件检测能力。

**多域联合预训练**：单域 encoder 直接迁移到其他域会完全失败（分布差异太大）。联合预训练通过 per-dataset 归一化消除分布差异，利用多域数据的互补性提升表征质量。实验表明联合训练使 Phase1 offset 准确率从 0.810 提升至 0.900（+11.1%），直接有利于 Phase 2 时间定位精度。

## 2. 模型架构

```
x̃ ∈ (B, 96)  z-score 归一化后的时间序列
  → unfold(patch_len=8, stride=4) → patches (B, 23, 8)
  → Linear(8→128, bias=True) + SinusoidalPE(non-learnable) + LayerNorm
  → 2层 Transformer Encoder (d=128, h=4, d_ff=256, GELU, dropout=0.1)
    每层: MHSA + Residual + LN + Conv1d-FFN + Residual + LN
  → z ∈ (B, 23, 128)
  → 4个预测头:
      has_apex    (B, 23)      sigmoid → [0,1]     二分类
      d_to_apex   (B, 23)      ReLU → ≥0           距离回归
      phase_dist  (B, 23, 4)   softmax → Δ³        阶段分布
      apex_offset (B, 23, 8)   raw logits           8分类 (masked in loss)
```

- **Encoder 来源**: 沿用 iTransformer / PatchTST 的 Encoder + EncoderLayer 结构（`layers/Transformer_EncDec.py`），FFN 使用 Conv1d(kernel=1) 替代 Linear
- **参数量**: ~268K（全部可训练）

## 3. 预训练损失函数

```
L = λ_bce · FocalBCE(has_apex)        # γ=2.0, 缓解正负样本不均衡(~3% apex)
  + λ_d   · SmoothL1(d_to_apex)       # 对异常值更鲁棒
  + λ_kl  · WeightedKL(phase_dist)    # per-class权重, 强调稀有阶段
  + λ_off · MaskedCE(apex_offset)     # 仅对has_apex=1的patch计算, label_smoothing=0.1
```

| 损失项 | 损失函数 | 权重 λ | 设计动机 |
|--------|----------|--------|----------|
| has_apex | Focal BCE (γ=2.0) | 1.0 | 背景 patch 占 97%，Focal Loss 降低易分类样本权重 |
| d_to_apex | Smooth L1 | 0.5 | 比 MSE 对离群距离值更鲁棒 |
| phase_dist | Weighted KL | 1.0 | per-class 权重 [1.0, 2.0, 2.0, 2.0]，强调 rising/apex/falling |
| apex_offset | Masked CE (smoothing=0.1) | 0.5 | 仅在含峰值 patch 上计算，避免背景噪声干扰 |

## 4. 训练流程

### 4.1 数据加载

- **单域**: `build_dataloaders()` — 从一个域的 series CSV + patch labels CSV 构建
- **多域**: `build_multi_domain_dataloaders()` — 每个域独立 z-score 归一化后通过 `ConcatDataset` 混合
- **窗口**: seq_len=96h, patch_len=8h, stride=4h → 23 patches/window
- **数据划分**: 按时间顺序 train:val:test = 7:1:2

### 4.2 优化器与调度

- **优化器**: Adam (lr=1e-3, weight_decay=1e-4)
- **学习率调度**: Cosine Annealing (epoch-level)
- **梯度裁剪**: max_norm=1.0
- **Early Stopping**: patience=5 (基于 val_loss)

### 4.3 训练循环

1. 每 epoch 遍历 train_loader，计算 4 项 loss，反向传播
2. 在 val_loader 上评估 loss 和 per-task 指标
3. 保存 val_loss 最优的 checkpoint（含 model_state, args, mean, std, 可选 domain_norms）
4. 训练结束后加载最优 checkpoint，在 test_loader 上做完整评估
5. 多域训练时额外对每个域的 test 集分别评估

### 4.4 评估指标

| 任务 | 指标 | 说明 |
|------|------|------|
| has_apex | F1, Precision, Recall, AUC, Accuracy | 二分类检测 |
| d_to_apex | MAE, RMSE, R² | 距离回归质量 |
| phase_dist | CosSim, KL, EventF1, per-class MAE | 软分布预测质量 |
| apex_offset | Accuracy, Top2 Accuracy, MAE | 分类定位精度 |

## 5. 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| seq_len | 96 | 输入窗口长度 (96小时=4天) |
| patch_len | 8 | Patch长度 |
| stride | 4 | Patch步长 |
| d_model | 128 | 模型维度 |
| n_heads | 4 | 注意力头数 |
| e_layers | 2 | Encoder层数 |
| d_ff | 256 | FFN维度 |
| dropout | 0.1 | Dropout率 |
| learning_rate | 1e-3 | 学习率 |
| weight_decay | 1e-4 | 权重衰减 |
| train_epochs | 50 | 最大训练轮数 |
| patience | 5 | Early stopping 耐心值 |
| lradj | cosine | 学习率调度 |
| log1p_d | 1 | 对 d_to_apex 标签做 log(1+x) 变换 |
| lambda_bce / d / kl / off | 1.0 / 0.5 / 1.0 / 0.5 | 损失权重 |
| kl_apex_weight | 2.0 | KL 中 apex 阶段权重 |
| kl_falling_weight | 2.0 | KL 中 falling 阶段权重 |
| lambda_npp | 0.5 | NPP 辅助目标权重（0=禁用） |

## 6. 使用方式

### 单域训练 (批量, 推荐)

```bash
bash phase1_encoder_pretrain_final/train_single_domain.sh
```

输出: `checkpoints/{wlel,ett,elc}/s{42,123,456}/best_model.pth`

### 单域训练 (手动)

```bash
python phase1_encoder_pretrain_final/train.py \
  --train_mode single --single_domain wlel \
  --seed 42 \
  --output_dir phase1_encoder_pretrain_final/checkpoints/wlel/s42 \
  --train_epochs 50 --batch_size 64 --learning_rate 1e-3
```

### 多域联合训练 + NPP (批量, 推荐)

```bash
bash phase1_encoder_pretrain_final/train_multi_domain.sh
```

输出: `checkpoints/multi_npp05/s{42,123,456}/best_model.pth`（注：联合训练 checkpoint 在当前 repo 中未保留）

### 多域联合训练 (手动)

```bash
python phase1_encoder_pretrain_final/train.py \
  --train_mode multi --multi_domains wlel ett electricity \
  --lambda_npp 0.5 --seed 42 \
  --output_dir phase1_encoder_pretrain_final/checkpoints/multi_npp05/s42 \
  --train_epochs 50 --batch_size 64 --learning_rate 1e-3
```

### 域预设路径

| 域名 | series_path | labels_path |
|------|-------------|-------------|
| wlel | `dataset/wlel/event_v1/data/wlel_event_series_v1.csv` | `dataset/wlel/event_v1/data/wlel_patch_labels_v1.csv` |
| ett | `dataset/ett/event_v1/data/ett_event_series_v1.csv` | `dataset/ett/event_v1/data/ett_patch_labels_v1.csv` |
| electricity | `dataset/electricity/event_v1/data/elc_event_series_v1.csv` | `dataset/electricity/event_v1/data/elc_patch_labels_v1.csv` |

## 7. 消融实验结果

### 7.1 预训练策略消融 (2×2: 单域/多域 × 有/无NPP)

| 配置 | 训练域 | NPP | F1 | OffAcc | dMAE | dR² | CosSim | Top2 |
|------|--------|-----|-----|--------|------|-----|--------|------|
| P1-C | WLEL单域 | 无 | 0.920±0.001 | 0.810±0.003 | 0.197±0.003 | — | 0.966±0.001 | — |
| P1-D | WLEL单域 | λ=0.5 | 0.921±0.011 | 0.816±0.027 | 0.195±0.009 | — | 0.964±0.003 | — |
| ETT单域(la=3) | ETT | 无 | 0.909±0.012 | 0.765±0.004 | 0.220±0.010 | 0.807 | 0.975±0.002 | 0.936±0.005 |
| P1-A | 3域联合 | 无 | 0.939 | 0.917 | 0.213 | 0.821 | 0.973 | 0.987 |
| **P1-B** | **3域联合** | **λ=0.5** | **0.940±0.001** | **0.900±0.007** | **0.195±0.003** | **0.831** | **0.971±0.001** | **0.986±0.001** |

注: P1-C/D 为 3-seed 平均±std (seeds=42,123,456)；P1-A 为旧版 1-seed 结果；P1-B 为 3-seed 平均。

**关键结论**:

1. **多域联合 >> 单域**: OffAcc 0.810→0.900 (+11%)，F1 0.920→0.940 (+2.2%)。多域数据的互补性大幅提升了 apex offset 定位精度。
2. **NPP 改善 dMAE**: 单域下 dMAE 0.197→0.195 (-1.0%)，多域下 dMAE 0.213→0.195 (-8.5%)。NPP 让表征编码前瞻信息，对距离回归有稳定正向贡献。
3. **NPP 对 OffAcc 的影响因域而异**: 单域下 OffAcc +0.7%，多域下 OffAcc -1.7%。推测多域联合已提供了足够的表征多样性，NPP 的边际贡献缩小。
4. **推荐配置**: 3域联合 + NPP(λ=0.5)，综合 dMAE 最低且 F1 最高。

### 7.2 自监督目标消融 (MPR vs NPP)

| 方法 | 训练域 | F1 | OffAcc | dMAE |
|------|--------|-----|--------|------|
| Base (4 heads) | WLEL | 0.920±0.001 | 0.810±0.003 | 0.197±0.003 |
| +MPR (mask=0.4, λ=0.5) | WLEL | 0.794±0.047 | 0.451±0.030 | 0.472±0.035 |
| +MPR (mask=0.15, λ=0.1) | WLEL | 0.775±0.027 | 0.530±0.049 | 0.395±0.021 |
| **+NPP (λ=0.5)** | **WLEL** | **0.921±0.011** | **0.816±0.027** | **0.195±0.009** |

**MPR 失败原因**: Masked Patch Reconstruction 遮挡输入 patch 后，被 mask 的位置无法提供正确的事件标签（has_apex 等），严重干扰事件监督头的学习。

**NPP 成功原因**: Next-Patch Prediction 不遮挡输入，4 个事件监督头正常工作。NPP 只是额外添加辅助 loss（预测下一个 patch 的波形），让表征编码前瞻信息，与事件监督互补而非冲突。

### 7.3 Phase 1 子任务消融 (WLEL, 3-seed)

逐一将某个预训练子任务的 loss 权重设为 0（其余保持不变），训练消融版 encoder，评估 Phase 1 全指标及 Phase 2 下游效果。

#### Phase 1 层面：各子任务对自身及其他指标的影响

| 消融配置 | 去掉的 Loss | has_apex F1 | d_to_apex MAE | CosSim | Phase KL | OffAcc | OffTop2 |
|----------|------------|------------|--------------|--------|---------|--------|--------|
| **P0 baseline** | — | **0.920±0.001** | **0.197±0.003** | **0.966±0.001** | **0.096±0.002** | **0.810±0.003** | **0.953±0.002** |
| P1a w/o has_apex | λ_bce=0 | **0.000 (崩)** | 0.221±0.008 (+12%) | 0.962±0.000 (-0.4%) | 0.109±0.004 (+14%) | 0.794±0.014 (-2.0%) | 0.943±0.002 (-1.0%) |
| P1b w/o distance | λ_d=0 | 0.916±0.003 (-0.4%) | **0.655±0.000 (崩, +232%)** | 0.964±0.003 (-0.2%) | 0.101±0.006 (+5%) | 0.807±0.033 (-0.4%) | 0.950±0.010 (-0.3%) |
| P1c w/o phase | λ_kl=0 | 0.915±0.002 (-0.5%) | 0.205±0.003 (+4.1%) | **0.680±0.000 (崩, -30%)** | **0.730±0.000 (崩, +663%)** | 0.808±0.023 (-0.3%) | 0.950±0.008 (-0.3%) |
| P1d w/o offset | λ_off=0 | 0.913±0.004 (-0.8%) | 0.199±0.008 (+1.0%) | 0.964±0.001 (-0.2%) | 0.105±0.001 (+9%) | **0.104±0.010 (崩, -87%)** | **0.202±0.011 (崩, -79%)** |

**Phase 1 层面关键结论**:
- 每个子任务的 loss 去掉后，**对应直接指标精确崩塌**，对其他指标影响很小——说明 4 个子任务高度正交、相对独立
- P1a 去掉后 has_apex F1=0（预期），但 OffAcc 仍有 0.794（-2%），说明 offset 任务不依赖 has_apex
- P1b 去掉后 dMAE 从 0.197 崩到 0.655（+232%），其余指标几乎不变
- P1c 去掉后 CosSim 从 0.966 崩到 0.680（-30%），KL 散度从 0.096 飙到 0.730（+663%）
- P1d 去掉后 OffAcc 从 0.810 崩到 0.104（-87%），Top2Acc 从 0.953 崩到 0.202

#### Phase 2 下游影响 (WLEL, 使用消融版 encoder 作为 G0_full 输入)

| 消融配置 | P1 直接崩塌指标 | P2 F1 | P2 OnsetMAE | P2 ApexMAE | P2 IntMAPE |
|----------|----------------|-------|------------|-----------|----------|
| **G0_full (P0 baseline)** | — | **0.841±0.006** | **0.747±0.021** | **0.707±0.021** | 0.077±0.001 |
| GP1a w/o has_apex | has_apex F1→0 | 0.837±0.012 (-0.5%) | 0.760±0.054 (+1.7%) | 0.715±0.037 (+1.1%) | 0.075±0.003 |
| GP1b w/o distance | dMAE +232% | 0.837±0.011 (-0.5%) | 0.741±0.018 (-0.8%) | 0.707±0.018 (=) | 0.075±0.002 |
| GP1c w/o phase | CosSim −30% | 0.841±0.004 (=) | 0.735±0.021 (-1.6%) | 0.697±0.017 (-1.4%) | 0.076±0.004 |
| **GP1d w/o offset** | **OffAcc −87%** | 0.841±0.004 (=) | **0.802±0.010 (+7.4%)** | **0.752±0.010 (+6.4%)** | **0.072±0.002** |

**Phase 2 下游关键结论**:

1. **offset 是唯一影响 Phase 2 时间定位的关键子任务**: GP1d 的 OnsetMAE 劣化 7.4%、ApexMAE 劣化 6.4%，与 Phase 1 OffAcc 崩塌（-87%）直接对应。Apex offset 分类教编码器学习精确的 patch 内时间位置，这种位置先验直接被 Phase 2 decoder 的 cross-attention 利用。

2. **Phase 1 CosSim 崩塌（P1c）不传导到 Phase 2**: 去掉 phase KL 后 CosSim 从 0.966 崩到 0.680，但 Phase 2 F1 和 OnMAE 几乎不受影响（甚至略有改善）。说明**相位分布信息不是事件检测的核心特征**，decoder 的事件预测主要依赖 offset 编码的位置信号和 has_apex 编码的存在性信号。

3. **has_apex 和 distance 对 Phase 2 有轻微但一致的贡献**: 去掉后 F1 各降 0.5%，方向一致。它们提供的存在性和远近感知有助于 decoder 的检测能力，但对定位精度影响较小。

4. **子任务贡献排序（Phase 2 视角）**: offset > has_apex ≈ distance > phase

#### 实验脚本

```bash
# Phase 1 子任务消融训练
python phase2_small_decoder_final/run_p1_subtask_ablation.py

# Phase 2 下游评估
python phase2_small_decoder_final/run_p1_ablation_phase2.py

# 完整 Phase 1 指标评估（含 CosSim）
python phase1_encoder_pretrain_final/eval_p1_ablation.py
```

输出检查点: `phase1_encoder_pretrain_final/checkpoints/wlel_ablation/{P1a,P1b,P1c,P1d}_wo_*/s{42,123,456}/`

## 8. 输出

- `best_model.pth`: 最优 checkpoint，包含 `model_state`, `args`, `mean`, `std`, (多域时) `domain_norms`
- `training_history.json`: 训练日志（train/val loss、F1、per-task metrics）

## 9. 文件结构

```
phase1_encoder_pretrain_final/
├── model.py                  # PatchEncoder
├── loss.py                   # Phase1Loss: Focal BCE + Smooth L1 + Weighted KL + Masked CE
├── dataset.py                # WlelPatchDataset + 单域/多域 DataLoader 构建
├── train.py                  # 训练脚本: 域预设 + 评估 + early stopping
├── train_single_domain.sh    # 单域训练脚本 (3域 × 3seeds)
├── train_multi_domain.sh     # 多域联合训练脚本 (3域+NPP × 3seeds)
├── eval_p1_ablation.py       # 子任务消融完整指标评估（含 CosSim）
├── layers/                   # Transformer Encoder 底层模块
│   ├── Transformer_EncDec.py # Encoder, EncoderLayer, Decoder, DecoderLayer
│   └── SelfAttention_Family.py # FullAttention, AttentionLayer 等
├── utils/masking.py          # 注意力掩码
├── checkpoints/
│   ├── wlel/s{42,123,456}/              # WLEL 单域 encoder
│   ├── ett/s{42,123,456}/               # ETT 单域 encoder
│   ├── elc/s{42,123,456}/               # ELC 单域 encoder
│   ├── wlel_ablation/                   # Phase 1 子任务消融 encoder
│   │   ├── P1a_wo_has_apex/s{42,123,456}/
│   │   ├── P1b_wo_distance/s{42,123,456}/
│   │   ├── P1c_wo_phase/s{42,123,456}/
│   │   └── P1d_wo_offset/s{42,123,456}/
│   └── p1_ablation_full_metrics.json    # 子任务消融完整指标汇总
│   注: 联合训练 (multi_npp05/) checkpoint 在当前 repo 中未保留
└── README.md                 # 本文件
```
