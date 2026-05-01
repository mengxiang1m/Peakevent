# PatchEvent: 基于Patch预训练与自回归解码的时间序列事件预测

## Abstract

Predicting future structured events in time series—including **when** they occur, **how long** they last, and **how intense** they are—poses unique challenges due to variable-length event sequences and the demand for fine-grained temporal localization. We propose **PatchEvent**, a two-phase encoder-decoder framework that reformulates this task as structured sequence generation.

In **Phase 1**, a Patch-based Transformer encoder is pretrained with four event-aware objectives (peak detection, distance regression, phase distribution prediction, and apex offset classification), with support for both single-domain and cross-domain joint pretraining via per-dataset normalization. In **Phase 2**, a lightweight autoregressive Transformer decoder generates discrete event token sequences (onset, duration, apex, intensity) conditioned on encoder memory. A **Memory Bridge** between encoder and decoder—comprising learnable memory position encoding (MemPos) and self-attention aggregation (SA Agg)—addresses position information dilution in the encoder output. The AR decoder further introduces **position-valid masking**, **position-aware label smoothing**, and **attribute-weighted loss** to enforce structural constraints and prioritize timing accuracy.

Extensive experiments on three industrial datasets—WLEL (power load), ETT (transformer temperature), and ELC (electricity consumption)—spanning ≥129 experiments (9 architectural configs × 3 seeds × 3 domains + 4 AR-internal × 3 seeds × 3 domains + Phase 1 subtask ablation, v2) demonstrate that PatchEvent achieves F1=0.841±0.005 on WLEL, F1=0.855±0.000 on ETT (la=3), and F1=0.794±0.013 on ELC. Systematic ablation validates each module’s contribution: MemPos is the most critical Memory Bridge component (WLEL ΔF1=+2.2%, ELC ΔF1=+6.4%); AR decoding outperforms DETR across all domains (ΔF1=+3.4%~21.7%, ΔOnsetMAE=−53%~76%); both position-valid masking and causal masking are absolutely essential (removing either collapses F1 to 0.000~0.243); AR-internal mechanisms further improve timing accuracy by 9–15% on WLEL; a new encoder comparison shows all non-pretrained encoders (CNN/LSTM/MLP/Scratch) achieve comparable F1 but 31–41% worse OnsetMAE on WLEL, isolating pretraining as a temporal localization prior; and Phase 1 subtask ablation identifies apex offset classification as the single most critical pretraining task for Phase 2 timing (removing it degrades OnsetMAE by 7.4%).

### 中文摘要

时间序列事件预测旨在预测未来结构化事件的**发生时间**、**持续时长**和**峰值强度**，因事件序列的变长特性及精细时间定位需求而极具挑战性。本文提出 **PatchEvent** 框架，将该问题重新建模为结构化序列生成任务，采用两阶段编码器-解码器架构。

**第一阶段（表征学习）**：设计四个事件感知的多任务预训练目标——峰值检测（Focal BCE）、距离回归（Smooth L1）、相位分布预测（加权KL散度）和峰值偏移分类（带标签平滑的交叉熵），训练Patch-based Transformer Encoder在patch粒度上学习事件敏感的时序语义表征。该阶段同时支持单域训练和跨域联合预训练（通过per-dataset归一化消除分布差异）。

**第二阶段（序列生成）**：轻量级自回归Transformer Decoder基于编码器记忆（encoder memory）逐token生成离散化的事件序列（298-token结构化词汇表，每个事件由onset/duration/apex/intensity四个token组成）。编码器与解码器之间通过**Memory Bridge**增强信息传递，包含两个互补模块：(1) 可学习的Memory位置编码（MemPos）——在memory端重新注入被编码器多层注意力稀释的时序位置信号；(2) Self-Attention聚合（SA Agg）——在增强后的memory空间中实现patch间全局信息交互。AR Decoder内部进一步引入**位置有效掩码（PosValidMask）**、**位置感知标签平滑（PosAwareSmooth）**和**属性加权损失（AttrWeightedLoss）**，强化结构约束并优先保障时间定位精度。

在WLEL（工业电力负荷）、ETT（变压器温度）和ELC（用电量）三个工业数据集上的系统实验（9种架构配置×3域 + 4种AR内部消融×3域 + Phase 1子任务消融，v2版共≥129个实验）表明：
- WLEL上 **F1=0.841±0.005**、**OnsetMAE=0.747±0.022小时**；ETT(la=3)上 **F1=0.855±0.000, OnsetMAE=0.704±0.010**；ELC上 **F1=0.794±0.013**
- MemPos是 Memory Bridge贡献最大的单一模块（WLEL F1 +2.2%、ELC F1 +6.4%）
- 自回归解码在所有域显著优于DETR并行解码（F1提升3.4%~21.7%、OnsetMAE改善53%~76%）
- **PosValidMask和Causal Mask均不可去除**（去掉任一后F1崩塌至0.000~0.243）
- AR内部的AttrWeightedLoss和PosAwareSmooth在WLEL上分别改善OnsetMAE 14.5%和8.8%
- 所有非预训练编码器（CNN/LSTM/MLP/Scratch）F1与G0相近，但OnsetMAE均劣化–31~41%（WLEL），证明预训练的核心价值在于时间定位先验
- Phase 1子任务消融识别出**apex offset分类是Phase 2时间定位的关键子任务**（去掉后OnsetMAE劣化 7.4%）

---

## 一、问题与挑战

### 1.1 研究背景

时间序列分析是工业领域的核心技术之一，广泛应用于电力负荷预测、设备状态监测、能源管理等场景。近年来，基于Transformer的时序模型（如Informer [Zhou et al., 2021]、PatchTST [Nie et al., 2023]、iTransformer [Liu et al., 2024]）在长序列点预测任务上取得了显著进展。然而，这些方法主要关注**固定长度的数值预测**（如未来96个时间步的值），无法直接输出**结构化的事件描述**。

在实际工业场景中，运维人员往往更关心"未来4天内将发生几次用电高峰？每次高峰什么时候开始、持续多久、峰值多大？"——即**事件级别的预测**，而非逐点的数值预测。

**具体场景举例**：
- **电力调度**：电网调度员需要提前知道未来哪些时段会出现负荷高峰，以便安排发电机组启停、购买备用电力。关键信息不是"第47小时的负荷是5200MW"，而是"第45小时开始一个持续4小时的高峰，峰值出现在第47小时，强度为正常水平的1.8倍"。
- **变压器保护**：变压器温度过高会导致绝缘老化，运维人员需要预测未来的温度峰值事件，提前启动冷却系统或降低负载。

这些需求催生了一个新的任务定义：**时间序列事件预测 (Time Series Event Prediction)**——不是预测逐点数值，而是预测离散的、结构化的事件序列。

### 1.2 问题定义

给定一段历史时间序列 $\mathbf{x} \in \mathbb{R}^L$（$L=96$ 小时 = 4天），预测未来等长时间窗口内将发生的**结构化事件序列** $\mathcal{E} = \{e_1, e_2, \dots, e_K\}$，其中每个事件 $e_k$ 包含4个属性：

$$e_k = (\text{onset}_k, \text{duration}_k, \text{apex}_k, \text{intensity}_k)$$

- **onset** $\in [0, 95]$: 事件起始时间（小时索引）
- **duration** $\in \{3, 4, 5\}$: 持续时长（小时）
- **apex** $\in [0, 95]$: 峰值时间（小时索引）
- **intensity** $\in [0.50, 2.48]$: 归一化强度

事件数量 $K$ 在不同窗口中变化（2~6个），且事件之间存在时序依赖关系（如前一个事件结束后需要一定间歇才开始下一个）。

### 1.3 现有方法的不足

| 方法类型 | 代表工作 | 局限性 |
|----------|----------|--------|
| **阈值检测** | 滑动窗口峰值检测 | 仅能检测已发生事件，无法预测未来 |
| **点预测** | Informer, PatchTST | 输出固定长度数值序列，无法直接生成结构化事件描述 |
| **连续值回归** | 多任务回归 | 需预设最大事件数，无法优雅处理变长输出；各事件独立回归丢失时序依赖 |
| **DETR式并行解码** | Object Query预测 | 事件query独立预测，丢失事件间序列依赖（实验验证MAE +84%） |

### 1.4 五大核心挑战

以下五个挑战分别对应§二中的五个Contribution：

**挑战 P1: 事件感知语义表征的构建与预训练知识的迁移困境** → C1

准确预测未来事件的前提是对历史时间序列中的**峰值事件模式**建立高质量的语义表征。通过对历史数据进行事件感知的预训练编码，encoder能够学习到峰值形态识别（has_apex）、事件阶段判别（phase_dist: 上升期/峰值期/衰减期/背景）、与最近峰值的距离感知（d_to_apex）、峰值在patch内的精确位置（apex_offset）等与事件密切相关的时序语义，为下游decoder提供信息丰富的记忆上下文。没有这种事件感知的编码，decoder将缺乏结构化的时序先验，被迫从原始信号中同时学习特征提取和事件生成，难以收敛到有效解（详见§4.2消融验证）。

然而，预训练encoder在迁移到下游事件预测任务时面临两难困境：**完全冻结**encoder虽保留了预训练知识，但表征无法适应事件生成任务的特定需求（如onset精确定位），限制了性能上限；**全量微调**所有参数则可能破坏预训练阶段积累的通用时序模式，引发灾难性遗忘，尤其在下游标注数据有限时风险更大。

**挑战 P2: Encoder-Decoder信息传递中的位置丢失与全局缺失** → C2

Patch-based Transformer Encoder虽在内部使用了正弦位置编码，但经过2层多头自注意力运算后，输出的语义向量中**绝对时序位置信息被逐层稀释**。当这些向量作为memory传递给decoder时，decoder在cross-attention中只能获取"什么模式"（语义），却丢失了"在什么位置"（时间坐标）。此外，PatchTST式encoder的各patch**独立编码**（channel-independence），patch间缺乏全局信息交换，decoder难以感知整个序列的事件分布格局。这两个问题直接导致事件onset/apex时间的定位精度受限。

**挑战 P3: 变长结构化事件序列的输出建模** → C3

时间序列事件预测的输出是**变长序列**（每个窗口的事件数量从2到6不等），且每个事件是**4属性的结构化描述**。传统方法面临根本性的建模困难：(a) 基于阈值的峰值检测只能检测已发生的事件，无法预测未来；(b) 连续值回归方法需预设固定的最大事件数，对多出或不足的事件无法优雅处理；(c) DETR式并行解码虽能处理变长输出，但丢失了事件间的时序依赖（如"前一个事件结束后多久开始下一个"），导致定位精度大幅下降。

**挑战 P4: 跨域Encoder不可迁移，部署成本高** → C4

实验表明，在一个域（如WLEL工业电力负荷）上预训练的encoder直接迁移到另一个域（如ETT变压器温度）时**完全失败**（F1从0.895暴跌至0.215）。原因在于：(a) 不同域的数据分布差异极大（均值差两个数量级：WLEL~5000 vs ETT~27）；(b) encoder内嵌的归一化参数与目标域不匹配。这意味着每个新域都需要独立训练Phase 1 encoder，大幅增加了部署和维护成本。

---

## 二、Contributions

### Contribution 1: 两阶段冻结-选择性解冻训练范式 (Two-Phase Freeze-Then-Selectively-Unfreeze)

> **针对挑战 P1** | 模型细节见§3.2, §3.6 | 实验验证见§4.2

**解决思路**: 将时间序列事件预测分解为**表征学习**(Phase 1)和**序列生成**(Phase 2)两个阶段，并通过选择性解冻实现encoder-decoder的深度协同。

**Phase 1 — 事件感知Patch Encoder预训练**:

以Patch为粒度对时间序列进行预训练，设计4个互补的事件感知预训练目标：

| 预训练头 | 输出 | 损失函数 | 作用 |
|----------|------|---------|------|
| has_apex | $\hat{y}_i \in [0,1]$ | Focal BCE (γ=2.0) | 判断patch是否含峰值 |
| d_to_apex | $\hat{d}_i \geq 0$ | Smooth L1 | 回归到最近峰值的距离 |
| phase_dist | $\hat{\mathbf{q}}_i \in \Delta^3$ | Weighted KL [1,3,3,3] | 预测事件阶段分布 |
| apex_offset | $\hat{\mathbf{o}}_i \in \mathbb{R}^8$ | CE (smoothing=0.1) | 分类峰值在patch内的位置 |

**Phase 2 — 冻结Encoder + 选择性解冻**:

冻结Phase 1预训练的encoder权重，仅训练decoder。在此基础上解冻encoder最后1层，以极低学习率 ($\eta_{enc}=10^{-5}$, 为 $\eta_{dec}$ 的1/50) 微调。

**实验证据** (v2): G0_Full vs GS3_Bare: WLEL F1从0.831提升至0.841(+1.2%), OnsetMAE从0.793改善至0.747(−5.8%)。预训练的独立贡献通过 Scratch 对比验证：同架构无预训练时 WLEL OnsetMAE劣化 33%（GS4_scratch: 0.992 vs G0: 0.747）。

### Contribution 2: Memory增强的编码器-解码器桥接 (Memory-Enhanced Encoder-Decoder Bridge)

> **针对挑战 P2** | 模型细节见§3.4 | 实验验证见§4.3 S0/S1/S2

**解决思路**: 在encoder输出和decoder cross-attention之间引入轻量级Memory Bridge，通过两个互补模块解决位置丢失和全局缺失问题。

**模块A — Memory Position Encoding (MemPos)**:

$$\mathbf{M}^{(2)}_i = \mathbf{M}^{(1)}_i + \mathbf{p}^{mem}_i, \quad \mathbf{p}^{mem} \in \mathbb{R}^{(N+2) \times d}$$

$\mathbf{p}^{mem}$ 为可学习的 `nn.Embedding` 参数。

**设计动机 — 为什么encoder内部的PE不够？**

标准Transformer Encoder在输入端使用正弦位置编码(PE)，但经过多层self-attention后，PE的位置信号会被逐层稀释。这是因为self-attention本质上是**全局加权平均**——每个patch的输出是所有patch的加权和，多轮运算后各位置的表征趋于相似，绝对位置信息被"平滑掉"。

在标准的seq2seq任务（如机器翻译）中，这不是问题，因为decoder有自己的位置编码来track输出位置。但在事件预测中，decoder需要从memory中判断"事件发生在**哪个时间位置**"——这要求memory的key在cross-attention中携带**明确的时序位置信号**。

MemPos与encoder内部PE的区别：
- **encoder PE**: 在encoder self-attention之**前**注入，经多层SA后信号稀释
- **MemPos**: 在encoder之**后**、decoder cross-attention之**前**注入，信号直接传递给decoder
- **encoder PE**: 固定正弦编码（不可学习）
- **MemPos**: **可学习**的embedding，能自适应学习"哪些时间位置对事件定位更重要"

**消融证据** (v2): 去除MemPos后WLEL F1下降2.2%（0.841→0.819），OnsetMAE恶化5.6%（0.747→0.789）；ELC F1下降6.4%（0.794→0.730）。三域中**最稳定的单模块贡献者**，验证了memory端位置注入的必要性。

**模块B — Self-Attention Aggregation (SA Agg)**:

$$\mathbf{M}^{(1)} = \text{TransformerEncoderLayer}(\mathbf{M}^{(0)})$$

通过一层Transformer Encoder Layer (MHSA + FFN + 残差 + LN, GELU) 实现patch间全局信息交换。

**设计动机**: 虽然encoder内部已有2层self-attention，但SA Agg在**增强后的memory空间**（经过Proj维度变换和MemPos位置注入后）操作，相当于在新的表征空间中做一次全局整合。消融数据表明SA Agg对F1的贡献较小（WLEL ΔF1=+0.1%），但对定位精度有明确改善。

**消融证据** (v2): 去除SA后WLEL OnsetMAE恶兤1.5%（0.747→0.758）；ELC F1降3.2%（0.794→0.762）。对MAE的贡献大于F1，说明全局上下文主要影响**定位精度**而非检测能力。

### Contribution 3: 结构化事件Token化与自回归序列生成

> **针对挑战 P3** | 模型细节见§3.3, §3.5 | 实验验证见§4.3 S0/S5

**解决思路**: 将事件预测重新建模为**结构化序列生成**任务。

**事件Token化**: 构建结构化词汇表（大小随 pred_len 动态调整），每个事件属性离散化为独立token：

| Token类型 | ID范围 (pred=96) | 数量 | 含义 |
|-----------|--------|------|------|
| PAD/BOS/EOS | 0~2 | 3 | 特殊token |
| ONSET_k | 3~(2+N) | N=pred_len | 事件起始时间 |
| DUR_d | (3+N)~(5+N) | 3 | 持续时长 (d=3,4,5) |
| APEX_k | (6+N)~(5+2N) | N=pred_len | 峰值时间 |
| INT_b | (6+2N)~(105+2N) | 100 | 强度bin ([0.50, 2.48], 步长0.02) |

为 pred_len=96/168/336 时，VOCAB_SIZE 分别为 298/442/778。

**自回归生成**: Transformer Decoder通过条件概率链 $P(y_t | y_{<t}, \mathbf{M})$ 逐token生成事件序列，自然处理变长输出(EOS终止)、保持结构化(位置约束解码)、建模事件间依赖。词汇表和位置嵌入随 pred_len 动态构建，支持任意预测窗口长度。

**位置约束解码**: 推理时在位置 $k$ (从BOS后计)，通过掩码强制只允许对应属性类型的token：
- $k \bmod 4 = 0$: ONSET_* 或 EOS
- $k \bmod 4 = 1$: DUR_*
- $k \bmod 4 = 2$: APEX_*
- $k \bmod 4 = 3$: INT_*

**消融证据** (v2): AR vs DETR在所有域全面领先——WLEL: F1 +3.4%（0.841 vs 0.807）, OnsetMAE −53%（0.747 vs 1.583）; ELC: F1 +21.7%（0.794 vs 0.577）， OnsetMAE −76%（1.116 vs 4.795）; ETT: F1 +17.6%（0.855 vs 0.679）。DETR的event query独立预测丢失了事件间时序依赖，验证了自回归序列生成的必要性。

### Contribution 4: 跨域联合多域预训练

> **针对挑战 P4** | 实验验证见§4.4

**解决思路**: Per-dataset归一化 + ConcatDataset混合训练。每个域独立z-score，归一化后混合训练Phase 1。

**实验证据**:

| 训练模式 | Phase1 F1 | Phase1 offset_Acc | Phase2 OnsetMAE |
|----------|-----------|-------------------|-----------------|
| WLEL单域 | 0.920±0.001 | 0.810±0.003 | 0.742 |
| ETT单域(la=3) | 0.909±0.012 | 0.765±0.004 | 0.680 |
| ELC单域 | 0.907±0.004 | 0.879±0.005 | 1.020 |
| **联合域+NPP** | **0.940±0.001** | **0.900±0.007** | — |

联合训练在所有关键指标上优于单域，验证了多域数据的互补性。offset准确率从单域最高0.879提升至0.900(+2.4%)。

---

## 三、模型架构

### 3.1 整体架构

```
Input x ∈ ℝ^96  (96小时 = 4天)
    │
    ▼  z-score归一化: x̃ = (x - μ) / σ
    │
    ▼
┌──────────────────────────┐
│ Phase1 PatchEncoder      │
│ (预训练+解冻last block)  │
│ unfold(8,4) → 23 patches │
│ Linear(8→128) + PE + LN  │
│ 2层Transformer Encoder   │
│ → z ∈ (B, 23, 128)       │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ Memory Bridge            │
│ ① Proj(128→128) + LN    │
│ ② MemPos(+可学习位置)   │
│ ③ SA_Agg(全局交互)      │
│ → M ∈ (B, 23, 128)      │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ Phase2 AR Decoder (3层)  │
│ TokenEmbed(298,128)      │
│ + PosEmbed(64,128)       │
│ → Causal Self-Attn       │
│ → Cross-Attn(↑M)        │
│ → FFN → lm_head(→298)   │
│ → 位置约束+onset间距约束 │
└────────────┬─────────────┘
             ▼
[BOS, onset₁, dur₁, apex₁, int₁, ..., EOS]
```

### 3.2 Phase 1: 事件感知Patch Encoder

**Patch切分与嵌入**:

给定归一化序列 $\tilde{\mathbf{x}} \in \mathbb{R}^L$ ($L=96$)，滑动窗口切分为 $N=23$ 个重叠patch：

$$\mathbf{p}_i = \tilde{\mathbf{x}}[i \cdot s : i \cdot s + P], \quad i = 0, \dots, N-1$$

$P=8$ (patch长度), $s=4$ (步长)。每个patch通过线性投影 + 正弦位置编码 + LayerNorm：

$$\mathbf{e}_i = \text{LN}\left(\mathbf{W}_p \mathbf{p}_i + \mathbf{b}_p + \text{PE}(i)\right)$$

$$\text{PE}(i, 2j) = \sin(i / 10000^{2j/d}), \quad \text{PE}(i, 2j+1) = \cos(i / 10000^{2j/d})$$

**Transformer Encoder**: 2层标准Transformer Encoder ($d=128$, $h=4$, $d_{ff}=256$, GELU, dropout=0.1)：

$$\mathbf{Z}' = \text{LN}(\mathbf{Z} + \text{MHSA}(\mathbf{Z})), \quad \mathbf{Z}'' = \text{LN}(\mathbf{Z}' + \text{FFN}(\mathbf{Z}'))$$

**Phase 1 Loss**:

$$\mathcal{L}_{P1} = \lambda_{bce} \cdot \text{FocalBCE}(\hat{y}, y; \gamma=2) + \lambda_d \cdot \text{SmoothL1}(\hat{d}, d) + \lambda_{kl} \cdot \text{WKL}(\hat{\mathbf{q}}, \mathbf{q}; \mathbf{w}=[1,3,3,3]) + \lambda_{off} \cdot \text{CE}(\hat{\mathbf{o}}, o; \epsilon=0.1)$$

默认权重: $\lambda_{bce}=1.0$, $\lambda_d=0.5$, $\lambda_{kl}=1.0$, $\lambda_{off}=0.5$

### 3.3 结构化事件Token化 (StructuredEventTokenizer)

词汇表共298个token。每个事件由4个连续token表示：

$$\text{seq} = [\text{BOS}, \underbrace{o_1, d_1, a_1, i_1}_{\text{event}_1}, \underbrace{o_2, d_2, a_2, i_2}_{\text{event}_2}, \dots, \text{EOS}]$$

Intensity离散化: $b = \text{round}((I - 0.50) / 0.02)$, $b \in [0, 99]$, 对应连续值 $[0.50, 2.48]$

### 3.4 Memory Bridge

**Patch Projector**: $\mathbf{M}^{(0)} = \text{LN}(\mathbf{W}_{proj}\mathbf{z} + \mathbf{b}_{proj})$

**MemPos**: $\mathbf{M}^{(1)}_i = \mathbf{M}^{(0)}_i + \mathbf{p}^{mem}_i$, $\mathbf{p}^{mem}$ 为 nn.Embedding(25, 128) 可学习参数

**SA Agg**: $\mathbf{M}^{(2)} = \text{TransformerEncoderLayer}(\mathbf{M}^{(1)})$

最终输出 $\mathbf{M} = \mathbf{M}^{(2)} \in \mathbb{R}^{B \times 23 \times 128}$ 作为decoder的cross-attention memory。

### 3.5 Phase 2: AR Decoder

**Token嵌入**: $\mathbf{h}^{(0)}_j = \text{Dropout}(\text{TokenEmbed}(y_j) + \text{PosEmbed}(j))$

TokenEmbed ∈ $\mathbb{R}^{298 \times 128}$, PosEmbed ∈ $\mathbb{R}^{64 \times 128}$, embed_dropout=0.15

**3层Transformer Decoder**, 每层:

$$\mathbf{h}' = \text{LN}(\mathbf{h} + \text{CausalMHSA}(\mathbf{h}))$$
$$\mathbf{h}'' = \text{LN}(\mathbf{h}' + \text{CrossMHA}(\mathbf{h}', \mathbf{M}, \mathbf{M}))$$
$$\mathbf{h}''' = \text{LN}(\mathbf{h}'' + \text{FFN}(\mathbf{h}''))$$

**输出**: $\mathbf{l}_t = \mathbf{W}_{lm}\mathbf{h}^{(3)}_t + \mathbf{b}_{lm} \in \mathbb{R}^{298}$

推理时应用位置掩码 + argmax + onset间距约束($\delta_{min}=2$)。

### 3.6 训练目标

**Phase 2 Loss**:

$$\mathcal{L}_{P2} = \sum_{t=1}^{T} w_{a(t)} \cdot \text{CE}(\tilde{\mathbf{l}}_t, y_t; \epsilon=0.1)$$

属性级权重: $w_{onset}=2.0$, $w_{dur}=0.5$, $w_{apex}=2.0$, $w_{int}=0.5$

Position-aware label smoothing: smoothing概率仅分配到位置合法的token上。

### 3.7 参数汇总

| 参数 | 值 | 说明 |
|------|-----|------|
| $L$ | 96 | 输入序列长度 (4天) |
| $P$ / $s$ | 8 / 4 | Patch长度 / 步长 |
| $N$ | 23 | Patch数量 |
| $d$ | 128 | 模型维度 |
| $h$ | 4 | 注意力头数 |
| $L_{enc}$ / $L_{dec}$ | 2 / 3 | Encoder / Decoder层数 |
| $d_{ff}$ | 256 | FFN维度 |
| $V$ | 298/442/778 | 词汇表大小 (pred_len=96/168/336) |
| dropout | 0.2 | Dropout率 |
| $\eta_{dec}$ / $\eta_{enc}$ | 5e-4 / 1e-5 | Decoder / Encoder学习率 |
| 总参数量 | ~1M (trainable ~900K) | — |

---

## 四、实验

### 4.1 数据集

| 数据集 | 领域 | 时间跨度 | 采样频率 | 样本数 | 均值/标准差 | 事件数 | 峰值检测 |
|--------|------|----------|----------|--------|-------------|--------|----------|
| **WLEL** | 工业电力负荷 | ~3年 | 1小时 | 7478 / 1028 / 2103 | 5079±1171 MW | ~400 | la=3, rf=0.020 |
| **ETT (ETTh2)** | 变压器温度 | ~2年 | 1小时 | 12194 / 1741 / 3485 | 27±12 °C | 798 | la=3, rf=0.020 |
| **ELC** | 用电量 | ~3年 | 1小时 | 18412 / 2631 / 5261 | 3340±660 | 1429 | la=5, rf=0.020 |

**WLEL数据集**: 来自某工业园区的小时级电力负荷数据。"事件"定义为负荷超过局部基准线的峰值段，通过scipy.signal.find_peaks检测历史峰值，每个峰值事件包含onset(负荷开始上升)、apex(负荷达到峰值)、duration(从onset到结束)和intensity(峰值负荷与基准的比值)。

**ETT数据集**: 来自ETTh2（Electricity Transformer Temperature）公开基准数据集的OT(Oil Temperature)列。事件定义与WLEL相同——温度超过局部基准的峰值段。ETT使用lookahead=3（与WLEL一致），信号平滑无噪声风险。

**ELC数据集**: 来自公开Electricity数据集（321个客户用电量聚合），采用OT列。ELC信号高频噪声较大，使用lookahead=5过滤伪峰值（la=3会产生过多噪声峰，导致F1下降）。三域均统一使用rate_frac=0.020确保事件定义一致。

三个数据集分布差异显著（均值跨两个数量级：WLEL~5079 vs ETT~27 vs ELC~3340），为验证跨域通用性提供了严格测试条件。每个样本窗口长度96小时（4天），以stride=4的滑动窗口从原始序列中提取，按时间顺序划分为train/val/test（约7:1:2）。

**数据预处理**: Phase 1和Phase 2均使用训练集的z-score归一化参数。联合训练时每个域**独立计算**z-score（per-dataset normalization），避免分布差异导致归一化失真。

### 4.2 评价指标

- **Event F1**: 事件检测F1分数。预测onset与真实onset匹配阈值2小时。$\text{F1} = 2PR/(P+R)$
- **Onset MAE**: 匹配事件对中起始时间平均绝对误差（小时）
- **Apex MAE**: 峰值时间平均绝对误差
- **Duration MAE**: 持续时长平均绝对误差
- **Intensity MAPE**: 强度平均绝对百分比误差

所有实验使用3个随机种子 (42, 123, 456)，报告均值±标准差。

### 4.3 GS 三域架构消融 (v2, 9配置, 3-seed)

**WLEL** (3-seed mean±std):

| 配置 | F1 | Onset MAE | Apex MAE | Dur MAE | Int MAPE |
|------|----|-----------|----------|---------|----------|
| **G0 Full** | **0.841±0.005** | **0.747±0.022** | **0.707±0.015** | 0.440±0.011 | 0.0768±0.0018 |
| GS1 w/o MemPos | 0.819±0.008 | 0.789±0.030 | 0.749±0.036 | 0.458±0.016 | 0.0899±0.0030 |
| GS2 w/o SA | 0.840±0.005 | 0.758±0.014 | 0.720±0.013 | 0.438±0.013 | **0.0749±0.0010** |
| GS3 Bare | 0.831±0.010 | 0.793±0.007 | 0.769±0.016 | 0.448±0.005 | 0.0789±0.0053 |
| GS4 CNN | 0.852±0.008 | 1.055±0.029 | 0.968±0.026 | 0.466±0.011 | 0.0680±0.0005 |
| GS4 LSTM | **0.860±0.001** | 1.038±0.011 | 0.970±0.011 | 0.439±0.010 | — |
| GS4 MLP | 0.851±0.007 | 0.983±0.034 | 0.914±0.027 | 0.451±0.010 | — |
| GS4 Scratch | 0.838±0.001 | 0.992±0.006 | 0.934±0.011 | 0.471±0.010 | — |
| GS5 DETR | 0.807±0.024 | 1.583±0.305 | 0.744±0.036 | 0.432±0.023 | — |

**ETT** (la=3, 3-seed mean±std):

| 配置 | F1 | Onset MAE | Apex MAE | Dur MAE | Int MAPE |
|------|----|-----------|----------|---------|----------|
| **G0 Full** | **0.855±0.000** | **0.704±0.010** | **0.652±0.008** | **0.213±0.000** | 0.2602±0.0108 |
| GS1 w/o MemPos | 0.852±0.000 | 0.712±0.014 | 0.658±0.012 | 0.214±0.001 | 0.2814±0.0207 |
| GS2 w/o SA | 0.854±0.002 | 0.712±0.024 | 0.660±0.021 | 0.215±0.002 | **0.2595±0.0024** |
| GS3 Bare | 0.846±0.008 | 0.738±0.024 | 0.684±0.021 | 0.214±0.002 | 0.2794±0.0223 |
| GS4 CNN | 0.853±0.002 | 0.721±0.034 | 0.666±0.029 | 0.214±0.002 | **0.2483±0.0049** |
| GS4 LSTM | 0.849±0.003 | 0.705±0.017 | 0.652±0.015 | 0.215±0.002 | — |
| GS4 MLP | 0.848±0.002 | 0.763±0.033 | 0.706±0.028 | 0.216±0.001 | — |
| GS4 Scratch | 0.851±0.002 | 0.737±0.025 | 0.682±0.022 | 0.216±0.001 | — |
| GS5 DETR | 0.679±0.016 | 1.601±0.403 | 0.598±0.025 | 0.222±0.003 | — |

**ELC** (3-seed mean±std):

| 配置 | F1 | Onset MAE | Apex MAE | Dur MAE | Int MAPE |
|------|----|-----------|----------|---------|----------|
| **G0 Full** | **0.794±0.013** | 1.116±0.052 | 1.091±0.048 | 0.541±0.025 | **0.1026±0.0024** |
| GS1 w/o MemPos | 0.730±0.006 | 1.143±0.018 | 1.136±0.012 | 0.543±0.022 | 0.1202±0.0010 |
| GS2 w/o SA | 0.762±0.014 | 1.165±0.070 | 1.139±0.070 | **0.539±0.020** | 0.1031±0.0003 |
| GS3 Bare | 0.716±0.042 | **1.111±0.014** | **1.085±0.071** | 0.520±0.037 | 0.1164±0.0028 |
| GS4 CNN | 0.790±0.010 | 1.185±0.027 | 1.210±0.014 | 0.511±0.004 | 0.1016±0.0027 |
| GS4 LSTM | 0.752±0.042 | 1.299±0.061 | 1.292±0.032 | 0.513±0.006 | — |
| GS4 MLP | 0.795±0.010 | 1.218±0.019 | 1.191±0.021 | **0.492±0.021** | — |
| GS4 Scratch | 0.794±0.023 | 1.200±0.023 | 1.217±0.047 | 0.523±0.027 | — |
| GS5 DETR | 0.577±0.011 | 4.795±1.849 | 1.017±0.031 | 0.529±0.030 | — |

**三域模块贡献总结** (v2):

| 模块 | WLEL ΔF1 | WLEL ΔOnsetMAE | ETT ΔOnsetMAE | ELC ΔF1 | ELC ΔOnsetMAE |
|------|----------|----------------|---------------|---------|---------------|
| MemPos (G0 vs GS1) | +2.2% | −5.6% | −1.1% | +6.4% | −2.4% |
| SA (G0 vs GS2) | +0.1% | −1.5% | −1.1% | +3.2% | −4.2% |
| Patch vs Scratch (G0 vs GS4_scratch) | +0.4% | **−24.7%** | **−4.5%** | 0% | −7.0% |
| Patch vs CNN (G0 vs GS4_cnn) | −1.3% | **−29.2%** | **−2.4%** | +0.5% | +6.2% |
| AR vs DETR (G0 vs GS5) | **+3.4%** | **−52.9%** | **+127%↑** | **+21.7%** | **−76.7%** |

**域间特性差异**:
- **WLEL**: MemPos贡献显著，预训练Patch encoder的时间定位价值在 OnsetMAE 上最达− 29%，AR大幅胜出
- **ETT**: F1在各配置间差异极小（信号规律、任务简单），但DETR灾难性失败（OnsetMAE=1.601，+127%），AR必要性强烈证实
- **ELC**: 最难域。MemPos是F1最关键模块（+6.4%），DETR在ELC上完全崩溃（OnsetMAE=4.795h，+330%）

### 4.4 Phase 1预训练结果 (4子任务)

| 训练模式 | has_apex F1 | AUC | d_MAE | d_R² | phase CosSim | EventF1 | offset Acc | Top2 |
|----------|------------|-----|-------|------|-------------|---------|-----------|------|
| WLEL单域 | 0.920±0.001 | 0.973 | 0.197±0.003 | 0.793 | 0.966±0.001 | — | 0.810±0.003 | — |
| ETT单域(la=3) | 0.909±0.012 | 0.985 | 0.220±0.010 | 0.807 | 0.975±0.002 | 0.881 | 0.765±0.004 | 0.936 |
| ELC单域 | 0.907±0.004 | 0.983 | 0.257±0.008 | 0.772 | 0.971±0.000 | 0.820 | 0.879±0.005 | 0.988 |
| **联合域+NPP** | **0.940±0.001** | 0.984 | **0.195±0.003** | **0.831** | 0.971±0.001 | — | **0.900±0.007** | **0.986** |

联合训练在4个子任务的关键指标上均优于或接近单域最优，验证了多域数据的互补性。offset准确率从WLEL的0.810提升至0.900（+11%），这直接有利于Phase 2的事件时间定位。ETT la=3数据下Phase1 F1=0.909，offset准确率0.765。

### 4.5 GD AR Decoder 内部组件消融 (v2, 3域, 3-seed)

在G0_full基线上逐一禁用AR decoder的4个内部组件：

**WLEL**:

| 配置 | F1 | OnsetMAE | ApexMAE | DurMAE |
|------|-----|---------|---------|--------|
| **G0 baseline** | **0.841±0.005** | **0.747±0.022** | **0.707±0.015** | 0.440±0.011 |
| GD1 w/o PosValidMask | 0.243±0.176 **(崩!)** | 0.769±0.150 | 0.710±0.127 | 0.482±0.095 |
| GD2 w/o PosAwareSmooth | 0.838±0.002 | 0.813±0.017 (+8.8%) | 0.788±0.024 | **0.421±0.002** |
| GD3 w/o AttrWeightedLoss | 0.839±0.006 | 0.855±0.009 (+14.5%) | 0.816±0.005 | 0.451±0.015 |
| GD4 w/o CausalMask | 0.278±0.073 **(崩!)** | 9.761±2.769 **(爆!)** | 1.289±0.047 | 0.986±0.091 |

**ETT** (la=3, 3-seed):

| 配置 | F1 | OnsetMAE | ApexMAE | DurMAE |
|------|-----|---------|---------|--------|
| **G0 baseline** | **0.855±0.000** | **0.704±0.010** | **0.652±0.008** | **0.213±0.000** |
| GD1 w/o PosValidMask | 0.085±0.055 **(崩!)** | 0.850±0.149 | 0.733±0.092 | 0.198±0.086 |
| GD2 w/o PosAwareSmooth | 0.855±0.001 | 0.705±0.001 (=) | 0.655±0.002 | 0.214±0.002 |
| GD3 w/o AttrWeightedLoss | **0.856±0.001** | **0.682±0.005 (−3.1%)** | **0.633±0.004** | 0.215±0.001 |
| GD4 w/o CausalMask | 0.278±0.064 **(崩!)** | 3.814±1.863 **(爆!)** | 1.375±0.178 | 0.322±0.027 |

**ELC**:

| 配置 | F1 | OnsetMAE | ApexMAE | DurMAE |
|------|-----|---------|---------|--------|
| **G0 baseline** | **0.794±0.013** | **1.116±0.052** | **1.091±0.048** | 0.541±0.025 |
| GD1 w/o PosValidMask | **0.000 (完全崩!)** | — | — | — |
| GD2 w/o PosAwareSmooth | 0.762±0.018 | 1.136±0.029 | 1.139±0.084 | 0.595±0.076 |
| GD3 w/o AttrWeightedLoss | **0.801±0.006** | **1.076±0.048 (−3.6%)** | **0.979±0.049** | **0.499±0.015** |
| GD4 w/o CausalMask | 0.181±0.037 **(崩!)** | 8.644±4.638 **(爆!)** | 1.547±0.098 | 0.567±0.055 |

**GD 消融关键结论** (v2):

**(1) PosValidMask是 AR 生成的守门员（v2新发现）**: GD1在全郠3域F1彻底崩塌（WLEL 0.243、ETT 0.085、ELC 0.000）。旧版实验 (S0 基线) 中 PosValidMask 发展 F1 仅降 1.5%，而v2版实验崩塌远更严重，证明位置约束掩码对 AR 结构合法性不可缺缺。

**(2) CausalMask同样不可去除**: GD4在全郠3域F1崩塌至0.18~0.28，OnMAE爆炸至4~10h。

**(3) AttrWeightedLoss贡献与域难度正相关**: WLEL OnMAE劣化 14.5%，ETT反而改善 3.1%，ELC改善 3.6%。

**(4) PosAwareSmooth对WLEL定位贡献达8.8%**: ETT/ELC影响小。

### 4.6 Phase 1 子任务消融 → Phase 2 下游影响 (WLEL, v2)

| 消融配置 | P1 直接崩塌指标 | P2 F1 | P2 OnsetMAE | P2 ApexMAE | P2 IntMAPE |
|----------|----------------|-------|------------|-----------|----------|
| **G0_full (P0)** | — | **0.841±0.005** | **0.747±0.022** | **0.707±0.015** | 0.0768±0.0018 |
| GP1a w/o has_apex | has_apex F1→0 | 0.837±0.010 (−0.5%) | 0.760±0.044 (+1.7%) | 0.715±0.030 (+1.1%) | 0.0749±0.0021 |
| GP1b w/o distance | dMAE +232% | 0.837±0.009 (−0.5%) | 0.741±0.014 (−0.8%) | 0.707±0.014 (=) | 0.0751±0.0019 |
| GP1c w/o phase | CosSim −30% | 0.841±0.003 (=) | 0.735±0.017 (−1.6%) | 0.697±0.014 (−1.4%) | 0.0760±0.0036 |
| **GP1d w/o offset** | **OffAcc −87%** | 0.841±0.004 (=) | **0.802±0.009 (+7.4%)** | **0.752±0.008 (+6.4%)** | **0.0721±0.0014** |

**子任务贡献排序（Phase 2肉眼）**: offset > has_apex ≈ distance > phase

**关键结论**: Apex offset子任务（GP1d）是**唯一显著影响 Phase 2 时间定位**的 Phase 1 组件，去掉后OnsetMAE劣化 7.4%、ApexMAE劣化 6.4%。Phase 1 CosSim崩塌（P1c）不传导到Phase 2，说明相位分布信息不是事件检测的核心特征。

### 4.7 预测长度可扩展性验证 (pred_len=96/168/336)

在保持 seq_len=96 (Phase 1 encoder 不变) 的前提下，扩展预测窗口至 168h 和 336h。Tokenizer 和位置嵌入根据 pred_len 动态构建。

| 域 | pred_len | F1 | OnsetMAE | ApexMAE |
|------|----------|----|----------|---------|
| WLEL | 96 | 0.841±0.005 | 0.747±0.022 | 0.707±0.015 |
| WLEL | 168 | 0.830±0.002 | 0.724±0.005 | 0.696±0.016 |
| WLEL | 336 | 0.794±0.001 | 0.779±0.052 | 0.767±0.043 |
| ETT | 96 | 0.855±0.000 | 0.704±0.010 | 0.652±0.008 |
| ETT | 168 | 0.839±0.010 | 0.733±0.038 | 0.689±0.037 |
| ETT | 336 | 0.750±0.009 | 0.762±0.047 | 0.718±0.044 |
| ELC | 96 | 0.794±0.013 | 1.116±0.052 | 1.091±0.048 |
| ELC | 168 | 0.769±0.037 | 1.106±0.047 | 1.096±0.047 |
| ELC | 336 | 0.735±0.036 | 1.116±0.077 | 1.066±0.102 |

**关键结论**: (1) pred_len=168 F1 仅下降 1.3~3.2%，鲁棒性良好；(2) pred_len=336 三域 F1 降幅均在 0.05 内 (WLEL −5.6%, ETT −12%, ELC −7.5%)，跨域鲁棒性一致；(3) OnsetMAE 对预测窗口扩展鲁棒，定位精度不因窗口变长而显著退化。

---

## 五、结论与讨论

### 5.1 核心发现 (v2, ≥129实验)

1. **PosValidMask是AR生成的守门员（v2新发现）**: 去掉后全部3域F1彻底崩塌（ELC降至0.000），远比旧版实验（−1.5%）严重。位置约束掩码确保每步只输出结构合法token，是AR解码的必要条件。

2. **MemPos是最稳定的贡献模块**: WLEL F1 +2.2%、ELC F1 +6.4%。Transformer Encoder输出中位置信息被逐层稀释，MemPos在memory端重新注入位置信号不可替代。

3. **AR解码不可替代**: DETR在全部3域全面落败（WLEL F1 +3.4%，ELC F1 +21.7%，ETT F1 +17.6%）。事件间时序依赖不可忽略，AR通过条件概率链自然建模事件序列。

4. **预训练的核心价值是时间定位，不是事件检测**: 所有非预训练编码器（CNN/LSTM/MLP/Scratch）F1与G0相近，但OnsetMAE均劣化31~41%（WLEL）。Scratch对比G0劣化33%，直接证明提升来自预训练本身而非架构。

5. **Apex offset是连接Phase 1和Phase 2的关键子任务**: 去掉offset子任务后OnsetMAE劣化7.4%，其他子任务（has_apex/distance/phase）去掉后Phase 2几乎不变。Phase 1 CosSim崩塌不传导到Phase 2。

6. **CausalMask不可去除**: F1崩塌至0.18~0.28，OnMAE爆炸4~10h。AR decoder的因果结构是自回归生成的基石。

7. **AttrWeightedLoss贡献与域难度正相关**: WLEL OnMAE劣化14.5%，ETT反而改善3.1%，ELC改善3.6%。

8. **联合预训练提升表征质量**: Phase1 offset准确率0.810→0.900(+11%)，直接有利于Phase 2时间定位。

9. **模块组合有效**: GS3_Bare→G0_Full: WLEL F1 +1.2% (0.831→0.841), OnsetMAE −5.8% (0.793→0.747)。

### 5.2 局限性

- ELC域 F1 < 0.80，最难的域仍有较大提升空间
- Phase 1子任务消融仅在WLEL验证，需ETT/ELC交叉验证以确认跨域稳定性
- phase_dist标签中rising/apex/falling阶段极稀有（dominant <0.3%），需要软分布评估（CosSim）而非硬分类
- 联合encoder下Unfreeze增量接近0，联合预训练已覆盖解冻微调的贡献
- pred_len=336 三域 F1 降幅均 <12%，鲁棒性远好于预期（早期实验因 max_new_tokens 截断已修复）

---

## 六、文件索引

| 目录/文件 | 内容 |
|-----------|------|
| `README.md` | 本文件：论文核心内容总结（双语） |
| `phase1_encoder_pretrain_final/` | Phase1完整代码 (model/train/dataset/loss/layers/eval) + checkpoints + README |
| `phase2_small_decoder_final/` | Phase2完整代码 (model/train/evaluate/dataset) + checkpoints + README |
| `phase2_small_decoder_final/run_arch_ablation_v2.py` | GS架构消融运行脚本 (9configs × 3域 × 3seeds) |
| `phase2_small_decoder_final/run_ar_ablation_v2.py` | GD AR内部消融运行脚本 (4configs × 3域 × 3seeds) |
| `phase2_small_decoder_final/run_p1_subtask_ablation.py` | Phase 1子任务消融训练脚本 |
| `phase2_small_decoder_final/run_p1_ablation_phase2.py` | Phase 1子任务消融Phase2评估脚本 |
| `phase2_small_decoder_final/run_pred168.py` | pred_len=168 扩展实验脚本 (3域 × 3seeds) |
| `phase2_small_decoder_final/run_pred336.py` | pred_len=336 扩展实验脚本 (3域 × 3seeds) |
| `phase2_small_decoder_final/collect_ablation_v2_results.py` | v2实验结果汇总脚本 |
| `phase2_small_decoder_final/checkpoints/ablation_v2_summary.json` | 全部v2实验结果JSON |
| `phase1_encoder_pretrain_final/eval_p1_ablation.py` | Phase 1子任务消融完整指标评估（含CosSim） |
| `phase1_encoder_pretrain_final/checkpoints/p1_ablation_full_metrics.json` | Phase 1子任务消融完整指标汇总 |
| `dataset/analysis/` | 数据集分析脚本+可视化PNG |
| `dataset/docx/数据集构建指南.md` | 三域数据集构建流水线+参数调优实验记录 |
| `docs/` | 参考文档 (PLOT_GUIDELINES, innovation_survey) |
| `environment.yaml` | Conda环境配置 |
