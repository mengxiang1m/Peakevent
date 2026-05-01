# 周报：PeakFocus (ICDE 2026) 实验与论文修订

> **周期**：2026-04-09 ~ 2026-04-16
> **投稿**：ICDE 2027 Research Track（第一轮，截止 2026-06-11）
> **论文**：PeakFocus: An End-to-End Framework Bridging Localization and Regression for Electricity Load Peak Forecasting
> **投稿文件**：`Paper/ICDE26/69c8b883b8ca8153613be0fc/ywz1-PeakFocus-ICDE26.tex`（12 页 PDF）

---

## 一、本周工作量概览

| 类别 | 工作量 |
|---|---|
| 实验训练 | ~700 runs（5 seeds × 2 数据集 × 2 horizons × ~35 配置），总训练时间约 50+ GPU 小时 |
| 代码文件修改 | ~100+ 个文件 |
| 论文 tex 修改 | 实验分析全面重写 + 表格数据更新 + 超参数表修正 |
| 图片重绘 | 6 张论文图全部重做（300 DPI, Comic Sans, 统一 burgundy/navy/teal 配色） |
| 代码整理 | 项目重构 + 10 个目录中文 README + 数据管线自动化 |

---

## 二、实验工作详情

### 2.1 发现并修正的实验问题（重跑 ~70 runs）

**（1）ELC 数据集 λ 权重不一致**
- 问题：部分 ELC 实验使用旧的 (0.4, 0.4, 0.2) 权重，与 WLEL 的 (0.2, 0.4, 0.4) 不统一
- 修复：ELC PeakFocus 主表 + Hard mask 用 (0.2, 0.4, 0.4) 重跑
- 结果变化：ELC PeakFocus H=336 MSE 从 0.602 → **0.443**（改善 26%），F1 从 0.741 → **0.747**

**（2）ELC Seq2Peak peak_lookahead 参数错误**
- 问题：ELC 应使用 `peak_lookahead=5`，但 Seq2Peak 脚本错误设为 3
- 修复：修正脚本 + 重跑 10 runs
- 结果变化：F1 从 0.412 → **0.485**（H=336），0.437 → **0.516**（H=720），提升 18%

**（3）PatchTST 过参数化**
- 问题：PatchTST 使用 d_model=512（10.4M 参数），是 PeakFocus（1.1M）的 9 倍，对比不公平
- 修复：统一为 d_model=256, d_ff=512（4.1M 参数），重跑 40 runs，删除旧版所有残留
- 结果变化：参数量 10.4M → **4.1M**；ELC 上 F1 反而从 0.709 → **0.737**（说明 d_model=512 在 ELC 上过拟合）

### 2.2 新增实验（~200 runs）

| 实验 | 目的 | 数据集 | 配置数 | runs |
|---|---|---|---|---|
| w/o UPAP canonical 消融 | 消融表新增行 | WLEL + ELC | 2H × 2ds | 20 |
| d_model sweep | 参数面板新增列 | WLEL + ELC | 4 values × 2H × 2ds | 60 |
| e_layers sweep | 参数面板新增列 | WLEL + ELC | 3 values × 2H × 2ds | 40 |
| d_ff sweep | 参数面板新增列 | WLEL + ELC | 3 values × 2H × 2ds | 40 |
| ELC K sweep | 参数面板补全 | ELC | K=0/1/2/3 × 5seeds | 20 |
| ELC Loss config | 参数面板补全 | ELC | 4 configs × 3seeds | 12 |
| Hard mask H=720 | 消融表补全 | WLEL + ELC | 2ds × 5seeds | 10 |
| SegRNN/TimeMixer Vanilla | 泛化图补缺 | WLEL/ELC | 2ds × 5seeds × 2H | 20 |

### 2.3 实验结论变化

**Table I 主对比表**：
- PeakFocus 在 ELC H=336 上 BCS 从非最优变为**最优**（0.372，超过 TimeMixer 的 0.374）
- PatchTST 参数量减半后 ELC 性能反而提升，说明轻量化配置更适合噪声数据
- Seq2Peak 修正后 ELC 性能大幅提升，但仍远弱于端到端方法

**参数敏感性结论**：
- K sweep：双数据集验证 K=2 最优（之前只有 WLEL），ELC 上趋势一致
- d_model=512 在 ELC 上 F1 下降 6%，论文新增"over-parameterization hurts on noisier data"的讨论
- e_layers 和 d_ff 在大范围变化下几乎无影响，支持"单层 + 中等宽度设计已足够"的结论

---

## 三、论文修订详情

### 3.1 图表变化（与上一版对比）

| 元素 | 旧版 | 新版 | 变化说明 |
|---|---|---|---|
| Figure 总数 | 7 | **8** | 新增 Fig.8 LAD 注意力机制可视化（gate_combined） |
| Table 总数 | 4 | **2** | K sweep 表和 Loss weight 表**合并进参数面板图**（节省空间） |
| Algorithm 总数 | 2 | 2 | 不变 |
| Fig.5 泛化柱状图 | 仅 WLEL 部分模型有数据 | **WLEL+ELC 全部 7 backbone 完整** | 补齐 SegRNN/TimeMixer Vanilla |
| Fig.6 参数面板 | 2×3 格（3 列：K/λ/d_model） | **2×5 格**（5 列：+e_layers+d_ff） | 新增两列；ELC K/λ 也补齐 |
| Fig.8 注意力可视化 | **不存在** | 4 行子图（预测曲线+注意力聚合+热力图+峰值概率） | 全新增加 |
| Fig.9 定性对比 CD | 样本 7056（几乎无差异） | **样本 7145**（PatchTST 峰值 MSE 是 PeakFocus 的 6 倍） | 选了差异更明显的样本 |

### 3.2 实验分析文字修改逻辑

**修改原则**：
- 每个消融 observation 回扣 Introduction 中提出的三个 motivation（paradigm disconnect / multi-scale conflict / intensity smoothing）
- 减少数字堆砌（A=0.756, B=0.708 式罗列），改用相对比较（"outperforming by ~7%"）
- 保留 "As shown in Figure X" 保持上下文连贯

**Observation 结构调整**：
- 旧版 7 个 Observation → 新版 **6 个**
- Obs 1（w/o MSM-PL 消融）+ Obs 5（K sweep）→ 合并为 **"Multi-Scale Cascade Is Essential"**（消融图 + 参数图两个视角讲同一个故事）
- 新增 **"Robustness to Architectural Hyperparameters"**（4 个结构参数的鲁棒性，区别于 K 的有效性验证）

**参数敏感性分析角度拆分**：
- K sweep：**核心创新点有效性验证**（不是鲁棒性），回扣"多尺度级联"设计
- loss weights / d_model / e_layers / d_ff：**结构鲁棒性验证**（证明不需要精细调参）

### 3.3 超参数表修正

| 问题 | 修正 |
|---|---|
| τ = 0.5（论文）vs 0.4（代码） | → **0.4** |
| LR Scheduler 描述不准确 | → **Warm-up 3 + ExpDecay 0.9^{e-3}** |
| 左栏 15 行 vs 右栏 14 行 | → **14 = 14** 对齐 |

### 3.4 其他修订
- 删除 Impact Statement（ICDE Research Track 不要求）
- 消除 References 前大空白
- 所有图片统一 300 DPI + Comic Sans + burgundy/navy/teal 配色
- 清除部分审稿颜色标记（仍有 21 处 `\textcolor{red}` 待最终清理）

---

## 四、ICDE 2027 投稿要求对照

> 官网：https://icde2027.github.io/
> 地点：丹麦哥本哈根，2027 年 5 月 17-21 日
> **第一轮截止：2026-06-11（距今约 2 个月）**
> 审稿流程：两轮，每轮 Accept/Reject（无 R&R），有 Author Rebuttal（08-08 ~ 08-15）

### 4.1 格式合规性

| ICDE 2027 要求 | 当前状态 | 合规？ |
|---|---|---|
| IEEE conference 模板 | IEEEtran.cls | ✅ |
| 正文 ≤12 页（不含 References 和 AI 声明） | 当前 12 页（含 References） | ⚠️ 需确认正文是否超页 |
| 不允许附录 | 无附录 | ✅ |
| 单盲审稿 | 作者信息已填 | ✅ |
| AI-Generated Content Acknowledgement | **缺失** | ❌ 必须在 Acknowledgements 中声明使用了哪些 AI 工具（不计入页数） |
| Impact Statement | ICDE 2027 不要求 | ✅ 已删除 |
| 补充材料（代码/数据） | Anonymous GitHub repo | ✅ |
| 每位作者最多 5 篇（两轮合计） | 1 篇 | ✅ |
| ORCID | 投稿时所有作者需提供 | ⚠️ 待确认 |
| Out-of-scope 说明 | 投稿表单可填 1000 字符相关性说明 | 建议准备 |

### 4.2 ICDE 2027 主题匹配分析

ICDE 2027 的 Topics of Interest（共 30 个）中，与本论文直接相关的：

| ICDE 2027 Topic | 论文对应内容 | 匹配度 |
|---|---|---|
| **15. Time-series, Temporal and Spatial Data Management** | 时间序列峰值预测、WLEL 大规模时序数据、时态评估协议 | ⭐⭐⭐ 直接匹配 |
| **7. Data Streams, Complex Event Processing** | 峰值事件 = 时间序列流中的极端事件检测与处理 | ⭐⭐⭐ 直接匹配 |
| **17. Data Mining with Data-centric Focus** | 端到端 UPAP 管线、以数据质量为导向的 soft mask 设计 | ⭐⭐ 匹配 |
| **2. Data Systems for AI** | 面向 AI 模型的峰值标注管线、容忍度评估系统 | ⭐⭐ 匹配 |
| **22. Data Quality, Curation, Provenance, Workflows** | 软掩码处理标注噪声、tolerance-aware 数据匹配协议 | ⭐⭐ 匹配 |
| **29. Domain-specific Data Engineering (Science, Medical)** | 电力负荷 = 典型的领域特定数据工程 | ⭐⭐ 匹配 |

**注意**：ICDE 2027 新增了 "Data Mining with **Data-centric** Focus"（比 2026 版更强调数据中心），以及 "Domain-specific Data Engineering" 和 "Energy-efficient Data Systems"——后两个对我们有利。

### 4.3 针对上次拒稿意见的回应

上次组里收到的拒稿意见：**"论文偏 ML conference，缺乏 Data Engineering 深度"**（原文："it lacks the depth of Data Engineering insights—such as data management, system architecture for large-scale pipelines, or storage optimizations"）。

**当前论文已有的 DE 元素**：
1. **UPAP 端到端数据处理管线**：raw time series → soft-mask peak annotation → dual-head prediction → tolerance-aware evaluation，本质是一个数据处理 pipeline
2. **Tolerance-based Condense-and-Match Protocol**（Algorithm 1）：类似数据库中的 fuzzy join / approximate matching
3. **WLEL 大规模真实工业数据**：世界最大电力负荷数据集之一（2021-2025 年，小时级，浦东电网）
4. **数据质量处理**：RevIN 归一化、lookahead-based peak annotation、soft Gaussian mask 处理标注不确定性
5. **效率分析**：雷达图展示参数量+推理延迟等系统指标

**与被拒论文（MRTGL）的差异**：
- MRTGL 被拒因为"纯模型贡献，缺 DE 深度"
- PeakFocus 的差异化：(1) 提出完整的评估协议（Algorithm 1，不只是模型）(2) 数据标注质量问题（soft mask）(3) 电力领域特定的数据工程

**仍存在的风险和补强建议**：
1. §3 方法论核心仍偏模型架构 → **建议在方法描述中强调数据处理视角（pipeline 而非 model）**
2. 缺少系统性能指标 → **建议加 throughput (samples/sec)**
3. 投稿表单的 **1000 字符相关性说明**建议写明：本文属于 "Time-series Data Management" + "Complex Event Processing" + "Domain-specific Data Engineering (Energy)"

**Desk Reject 风险**：ICDE 明确说"purely advancing data mining or ML without touching data management → out of scope"。需要在写作上确保 DE 属性够明显。

### 4.4 ICDE 收录时序论文的趋势分析（基于前期调研）

> 详见 `docs/ICDE_VLDB_SIGMOD_时序论文调研.md`

**ICDE 2023-2025 时序论文趋势**：
- 近三年共约 50 篇时序 Research Track 论文
- **异常检测**是最稳定的子方向（累计 10+ 篇）
- **预测方向明显增长**：2023 仅 1 篇 → 2025 增至 5 篇（含 Patch Transformer、LLM 蒸馏等新方法）
- **数据管理类**（压缩/存储/数据库）每年稳定 2-3 篇
- **事件预测始终空白**：三年 0 篇 — **这是 PeakFocus 的差异化机会**

**对 PeakFocus 的启示**：
1. ICDE 2025 已接收 Patch-based 时序预测论文（《Towards Lightweight TS Forecasting》，西安工程大学+华东师范大学），说明 ICDE 对预测方法有接受度
2. ICDE 2024 接收了 1 篇 CEP（复杂事件处理）论文，PeakFocus 的峰值检测可以定位为"时序流中的极端事件处理"
3. 中国大陆机构参与的时序论文占 ~60%（清华、华东师范、哈工大、浙大等），投稿环境熟悉

**DB 会议 vs ML 会议的核心差异**：

| 维度 | ICDE 看重 | KDD/NeurIPS 看重 |
|---|---|---|
| 核心 | 效率、可扩展性、系统贡献 | 模型精度、新架构 |
| 评审风格 | "比 SOTA 快 100x" | "比 SOTA 准 3%" |
| 对预测论文 | 偏系统/效率/AutoML 角度 | 偏模型创新 |

**PeakFocus 在 ICDE 的定位策略**：
- **不要定位为"新的时序预测模型"**（这会被认为是 KDD 论文）
- **应定位为"面向电力负荷峰值事件的端到端数据处理框架"**
  - UPAP = 数据管线（pipeline contribution）
  - Tolerance-based Protocol = 数据匹配算法（data management contribution）
  - Soft mask = 数据质量处理（data quality contribution）
  - 大规模 WLEL 数据 = 工业级验证（domain-specific data engineering）

---

## 五、代码整理

- 项目重构为 release-ready 结构（`experiments/` 按论文编号、`tools/` 合并、`visualization/` 合并）
- 10 个关键目录（models / exp / layers / data_provider / utils / tools / experiments / visualization / dataset / paper_data）全部创建中文 README.md
- 数据管线自动化：`parse_epoch_time.py` → `build_figure_data.py` → `plot_*.py`（一键从 log 到图）
- 清理 PatchTST unfair 历史残留（结果/脚本/代码注释全部清除）
- 新增 `redraw_qualitative_comparison.py`（统一风格重绘定性对比图）

---

## 六、现存问题

| 问题 | 严重程度 | 说明 |
|---|---|---|
| 论文中仍有 21 处 `\textcolor{red}{}` + 1 处 `\textcolor{blue}{}` | **高** | 提交前必须全部清除 |
| 缺少 AI-Generated Content Acknowledgement | **高** | ICDE 2026 要求必须有此声明 |
| Overleaf 和本地有冲突 | 中 | 本地修改了大量内容，需要手动同步 |
| 论文 DE 定位可以更突出 | 中 | 按老师建议在写作上"靠一靠 Data Engineering" |
| 代码中 `proposed_model` 未重命名为 `PeakFocus` | 低 | release 前需要处理 |

---

## 七、下一步计划（距第一轮截止 2026-06-11 约 2 个月）

- [ ] 清除论文中所有 21 处 `\textcolor{red}{}` + 1 处 `\textcolor{blue}{}` 审稿标记
- [ ] 添加 AI-Generated Content Acknowledgement（ICDE 2027 强制要求，不计入页数）
- [ ] 同步本地修改到 Overleaf
- [ ] 在 Introduction/§3/Conclusion 中增强 Data Engineering 定位表述（回应老师建议）
- [ ] 准备投稿表单中的 1000 字符 scope 说明
- [ ] 确认所有作者 ORCID
- [ ] 代码 release 准备（GitHub anonymous repo + 重命名 proposed_model → PeakFocus）
- [ ] 最终校对 + 提交
