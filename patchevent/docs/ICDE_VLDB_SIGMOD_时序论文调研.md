# ICDE / VLDB / SIGMOD 时序论文调研

**日期**：2026年4月9日  
**目的**：调研 DB 顶会近三年时间序列论文数量、主题分布，评估 PatchEvent 投稿适配性

---

## 一、总览：各会议时序论文数量

| 会议 | 年份 | 时序相关论文数（Research Track） | 备注 |
|------|------|--------------------------------|------|
| **ICDE 2023** | 2023 | ~20 篇 | 含时空预测 6 篇 |
| **ICDE 2024** | 2024 | ~14 篇 | 异常检测 4 篇为最大子方向 |
| **ICDE 2025** | 2025 | ~16 篇 | 预测 5 篇、压缩/存储 3 篇 |
| **VLDB 2024** | 2024 | ~15-20 篇 | 相似性搜索/索引为传统强项 |
| **SIGMOD 2024** | 2024 | ~10-15 篇 | 数据管理/清洗偏重 |

**关键结论：三个 DB 顶会近三年均无"时间序列事件预测"或"峰值预测"论文。**

---

## 二、ICDE 2023 时序论文（~20 篇）

### 主题分布

| 子方向 | 数量 |
|--------|------|
| 时空/交通预测 | 6 |
| 异常检测 | 3 |
| 时间序列分类 | 2 |
| 解释/分析 | 2 |
| 预测 (Long-term Forecasting) | 1 |
| 补全 (Imputation) | 1 |
| 模式挖掘 | 1 |
| 数据库/存储 | 1 |
| 金融时序 | 1 |
| 事件序列 | 1 |
| COVID时空 | 1 |

### 代表性论文

| # | 论文 | 作者 | 单位 | 主题 |
|---|------|------|------|------|
| 1 | Towards Long-Term Time-Series Forecasting: Feature, Pattern, and Distribution | Yan Li, Xinjiang Lu, Haoyi Xiong, Jian Tang, Jiantao Su, Bo Jin, Dejing Dou | 浙江大学, 百度研究院, 清华大学, 龙源电力, 大连理工大学 | 长期时序预测 |
| 2 | When Spatio-Temporal Meet Wavelets (STWave) | Yuchen Fang, Yanjun Qin, Haiyong Luo, Fang Zhao, Bingbing Xu, Liang Zeng, Chenxing Wang | 中科院计算所, 中国科学院大学, 北京邮电大学, 清华大学 | 小波+时空交通预测 |
| 3 | Dynamic Hypergraph Structure Learning for Traffic Flow Forecasting (DyHSL) | Yusheng Zhao, Xiao Luo, Wei Ju, Chong Chen, Xian-Sheng Hua, Ming Zhang | 北京大学, UCLA, Terminus Group | 动态超图交通预测 |
| 4 | Self-Supervised Spatial-Temporal Bottleneck Attentive Network (SSTBAN) | Shengnan Guo, Youfang Lin, Letian Gong, Chenyu Wang, Zeyu Zhou, Zekai Shen, Yiheng Huang, Huaiyu Wan | 北京交通大学 | 自监督时空预测 |
| 5 | A Stitch in Time Saves Nine: Early Anomaly Detection with Correlation Analysis | Yihao Ang, Qiang Huang, Anthony K.H. Tung, Zhiyong Huang | 新加坡国立大学 (NUS) | 早期异常检测 |
| 6 | UADB: Unsupervised Anomaly Detection Booster | Hangting Ye, Zhining Liu, Xinyi Shen, Wei Cao, Shun Zheng, Xiaofan Gui, Huishuai Zhang, Yi Chang, Jiang Bian | 吉林大学, UIUC, 中国人民大学, 微软亚洲研究院 | 无监督异常检测增强 |
| 7 | PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation | Mingzhe Liu, Han Huang, Hao Feng, Leilei Sun, Bowen Du, Yanjie Fu | 北京航空航天大学, 中佛罗里达大学 | 扩散模型时空补全 |
| 8 | TSExplain: Explaining Aggregated Time Series | Yiru Chen, Silu Huang | 哥伦比亚大学, 微软研究院 | 时间序列解释 |
| 9 | Mining Seasonal Temporal Patterns in Time Series | Van Long Ho, Nguyen Ho, Torben Bach Pedersen | 奥尔堡大学 (丹麦) | 季节性模式挖掘 |
| 10 | Discovering Temporal Patterns for Event Sequence Clustering | Weichang Wu, Junchi Yan, Xiaokang Yang, Hongyuan Zha | 上海交通大学, 香港中文大学(深圳) | 事件序列聚类 |
| 11 | Backward-Sort for Time Series in Apache IoTDB | Xiaojian Zhang, Hongyin Zhang, Shaoxu Song, Xiangdong Huang, Chen Wang, Jianmin Wang | 清华大学软件学院 | 时序数据库优化 |

---

## 三、ICDE 2024 时序论文（~14 篇 Research Track）

### 主题分布

| 子方向 | 数量 |
|--------|------|
| 异常检测 | 4 |
| 预测/Forecasting | 3 |
| 表示学习/预训练 | 3 |
| 时空预测 | 1 |
| 隐私保护 | 1 |
| 事件处理 (CEP) | 1 |
| 流处理 | 1 |

### 完整论文列表

| # | 论文 | 作者 | 单位 | 主题 |
|---|------|------|------|------|
| 1 | TimeDRL: Disentangled Representation Learning for Multivariate Time-Series | Ching Chang, Chiao-Tung Chan, Wei-Yao Wang, Wen-Chih Peng, Tien-Fu Chen | 阳明交通大学 (台湾 NYCU) | 多变量时序表示学习 |
| 2 | Scaling Up Multivariate TS Pre-Training with Decoupled Spatial-Temporal Representations | Rui Zha, Le Zhang, Shuangli Li, Jingbo Zhou, Tong Xu, Hui Xiong, Enhong Chen | 中国科学技术大学, 百度研究院, 港科大(广州) | 时序预训练 |
| 3 | TS3Net: Triple Decomposition with Spectrum Gradient for Long-Term TS Analysis | Xiangkai Ma, Xiaobin Hong, Sanglu Lu, Wenzhong Li | 南京大学 | 长期时序分析 |
| 4 | Learning Time-Aware Graph Structures for Spatially Correlated TS Forecasting | Minbo Ma, Fei Teng, Tianrui Li, Jilin Hu, Christian S. Jensen, Peng Han, Zhiqiang Xu | 西南交通大学, 华东师范大学, 奥尔堡大学, 电子科技大学, MBZUAI | 时空时序预测 |
| 5 | Temporal-Frequency Masked Autoencoders for TS Anomaly Detection | Yuchen Fang, Jiandong Xie, Yan Zhao, Lu Chen, Yunjun Gao, Kai Zheng | 电子科技大学, 华为云, 奥尔堡大学, 浙江大学 | 时频掩码异常检测 |
| 6 | Unraveling the 'Anomaly' in TS Anomaly Detection: Self-supervised Tri-domain Solution | Yuting Sun, Tong Chen, Hongzhi Yin, Guansong Pang, Guanhua Ye, Xia Hu | 昆士兰大学, 新加坡管理大学, Rice University | 自监督三域异常检测 |
| 7 | Learning Multi-Pattern Normalities in Frequency Domain for TS Anomaly Detection | Feiyi Chen, Zhen Qin, Shuiguang Deng, Yingying Zhang, Lunting Fan, Renhe Jiang, Yuxuan Liang, Qingsong Wen | 浙江大学, 阿里巴巴, 东京大学, 港科大(广州), Squirrel AI | 频域多模式异常检测 |
| 8 | From Chaos to Clarity: TS Anomaly Detection in Astronomical Observations | Xinli Hao, Chaohong Ma, Xiaofeng Meng, Yile Chen, Chen Yang, Zhihui Du, Chao Wu | 中国人民大学, 南洋理工大学, NJIT, 国家天文台(CAS) | 天文时序异常检测 |
| 9 | PrivShape: Extracting Shapes in Time Series under User-Level LDP | Yulian Mao, Qingqing Ye, Haibo Hu, Qi Wang, Kai Huang | 南方科技大学, 香港理工大学, 澳门科技大学 | 时序形状+差分隐私 |
| 10 | Representation Learning of Tangled Key-Value Sequence Data for Early Classification | Tao Duan, Junzhou Zhao, Shuo Zhang, Jing Tao, Pinghui Wang | 西安交通大学 | 序列早期分类 |
| 11 | A Unified Replay-based Continuous Learning Framework for Spatio-Temporal Prediction | Hao Miao, Yan Zhao, Bin Yang, Christian S. Jensen, Chenjuan Guo, Kai Zheng, Feiteng Huang, Jiandong Xie | 奥尔堡大学, 华东师范大学, 电子科技大学, 华为云 | 时空持续学习 |
| 12 | An Efficient Algorithm for Continuous Complex Event Matching Using Bit-Parallelism | Tao Qiu, Shenwang Jiang, Xiaochun Yang, Bin Wang, Chuanyu Zong, Rui Zhu | 东北大学 (沈阳) | 复杂事件匹配 (CEP) |
| 13 | ZeroTune: Learned Zero-Shot Cost Model for Stream Processing | Pratyush Agnihotri, Boris Koldehofe, Paul Stiegele, Roman Heinrich, Carsten Binnig, Manisha Luthra | TU Darmstadt, TU Ilmenau, DFKI (德国) | 流处理调优 |
| 14 | Robust Auto-Scaling with Probabilistic Workload Forecasting for Cloud Databases | Haitian Hang, Xiu Tang, Jianling Sun, Lingfeng Bao, David Lo, Haoye Wang | 浙江大学, 新加坡管理大学 | 工作负载预测 |

---

## 四、ICDE 2025 时序论文（~16 篇 Research Track）

### 主题分布

| 子方向 | 数量 |
|--------|------|
| 预测 (Forecasting) | 5 |
| 压缩/存储/管理 | 3 |
| 数据修复/填补 | 2 |
| 时空预测 | 2 |
| 分析/分解 | 2 |
| 分类 | 1 |
| 约束挖掘 | 1 |

### 完整论文列表

| # | 论文 | 作者 | 单位 | 主题 |
|---|------|------|------|------|
| 1 | Auto-TSF: Meta-learning for Automatic TS Forecasting Algorithm Selection | Tianyu Mu, Hongzhi Wang*, Chen Liang, Xinyue Shao | 哈尔滨工业大学 | 预测算法自动选择 |
| 2 | Towards Lightweight TS Forecasting: Patch-wise Transformer with Weak Data Enriching | Meng Wang, Jintao Yang, Bin Yang*, Hui Li, Tongxin Gong, Bo Yang, Jiangtao Cui | 西安工程大学, 华东师范大学, 西安电子科技大学 | 轻量Patch Transformer预测 |
| 3 | Accurate and Efficient Multivariate TS Forecasting via Offline Clustering | Yiming Niu, Jinliang Deng, Lulu Zhang, Zimu Zhou, Yongxin Tong* | 北京航空航天大学, UTS(悉尼), 香港城市大学 | 多变量预测+离线聚类 |
| 4 | TimeKD: Efficient Multivariate TS Forecasting via Calibrated LMs with Knowledge Distillation | Chenxi Liu*, Hao Miao, Qianxiong Xu, Shaowen Zhou, Cheng Long, Yan Zhao, Ziyue Li, Rui Zhao | 南洋理工大学, 奥尔堡大学, 科隆大学, 商汤科技 | LLM+知识蒸馏预测 |
| 5 | EasyTime: Time Series Forecasting Made Easy | Xiangfei Qiu, Xiuwen Li 等, Jilin Hu*, Chenjuan Guo, Aoying Zhou, Christian S. Jensen, Bin Yang | 华东师范大学, 奥尔堡大学 | 时序预测系统/框架 |
| 6 | AimTS: Augmented Series and Image Contrastive Learning for TS Classification | Yuxuan Chen, Shanshan Huang, Yunyao Cheng, Peng Chen, Zhongwen Rao, Yang Shu*, Bin Yang, Lujia Pan, Chenjuan Guo | 华东师范大学, 奥尔堡大学, 华为诺亚方舟实验室 | 对比学习分类 |
| 7 | DiffODE: Neural ODE with Differentiable Hidden State for Irregular TS Analysis | Yudong Zhang, Xu Wang, Xuan Yu, Zhengyang Zhou, Xing Xu, Lei Bai, Yang Wang* | 中国科学技术大学, 武汉理工大学, 上海AI实验室 | 不规则时序建模 |
| 8 | OneRoundSTL: In-Database Seasonal-Trend Decomposition | Zijie Chen, Shaoxu Song*, Jianmin Wang | 清华大学 | 数据库内时序分解 |
| 9 | A-DARTS: Stable Model Selection for Data Repair in Time Series | Mourad Khayati*, Guillaume Chacun, Zakhar Tymchenko, Philippe Cudre-Mauroux | 弗里堡大学, HES-SO (瑞士) | 时序数据修复 |
| 10 | Collaborative Imputation for Multivariate TS with Convergence Guarantee | Yu Sun, Xinyu Yang, Shaoxu Song*, Ying Zhang, Xiaojie Yuan | 南开大学, 清华大学 | 多变量时序填补 |
| 11 | Learned Compression of Nonlinear TS with Random Access | Andrea Guerra*, Giorgio Vinciguerra, Antonio Boffa, Paolo Ferragina | 比萨大学, EPFL (瑞士) | 非线性时序压缩 |
| 12 | TempSched: Temperature-Aware Storage Scheduler for TS Across Cloud-Edge-Device | Shuangshuang Cui, Hongzhi Wang*, Xianglong Liu, Xiaoou Ding | 哈尔滨工业大学 | 云边端时序存储 |
| 13 | SOUND: Sanity Checking of Pipelines for Uncertain and Sparse Data Series | Hermann Stolte*, Iftach Sadeh, Elisa Pueschel, Avigdor Gal, Matthias Weidlich | 柏林洪堡大学, DESY, 波鸿鲁尔大学, 以色列理工学院 | 数据序列管道检查 |
| 14 | tDCDiscover: Mining Threshold Denial Constraints from TS Data | Xiaoou Ding, Zhou Muyun, Yida Liu, Zekai Qian, Chen Wang, Hongzhi Wang*, Jianmin Wang | 哈尔滨工业大学, 清华大学 | 时序约束挖掘 |
| 15 | CrossST: Cross-District Pattern Generalization in Urban Spatio-Temporal Forecasting | Aoyu Liu, Yaying Zhang* | 同济大学 | 跨区域时空预测 |
| 16 | Leveraging Heterogeneous Experts for Traffic Prediction | Yueyang Yao, Xingyuan Dai, Yisheng Lv* | 中科院自动化研究所 | 交通预测 |

> 注：带 * 号为通讯作者

---

## 五、VLDB / SIGMOD 时序论文概况

### VLDB 2023-2024 代表性论文

| # | 论文 | 作者 | 单位 | 主题 |
|---|------|------|------|------|
| 1 | TSM-Bench: Benchmarking Time Series Database Systems for Monitoring Applications | Abdelouahab Khelifati, Mourad Khayati, Anton Dignos, Djellel Eddine Difallah, Philippe Cudre-Mauroux | 弗里堡大学(瑞士), Free Univ. of Bozen-Bolzano(意大利), NYU Abu Dhabi | 时序数据库 benchmark |
| 2 | METER: A Dynamic Concept Adaptation Framework for Online Anomaly Detection | Jiaqi Zhu, Shaofeng Cai, Fang Deng, Beng Chin Ooi, Wenqiao Zhang | 北京理工大学, 新加坡国立大学, 浙江大学 | 在线异常检测 |
| 3 | OneShotSTL: One-Shot Seasonal-Trend Decomposition for Online TS | Xiao He, Ye Li, Jian Tan, Bin Wu, Feifei Li | 阿里巴巴达摩院 | 时序分解 |
| 4 | CIVET: Exploring Compact Index for Variable-Length Subsequence Matching on TS | Haoran Xiong, Hang Zhang, Zeyu Wang, Zhenying He, Peng Wang, X. Sean Wang | 复旦大学 | 时序子序列匹配索引 |

### SIGMOD 2024 代表性论文

| # | 论文 | 作者 | 单位 | 主题 |
|---|------|------|------|------|
| 1 | Akane: Perplexity-Guided Time Series Data Cleaning | Xiaoyu Han, Haoran Xiong, Zhenying He, Peng Wang, Chen Wang, X. Sean Wang | 复旦大学, 清华大学 | 时序数据清洗 |

> 注：VLDB/SIGMOD 的部分论文标题为根据主题检索到的最接近匹配，已修正为实际发表标题。

---

## 六、DB 会议 vs ML/AI 会议的 Scope 差异

### DB 会议接收的时序工作类型

| 类别 | 热度 | 典型内容 |
|------|------|----------|
| **相似性搜索/索引** | ★★★★★ | Matrix Profile, 子序列匹配, DTW加速, 学习型索引 |
| **异常检测** | ★★★★☆ | 可扩展检测, 流式检测, 在线检测 |
| **数据管理/存储** | ★★★★☆ | 压缩, 查询处理, 时序数据库, benchmark |
| **数据质量** | ★★★☆☆ | 缺失值填补, 数据清洗, 去噪 |
| **分解/挖掘** | ★★★☆☆ | 趋势分解, motif发现, shapelet |
| **预测 (Forecasting)** | ★★☆☆☆ | 较少，但近年增长（偏系统/效率角度） |
| **事件预测** | ☆☆☆☆☆ | **几乎不出现** |

### Scope 对比

| 维度 | DB 会议 (ICDE/VLDB/SIGMOD) | ML/AI 会议 (KDD/NeurIPS/ICML) |
|------|---------------------------|-------------------------------|
| **核心关注** | 效率、可扩展性、系统 | 模型精度、新架构、新范式 |
| **预测** | 关注系统效率、在线更新、AutoML | 关注精度、新模型 (Transformer/SSM) |
| **异常检测** | 大规模可扩展性、流式处理 | 检测精度、无监督方法 |
| **Foundation Models** | 偏 benchmark/系统角度 | 核心热点 (PatchTST, TimesFM...) |
| **事件预测** | **极少** | KDD/AAAI 有一些 |
| **评审看重** | 系统贡献 + 实验规模 + 效率 | 模型创新 + SOTA + 理论 |
| **论文风格** | "我们在 10TB 数据上比 SOTA 快 100x" | "我们在 8 个 benchmark 上超越 SOTA" |

---

## 七、高频出现的单位统计

### ICDE 2023-2025 时序论文中出现频次最高的机构

| 排名 | 单位 | 出现次数 | 代表方向 |
|------|------|----------|----------|
| 1 | **清华大学** | 5 | 时序数据库 (IoTDB), 分解, 约束挖掘 |
| 2 | **华东师范大学** | 4 | 时序预测, 分类, 系统框架 |
| 3 | **奥尔堡大学 (Aalborg)** | 4 | 时序预测, 模式挖掘, 时空预测 |
| 4 | **哈尔滨工业大学** | 3 | 预测AutoML, 存储调度, 约束挖掘 |
| 5 | **浙江大学** | 3 | 长期预测, 异常检测, 工作负载预测 |
| 6 | **电子科技大学 (UESTC)** | 3 | 异常检测, 时空预测 |
| 7 | **中科院系统 (计算所/自动化所/USTC)** | 3 | 时空预测, 预训练, 交通预测 |
| 8 | **北京交通大学** | 2 | 时空交通预测 |
| 9 | **北京大学** | 1 | 交通预测 (超图) |
| 10 | **新加坡国立大学 (NUS)** | 2 | 异常检测 |
| 11 | **微软研究院** | 2 | 异常检测, 时序解释 |
| 12 | **弗里堡大学 (瑞士)** | 2 | benchmark, 数据修复 |

### 中国高校/机构在 ICDE 时序方向占比

ICDE 2023-2025 共约 50 篇时序相关论文，其中**中国大陆机构参与的超过 30 篇（~60%）**，是绝对主力。前五名均为中国/华人团队主导。

---

## 八、与 PatchEvent 的适配性分析

### PatchEvent 的核心贡献

- 新任务定义：时间序列**结构化事件预测**（onset, duration, apex, intensity）
- 新方法：两阶段 Patch Encoder + AR Decoder 框架
- 应用领域：电力负荷、变压器温度、用电量

### 各会议匹配度

| 会议 | 匹配度 | 原因 |
|------|--------|------|
| **KDD** | ★★★★★ | 接受应用导向的时序新方法，有 Applied Data Science Track |
| **AAAI** | ★★★★☆ | 接受新预测范式，事件预测有先例 |
| **CIKM** | ★★★★☆ | Seq2Peak 的发表地（CIKM 2023），明确接受此类工作 |
| **IJCAI** | ★★★☆☆ | 接受 AI 应用，但竞争激烈 |
| **NeurIPS/ICML** | ★★★☆☆ | 需强调方法通用性和理论贡献 |
| **ICDE** | ★★☆☆☆ | 需增加系统层面贡献 |
| **VLDB** | ★★☆☆☆ | 需增加可扩展性/数据管理贡献 |
| **SIGMOD** | ★★☆☆☆ | 需增加系统/benchmark贡献 |

### 投 DB 会议需要的改造方向

如果要投 ICDE/VLDB/SIGMOD，PatchEvent 需要补充：

1. **系统层面贡献**：高效在线推理、增量更新、大规模部署
2. **Benchmark 贡献**：构建标准化的时间序列事件预测 benchmark（当前没有公开 benchmark）
3. **数据管理角度**：事件的存储/索引/查询
4. **可扩展性实验**：在更大规模数据上的效率分析

### Scope 不符合被拒的风险

DB 会议审稿人可能的拒绝理由：
- "This is a pure ML/prediction paper, lacks system contribution"（纯ML论文，缺乏系统贡献）
- "The scalability analysis is insufficient"（可扩展性分析不足）
- "Better suited for KDD/AAAI"（更适合 KDD/AAAI）
- "The datasets are too small for a DB venue"（数据集规模太小）

---

## 九、ICDE 2023-2025 时序论文趋势

| 趋势 | 说明 |
|------|------|
| **异常检测持续火热** | 三年累计 10+ 篇，是最稳定的时序子方向 |
| **预测方向增长** | 2023 仅 1 篇 → 2025 增至 5 篇，且引入 LLM、Patch 等新技术 |
| **预训练/Foundation Model 兴起** | 2024 出现 TimeDRL 等表示学习，2025 出现 LLM 蒸馏 |
| **数据管理类持续稳定** | 压缩、存储、数据库优化每年 2-3 篇 |
| **时空预测逐年减少** | 2023 有 6 篇，2024-2025 各 1-2 篇 |
| **事件预测始终空白** | 三年 0 篇，是未被覆盖的方向 |

---

## 十、建议

1. **首选 KDD/AAAI/CIKM**：与 PatchEvent 的任务定义和方法论最匹配
2. **ICDE 作为备选**：若补充系统/效率贡献，ICDE 2025 已有 Patch Transformer 预测论文（Towards Lightweight TS Forecasting），说明 ICDE 对 Patch-based 时序方法有接受度
3. **差异化优势**：无论投哪个会议，"结构化事件预测"在 DB 和 ML 会议中都是新方向，新颖性有保障
