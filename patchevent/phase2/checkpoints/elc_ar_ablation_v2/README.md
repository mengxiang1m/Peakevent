# elc_ar_ablation_v2

**用途**：论文 Appendix
**说明**：ELC AR消融
**规模**：4配置×3种子

## 实验结果

| Run | F1 | OnsetMAE | ApexMAE |
|-----|-----|----------|---------|
| GD1_no_pos_mask_s123 | 0.0000 | nan | nan |
| GD1_no_pos_mask_s42 | 0.0004 | 0.0000 | 0.0000 |
| GD1_no_pos_mask_s456 | 0.0000 | nan | nan |
| GD2_no_pos_smooth_s123 | 0.7587 | 1.1303 | 1.0295 |
| GD2_no_pos_smooth_s42 | 0.7425 | 1.1747 | 1.2346 |
| GD2_no_pos_smooth_s456 | 0.7851 | 1.1035 | 1.1534 |
| GD3_no_attr_weight_s123 | 0.8071 | 1.0322 | 0.9626 |
| GD3_no_attr_weight_s42 | 0.7924 | 1.1426 | 1.0455 |
| GD3_no_attr_weight_s456 | 0.8024 | 1.0544 | 0.9290 |
| GD4_no_causal_s123 | 0.2334 | 4.4678 | 1.4293 |
| GD4_no_causal_s42 | 0.1575 | 15.1120 | 1.5422 |
| GD4_no_causal_s456 | 0.1532 | 6.3528 | 1.6700 |