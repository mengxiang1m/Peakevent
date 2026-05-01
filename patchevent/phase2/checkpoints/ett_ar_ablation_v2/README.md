# ett_ar_ablation_v2

**用途**：论文 Appendix
**说明**：ETT AR消融
**规模**：4配置×3种子

## 实验结果

| Run | F1 | OnsetMAE | ApexMAE |
|-----|-----|----------|---------|
| GD1_no_pos_mask_s123 | 0.0173 | 1.0312 | 0.8125 |
| GD1_no_pos_mask_s42 | 0.1525 | 0.8520 | 0.7829 |
| GD1_no_pos_mask_s456 | 0.0844 | 0.6667 | 0.6049 |
| GD2_no_pos_smooth_s123 | 0.8552 | 0.7054 | 0.6547 |
| GD2_no_pos_smooth_s42 | 0.8543 | 0.7070 | 0.6571 |
| GD2_no_pos_smooth_s456 | 0.8566 | 0.7036 | 0.6529 |
| GD3_no_attr_weight_s123 | 0.8562 | 0.6752 | 0.6275 |
| GD3_no_attr_weight_s42 | 0.8542 | 0.6882 | 0.6377 |
| GD3_no_attr_weight_s456 | 0.8565 | 0.6813 | 0.6340 |
| GD4_no_causal_s123 | 0.3246 | 1.4747 | 1.2125 |
| GD4_no_causal_s42 | 0.1869 | 6.0325 | 1.6227 |
| GD4_no_causal_s456 | 0.3220 | 3.9357 | 1.2887 |