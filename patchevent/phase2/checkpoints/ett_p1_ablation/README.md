# ett_p1_ablation

**用途**：论文 Appendix
**说明**：ETT Phase1消融
**规模**：5配置×3种子

## 实验结果

| Run | F1 | OnsetMAE | ApexMAE |
|-----|-----|----------|---------|
| GP1a_wo_has_apex_s123 | 0.8524 | 0.7182 | 0.6662 |
| GP1a_wo_has_apex_s42 | 0.8553 | 0.7317 | 0.6779 |
| GP1a_wo_has_apex_s456 | 0.8524 | 0.7305 | 0.6776 |
| GP1b_wo_distance_s123 | 0.8496 | 0.7485 | 0.6950 |
| GP1b_wo_distance_s42 | 0.8535 | 0.7018 | 0.6496 |
| GP1b_wo_distance_s456 | 0.8539 | 0.6864 | 0.6349 |
| GP1c_wo_phase_s123 | 0.8531 | 0.7069 | 0.6565 |
| GP1c_wo_phase_s42 | 0.8508 | 0.7267 | 0.6741 |
| GP1c_wo_phase_s456 | 0.8573 | 0.6877 | 0.6367 |
| GP1d_wo_offset_s123 | 0.8492 | 0.7206 | 0.6672 |
| GP1d_wo_offset_s42 | 0.8506 | 0.7440 | 0.6838 |
| GP1d_wo_offset_s456 | 0.8563 | 0.7003 | 0.6495 |
| GP1e_generic_npp_s123 | 0.8461 | 0.7446 | 0.6884 |
| GP1e_generic_npp_s42 | 0.8465 | 0.7571 | 0.6986 |
| GP1e_generic_npp_s456 | 0.8588 | 0.7006 | 0.6490 |