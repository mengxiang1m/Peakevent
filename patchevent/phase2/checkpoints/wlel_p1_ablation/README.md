# wlel_p1_ablation

**用途**：论文主表 Table 5
**说明**：Phase1 head消融 w=(2,0.5,2,0.5)
**规模**：5配置×3种子

## 实验结果

| Run | F1 | OnsetMAE | ApexMAE |
|-----|-----|----------|---------|
| GP1a_wo_has_apex_s123 | 0.8456 | 0.6989 | 0.6779 |
| GP1a_wo_has_apex_s42 | 0.8415 | 0.7815 | 0.7173 |
| GP1a_wo_has_apex_s456 | 0.8227 | 0.8007 | 0.7509 |
| GP1b_wo_distance_s123 | 0.8485 | 0.7269 | 0.6902 |
| GP1b_wo_distance_s42 | 0.8259 | 0.7605 | 0.7252 |
| GP1b_wo_distance_s456 | 0.8363 | 0.7344 | 0.7058 |
| GP1c_wo_phase_s123 | 0.8450 | 0.7578 | 0.7147 |
| GP1c_wo_phase_s42 | 0.8399 | 0.7326 | 0.6926 |
| GP1c_wo_phase_s456 | 0.8371 | 0.7154 | 0.6823 |
| GP1d_wo_offset_s123 | 0.8383 | 0.8093 | 0.7631 |
| GP1d_wo_offset_s42 | 0.8464 | 0.7899 | 0.7434 |
| GP1d_wo_offset_s456 | 0.8395 | 0.8065 | 0.7504 |
| GP1e_generic_npp_s123 | 0.8462 | 0.9442 | 0.8806 |
| GP1e_generic_npp_s42 | 0.8423 | 0.9303 | 0.8531 |
| GP1e_generic_npp_s456 | 0.8382 | 0.8222 | 0.7650 |