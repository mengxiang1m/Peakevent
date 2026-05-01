# elc_p1_ablation

**用途**：论文 Appendix
**说明**：ELC Phase1消融
**规模**：5配置×3种子

## 实验结果

| Run | F1 | OnsetMAE | ApexMAE |
|-----|-----|----------|---------|
| GP1a_wo_has_apex_s123 | 0.8017 | 1.1131 | 1.0547 |
| GP1a_wo_has_apex_s42 | 0.8086 | 1.1460 | 1.1266 |
| GP1a_wo_has_apex_s456 | 0.8052 | 1.0805 | 1.1453 |
| GP1b_wo_distance_s123 | 0.7968 | 1.1091 | 1.0681 |
| GP1b_wo_distance_s42 | 0.7492 | 1.1721 | 1.1438 |
| GP1b_wo_distance_s456 | 0.7764 | 1.1254 | 1.0974 |
| GP1c_wo_phase_s123 | 0.7677 | 1.1793 | 1.1598 |
| GP1c_wo_phase_s42 | 0.7726 | 1.1544 | 1.0871 |
| GP1c_wo_phase_s456 | 0.7947 | 1.0340 | 1.0263 |
| GP1d_wo_offset_s123 | 0.8009 | 1.0700 | 1.0223 |
| GP1d_wo_offset_s42 | 0.7780 | 1.1328 | 1.1086 |
| GP1d_wo_offset_s456 | 0.7444 | 1.0842 | 1.0550 |
| GP1e_generic_npp_s123 | 0.8051 | 1.2038 | 1.2408 |
| GP1e_generic_npp_s42 | 0.8293 | 1.1813 | 1.1566 |
| GP1e_generic_npp_s456 | 0.7764 | 1.1991 | 1.2343 |