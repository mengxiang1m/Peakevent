# Hybrid DETR-AR Ablation Summary

## Hybrid Internal Ablation

| Domain | Config | n | F1 (mean+/-std) | OnMAE | ApMAE | DurMAE | IntMAPE |
|---|---|---:|---:|---:|---:|---:|---:|
| wlel | H0_hybrid | 1 | 0.8314+/-0.0000 | 1.3240 | 1.2012 | 0.4139 | 0.0655 |
| wlel | HQ8_queries | 2 | 0.8116+/-0.0165 | 1.4070 | 1.2951 | 0.4219 | 0.0751 |
| wlel | HQ12_queries | 1 | 0.8195+/-0.0000 | 1.3782 | 1.2681 | 0.3975 | 0.0705 |
| wlel | HR0_no_refine | 1 | 0.8091+/-0.0000 | 1.4360 | 1.3520 | 0.4205 | 0.0769 |
| wlel | HR2_refine2 | 1 | 0.7406+/-0.0000 | 1.5411 | 1.4500 | 0.4291 | 0.0784 |
| wlel | HO_query_order | 0 | NA | NA | NA | NA | NA |
| wlel | HC_no_causal_refine | 0 | NA | NA | NA | NA | NA |
| wlel | HM_no_intensity_cost | 1 | 0.8209+/-0.0000 | 1.2914 | 1.1981 | 0.4112 | 0.0664 |
| wlel | HP_no_proposal_aux | 1 | 0.7504+/-0.0000 | 1.5092 | 1.4505 | 0.4225 | 0.0742 |
| wlel | HN_noobj05 | 0 | NA | NA | NA | NA | NA |
| wlel | HF_focal2 | 0 | NA | NA | NA | NA | NA |
| wlel | HT_count_head | 0 | NA | NA | NA | NA | NA |
| wlel | HS_structure | 0 | NA | NA | NA | NA | NA |
| ett | H0_hybrid | 0 | NA | NA | NA | NA | NA |
| ett | HQ8_queries | 0 | NA | NA | NA | NA | NA |
| ett | HQ12_queries | 0 | NA | NA | NA | NA | NA |
| ett | HR0_no_refine | 0 | NA | NA | NA | NA | NA |
| ett | HR2_refine2 | 0 | NA | NA | NA | NA | NA |
| ett | HO_query_order | 0 | NA | NA | NA | NA | NA |
| ett | HC_no_causal_refine | 0 | NA | NA | NA | NA | NA |
| ett | HM_no_intensity_cost | 0 | NA | NA | NA | NA | NA |
| ett | HP_no_proposal_aux | 0 | NA | NA | NA | NA | NA |
| ett | HN_noobj05 | 0 | NA | NA | NA | NA | NA |
| ett | HF_focal2 | 0 | NA | NA | NA | NA | NA |
| ett | HT_count_head | 0 | NA | NA | NA | NA | NA |
| ett | HS_structure | 0 | NA | NA | NA | NA | NA |
| elc | H0_hybrid | 0 | NA | NA | NA | NA | NA |
| elc | HQ8_queries | 0 | NA | NA | NA | NA | NA |
| elc | HQ12_queries | 0 | NA | NA | NA | NA | NA |
| elc | HR0_no_refine | 0 | NA | NA | NA | NA | NA |
| elc | HR2_refine2 | 0 | NA | NA | NA | NA | NA |
| elc | HO_query_order | 0 | NA | NA | NA | NA | NA |
| elc | HC_no_causal_refine | 0 | NA | NA | NA | NA | NA |
| elc | HM_no_intensity_cost | 0 | NA | NA | NA | NA | NA |
| elc | HP_no_proposal_aux | 0 | NA | NA | NA | NA | NA |
| elc | HN_noobj05 | 0 | NA | NA | NA | NA | NA |
| elc | HF_focal2 | 0 | NA | NA | NA | NA | NA |
| elc | HT_count_head | 0 | NA | NA | NA | NA | NA |
| elc | HS_structure | 0 | NA | NA | NA | NA | NA |
