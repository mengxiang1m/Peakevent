# Paper Data

Auto-generated experiment results for paper tables. Total: 541 runs across 28 experiments × 3 domains.

## Files

| File | Description | Rows |
|------|-------------|------|
| `parsed_logs.csv` | Master table: every single run | 541 |
| `table_ar_ablation.csv` | tab:ar_abl — AR decoder ablation | 12 |
| `table_baseline.csv` | tab:baseline — External baseline comparison | 45 |
| `table_cross_domain.csv` | tab:cross — Cross-domain validation | 12 |
| `table_grid_search.csv` | Hyperparameter grid search | 33 |
| `table_horizon.csv` | tab:horizon — Horizon scalability | 9 |
| `table_hw_ar.csv` | tab:hw_ar — Stronger weight AR ablation | 12 |
| `table_hw_arch.csv` | tab:hw_arch — Stronger weight arch ablation | 12 |
| `table_main_arch_ablation.csv` | tab:main — Component study (arch ablation) | 27 |
| `table_p1_ablation.csv` | tab:p1_abl — Phase 1 pretraining ablation | 15 |
| `table_phase1_metrics.csv` | tab:p1 — Phase 1 metrics | 5 |

## Regenerate

```bash
python patchevent/paper_data/extract_all_results.py
```

## Paper Table ↔ CSV Mapping

| Paper Table | CSV File | Experiment Source |
|-------------|----------|-------------------|
| tab:main (WLEL component) | table_main_arch_ablation.csv | phase2/checkpoints/*_arch_ablation_v2 |
| tab:baseline (External) | table_baseline.csv | baselines/results/posthoc/summary_agg.csv |
| tab:ar_abl (AR ablation) | table_ar_ablation.csv | phase2/checkpoints/*_ar_ablation_v2 |
| tab:p1_abl (Phase1 ablation) | table_p1_ablation.csv | phase2/checkpoints/*_p1_ablation |
| tab:cross (Cross-domain) | table_cross_domain.csv | phase2/checkpoints/*_arch_ablation_v2 |
| tab:horizon (Horizon) | table_horizon.csv | phase2/checkpoints/*_pred{168,336} |
| tab:hw_arch/tab:hw_ar | table_hw_*.csv | phase2/checkpoints/*_hw_ablation |
| tab:p1 (Phase1 metrics) | table_phase1_metrics.csv | phase1/checkpoints/p1_ablation_full_metrics.json |
