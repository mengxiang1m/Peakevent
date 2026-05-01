# PatchEvent Comparison Experiments

This folder is an isolated workspace for external baseline comparison with PatchEvent.

## Scope

- Predict-then-Detect baselines from `tslib/`: `PatchTST`, `iTransformer`, `TimeMixer`, `DLinear` (optional: `Autoformer`).
- Peak-aware baseline from `seq2peak/`: `peak_Autoformer` (Seq2Peak).
- Unified post-hoc event evaluation using:
  - `dataset.build_wlel_events.detect_events_from_peak_labels`
  - `phase2_small_decoder_final.evaluate.{match_events, compute_sample_errors, aggregate_metrics}`

## Layout

- `data_prep.py`: converts STSEP csv (`timestamp,value`) to `date,OT` and creates `data/dataset_configs.json`.
- `posthoc_evaluate.py`: detects peaks from forecast curves and computes event metrics.
- `collect_results.py`: aggregates run-level JSON to CSV + LaTeX tables.
- `scripts/`: training/eval shell scripts.
- `tslib/`, `seq2peak/`: copied baseline code with local modifications for STSEP.

## Quick Start

```bash
cd comparison_experiments
python data_prep.py

# WLEL training
bash scripts/run_tslib_wlel.sh
bash scripts/run_seq2peak_wlel.sh

# Event evaluation
python posthoc_evaluate.py --dataset wlel --pred_len 96
python posthoc_evaluate.py --dataset wlel --pred_len 168
python posthoc_evaluate.py --dataset wlel --pred_len 336
python collect_results.py
```

## Notes

- Use `--data stsep --dataset_config ./data/dataset_configs.json --dataset_name <wlel|ett|elc>` for both baseline codebases.
- `per_window_preds.npy` and `per_window_starts.npy` are saved by modified `test()` functions and consumed by `posthoc_evaluate.py`.
- For fair protocol alignment, set `--seq_len 96 --label_len 48` and default `--window_stride 4`.
