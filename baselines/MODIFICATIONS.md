# Modifications Log

This file records all code changes made in `comparison_experiments/` relative to copied upstream baseline repos.

## TSLib (`comparison_experiments/tslib`)

### `data_provider/data_loader.py`

- Added optional imports with graceful fallback for `datasets`, `sktime`, and `huggingface_hub`.
- Added `Dataset_STSEP`:
  - Reads fixed split boundaries from `--dataset_config` + `--dataset_name`.
  - Uses exact train/val/test indices from STSEP data config.
  - Supports aligned sliding windows with `--window_stride` (default 1; scripts use 4).
  - Keeps `scale=True` behavior and exposes `inverse_transform`.
- Added helper `_read_stsep_split`.

### `data_provider/data_factory.py`

- Registered `'stsep': Dataset_STSEP` in `data_dict`.

### `exp/exp_long_term_forecasting.py`

- In `test()`:
  - Added export of per-window forecasts:
    - `per_window_preds.npy` (shape `[N_test, pred_len]` for univariate)
    - `per_window_starts.npy` (global window starts aligned to dataset loader)
  - Existing outputs (`pred.npy`, `true.npy`, `metrics.npy`) kept.

### `run.py`

- Added new args:
  - `--dataset_config`
  - `--dataset_name`
  - `--window_stride`
- Changed seed behavior:
  - Uses `--seed` instead of hard-coded `2021` for `random`, `numpy`, `torch`.

## Seq2Peak (`comparison_experiments/seq2peak`)

### `data_provider/data_loader.py`

- Added helper `_read_stsep_split`.
- Added `Dataset_STSEP_Seq2Peak`:
  - Same split/stride alignment logic as TSLib STSEP dataset.
  - Compatible with Seq2Peak data factory signature (`args` passed explicitly when needed).

### `data_provider/data_factory.py`

- Registered `'stsep': Dataset_STSEP_Seq2Peak` in `data_dict`.
- For `args.data == 'stsep'`, passes `args` into dataset constructor.

### `exp/exp_peak.py`

- In `test()`:
  - Added inverse transform support for full-curve outputs when `--inverse` is enabled.
  - Added export:
    - `metrics.npy`
    - `pred.npy`
    - `true.npy`
    - `per_window_preds.npy`
    - `per_window_starts.npy`

### `run.py`

- Added new args:
  - `--dataset_config`
  - `--dataset_name`
  - `--window_stride`
  - `--inverse`
- Fixed eval-only setting string typo:
  - replaced `args.shift` with `args.with_shift`.

## New Comparison Utilities

### `data_prep.py`

- Converts STSEP series csv to `data/<dataset>.csv` (`date,OT`).
- Builds `data/dataset_configs.json` with:
  - split boundaries
  - row counts
  - default event/meta params
  - expected test-window counts for default horizons.

### `posthoc_evaluate.py`

- Loads per-window curve forecasts.
- Applies `scipy.signal.find_peaks`.
- Converts peak labels to structured events via `detect_events_from_peak_labels`.
- Computes metrics with PatchEvent evaluation functions:
  - `match_events`
  - `compute_sample_errors`
  - `aggregate_metrics`
- Saves run-level JSON into `results/posthoc/<dataset>/pred<pred_len>/`.

### `collect_results.py`

- Aggregates run-level JSON files to:
  - `summary_all_runs.csv`
  - `summary_agg.csv` (mean/std by dataset, pred_len, method)
  - `<dataset>_table.tex` (LaTeX table).

### `scripts/*`

- Added runnable bash scripts for WLEL baseline training and evaluation orchestration.
