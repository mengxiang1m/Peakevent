# PeakFocus: An End-to-End Framework Bridging Localization and Regression for Electricity Load Peak Forecasting

## 1. Overview

PeakFocus is an end-to-end peak forecasting framework that jointly performs **peak localization** and **value regression** for electricity load time series. The framework introduces two key components:

- **MSM-PL (Multi-Scale Mixing Peak Locator)**: A multi-scale convolutional module that localizes potential peak positions and generates soft/hard peak masks.
- **LAD (Location-Aware Decoder)**: A cross-attention decoder that uses peak features from MSM-PL to guide the value regression, focusing model capacity on peak-relevant regions.

The core model outputs two predictions simultaneously:
1. **Peak probability mask** (from MSM-PL): Binary/soft classification of each time step as peak or non-peak.
2. **Value prediction** (from LAD): The forecasted time series, with enhanced accuracy at peak positions.

---

## 2. Environment Setup

### 2.1 Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate tslib_findpeaks
```

### 2.2 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.12.0 | Runtime |
| PyTorch | 2.8.0+cu128 | Deep learning framework |
| findpeaks | 2.7.5 | Peak detection in evaluation |
| scikit-learn | 1.7.2 | Metrics and preprocessing |
| pandas | 2.3.3 | Data loading |
| matplotlib | 3.10.7 | Visualization |
| sktime | 0.39.0 | Time series utilities |

### 2.3 Hardware Requirements

- GPU: NVIDIA GPU with CUDA 12.8+ support (tested on single GPU)
- RAM: 16GB+ recommended

---

## 3. Dataset Preparation

### 3.1 Supported Datasets

| Dataset ID | Name | Description | Data Path |
|-----------|------|-------------|-----------|
| `load_data_mixed` (WLEL) | Wuhan Load & Electricity | High-frequency load data (2021-2025) | `./dataset/load_data/hf_load_data/` |
| `electricity_mixed` (ELC) | Electricity | Standard electricity dataset (2016-2019) | `./dataset/electricity/` |

### 3.2 Data Format

The mixed dataset CSV must contain the following columns:

```
date, value_60min, value_max, is_peak
```

- `date`: Timestamp column (hourly frequency)
- `value_60min`: Hourly average load value (used as encoder input)
- `value_max`: Hourly maximum load value (used as prediction target)
- `is_peak`: Binary peak label (1=peak, 0=non-peak), auto-generated via `peak_lookahead`

**File naming convention**: `{dataset_name}_mixed_with_peaks_lookahead_{N}.csv`

- WLEL: `hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv`
- ELC: `electricity_mixed_with_peaks_lookahead_5.csv`

### 3.3 Peak Label Generation

Peak labels in the CSV are pre-generated using the `findpeaks` library with `lookahead` parameter:
- WLEL dataset: `peak_lookahead=3`
- ELC dataset: `peak_lookahead=5`

The `is_peak` column marks hourly time steps where a local peak occurs within the lookahead window.

### 3.4 Data Splitting

The framework uses **date-range-based splitting** (automatic detection in `data_factory.py`):

| Dataset | Train | Validation | Test |
|---------|-------|------------|------|
| WLEL | 2021-01 ~ 2023-08 | 2023-09 ~ 2024-08 | 2024-09 ~ end |
| ELC | 2016-07 ~ 2018-04 (22 months) | 2018-05 ~ 2018-11 (7 months) | 2018-12 ~ 2019-07 (8 months) |

### 3.5 Peak Detection Example

To understand the peak detection pipeline, refer to `peak_detect_example/detect_peak.py`, which demonstrates:
- AMPD (Automatic Multiscale-based Peak Detection)
- `scipy.signal.find_peaks`
- `findpeaks` library usage

---

## 4. Project Structure

```
code/
├── run.py                          # Main entry point
├── environment.yaml                # Conda environment specification
│
├── models/
│   ├── proposed_model.py           # PeakFocus (MSM-PL + LAD)
│   ├── peak_Transformer.py         # Peak Transformer (seq2peak baseline)
│   ├── naive_baselines.py          # Persistence / Seasonal / MovAvg
│   ├── Transformer.py              # Standard Transformer
│   ├── PatchTST.py                 # PatchTST
│   ├── DLinear.py                  # DLinear
│   ├── iTransformer.py             # iTransformer
│   ├── TimeMixer.py                # TimeMixer
│   ├── CycleNet.py                 # CycleNet
│   ├── SegRNN.py                   # SegRNN
│   ├── STID.py                     # STID
│   ├── Informer.py                 # Informer
│   ├── MetaEformer.py              # MetaEformer
│   └── ...                         # 30+ other baseline models
│
├── exp/
│   ├── exp_basic.py                # Base experiment class (model registry)
│   ├── exp_long_term_forecasting.py            # Standard long-term forecasting
│   ├── exp_peak_detect_based_on_long_term_forecasting.py       # Main: dual-head peak detection
│   ├── exp_peak_detect_based_on_long_term_forecasting_basic.py # Simplified: no peak classification head
│   └── exp_peak_detect_based_on_long_term_forecasting_seq2peak.py  # For peak_Transformer (seq2peak)
│
├── data_provider/
│   ├── data_factory.py             # Dataset routing and date-range auto-detection
│   └── data_loader.py              # Dataset classes (Dataset_Custom_Mixed, etc.)
│
├── layers/                         # Neural network building blocks
│   ├── Embed.py                    # Data/temporal embedding layers
│   ├── SelfAttention_Family.py     # Attention mechanisms
│   ├── Transformer_EncDec.py       # Encoder/decoder blocks
│   └── ...
│
├── utils/
│   ├── tools.py                    # EarlyStopping, learning rate adjustment
│   ├── losses.py                   # MAPE, SMAPE, MASE losses
│   ├── masking.py                  # Masking utilities
│   ├── timefeatures.py             # Time feature encoding
│   └── ...
│
├── scripts/
│   └── peak_detect_based_on_LTF/
│       ├── ELC/                    # ELC dataset experiment scripts
│       │   ├── test_proposed_model.sh
│       │   ├── test_all.sh
│       │   └── ...
│       ├── load_data/              # WLEL dataset experiment scripts
│       │   ├── test_proposed_model.sh
│       │   ├── test_all.sh
│       │   └── ...
│       ├── ablation_loss_weights.sh
│       ├── ablation_mask_type.sh
│       ├── ablation_n_scales.sh
│       └── ablation_naive_baselines.sh
│
├── metric_tools/
│   ├── calculate_mean_metrics.py   # Aggregate metrics across iterations
│   ├── cal.sh                      # Batch metric calculation
│   └── cal_single.sh              # Single-config metric calculation
│
├── extract_results.py              # Extract ablation results from logs
├── extract_sc_f1.py                # Extract self-consistency F1
├── compute_sc_f1.py                # Recompute SC-F1 from checkpoints
├── verify_results.py               # Verify experiment results
│
├── visualize_attn_heatmap.py       # LAD cross-attention heatmap
├── visualize_gate.py               # Gate activation visualization
├── visualize_gate_combined.py      # Combined visualization (20 samples)
│
├── analysis/
│   ├── compute_r2.py               # Compute R² from saved predictions
│   ├── load_data_analysis.py       # Data analysis
│   └── visualize_interpretability.py  # Interpretability analysis
├── predict/                        # Standalone prediction demo
├── peak_detect_example/            # Peak detection tutorial
└── dataset_describe/               # Dataset statistics and visualization
```

---

## 5. Training & Testing

### 5.1 Task Types

| Task Name | Experiment Class | Model Output | Use Case |
|-----------|-----------------|--------------|----------|
| `peak_detect_ltf` | `Exp_Peak_Detect_LTF` | (value, peak_mask) | **PeakFocus** and baselines with dual-head |
| `peak_detect_ltf_basic` | `Exp_Peak_Detect_LTF_basic` | (value) | Baselines without peak classification head |
| `seq2peak` | `Exp_Peak_Detect_LTF_Seq2Peak` | (value, peak_values) | `peak_Transformer` only |
| `long_term_forecast` | `Exp_Long_Term_Forecast` | (value) | Standard long-term forecasting |

### 5.2 Train PeakFocus (Proposed Model)

**On WLEL dataset (pred_len=336)**:
```bash
python -u run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model proposed_model \
  --data load_data_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaxIn_MaxOut' \
  --batch_size 128 \
  --mlp_layers 2 \
  --enc_in 1 --dec_in 1 --c_out 1 \
  --d_model 256 \
  --d_ff 256 \
  --n_heads 4 \
  --e_layers 1 \
  --factor 3 \
  --patience 5 \
  --train_epochs 20 \
  --learning_rate 0.001 \
  --lradj type3 \
  --loss MSE \
  --freq t \
  --gpu 0 \
  --itr 5 \
  --enable_peak_eval 1 \
  --peak_tolerance 1 \
  --peak_lookahead 3 \
  --if_lad 1 \
  --if_msm_pl 1 \
  --value_loss_weight 0.4 \
  --peak_loss_weight 0.4 \
  --tp_mse_loss_weight 0.2
```

**On ELC dataset (pred_len=336)**:
```bash
python -u run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/electricity \
  --data_path electricity_mixed_with_peaks_lookahead_5.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model proposed_model \
  --data electricity_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaxIn_MaxOut' \
  --batch_size 128 \
  --mlp_layers 2 \
  --enc_in 1 --dec_in 1 --c_out 1 \
  --d_model 256 \
  --d_ff 256 \
  --n_heads 4 \
  --e_layers 1 \
  --factor 3 \
  --patience 5 \
  --train_epochs 20 \
  --learning_rate 0.001 \
  --lradj type3 \
  --loss MSE \
  --freq t \
  --gpu 0 \
  --itr 5 \
  --enable_peak_eval 1 \
  --peak_tolerance 1 \
  --peak_lookahead 5 \
  --if_lad 1 \
  --if_msm_pl 1 \
  --value_loss_weight 0.4 \
  --peak_loss_weight 0.4 \
  --tp_mse_loss_weight 0.2
```

To change prediction horizon, set `--pred_len 720`.

### 5.3 Train Baseline Models

**Baselines use `peak_detect_ltf_basic` task** (no peak classification head):

```bash
# Example: PatchTST on WLEL-336
python -u run.py \
  --task_name peak_detect_ltf_basic \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model PatchTST \
  --data load_data_mixed \
  --features S \
  --seq_len 168 --label_len 48 --pred_len 336 \
  --input_col value_max --target_col value_max \
  --des 'MaxIn_MaxOut' \
  --batch_size 128 \
  --enc_in 1 --dec_in 1 --c_out 1 \
  --d_model 256 --d_ff 256 --n_heads 4 \
  --e_layers 1 --d_layers 1 \
  --patience 5 --train_epochs 20 \
  --learning_rate 0.001 --lradj type3 \
  --loss MSE --freq t --gpu 0 --itr 5 \
  --enable_peak_eval 1 --peak_tolerance 1 --peak_lookahead 3
```

Replace `--model` with any of: `Transformer`, `DLinear`, `Informer`, `TimeMixer`, `SegRNN`, `CycleNet`, `STID`, `MetaEformer`, `PatchTST`.

**Naive baselines** (no training required):
```bash
python -u run.py \
  --task_name peak_detect_ltf_basic \
  --is_training 0 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model NaiveBaseline \
  --data load_data_mixed \
  --naive_type persistence \
  --features S \
  --seq_len 168 --label_len 48 --pred_len 336 \
  --input_col value_max --target_col value_max \
  --enc_in 1 --dec_in 1 --c_out 1 \
  --enable_peak_eval 1 --peak_tolerance 1 --peak_lookahead 3 \
  --itr 1
```

Available `--naive_type`: `persistence`, `seasonal` (with `--seasonal_period 168`), `movavg` (with `--movavg_window 24`).

### 5.4 Batch Experiment Scripts

Run all comparison experiments at once:

```bash
# All baselines on WLEL dataset
bash scripts/peak_detect_based_on_LTF/load_data/test_all.sh

# All baselines on ELC dataset
bash scripts/peak_detect_based_on_LTF/ELC/test_all.sh

# PeakFocus on WLEL
bash scripts/peak_detect_based_on_LTF/load_data/test_proposed_model.sh

# PeakFocus on ELC
bash scripts/peak_detect_based_on_LTF/ELC/test_proposed_model.sh
```

---

## 6. Ablation Studies

### 6.1 Multi-Scale Depth (n_scales)

Tests MSM-PL with 0/1/2/3 scale levels on WLEL-336:

```bash
bash scripts/peak_detect_based_on_LTF/ablation_n_scales.sh
```

| n_scales | Architecture |
|----------|-------------|
| 0 | Simple linear projection (no multi-scale) |
| 1 | Single-scale convolution |
| 2 | Dual-scale with average pooling + upsampling (default) |
| 3 | Triple-scale cascaded injection |

### 6.2 Mask Type (soft vs. hard)

Tests Gaussian soft labels vs. binary hard labels:

```bash
bash scripts/peak_detect_based_on_LTF/ablation_mask_type.sh
```

- `soft`: Gaussian-smoothed labels with `sigma=2.0`
- `hard`: Binary 0/1 labels

Tested on both WLEL and ELC, pred_len=336 and 720.

### 6.3 Loss Weight Sensitivity

Tests 8 weight combinations for the triplet loss `(value, peak, tp_mse)`:

```bash
bash scripts/peak_detect_based_on_LTF/ablation_loss_weights.sh
```

| Config | value_lw | peak_lw | tp_mse_lw |
|--------|----------|---------|-----------|
| Default | 0.4 | 0.4 | 0.2 |
| Heavy global | 0.6 | 0.2 | 0.2 |
| Heavy TP-MSE | 0.2 | 0.2 | 0.6 |
| Equal | 0.33 | 0.33 | 0.33 |
| Swap peak/tp | 0.4 | 0.2 | 0.4 |
| Heavy peak | 0.2 | 0.6 | 0.2 |
| Low global | 0.2 | 0.4 | 0.4 |
| Dominant global | 0.8 | 0.1 | 0.1 |

### 6.4 Component Ablation (LAD / MSM-PL)

Included in `scripts/peak_detect_based_on_LTF/load_data/test_proposed_model.sh`:

| Variant | `if_lad` | `if_msm_pl` | Task |
|---------|----------|-------------|------|
| Full PeakFocus | 1 | 1 | `peak_detect_ltf` |
| w/o LAD | 0 | 1 | `peak_detect_ltf` |
| w/o MSM-PL | 0 | 0 | `peak_detect_ltf` |
| w/o both (basic) | 0 | 0 | `peak_detect_ltf_basic` |

### 6.5 Naive Baselines

```bash
bash scripts/peak_detect_based_on_LTF/ablation_naive_baselines.sh
```

Tests Persistence, Seasonal Naive (period=168), and Moving Average (window=24) on both datasets.

---

## 7. Loss Function

PeakFocus uses a **triplet loss**:

```
L_total = w_val * L_value + w_peak * L_peak + w_tp * L_tp_mse
```

| Component | Formula | Default Weight | Purpose |
|-----------|---------|----------------|---------|
| `L_value` | MSE(pred, true) | 0.4 | Overall prediction accuracy |
| `L_peak` | FocalLoss(peak_logits, peak_labels) | 0.4 | Peak localization (handles class imbalance) |
| `L_tp_mse` | MSE(pred[TP], true[TP]) | 0.2 | Value accuracy at correctly detected peaks |

**Focal Loss parameters**: `alpha=0.25`, `gamma=2.0`

**Soft labels**: When `use_soft_labels=1`, peak labels are Gaussian-smoothed with `sigma=2.0` instead of binary 0/1.

---

## 8. Evaluation Metrics

### 8.1 Prediction Metrics

| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error (overall) |
| MAE | Mean Absolute Error (overall) |
| RMSE | Root Mean Squared Error |

### 8.2 Peak Detection Metrics

| Metric | Description |
|--------|-------------|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-Score | 2 * Precision * Recall / (Precision + Recall) |
| TP / FP / FN | True Positives / False Positives / False Negatives |

Peak matching uses a **tolerance window** (`peak_tolerance=1` time step).

### 8.3 Peak Value Metrics

| Metric | Description |
|--------|-------------|
| TP_MSE / TP_MAE | Error at correctly detected peaks |
| FN_MSE / FN_MAE | Error at missed peaks |
| All_True_Peaks_MSE / MAE | Error at all ground-truth peak positions |

### 8.4 Composite Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| BPE | 0.5 * (1 - F1) + 0.5 * (1 - 1/(1+MSE)) | Balanced Peak Error (lower is better) |
| PIM | (1 + MSE) / (F1 + 0.01) | Peak Integrated Metric (lower is better) |

**Model selection** during training uses **BPE** (early stopping criterion).

---

## 9. Results Directory Structure

### 9.1 Naming Convention

```
results/{task}_{data}_{model}_{model_id}_{seq_len}_{pred_len}_{iteration}/
```

Example:
```
results/peak_detect_ltf_load_data_mixed_proposed_model_maxIn_maxOut_MSE_244_23_168_336_0/
```

### 9.2 Output Files

| File | Content |
|------|---------|
| `experiment_log.txt` | Complete training log with per-epoch metrics |
| `pred.npy` | Predicted values on test set |
| `true.npy` | Ground truth values on test set |
| `metrics.npy` | Array of [MAE, MSE, RMSE, MAPE, MSPE, DTW] |

### 9.3 Checkpoints

```
checkpoints/{task}_{data}_{model}_{model_id}_{seq_len}_{pred_len}_{iteration}/
└── checkpoint.pth
```

### 9.4 Visualization Outputs

```
visualization_outputs/{setting}/
├── peak_classification_sample_*.png
└── ...
```

---

## 10. Result Extraction & Aggregation

### 10.1 Calculate Mean Metrics Across Iterations

```bash
cd metric_tools/

# Single model
python calculate_mean_metrics.py \
  --task peak_detect_ltf \
  --data load_data_mixed \
  --model proposed_model \
  --input_type maxIn \
  --output_type maxOut \
  --loss MSE_244_23 \
  --seq_len 168 \
  --pred_len 336 \
  --results_dir ../results

# Batch calculation for all baselines
bash cal.sh
```

Output: Mean and standard deviation of all metrics across 5 iterations.

### 10.2 Extract Ablation Results

```bash
# Loss weight ablation
python extract_results.py --mode lossw

# Multi-scale ablation
python extract_results.py --mode nscales

# Mask type ablation
python extract_results.py --mode mask
```

### 10.3 Compute Additional Metrics

```bash
# R² score from saved predictions
python analysis/compute_r2.py

# Self-Consistency F1
python compute_sc_f1.py
```

---

## 11. Visualization

### 11.1 Attention Heatmap

Visualizes LAD cross-attention weights to show how peak features guide the decoder:

```bash
python visualize_attn_heatmap.py
```

Output: `visualization_outputs/model_internals/attn_heatmap_*.pdf`

### 11.2 Gate Activation

Visualizes the gate mechanism `G = tanh(H_pl) * sigmoid(X_en)`:

```bash
python visualize_gate.py
```

Output: `visualization_outputs/model_internals/gate_activation_*.pdf`

### 11.3 Combined Visualization

Generates comprehensive 4-panel figures (prediction + attention + heatmap + peak probability):

```bash
python visualize_gate_combined.py
```

Output: `visualization_outputs/model_internals/combined/gate_combined_*.pdf`

---

## 12. Key Parameters Reference

### 12.1 Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--d_model` | 256 | Hidden dimension |
| `--d_ff` | 256 | Feed-forward dimension |
| `--n_heads` | 4 | Number of attention heads |
| `--e_layers` | 1 | Encoder layers |
| `--mlp_layers` | 2 | MLP backbone depth (proposed_model) |
| `--if_lad` | 1 | Enable Location-Aware Decoder |
| `--if_msm_pl` | 1 | Enable Multi-Scale Mixing Peak Locator |
| `--n_scales` | 2 | Number of scales in MSM-PL |

### 12.2 Data & Forecasting

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seq_len` | 168 | Input sequence length (7 days) |
| `--label_len` | 48 | Label length for decoder (2 days) |
| `--pred_len` | 336/720 | Prediction horizon (14/30 days) |
| `--features` | S | Single-variate forecasting |
| `--input_col` | value_max | Input column name |
| `--target_col` | value_max | Target column name |

### 12.3 Peak Detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--peak_tolerance` | 1 | Matching tolerance (time steps) |
| `--peak_threshold` | 0.4 | Binarization threshold for peak logits |
| `--peak_lookahead` | 3 or 5 | findpeaks lookahead parameter |
| `--use_soft_labels` | 1 | Use Gaussian soft labels (vs. hard binary) |
| `--soft_label_sigma` | 2.0 | Gaussian smoothing sigma |
| `--enable_peak_eval` | 1 | Enable peak evaluation during test |

### 12.4 Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 128 | Training batch size |
| `--learning_rate` | 0.001 | Initial learning rate |
| `--lradj` | type3 | Learning rate scheduler type |
| `--train_epochs` | 20 | Maximum training epochs |
| `--patience` | 5 | Early stopping patience |
| `--itr` | 5 | Number of experiment iterations |

---

## 13. Quick Start

```bash
# 1. Setup environment
conda env create -f environment.yaml
conda activate tslib_findpeaks

# 2. Place dataset files in the correct directories
#    (see Section 3 for data format requirements)

# 3. Train PeakFocus on WLEL-336
bash scripts/peak_detect_based_on_LTF/load_data/test_proposed_model.sh

# 4. Train all baselines on WLEL
bash scripts/peak_detect_based_on_LTF/load_data/test_all.sh

# 5. Run ablation studies
bash scripts/peak_detect_based_on_LTF/ablation_n_scales.sh
bash scripts/peak_detect_based_on_LTF/ablation_mask_type.sh
bash scripts/peak_detect_based_on_LTF/ablation_loss_weights.sh
bash scripts/peak_detect_based_on_LTF/ablation_naive_baselines.sh

# 6. Aggregate results
cd metric_tools && bash cal.sh

# 7. Generate visualizations
python visualize_attn_heatmap.py
python visualize_gate_combined.py
```

---

## 14. Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{peakfocus2026,
  title={PeakFocus: An End-to-End Framework Bridging Localization and Regression for Electricity Load Peak Forecasting},
  author={},
  booktitle={},
  year={2026}
}
```

## Acknowledgement

This codebase is built upon [Time-Series-Library](https://github.com/thuml/Time-Series-Library) by THUML. We thank the authors for their excellent open-source contribution.
