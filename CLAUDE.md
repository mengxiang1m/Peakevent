# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PatchEvent: a two-phase encoder-decoder framework for **time series event prediction** on industrial data (electricity load, transformer temperature, consumption). Predicts structured events (onset, duration, apex, intensity) — not point values — using an event-aware patch encoder (Phase 1) followed by an autoregressive Transformer decoder (Phase 2).

Language: Chinese documentation, English code. Respond in Chinese when discussing with the user.

## Environment & Setup

```bash
conda activate tslib_findpeaks  # Python 3.12, PyTorch 2.8.0, CUDA 12.8
```

## Common Commands

### Phase 1: Encoder Pretraining

```bash
# Single-domain (recommended): trains on wlel/ett/elc × 3 seeds
bash patchevent/phase1/scripts/train_single_domain.sh

# Multi-domain joint training with NPP regularization
bash patchevent/phase1/scripts/train_multi_domain.sh

# Manual single run
python patchevent/phase1/train.py \
  --train_mode single --single_domain wlel \
  --seed 42 --output_dir patchevent/phase1/checkpoints/wlel/s42 \
  --train_epochs 50 --batch_size 64 --learning_rate 1e-3
```

### Phase 2: Decoder Training

```bash
python patchevent/phase2/train.py \
  --encoder_ckpt patchevent/phase1/checkpoints/wlel/s42/best_model.pth \
  --encoder_mode frozen \
  --batch_size 32 --train_epochs 100 --learning_rate 5e-4 --seed 42
```

Encoder modes: `frozen` (default, best), `scratch`, `cnn`, `lstm`, `mlp`.

### Ablation Scripts

```bash
python patchevent/phase2/scripts/run_ar_ablation_v2.py      # AR internal mechanisms
python patchevent/phase2/scripts/run_arch_ablation_v2.py     # Architecture variations
python patchevent/phase2/scripts/run_p1_ablation_phase2.py   # Phase 1 task importance
python patchevent/phase2/scripts/run_pred336.py              # Extended horizons
```

### Evaluation

Phase 2 metrics: F1, Precision, Recall, OnsetMAE, DurationMAE, ApexMAE, IntensityMAE. Event matching uses the Hungarian algorithm with tolerance-based TP (apex within 3h).

## Architecture

### Two-Phase Pipeline

```
Phase 1 (Pretrain)                     Phase 2 (Decode)
────────────────────                   ────────────────────
Input (B,96)                           Frozen Phase1 Encoder
  → Unfold patches (B,23,8)              → MemoryBridge (MemPos + SA Agg)
  → Linear(8→128) + sinusoidal PE          → SmallPatchDecoder (3-layer causal Transformer)
  → 2-layer Transformer Encoder              → Cross-attention to encoder memory
  → 4 heads: has_apex, d_to_apex,              → Autoregressive token generation
    phase_dist, apex_offset                      → [BOS, onset, dur, apex, int, ..., EOS]
```

### Key Design Decisions

- **MemoryBridge** is the most architecturally critical component. MemPos re-injects position info lost through encoder attention layers. Removing it drops F1 by 2-6%.
- **StructuredEventTokenizer**: 298-token vocabulary with positional constraints — token position k mod 4 determines attribute type (onset/duration/apex/intensity).
- **Position-valid masking** enforces structural constraints during decoding. Without it, F1 drops to near zero.
- **AR decoding** significantly outperforms DETR-style parallel decoding (F1 +3-22%, OnsetMAE −53-76%).
- Phase 1 pretraining provides a timing localization prior; without it, OnsetMAE degrades 31-41%.

### Key Hyperparameters

| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| seq_len | 96 | 96 |
| patch_len / stride | 8 / 4 (→ 23 patches) | inherited |
| d_model | 128 | 128 |
| n_heads | 4 | 4 |
| layers | 2 (encoder) | 3 (decoder) |
| lr | 1e-3 | 5e-4 (decoder), 1e-5 (encoder last layer) |
| optimizer | Adam (wd=1e-4) | Adam |

### Datasets

Three industrial domains: **WLEL** (power load), **ETT** (transformer temperature), **ELC** (electricity consumption). Per-dataset z-score normalization. Data lives in `dataset/`.

## Directory Layout

- `patchevent/` — Main project (self-contained)
  - `phase1/` — Encoder pretraining: model, loss, training, evaluation, scripts
  - `phase2/` — Decoder: model, tokenizer, training, ablation runners, scripts
  - `models/` — Shared model components (SmallPatchDecoder, encoders, tokenizer, etc.)
  - `losses/` — Shared loss functions (hungarian matching, intensity, value, etc.)
  - `training/` — Shared training engine and eval loop
  - `data/event_pipeline/` — Shared event detection, labeling, preprocessing
  - `dataset/` — Data building scripts (build_*_events.py)
  - `visualization/` — Plotting and visualization scripts
  - `configs/` — YAML configuration files (base + datasets + experiments)
  - `scripts/` — Cross-phase experiment runners (grid search, ablation, etc.)
- `baselines/` — Comparison experiments
  - `tslib/` — Time-Series-Library based baselines (PatchTST, iTransformer, etc.)
  - `seq2peak/` — Seq2Peak baseline (CIKM 2023)
  - `posthoc_evaluate.py` — Post-hoc event evaluation for baselines
- `dataset/` — Raw data files (WLEL, ETT, ELC)
- `papers/` — Paper LaTeX sources (Overleaf repos, separate git)

## Work Principles

- Explore code before modifying — search first, never blind-edit
- Check cross-file dependencies when modifying (encoder↔decoder, loss↔model)
- Save experiment checkpoints in structured folders; include summary markdown and code snapshots
- Root-cause analysis over surface-level patching

<!-- Claude + Codex collaboration protocol is defined in ~/.claude/CLAUDE.md (global rules) -->
