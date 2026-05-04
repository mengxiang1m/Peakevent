# Remote VM Commands

Run these on the remote GPU VM from the repository root. Do not run them on the local machine.

## 1. Single-batch smoke train

Use a tiny epoch/batch setting only to verify forward, Hungarian matching, backward, checkpoint save, and JSON evaluation plumbing.

```bash
python -u patchevent/phase2/train.py \
  --decoder_type hybrid \
  --series_path dataset/wlel/event_v1/data/wlel_event_series_v1.csv \
  --events_path dataset/wlel/event_v1/data/wlel_events_v1.jsonl \
  --output_dir patchevent/phase2/checkpoints/wlel_hybrid_v21_smoke/H0_v21_hybrid_s42 \
  --seq_len 96 --pred_len 96 \
  --patch_len 8 --patch_stride 4 \
  --d_model 128 --n_heads 4 --n_layers 3 --d_ff 256 \
  --backbone_layers 2 --refine_layers 1 \
  --hybrid_time_head structured --hybrid_use_count_head --hybrid_count_loss_weight 0.5 \
  --max_events 10 --batch_size 8 --eval_batch_size 16 \
  --train_epochs 1 --patience 1 --seed 42
```

## 2. WLEL single-seed hybrid train

```bash
python -u patchevent/phase2/train.py \
  --decoder_type hybrid \
  --series_path dataset/wlel/event_v1/data/wlel_event_series_v1.csv \
  --events_path dataset/wlel/event_v1/data/wlel_events_v1.jsonl \
  --output_dir patchevent/phase2/checkpoints/wlel_hybrid_v21_ablation/H0_v21_hybrid_s42 \
  --seq_len 96 --pred_len 96 \
  --patch_len 8 --patch_stride 4 \
  --d_model 128 --n_heads 4 --n_layers 3 --d_ff 256 \
  --backbone_layers 2 --refine_layers 1 \
  --hybrid_time_head structured --hybrid_use_count_head --hybrid_count_loss_weight 0.5 \
  --max_events 10 --batch_size 32 --eval_batch_size 128 \
  --train_epochs 50 --lr 5e-4 --weight_decay 0.05 \
  --seed 42
```

## 3. Internal structure ablations

These are not AR-vs-DETR baselines. They only analyze the internal Hybrid DETR-AR design.

```bash
# Query count: fewer queries
python -u patchevent/phase2/train.py --decoder_type hybrid \
  --series_path dataset/wlel/event_v1/data/wlel_event_series_v1.csv \
  --events_path dataset/wlel/event_v1/data/wlel_events_v1.jsonl \
  --output_dir patchevent/phase2/checkpoints/wlel_hybrid_v21_ablation/HQ8_queries_s42 \
  --max_events 8 --refine_layers 1 \
  --hybrid_time_head structured --hybrid_use_count_head --hybrid_count_loss_weight 0.5 \
  --seed 42

# Query count: more queries
python -u patchevent/phase2/train.py --decoder_type hybrid \
  --series_path dataset/wlel/event_v1/data/wlel_event_series_v1.csv \
  --events_path dataset/wlel/event_v1/data/wlel_events_v1.jsonl \
  --output_dir patchevent/phase2/checkpoints/wlel_hybrid_v21_ablation/HQ12_queries_s42 \
  --max_events 12 --refine_layers 1 \
  --hybrid_time_head structured --hybrid_use_count_head --hybrid_count_loss_weight 0.5 \
  --seed 42

# Refinement depth
python -u patchevent/phase2/train.py --decoder_type hybrid \
  --series_path dataset/wlel/event_v1/data/wlel_event_series_v1.csv \
  --events_path dataset/wlel/event_v1/data/wlel_events_v1.jsonl \
  --output_dir patchevent/phase2/checkpoints/wlel_hybrid_v21_ablation/HR0_no_refine_s42 \
  --max_events 10 --refine_layers 0 \
  --hybrid_time_head structured --hybrid_use_count_head --hybrid_count_loss_weight 0.5 \
  --seed 42

python -u patchevent/phase2/train.py --decoder_type hybrid \
  --series_path dataset/wlel/event_v1/data/wlel_event_series_v1.csv \
  --events_path dataset/wlel/event_v1/data/wlel_events_v1.jsonl \
  --output_dir patchevent/phase2/checkpoints/wlel_hybrid_v21_ablation/HR2_refine2_s42 \
  --max_events 10 --refine_layers 2 \
  --hybrid_time_head structured --hybrid_use_count_head --hybrid_count_loss_weight 0.5 \
  --seed 42

# Matching cost without intensity term
python -u patchevent/phase2/train.py --decoder_type hybrid \
  --series_path dataset/wlel/event_v1/data/wlel_event_series_v1.csv \
  --events_path dataset/wlel/event_v1/data/wlel_events_v1.jsonl \
  --output_dir patchevent/phase2/checkpoints/wlel_hybrid_v21_ablation/HM_no_intensity_cost_s42 \
  --max_events 10 --refine_layers 1 --hybrid_match_intensity_weight 0.0 \
  --hybrid_time_head structured --hybrid_use_count_head --hybrid_count_loss_weight 0.5 \
  --seed 42

# Proposal auxiliary loss disabled
python -u patchevent/phase2/train.py --decoder_type hybrid \
  --series_path dataset/wlel/event_v1/data/wlel_event_series_v1.csv \
  --events_path dataset/wlel/event_v1/data/wlel_events_v1.jsonl \
  --output_dir patchevent/phase2/checkpoints/wlel_hybrid_v21_ablation/HP_no_proposal_aux_s42 \
  --max_events 10 --refine_layers 1 --hybrid_proposal_loss_weight 0.0 \
  --hybrid_time_head structured --hybrid_use_count_head --hybrid_count_loss_weight 0.5 \
  --seed 42
```

## 4. Three-seed summary

Repeat each config with seeds `42`, `123`, and `456`, then aggregate:

```bash
python patchevent/phase2/collect_ablation_v2_results.py
```

## 5. Current Hybrid DETR-AR mainline runner

Use this script for the current end-to-end `HybridEventDecoder` v2.1. It replaces
legacy `GS/GD` switch-based ablations for the new mainline because the default
`decoder_type=hybrid` does not consume legacy AR options such as MemPos/SA
or position-valid masks.

```bash
python -u patchevent/phase2/scripts/run_hybrid_ablation.py
python patchevent/phase2/collect_ablation_v2_results.py
```

The expected configs are:

- `H0_v21_hybrid`
- `HQ8_queries`, `HQ12_queries`
- `HNC_no_count_head`, `HCD_count_train_only`, `HID_independent_time`
- `HR0_no_refine`, `HR2_refine2`
- `HO_query_order`
- `HC_no_causal_refine`
- `HM_no_intensity_cost`
- `HP_no_proposal_aux`
- `HN_noobj05`
- `HF_focal2`
- `HS_structure`

## 6. Validation-calibrated Hybrid summary

After Hybrid training finishes, run threshold/NMS calibration before collecting
the final table. The calibration script selects `exist_threshold` and
`hybrid_nms_onset_radius` on validation, then evaluates the chosen pair on test.

```bash
python -u patchevent/phase2/scripts/run_hybrid_calibration.py \
  --base_suffix hybrid_v21_ablation \
  --configs H0_v21_hybrid,HNC_no_count_head,HCD_count_train_only,HID_independent_time,HR0_no_refine,HO_query_order,HC_no_causal_refine,HM_no_intensity_cost,HP_no_proposal_aux \
  --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --nms_radii 0,1,2,3

python patchevent/phase2/collect_ablation_v2_results.py
```

For the optional effect-improvement configs:

```bash
python -u patchevent/phase2/scripts/run_hybrid_calibration.py \
  --base_suffix hybrid_v21_ablation \
  --configs HQ8_queries,HQ12_queries,HR2_refine2,HN_noobj05,HF_focal2,HS_structure \
  --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --nms_radii 0,1,2,3
```
