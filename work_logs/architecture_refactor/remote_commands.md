# Remote VM Commands

Run these on the remote GPU VM from the repository root. Do not run them on the local machine.

## 1. Single-batch smoke train

Use a tiny epoch/batch setting only to verify forward, Hungarian matching, backward, checkpoint save, and JSON evaluation plumbing.

```bash
python -u patchevent/phase2/train.py \
  --decoder_type hybrid \
  --series_path dataset/wlel/event_v1/data/wlel_event_series_v1.csv \
  --events_path dataset/wlel/event_v1/data/wlel_events_v1.jsonl \
  --output_dir patchevent/phase2/checkpoints/wlel_hybrid_smoke/H0_hybrid_s42 \
  --seq_len 96 --pred_len 96 \
  --patch_len 8 --patch_stride 4 \
  --d_model 128 --n_heads 4 --n_layers 3 --d_ff 256 \
  --backbone_layers 2 --refine_layers 1 \
  --max_events 10 --batch_size 8 --eval_batch_size 16 \
  --train_epochs 1 --patience 1 --seed 42
```

## 2. WLEL single-seed hybrid train

```bash
python -u patchevent/phase2/train.py \
  --decoder_type hybrid \
  --series_path dataset/wlel/event_v1/data/wlel_event_series_v1.csv \
  --events_path dataset/wlel/event_v1/data/wlel_events_v1.jsonl \
  --output_dir patchevent/phase2/checkpoints/wlel_hybrid_ablation/H0_hybrid_s42 \
  --seq_len 96 --pred_len 96 \
  --patch_len 8 --patch_stride 4 \
  --d_model 128 --n_heads 4 --n_layers 3 --d_ff 256 \
  --backbone_layers 2 --refine_layers 1 \
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
  --output_dir patchevent/phase2/checkpoints/wlel_hybrid_ablation/HQ8_queries_s42 \
  --max_events 8 --refine_layers 1 --seed 42

# Query count: more queries
python -u patchevent/phase2/train.py --decoder_type hybrid \
  --series_path dataset/wlel/event_v1/data/wlel_event_series_v1.csv \
  --events_path dataset/wlel/event_v1/data/wlel_events_v1.jsonl \
  --output_dir patchevent/phase2/checkpoints/wlel_hybrid_ablation/HQ12_queries_s42 \
  --max_events 12 --refine_layers 1 --seed 42

# Refinement depth
python -u patchevent/phase2/train.py --decoder_type hybrid \
  --series_path dataset/wlel/event_v1/data/wlel_event_series_v1.csv \
  --events_path dataset/wlel/event_v1/data/wlel_events_v1.jsonl \
  --output_dir patchevent/phase2/checkpoints/wlel_hybrid_ablation/HR0_no_refine_s42 \
  --max_events 10 --refine_layers 0 --seed 42

python -u patchevent/phase2/train.py --decoder_type hybrid \
  --series_path dataset/wlel/event_v1/data/wlel_event_series_v1.csv \
  --events_path dataset/wlel/event_v1/data/wlel_events_v1.jsonl \
  --output_dir patchevent/phase2/checkpoints/wlel_hybrid_ablation/HR2_refine2_s42 \
  --max_events 10 --refine_layers 2 --seed 42

# Matching cost without intensity term
python -u patchevent/phase2/train.py --decoder_type hybrid \
  --series_path dataset/wlel/event_v1/data/wlel_event_series_v1.csv \
  --events_path dataset/wlel/event_v1/data/wlel_events_v1.jsonl \
  --output_dir patchevent/phase2/checkpoints/wlel_hybrid_ablation/HM_no_intensity_cost_s42 \
  --max_events 10 --refine_layers 1 --hybrid_match_intensity_weight 0.0 --seed 42

# Proposal auxiliary loss disabled
python -u patchevent/phase2/train.py --decoder_type hybrid \
  --series_path dataset/wlel/event_v1/data/wlel_event_series_v1.csv \
  --events_path dataset/wlel/event_v1/data/wlel_events_v1.jsonl \
  --output_dir patchevent/phase2/checkpoints/wlel_hybrid_ablation/HP_no_proposal_aux_s42 \
  --max_events 10 --refine_layers 1 --hybrid_proposal_loss_weight 0.0 --seed 42
```

## 4. Three-seed summary

Repeat each config with seeds `42`, `123`, and `456`, then aggregate:

```bash
python patchevent/phase2/collect_ablation_v2_results.py
```
