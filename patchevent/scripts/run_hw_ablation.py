"""
Batch runner: WLEL architecture + AR ablation with HighWeight (4,0.5,4,0.5)
"""
import subprocess
import sys
import os
import json
import time

PYTHON = sys.executable
TRAIN_SCRIPT = "patchevent/phase2/train.py"
BASE_DIR = "patchevent/phase2/checkpoints/wlel_hw_ablation"
ENCODER_BASE = "patchevent/phase1/checkpoints/wlel"
SERIES = "dataset/wlel/event_v1/data/wlel_event_series_v1.csv"
EVENTS = "dataset/wlel/event_v1/data/wlel_events_v1.jsonl"

SEEDS = [42, 123, 456]

# Common args
COMMON = [
    "--series_path", SERIES,
    "--events_path", EVENTS,
    "--embed_dropout", "0.15",
    "--label_smoothing", "0.1",
    "--batch_size", "32",
    "--train_epochs", "50",
    "--lr", "5e-4",
    "--pred_len", "96",
    "--seq_len", "96",
]

# HighWeight for all configs except GD3_no_attr_weight
HW_WEIGHTS = ["--attr_loss_weights", "4.0,0.5,4.0,0.5"]

# Architecture ablation configs
ARCH_CONFIGS = {
    "G0_full": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1", "--use_memory_pos", "--use_self_attn_agg"],
    "GS1_wo_mempos": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1", "--use_self_attn_agg"],
    "GS2_wo_sa": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1", "--use_memory_pos"],
    "GS3_bare": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1"],
    "GS4_cnn": ["--encoder_mode", "cnn"],
    "GS4_lstm": ["--encoder_mode", "lstm"],
    "GS4_scratch": ["--encoder_mode", "scratch", "--unfreeze_last_n", "1", "--use_memory_pos", "--use_self_attn_agg"],
}

# AR ablation configs
AR_CONFIGS = {
    "GD1_no_pos_mask": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1", "--use_memory_pos", "--use_self_attn_agg", "--no_pos_valid_mask"],
    "GD2_no_pos_smooth": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1", "--use_memory_pos", "--use_self_attn_agg", "--no_pos_aware_smoothing"],
    "GD3_no_attr_weight": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1", "--use_memory_pos", "--use_self_attn_agg"],
    "GD4_no_causal": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1", "--use_memory_pos", "--use_self_attn_agg", "--no_causal_mask"],
}

# DETR (non-AR, separate)
DETR_CONFIG = {
    "GS5_detr": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1", "--use_memory_pos", "--use_self_attn_agg", "--use_non_ar", "--max_events", "10"],
}

def run_one(cfg_name, extra_args, seed, use_hw=True):
    out_dir = os.path.join(BASE_DIR, f"{cfg_name}_s{seed}")
    if os.path.exists(os.path.join(out_dir, "test_summary.json")):
        print(f"  SKIP {cfg_name}_s{seed} (already exists)")
        return True

    encoder_ckpt = os.path.join(ENCODER_BASE, f"s{seed}", "best_model.pth")
    cmd = [PYTHON, "-u", TRAIN_SCRIPT, "--encoder_ckpt", encoder_ckpt] + COMMON + extra_args
    if use_hw:
        cmd += HW_WEIGHTS
    cmd += ["--seed", str(seed), "--output_dir", out_dir]

    print(f"  RUN {cfg_name}_s{seed} ...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - t0

    # Check result
    summary_path = os.path.join(out_dir, "test_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            d = json.load(f)
        f1 = d.get("test_event_f1", 0)
        print(f"  DONE {cfg_name}_s{seed}: F1={f1:.4f} ({elapsed:.1f}s)")
        return True
    else:
        print(f"  FAIL {cfg_name}_s{seed} ({elapsed:.1f}s)")
        if result.stderr:
            # Print last 5 lines of stderr
            lines = result.stderr.strip().split('\n')
            for line in lines[-5:]:
                print(f"    {line}")
        return False


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    total = 0
    success = 0
    failed = []

    # Architecture ablation
    print("=== Architecture Ablation (HighWeight) ===")
    for cfg_name, flags in ARCH_CONFIGS.items():
        for seed in SEEDS:
            total += 1
            if run_one(cfg_name, flags, seed, use_hw=True):
                success += 1
            else:
                failed.append(f"{cfg_name}_s{seed}")

    # DETR (still uses HW for the CE loss part)
    print("\n=== DETR ===")
    for cfg_name, flags in DETR_CONFIG.items():
        for seed in SEEDS:
            total += 1
            if run_one(cfg_name, flags, seed, use_hw=True):
                success += 1
            else:
                failed.append(f"{cfg_name}_s{seed}")

    # AR ablation
    print("\n=== AR Ablation (HighWeight) ===")
    for cfg_name, flags in AR_CONFIGS.items():
        # GD3_no_attr_weight: NO attr_loss_weights (that's the point of this ablation)
        use_hw = (cfg_name != "GD3_no_attr_weight")
        for seed in SEEDS:
            total += 1
            if run_one(cfg_name, flags, seed, use_hw=use_hw):
                success += 1
            else:
                failed.append(f"{cfg_name}_s{seed}")

    print(f"\n=== Summary: {success}/{total} succeeded ===")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
