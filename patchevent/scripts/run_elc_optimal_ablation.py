"""
ELC ablation with optimal weight w=2 and standardized config
(unfreeze_last_n=1, bs=32, no raw_bypass, no ordinal_loss)
"""
import sys
import subprocess, os, json, time, numpy as np

PYTHON = sys.executable
TRAIN_SCRIPT = "patchevent/phase2/train.py"
SEEDS = [42, 123, 456]
BASE_OUT = "patchevent/phase2/checkpoints/elc_optimal_ablation"
ENCODER_BASE = "patchevent/phase1/checkpoints/elc"
P1_ABLATION_BASE = "patchevent/phase1/checkpoints/elc_ablation"
SERIES = "dataset/electricity/event_v1/data/elc_event_series_v1.csv"
EVENTS = "dataset/electricity/event_v1/data/elc_events_v1.jsonl"
OPTIMAL_WEIGHTS = "2.0,0.5,2.0,0.5"

COMMON = [
    "--series_path", SERIES, "--events_path", EVENTS,
    "--embed_dropout", "0.15", "--label_smoothing", "0.1",
    "--attr_loss_weights", OPTIMAL_WEIGHTS,
    "--batch_size", "32", "--train_epochs", "50",
    "--lr", "5e-4", "--dropout", "0.2",
    "--pred_len", "96", "--seq_len", "96",
]

def run_one(cfg_name, extra_args, seed):
    out_dir = os.path.join(BASE_OUT, f"{cfg_name}_s{seed}")
    summary = os.path.join(out_dir, "test_summary.json")
    if os.path.exists(summary):
        with open(summary) as f:
            return json.load(f)
    encoder = os.path.join(ENCODER_BASE, f"s{seed}", "best_model.pth")
    if cfg_name.startswith("GP1"):
        p1_map = {"GP1a_wo_has_apex": "P1a_wo_has_apex", "GP1b_wo_distance": "P1b_wo_distance",
                   "GP1c_wo_phase": "P1c_wo_phase", "GP1d_wo_offset": "P1d_wo_offset",
                   "GP1e_generic_npp": "P1e_generic_npp"}
        p1_name = p1_map.get(cfg_name)
        if p1_name:
            encoder = os.path.join(P1_ABLATION_BASE, p1_name, f"s{seed}", "best_model.pth")
    if not os.path.exists(encoder):
        return None
    cmd = [PYTHON, "-u", TRAIN_SCRIPT, "--encoder_ckpt", encoder,
           "--seed", str(seed), "--output_dir", out_dir] + COMMON + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(summary):
        with open(summary) as f:
            return json.load(f)
    if result.stderr:
        for line in result.stderr.strip().split('\n')[-2:]:
            print(f"      {line}")
    return None

def run_3seed(cfg_name, extra_args):
    f1s = []
    for seed in SEEDS:
        d = run_one(cfg_name, extra_args, seed)
        if d:
            f1s.append(d["test_event_f1"])
    if len(f1s) == 3:
        return f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}"
    return f"FAIL({len(f1s)}/3)"

def main():
    print("=== ELC Architecture Ablation (w=2, standardized config) ===")
    arch_cfgs = {
        "G0_full": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1", "--use_memory_pos", "--use_self_attn_agg"],
        "GS1_wo_mempos": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1", "--use_self_attn_agg"],
        "GS2_wo_sa": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1", "--use_memory_pos"],
        "GS3_bare": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1"],
        "GS4_cnn": ["--encoder_mode", "cnn"],
        "GS4_lstm": ["--encoder_mode", "lstm"],
        "GS4_scratch": ["--encoder_mode", "scratch", "--unfreeze_last_n", "1", "--use_memory_pos", "--use_self_attn_agg"],
        "GS5_detr": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1", "--use_memory_pos", "--use_self_attn_agg", "--use_non_ar", "--max_events", "10"],
    }
    for name, args in arch_cfgs.items():
        print(f"  {name}: ", end="", flush=True)
        print(run_3seed(name, args))

    print("\n=== ELC AR Ablation ===")
    ar_cfgs = {
        "GD1_no_pos_mask": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1", "--use_memory_pos", "--use_self_attn_agg", "--no_pos_valid_mask"],
        "GD2_no_pos_smooth": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1", "--use_memory_pos", "--use_self_attn_agg", "--no_pos_aware_smoothing"],
        "GD3_no_attr_weight": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1", "--use_memory_pos", "--use_self_attn_agg"],
        "GD4_no_causal": ["--encoder_mode", "frozen", "--unfreeze_last_n", "1", "--use_memory_pos", "--use_self_attn_agg", "--no_causal_mask"],
    }
    for name, args in ar_cfgs.items():
        print(f"  {name}: ", end="", flush=True)
        print(run_3seed(name, args))

    print("\n=== ELC P1 Ablation ===")
    base_args = ["--encoder_mode", "frozen", "--unfreeze_last_n", "1", "--use_memory_pos", "--use_self_attn_agg"]
    for name in ["GP1a_wo_has_apex", "GP1b_wo_distance", "GP1c_wo_phase", "GP1d_wo_offset", "GP1e_generic_npp"]:
        print(f"  {name}: ", end="", flush=True)
        print(run_3seed(name, base_args))

    print("\nDone!")

if __name__ == "__main__":
    main()
