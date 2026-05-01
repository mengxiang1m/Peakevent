"""
Batch runner: Phase1 ablation with HighWeight (4,0.5,4,0.5) on WLEL and ELC
(ETT skipped because HighWeight has no effect there)
"""
import subprocess
import sys
import os
import json
import time

PYTHON = sys.executable
TRAIN_SCRIPT = "patchevent/phase2/train.py"
SEEDS = [42, 123, 456]

DOMAINS = {
    "wlel": {
        "series": "dataset/wlel/event_v1/data/wlel_event_series_v1.csv",
        "events": "dataset/wlel/event_v1/data/wlel_events_v1.jsonl",
        "encoder_base": "patchevent/phase1/checkpoints/wlel",
        "p1_ablation_base": "patchevent/phase1/checkpoints/wlel_ablation",
    },
    "elc": {
        "series": "dataset/electricity/event_v1/data/elc_event_series_v1.csv",
        "events": "dataset/electricity/event_v1/data/elc_events_v1.jsonl",
        "encoder_base": "patchevent/phase1/checkpoints/elc",
        "p1_ablation_base": "patchevent/phase1/checkpoints/elc_ablation",
    },
}

COMMON = [
    "--embed_dropout", "0.15",
    "--label_smoothing", "0.1",
    "--attr_loss_weights", "4.0,0.5,4.0,0.5",
    "--batch_size", "32",
    "--train_epochs", "50",
    "--lr", "5e-4",
    "--pred_len", "96",
    "--seq_len", "96",
    "--encoder_mode", "frozen",
    "--unfreeze_last_n", "1",
    "--use_memory_pos",
    "--use_self_attn_agg",
]

# Phase1 ablation configs: which encoder checkpoint to use
P1_CONFIGS = {
    "GP1a_wo_has_apex": "P1a_wo_has_apex",
    "GP1b_wo_distance": "P1b_wo_distance",
    "GP1c_wo_phase": "P1c_wo_phase",
    "GP1d_wo_offset": "P1d_wo_offset",
    "GP1e_generic_npp": "P1e_generic_npp",
}

def find_encoder_ckpt(domain_info, ablation_name, seed):
    """Find the Phase1 ablated encoder checkpoint."""
    # Direct path: {p1_ablation_base}/{ablation_name}/s{seed}/best_model.pth
    p = os.path.join(domain_info["p1_ablation_base"], ablation_name, f"s{seed}", "best_model.pth")
    if os.path.exists(p):
        return p
    return None


def run_one(cfg_name, domain, seed):
    out_base = f"patchevent/phase2/checkpoints/{domain}_hw_p1_ablation"
    out_dir = os.path.join(out_base, f"{cfg_name}_s{seed}")

    if os.path.exists(os.path.join(out_dir, "test_summary.json")):
        print(f"  SKIP {domain}/{cfg_name}_s{seed}")
        return True

    domain_info = DOMAINS[domain]

    if cfg_name == "G0_full":
        enc_ckpt = os.path.join(domain_info["encoder_base"], f"s{seed}", "best_model.pth")
    else:
        ablation_name = P1_CONFIGS[cfg_name]
        enc_ckpt = find_encoder_ckpt(domain_info, ablation_name, seed)

    if enc_ckpt is None or not os.path.exists(enc_ckpt):
        print(f"  SKIP {domain}/{cfg_name}_s{seed} (encoder not found)")
        return False

    cmd = [PYTHON, "-u", TRAIN_SCRIPT,
           "--encoder_ckpt", enc_ckpt,
           "--series_path", domain_info["series"],
           "--events_path", domain_info["events"],
           "--seed", str(seed),
           "--output_dir", out_dir] + COMMON

    print(f"  RUN {domain}/{cfg_name}_s{seed} ...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - t0

    summary_path = os.path.join(out_dir, "test_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            d = json.load(f)
        f1 = d.get("test_event_f1", 0)
        print(f"  DONE {domain}/{cfg_name}_s{seed}: F1={f1:.4f} ({elapsed:.1f}s)")
        return True
    else:
        print(f"  FAIL {domain}/{cfg_name}_s{seed} ({elapsed:.1f}s)")
        if result.stderr:
            for line in result.stderr.strip().split('\n')[-3:]:
                print(f"    {line}")
        return False


def main():
    total = 0
    success = 0

    for domain in ["wlel", "elc"]:
        print(f"\n=== {domain.upper()} Phase1 Ablation (HighWeight) ===")
        # G0_full baseline
        for seed in SEEDS:
            total += 1
            if run_one("G0_full", domain, seed):
                success += 1
        # P1 ablation variants
        for cfg_name in P1_CONFIGS:
            for seed in SEEDS:
                total += 1
                if run_one(cfg_name, domain, seed):
                    success += 1

    print(f"\n=== Summary: {success}/{total} succeeded ===")


if __name__ == "__main__":
    main()
