"""
AR internal ablation on top of G0G1G2_v2 baseline.

All configs include:
  --use_raw_bypass --use_int_ordinal_loss
"""

from __future__ import annotations
import sys

import os
import subprocess


PYTHON = sys.executable
TRAIN_SCRIPT = "patchevent/phase2/train.py"
SEEDS = [42, 123, 456]

DOMAINS = {
    "wlel": {
        "series_path": "dataset/wlel/event_v1/data/wlel_event_series_v1.csv",
        "events_path": "dataset/wlel/event_v1/data/wlel_events_v1.jsonl",
        "encoder_dir": "patchevent/phase1/checkpoints/wlel",
    },
    "ett": {
        "series_path": "dataset/ett/event_v1/data/ett_event_series_v1.csv",
        "events_path": "dataset/ett/event_v1/data/ett_events_v1.jsonl",
        "encoder_dir": "patchevent/phase1/checkpoints/ett",
    },
    "elc": {
        "series_path": "dataset/electricity/event_v1/data/elc_event_series_v1.csv",
        "events_path": "dataset/electricity/event_v1/data/elc_events_v1.jsonl",
        "encoder_dir": "patchevent/phase1/checkpoints/elc",
    },
}

OUT_BASE_TMPL = "patchevent/phase2/checkpoints/{domain}_ar_ablation_v2"

COMMON = [
    "--encoder_mode", "frozen",
    "--use_memory_pos",
    "--use_self_attn_agg",
    "--d_model", "128",
    "--n_heads", "4",
    "--n_layers", "3",
    "--d_ff", "256",
    "--dropout", "0.2",
    "--embed_dropout", "0.15",
    "--lr", "5e-4",
    "--weight_decay", "0.05",
    "--label_smoothing", "0.1",
    "--attr_loss_weights", "2.0,0.5,2.0,0.5",
    "--train_epochs", "50",
    "--patience", "15",
    "--batch_size", "64",
    "--eval_batch_size", "128",
    "--warmup_steps", "200",
    "--max_grad_norm", "1.0",
    "--seq_len", "96",
    "--pred_len", "96",
    "--window_stride", "4",
    "--max_new_tokens", "60",
    "--tolerance", "3",
    "--gpu", "0",
    "--use_raw_bypass",
    "--use_int_ordinal_loss",
]

AR_CONFIGS = {
    "GD1_no_pos_mask": ["--no_pos_valid_mask"],
    "GD2_no_pos_smooth": ["--no_pos_aware_smoothing"],
    "GD3_no_attr_weight": [],
    "GD4_no_causal": ["--no_causal_mask"],
}


def is_done(out_dir: str) -> bool:
    return os.path.exists(os.path.join(out_dir, "test_summary.json"))


def build_base_for(cfg_name: str) -> list[str]:
    base = list(COMMON)
    if cfg_name == "GD3_no_attr_weight":
        idx = base.index("--attr_loss_weights")
        base.pop(idx)
        base.pop(idx)
    return base


def main():
    total = done = skip = fail = 0
    for domain, cfg in DOMAINS.items():
        out_base = OUT_BASE_TMPL.format(domain=domain)
        os.makedirs(out_base, exist_ok=True)
        for name, extra in AR_CONFIGS.items():
            for seed in SEEDS:
                total += 1
                out_dir = os.path.join(out_base, f"{name}_s{seed}")
                if is_done(out_dir):
                    print(f"[SKIP] {domain}/{name}_s{seed}")
                    skip += 1
                    continue
                enc_ckpt = os.path.join(cfg["encoder_dir"], f"s{seed}", "best_model.pth")
                cmd = [
                    PYTHON, TRAIN_SCRIPT,
                    "--encoder_ckpt", enc_ckpt,
                    "--series_path", cfg["series_path"],
                    "--events_path", cfg["events_path"],
                    "--output_dir", out_dir,
                    "--seed", str(seed),
                ] + build_base_for(name) + list(extra)

                print("\n" + "=" * 64)
                print(f"[START] {domain}/{name}_s{seed}")
                print(f" output: {out_dir}")
                print("=" * 64)
                ret = subprocess.run(cmd, cwd=os.getcwd())
                if ret.returncode != 0:
                    print(f"[FAIL] {domain}/{name}_s{seed}")
                    fail += 1
                else:
                    done += 1

    print("\n" + "=" * 64)
    print(f"AR v2 summary: total={total} done={done} skip={skip} fail={fail}")
    print("=" * 64)


if __name__ == "__main__":
    main()

