"""
Run Phase-2 G0_full using Phase-1 ablated encoders (WLEL only).
"""

from __future__ import annotations
import sys

import os
import subprocess


PYTHON = sys.executable
P2_TRAIN = "patchevent/phase2/train.py"
SEEDS = [42, 123, 456]

SERIES = "dataset/wlel/event_v1/data/wlel_event_series_v1.csv"
EVENTS = "dataset/wlel/event_v1/data/wlel_events_v1.jsonl"

P1_BASE = "patchevent/phase1/checkpoints/wlel_ablation"
OUT_BASE = "patchevent/phase2/checkpoints/wlel_p1_ablation"

COMMON = [
    "--encoder_mode", "frozen",
    "--use_memory_pos",
    "--use_self_attn_agg",
    "--use_raw_bypass",
    "--use_int_ordinal_loss",
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
]

GROUPS = {
    "GP1a_wo_has_apex": "P1a_wo_has_apex",
    "GP1b_wo_distance": "P1b_wo_distance",
    "GP1c_wo_phase": "P1c_wo_phase",
    "GP1d_wo_offset": "P1d_wo_offset",
    "GP1e_generic_npp": "P1e_generic_npp",
}


def main():
    total = done = skip = fail = 0
    for gp_name, p1_name in GROUPS.items():
        for seed in SEEDS:
            total += 1
            out_dir = os.path.join(OUT_BASE, f"{gp_name}_s{seed}")
            if os.path.exists(os.path.join(out_dir, "test_summary.json")):
                print(f"[SKIP] {gp_name}_s{seed}")
                skip += 1
                continue
            enc_ckpt = os.path.join(P1_BASE, p1_name, f"s{seed}", "best_model.pth")
            if not os.path.exists(enc_ckpt):
                print(f"[FAIL] missing encoder ckpt: {enc_ckpt}")
                fail += 1
                continue
            cmd = [
                PYTHON, P2_TRAIN,
                "--encoder_ckpt", enc_ckpt,
                "--series_path", SERIES,
                "--events_path", EVENTS,
                "--output_dir", out_dir,
                "--seed", str(seed),
            ] + list(COMMON)
            print("\n" + "=" * 64)
            print(f"[START] {gp_name}_s{seed}")
            print(f" encoder: {enc_ckpt}")
            print(f" output : {out_dir}")
            print("=" * 64)
            ret = subprocess.run(cmd, cwd=os.getcwd())
            if ret.returncode != 0:
                print(f"[FAIL] {gp_name}_s{seed}")
                fail += 1
            else:
                done += 1

    print("\n" + "=" * 64)
    print(f"P1->P2 ablation summary: total={total} done={done} skip={skip} fail={fail}")
    print("=" * 64)


if __name__ == "__main__":
    main()

