from __future__ import annotations
import sys

import os
import subprocess


PYTHON = sys.executable
TRAIN_SCRIPT = "patchevent/phase2/train.py"
SEEDS = [42, 123, 456]

DOMAINS = {
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
    "wlel": {
        "series_path": "dataset/wlel/event_v1/data/wlel_event_series_v1.csv",
        "events_path": "dataset/wlel/event_v1/data/wlel_events_v1.jsonl",
        "encoder_dir": "patchevent/phase1/checkpoints/wlel",
    },
}

HORIZONS = {
    96: {
        "max_new_tokens": 60,
        "max_seq_len": 64,
    },
    168: {
        "max_new_tokens": 60,
        "max_seq_len": 96,
    },
    336: {
        "max_new_tokens": 120,
        "max_seq_len": 160,
    },
}

COMMON = [
    "--encoder_mode", "frozen",
    "--use_memory_pos",
    "--use_self_attn_agg",
    "--unfreeze_last_n", "1",
    "--encoder_lr", "1e-5",
    "--d_model", "128",
    "--n_heads", "4",
    "--n_layers", "3",
    "--d_ff", "256",
    "--dropout", "0.2",
    "--embed_dropout", "0.15",
    "--lr", "5e-4",
    "--weight_decay", "0.05",
    "--label_smoothing", "0.1",
    "--attr_loss_weights", "2.0,0.5,2.0",
    "--value_loss_weight", "1.0",
    "--train_epochs", "50",
    "--patience", "15",
    "--batch_size", "64",
    "--eval_batch_size", "128",
    "--warmup_steps", "200",
    "--max_grad_norm", "1.0",
    "--seq_len", "96",
    "--window_stride", "4",
    "--tolerance", "3",
    "--gpu", "0",
    "--use_raw_bypass",
    "--use_tuple_guided_value",
    "--event_schema", "triplet",
    "--intensity_from_values", "apex",
]


def is_done(out_dir: str) -> bool:
    return os.path.exists(os.path.join(out_dir, "test_summary.json"))


def main():
    total = done = skip = fail = 0
    for pred_len, cfg_h in HORIZONS.items():
        for domain, cfg_d in DOMAINS.items():
            out_base = f"patchevent/phase2/checkpoints/{domain}_pred{pred_len}_hybrid"
            os.makedirs(out_base, exist_ok=True)
            for seed in SEEDS:
                total += 1
                out_dir = os.path.join(out_base, f"TG1_triplet_guided_s{seed}")
                if is_done(out_dir):
                    print(f"[SKIP] {domain}/pred{pred_len}/TG1_triplet_guided_s{seed}")
                    skip += 1
                    continue
                enc_ckpt = os.path.join(cfg_d["encoder_dir"], f"s{seed}", "best_model.pth")
                cmd = [
                    PYTHON, TRAIN_SCRIPT,
                    "--encoder_ckpt", enc_ckpt,
                    "--series_path", cfg_d["series_path"],
                    "--events_path", cfg_d["events_path"],
                    "--output_dir", out_dir,
                    "--seed", str(seed),
                    "--pred_len", str(pred_len),
                    "--max_new_tokens", str(cfg_h["max_new_tokens"]),
                    "--max_seq_len", str(cfg_h["max_seq_len"]),
                ] + list(COMMON)

                print("\n" + "=" * 72)
                print(f"[START] {domain}/pred{pred_len}/TG1_triplet_guided_s{seed}")
                print(f" output: {out_dir}")
                print("=" * 72)
                ret = subprocess.run(cmd, cwd=os.getcwd())
                if ret.returncode != 0:
                    print(f"[FAIL] {domain}/pred{pred_len}/TG1_triplet_guided_s{seed}")
                    fail += 1
                else:
                    done += 1

    print("\n" + "=" * 72)
    print(f"hybrid summary: total={total} done={done} skip={skip} fail={fail}")
    print("=" * 72)


if __name__ == "__main__":
    main()
