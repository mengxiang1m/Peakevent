"""
Phase-1 subtask ablation for ETT and ELC.
Mirrors run_p1_subtask_ablation.py (WLEL only) but for the other two domains.

5 ablations x 3 seeds x 2 domains = 30 experiments.
"""

from __future__ import annotations
import sys

import os
import subprocess

PYTHON = sys.executable
P1_TRAIN = "patchevent/phase1/train.py"
SEEDS = [42, 123, 456]
DOMAINS = ["ett", "elc"]

COMMON = [
    "--train_mode", "single",
    "--seq_len", "96",
    "--patch_len", "8",
    "--stride", "4",
    "--d_model", "128",
    "--n_heads", "4",
    "--e_layers", "2",
    "--d_ff", "256",
    "--dropout", "0.1",
    "--learning_rate", "1e-3",
    "--weight_decay", "1e-4",
    "--batch_size", "64",
    "--train_epochs", "50",
    "--patience", "5",
    "--lradj", "cosine",
    "--log1p_d", "1",
    "--use_gpu", "1",
]

ABLS = {
    "P1a_wo_has_apex": ["--lambda_bce", "0.0", "--lambda_d", "0.5", "--lambda_kl", "1.0", "--lambda_off", "0.5"],
    "P1b_wo_distance": ["--lambda_bce", "1.0", "--lambda_d", "0.0", "--lambda_kl", "1.0", "--lambda_off", "0.5"],
    "P1c_wo_phase": ["--lambda_bce", "1.0", "--lambda_d", "0.5", "--lambda_kl", "0.0", "--lambda_off", "0.5"],
    "P1d_wo_offset": ["--lambda_bce", "1.0", "--lambda_d", "0.5", "--lambda_kl", "1.0", "--lambda_off", "0.0"],
    "P1e_generic_npp": ["--lambda_bce", "0.0", "--lambda_d", "0.0", "--lambda_kl", "0.0", "--lambda_off", "0.0", "--lambda_npp", "0.5"],
}


def main():
    total = done = skip = fail = 0
    for domain in DOMAINS:
        out_base = f"patchevent/phase1/checkpoints/{domain}_ablation"
        for name, extra in ABLS.items():
            for seed in SEEDS:
                total += 1
                out_dir = os.path.join(out_base, name, f"s{seed}")
                ckpt = os.path.join(out_dir, "best_model.pth")
                if os.path.exists(ckpt):
                    print(f"[SKIP] {domain}/{name}/s{seed}")
                    skip += 1
                    continue
                os.makedirs(out_dir, exist_ok=True)
                cmd = [
                    PYTHON, P1_TRAIN,
                    "--single_domain", domain,
                    "--output_dir", out_dir,
                    "--seed", str(seed),
                ] + list(COMMON) + list(extra)
                print("\n" + "=" * 64)
                print(f"[START] {domain}/{name}/s{seed}")
                print(f" output: {out_dir}")
                print("=" * 64)
                ret = subprocess.run(cmd, cwd=os.getcwd())
                if ret.returncode != 0:
                    print(f"[FAIL] {domain}/{name}/s{seed}")
                    fail += 1
                else:
                    done += 1

    print("\n" + "=" * 64)
    print(f"P1 ablation ETT+ELC summary: total={total} done={done} skip={skip} fail={fail}")
    print("=" * 64)


if __name__ == "__main__":
    main()
