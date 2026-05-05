"""Run threshold/NMS calibration for completed Hybrid DETR-AR experiments."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


PYTHON = sys.executable
SWEEP_SCRIPT = "patchevent/phase2/scripts/sweep_hybrid_thresholds.py"
SEEDS = [42, 123, 456]
DOMAINS = ["wlel", "ett", "elc"]
CONFIGS = [
    "H0_hybrid",
    "HQ8_queries",
    "HQ12_queries",
    "HR0_no_refine",
    "HR2_refine2",
    "HO_query_order",
    "HC_no_causal_refine",
    "HM_no_intensity_cost",
    "HP_no_proposal_aux",
    "HN_noobj05",
    "HF_focal2",
    "HT_count_head",
    "HS_structure",
]
CONFIGS_V21 = [
    "H0_v21_hybrid",
    "HQ8_queries",
    "HQ12_queries",
    "HNC_no_count_head",
    "HCD_count_train_only",
    "HID_independent_time",
    "HID_independent_time_nocount",
    "HR0_no_refine",
    "HR2_refine2",
    "HO_query_order",
    "HC_no_causal_refine",
    "HM_no_intensity_cost",
    "HP_no_proposal_aux",
    "HN_noobj05",
    "HF_focal2",
    "HS_structure",
]
CONFIGS_V22 = [
    "H0_v22_hybrid",
    "HQ8_queries",
    "HQ12_queries",
    "HCT_count_aux_only",
    "HCD_count_decode",
    "HID_independent_time_nocount",
    "HR0_no_refine",
    "HR2_refine2",
    "HO_query_order",
    "HMO_matched_order",
    "HC_no_causal_refine",
    "HM_no_intensity_cost",
    "HP_no_proposal_aux",
    "HN_noobj05",
    "HF_focal2",
    "HS_structure",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-run Hybrid threshold/NMS calibration.")
    p.add_argument("--configs", default=None, help="Comma-separated config subset")
    p.add_argument("--domains", default=None, help="Comma-separated domain subset")
    p.add_argument("--seeds", default=None, help="Comma-separated seed subset")
    p.add_argument("--thresholds", default=None, help="Forwarded to sweep_hybrid_thresholds.py")
    p.add_argument("--nms_radii", default=None, help="Forwarded to sweep_hybrid_thresholds.py")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--base_suffix", default="hybrid_ablation",
                   choices=["hybrid_ablation", "hybrid_v21_ablation", "hybrid_v22_ablation"],
                   help="Checkpoint directory suffix after domain name")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _split_or_default(text: str | None, default: list[str]) -> list[str]:
    if not text:
        return list(default)
    return [v.strip() for v in text.split(",") if v.strip()]


def _split_int_or_default(text: str | None, default: list[int]) -> list[int]:
    if not text:
        return list(default)
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def main() -> None:
    args = parse_args()
    if args.base_suffix == "hybrid_v22_ablation":
        default_configs = CONFIGS_V22
    elif args.base_suffix == "hybrid_v21_ablation":
        default_configs = CONFIGS_V21
    else:
        default_configs = CONFIGS
    configs = _split_or_default(args.configs, default_configs)
    domains = _split_or_default(args.domains, DOMAINS)
    seeds = _split_int_or_default(args.seeds, SEEDS)
    total = done = skip = missing = fail = 0

    for domain in domains:
        base = f"patchevent/phase2/checkpoints/{domain}_{args.base_suffix}"
        for config in configs:
            for seed in seeds:
                total += 1
                run_dir = os.path.join(base, f"{config}_s{seed}")
                ckpt = os.path.join(run_dir, "best.pth")
                out_json = os.path.join(run_dir, "calibrated_eval", "threshold_sweep_test.json")
                if not os.path.exists(ckpt):
                    print(f"[MISSING] {run_dir}")
                    missing += 1
                    continue
                if os.path.exists(out_json) and not args.overwrite:
                    print(f"[SKIP] {run_dir}")
                    skip += 1
                    continue

                cmd = [
                    PYTHON,
                    SWEEP_SCRIPT,
                    "--checkpoint_dir",
                    run_dir,
                    "--gpu",
                    str(args.gpu),
                ]
                if args.thresholds:
                    cmd += ["--thresholds", args.thresholds]
                if args.nms_radii:
                    cmd += ["--nms_radii", args.nms_radii]

                print("\n" + "=" * 72)
                print(f"[CALIBRATE] {domain}/{config}_s{seed}")
                print("=" * 72)
                ret = subprocess.run(cmd, cwd=os.getcwd())
                if ret.returncode != 0:
                    print(f"[FAIL] {run_dir}")
                    fail += 1
                else:
                    done += 1

    print("\n" + "=" * 72)
    print(f"CALIBRATION summary: total={total} done={done} skip={skip} missing={missing} fail={fail}")
    print("=" * 72)


if __name__ == "__main__":
    main()
