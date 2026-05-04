"""Run Hybrid DETR-AR mainline ablations.

These configs target the current end-to-end `HybridEventDecoder` directly.
They are intentionally separate from legacy AR/DETR ablations whose switches
act on checkpoint-based decoders.
"""

from __future__ import annotations

import os
import subprocess
import sys


PYTHON = sys.executable
TRAIN_SCRIPT = "patchevent/phase2/train.py"
SEEDS = [42, 123, 456]

DOMAINS = {
    "wlel": {
        "series_path": "dataset/wlel/event_v1/data/wlel_event_series_v1.csv",
        "events_path": "dataset/wlel/event_v1/data/wlel_events_v1.jsonl",
    },
    "ett": {
        "series_path": "dataset/ett/event_v1/data/ett_event_series_v1.csv",
        "events_path": "dataset/ett/event_v1/data/ett_events_v1.jsonl",
    },
    "elc": {
        "series_path": "dataset/electricity/event_v1/data/elc_event_series_v1.csv",
        "events_path": "dataset/electricity/event_v1/data/elc_events_v1.jsonl",
    },
}

OUT_BASE_TMPL = "patchevent/phase2/checkpoints/{domain}_hybrid_v21_ablation"

COMMON = [
    "--decoder_type", "hybrid",
    "--d_model", "128",
    "--n_heads", "4",
    "--n_layers", "3",
    "--d_ff", "256",
    "--dropout", "0.2",
    "--lr", "5e-4",
    "--weight_decay", "0.05",
    "--train_epochs", "50",
    "--patience", "15",
    "--batch_size", "64",
    "--eval_batch_size", "128",
    "--warmup_steps", "200",
    "--max_grad_norm", "1.0",
    "--seq_len", "96",
    "--pred_len", "96",
    "--patch_len", "8",
    "--patch_stride", "4",
    "--backbone_layers", "2",
    "--max_new_tokens", "60",
    "--tolerance", "3",
    "--gpu", "0",
]

HYBRID_CONFIGS = {
    # v2.1 main candidate: structured temporal head + trained count controller.
    "H0_v21_hybrid": [
        "--max_events", "10",
        "--refine_layers", "1",
        "--hybrid_time_head", "structured",
        "--hybrid_use_count_head",
        "--hybrid_count_loss_weight", "0.5",
    ],

    # Capacity / event count sensitivity.
    "HQ8_queries": ["--max_events", "8", "--refine_layers", "1", "--hybrid_use_count_head", "--hybrid_count_loss_weight", "0.5"],
    "HQ12_queries": ["--max_events", "12", "--refine_layers", "1", "--hybrid_use_count_head", "--hybrid_count_loss_weight", "0.5"],

    # Count controller and temporal parameterization.
    "HNC_no_count_head": ["--max_events", "10", "--refine_layers", "1", "--hybrid_time_head", "structured"],
    "HCD_count_train_only": [
        "--max_events", "10",
        "--refine_layers", "1",
        "--hybrid_time_head", "structured",
        "--hybrid_use_count_head",
        "--hybrid_no_count_decoding",
        "--hybrid_count_loss_weight", "0.5",
    ],
    "HID_independent_time": [
        "--max_events", "10",
        "--refine_layers", "1",
        "--hybrid_time_head", "independent",
        "--hybrid_use_count_head",
        "--hybrid_count_loss_weight", "0.5",
    ],

    # DETR proposal vs event-order refinement.
    "HR0_no_refine": ["--max_events", "10", "--refine_layers", "0", "--hybrid_use_count_head", "--hybrid_count_loss_weight", "0.5"],
    "HR2_refine2": ["--max_events", "10", "--refine_layers", "2", "--hybrid_use_count_head", "--hybrid_count_loss_weight", "0.5"],
    "HO_query_order": [
        "--max_events", "10",
        "--refine_layers", "1",
        "--hybrid_refine_order", "query",
        "--hybrid_use_count_head",
        "--hybrid_count_loss_weight", "0.5",
    ],
    "HC_no_causal_refine": [
        "--max_events", "10",
        "--refine_layers", "1",
        "--hybrid_no_causal_refine_mask",
        "--hybrid_use_count_head",
        "--hybrid_count_loss_weight", "0.5",
    ],

    # Loss/matching mechanics.
    "HM_no_intensity_cost": [
        "--max_events", "10",
        "--refine_layers", "1",
        "--hybrid_match_intensity_weight", "0.0",
        "--hybrid_use_count_head",
        "--hybrid_count_loss_weight", "0.5",
    ],
    "HP_no_proposal_aux": [
        "--max_events", "10",
        "--refine_layers", "1",
        "--hybrid_proposal_loss_weight", "0.0",
        "--hybrid_use_count_head",
        "--hybrid_count_loss_weight", "0.5",
    ],
    "HN_noobj05": [
        "--max_events", "10",
        "--refine_layers", "1",
        "--hybrid_no_object_weight", "0.5",
        "--hybrid_use_count_head",
        "--hybrid_count_loss_weight", "0.5",
    ],
    "HF_focal2": [
        "--max_events", "10",
        "--refine_layers", "1",
        "--hybrid_object_focal_gamma", "2.0",
        "--hybrid_use_count_head",
        "--hybrid_count_loss_weight", "0.5",
    ],
    "HS_structure": [
        "--max_events", "10",
        "--refine_layers", "1",
        "--hybrid_structure_loss_weight", "1.0",
        "--hybrid_use_count_head",
        "--hybrid_count_loss_weight", "0.5",
    ],
}


def is_done(out_dir: str) -> bool:
    return os.path.exists(os.path.join(out_dir, "test_summary.json"))


def main() -> None:
    total = done = skip = fail = 0
    for domain, cfg in DOMAINS.items():
        out_base = OUT_BASE_TMPL.format(domain=domain)
        os.makedirs(out_base, exist_ok=True)
        for name, extra in HYBRID_CONFIGS.items():
            for seed in SEEDS:
                total += 1
                out_dir = os.path.join(out_base, f"{name}_s{seed}")
                if is_done(out_dir):
                    print(f"[SKIP] {domain}/{name}_s{seed}")
                    skip += 1
                    continue
                cmd = [
                    PYTHON, TRAIN_SCRIPT,
                    "--series_path", cfg["series_path"],
                    "--events_path", cfg["events_path"],
                    "--output_dir", out_dir,
                    "--seed", str(seed),
                ] + list(COMMON) + list(extra)

                print("\n" + "=" * 72)
                print(f"[START] {domain}/{name}_s{seed}")
                print(f" output: {out_dir}")
                print("=" * 72)
                ret = subprocess.run(cmd, cwd=os.getcwd())
                if ret.returncode != 0:
                    print(f"[FAIL] {domain}/{name}_s{seed}")
                    fail += 1
                else:
                    done += 1

    print("\n" + "=" * 72)
    print(f"HYBRID ablation summary: total={total} done={done} skip={skip} fail={fail}")
    print("=" * 72)


if __name__ == "__main__":
    main()
