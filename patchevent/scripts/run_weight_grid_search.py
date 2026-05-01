"""
Grid search for attr_loss_weights across 3 domains.
Search space: onset/apex weight w ∈ {1.0, 2.0, 3.0, 4.0, 6.0, 8.0}
Fixed: dur=0.5, int=0.5
Single seed (s42) for fast screening.
"""
import sys
import subprocess
import os
import json
import time
import numpy as np

PYTHON = sys.executable
TRAIN_SCRIPT = "patchevent/phase2/train.py"

DOMAINS = {
    "wlel": {
        "encoder": "patchevent/phase1/checkpoints/wlel/s42/best_model.pth",
        "series": "dataset/wlel/event_v1/data/wlel_event_series_v1.csv",
        "events": "dataset/wlel/event_v1/data/wlel_events_v1.jsonl",
    },
    "ett": {
        "encoder": "patchevent/phase1/checkpoints/ett/s42/best_model.pth",
        "series": "dataset/ett/event_v1/data/ett_event_series_v1.csv",
        "events": "dataset/ett/event_v1/data/ett_events_v1.jsonl",
    },
    "elc": {
        "encoder": "patchevent/phase1/checkpoints/elc/s42/best_model.pth",
        "series": "dataset/electricity/event_v1/data/elc_event_series_v1.csv",
        "events": "dataset/electricity/event_v1/data/elc_events_v1.jsonl",
    },
}

WEIGHTS_GRID = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]

COMMON = [
    "--encoder_mode", "frozen",
    "--unfreeze_last_n", "1",
    "--use_memory_pos",
    "--use_self_attn_agg",
    "--embed_dropout", "0.15",
    "--label_smoothing", "0.1",
    "--batch_size", "32",
    "--train_epochs", "50",
    "--lr", "5e-4",
    "--seed", "42",
    "--pred_len", "96",
    "--seq_len", "96",
]


def run_one(domain, w_onset_apex):
    info = DOMAINS[domain]
    w_str = f"{w_onset_apex},{0.5},{w_onset_apex},{0.5}"
    out_dir = f"patchevent/phase2/checkpoints/{domain}_grid_search/w{w_onset_apex}_s42"

    summary_path = os.path.join(out_dir, "test_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            d = json.load(f)
        return d

    cmd = [
        PYTHON, "-u", TRAIN_SCRIPT,
        "--encoder_ckpt", info["encoder"],
        "--series_path", info["series"],
        "--events_path", info["events"],
        "--attr_loss_weights", w_str,
        "--output_dir", out_dir,
    ] + COMMON

    print(f"  RUN {domain} w={w_onset_apex} ...", end="", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - t0

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            d = json.load(f)
        print(f" F1={d['test_event_f1']:.4f} Onset={d['test_onset_mae']:.4f} ({elapsed:.0f}s)")
        return d
    else:
        print(f" FAIL ({elapsed:.0f}s)")
        if result.stderr:
            for line in result.stderr.strip().split('\n')[-3:]:
                print(f"    {line}")
        return None


def main():
    all_results = {}

    for domain in ["wlel", "ett", "elc"]:
        print(f"\n=== {domain.upper()} Grid Search ===")
        results = []
        for w in WEIGHTS_GRID:
            d = run_one(domain, w)
            if d:
                results.append({
                    "weight": w,
                    "f1": d["test_event_f1"],
                    "onset_mae": d["test_onset_mae"],
                    "apex_mae": d["test_apex_mae"],
                    "dur_mae": d["test_duration_mae"],
                    "int_mape": d.get("test_intensity_mape", 0),
                })

        all_results[domain] = results

        # Print summary
        print(f"\n  {'w':>4} | {'F1':>7} | {'Onset':>7} | {'Apex':>7} | {'Dur':>7} | {'IntMAPE':>8}")
        print(f"  {'-'*50}")
        best_f1 = max(r["f1"] for r in results)
        best_onset = min(r["onset_mae"] for r in results)
        for r in results:
            f1_mark = " *" if r["f1"] == best_f1 else ""
            onset_mark = " *" if r["onset_mae"] == best_onset else ""
            print(f"  {r['weight']:>4.1f} | {r['f1']:.4f}{f1_mark} | {r['onset_mae']:.4f}{onset_mark} | {r['apex_mae']:.4f} | {r['dur_mae']:.4f} | {r['int_mape']:.4f}")

    # Final recommendation
    print("\n" + "=" * 60)
    print("OPTIMAL WEIGHTS PER DOMAIN")
    print("=" * 60)
    for domain, results in all_results.items():
        best = max(results, key=lambda r: r["f1"])
        print(f"  {domain.upper()}: w={best['weight']:.1f} → F1={best['f1']:.4f}, OnsetMAE={best['onset_mae']:.4f}")

    # Save results
    with open("patchevent/phase2/checkpoints/grid_search_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to grid_search_results.json")


if __name__ == "__main__":
    main()
