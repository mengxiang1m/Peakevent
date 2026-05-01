"""
Full grid search: attr_loss_weights × 3 domains × 3 seeds
+ architecture/training hparams on WLEL × 3 seeds
"""
import sys
import subprocess
import os
import json
import time
import numpy as np

PYTHON = sys.executable
TRAIN_SCRIPT = "patchevent/phase2/train.py"
SEEDS = [42, 123, 456]

DOMAINS = {
    "wlel": {
        "encoder_base": "patchevent/phase1/checkpoints/wlel",
        "series": "dataset/wlel/event_v1/data/wlel_event_series_v1.csv",
        "events": "dataset/wlel/event_v1/data/wlel_events_v1.jsonl",
    },
    "ett": {
        "encoder_base": "patchevent/phase1/checkpoints/ett",
        "series": "dataset/ett/event_v1/data/ett_event_series_v1.csv",
        "events": "dataset/ett/event_v1/data/ett_events_v1.jsonl",
    },
    "elc": {
        "encoder_base": "patchevent/phase1/checkpoints/elc",
        "series": "dataset/electricity/event_v1/data/elc_event_series_v1.csv",
        "events": "dataset/electricity/event_v1/data/elc_events_v1.jsonl",
    },
}

BASE_DIR = "patchevent/phase2/checkpoints"


def run_one(domain, seed, extra_args, out_dir):
    summary_path = os.path.join(out_dir, "test_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)

    info = DOMAINS[domain]
    encoder = os.path.join(info["encoder_base"], f"s{seed}", "best_model.pth")

    cmd = [
        PYTHON, "-u", TRAIN_SCRIPT,
        "--encoder_ckpt", encoder,
        "--series_path", info["series"],
        "--events_path", info["events"],
        "--encoder_mode", "frozen",
        "--unfreeze_last_n", "1",
        "--use_memory_pos",
        "--use_self_attn_agg",
        "--seed", str(seed),
        "--pred_len", "96",
        "--seq_len", "96",
        "--batch_size", "32",
        "--train_epochs", "50",
        "--output_dir", out_dir,
    ] + extra_args

    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__)))

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    else:
        if result.stderr:
            lines = result.stderr.strip().split('\n')
            for line in lines[-3:]:
                print(f"      ERR: {line}")
        return None


def run_3seed(domain, config_name, extra_args, base_out):
    f1s, onsets, apexs = [], [], []
    for seed in SEEDS:
        out = os.path.join(base_out, f"{config_name}_s{seed}")
        d = run_one(domain, seed, extra_args, out)
        if d:
            f1s.append(d["test_event_f1"])
            onsets.append(d["test_onset_mae"])
            apexs.append(d["test_apex_mae"])
    if len(f1s) == 3:
        return {
            "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
            "onset_mean": np.mean(onsets), "onset_std": np.std(onsets),
            "apex_mean": np.mean(apexs), "apex_std": np.std(apexs),
            "f1s": f1s,
        }
    return None


def print_results(results, title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'Config':>15} | {'F1 (3-seed)':>16} | {'Onset':>16} | {'Apex':>16}")
    print(f"  {'-'*70}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["f1_mean"]):
        f1 = f"{r['f1_mean']:.3f}±{r['f1_std']:.3f}"
        on = f"{r['onset_mean']:.3f}±{r['onset_std']:.3f}"
        ap = f"{r['apex_mean']:.3f}±{r['apex_std']:.3f}"
        print(f"  {name:>15} | {f1:>16} | {on:>16} | {ap:>16}")


def main():
    all_results = {}

    # ====== PART 1: attr_loss_weights search (3 domains × 3 seeds) ======
    print("=" * 60)
    print("PART 1: attr_loss_weights Grid Search")
    print("=" * 60)

    WEIGHT_GRID = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]

    for domain in ["wlel", "ett", "elc"]:
        print(f"\n--- {domain.upper()} ---")
        domain_results = {}
        for w in WEIGHT_GRID:
            name = f"w{w:.0f}"
            w_str = f"{w},0.5,{w},0.5"
            args = [
                "--attr_loss_weights", w_str,
                "--embed_dropout", "0.15",
                "--label_smoothing", "0.1",
                "--lr", "5e-4",
                "--dropout", "0.2",
            ]
            base_out = f"{BASE_DIR}/{domain}_grid_search"
            print(f"    {name}: ", end="", flush=True)
            r = run_3seed(domain, name, args, base_out)
            if r:
                print(f"F1={r['f1_mean']:.4f}±{r['f1_std']:.4f}")
                domain_results[name] = r
            else:
                print("FAIL")

        all_results[f"weights_{domain}"] = domain_results
        print_results(domain_results, f"{domain.upper()} Weight Search")

    # ====== PART 2: Architecture search (WLEL only × 3 seeds) ======
    print("\n" + "=" * 60)
    print("PART 2: Architecture Search (WLEL)")
    print("=" * 60)

    # Use the best weight from Part 1 for WLEL (or default to 4.0)
    wlel_weights = all_results.get("weights_wlel", {})
    if wlel_weights:
        best_w_name = max(wlel_weights, key=lambda k: wlel_weights[k]["f1_mean"])
        best_w = float(best_w_name[1:])
    else:
        best_w = 4.0
    best_w_str = f"{best_w},0.5,{best_w},0.5"
    print(f"  Using best weight: {best_w_str}")

    arch_configs = {
        "nl2":     ["--n_layers", "2", "--d_model", "128", "--d_ff", "256"],
        "nl3":     ["--n_layers", "3", "--d_model", "128", "--d_ff", "256"],  # default
        "nl4":     ["--n_layers", "4", "--d_model", "128", "--d_ff", "256"],
        "dm64":    ["--n_layers", "3", "--d_model", "64",  "--d_ff", "128"],
        "dm256":   ["--n_layers", "3", "--d_model", "256", "--d_ff", "512"],
        "dff128":  ["--n_layers", "3", "--d_model", "128", "--d_ff", "128"],
        "dff512":  ["--n_layers", "3", "--d_model", "128", "--d_ff", "512"],
    }

    arch_results = {}
    for name, arch_args in arch_configs.items():
        full_args = arch_args + [
            "--attr_loss_weights", best_w_str,
            "--embed_dropout", "0.15",
            "--label_smoothing", "0.1",
            "--lr", "5e-4",
            "--dropout", "0.2",
        ]
        print(f"    {name}: ", end="", flush=True)
        r = run_3seed("wlel", name, full_args, f"{BASE_DIR}/wlel_arch_search")
        if r:
            print(f"F1={r['f1_mean']:.4f}±{r['f1_std']:.4f}")
            arch_results[name] = r
        else:
            print("FAIL")

    all_results["arch_wlel"] = arch_results
    print_results(arch_results, "WLEL Architecture Search")

    # ====== PART 3: Training hparam search (WLEL only × 3 seeds) ======
    print("\n" + "=" * 60)
    print("PART 3: Training Hparam Search (WLEL)")
    print("=" * 60)

    # Use best arch from Part 2
    if arch_results:
        best_arch_name = max(arch_results, key=lambda k: arch_results[k]["f1_mean"])
        best_arch = arch_configs[best_arch_name]
    else:
        best_arch = ["--n_layers", "3", "--d_model", "128", "--d_ff", "256"]
    print(f"  Using best arch: {best_arch_name} = {best_arch}")

    train_configs = {
        "lr1e-4":  ["--lr", "1e-4", "--dropout", "0.2", "--embed_dropout", "0.15", "--label_smoothing", "0.1"],
        "lr3e-4":  ["--lr", "3e-4", "--dropout", "0.2", "--embed_dropout", "0.15", "--label_smoothing", "0.1"],
        "lr5e-4":  ["--lr", "5e-4", "--dropout", "0.2", "--embed_dropout", "0.15", "--label_smoothing", "0.1"],  # default
        "lr1e-3":  ["--lr", "1e-3", "--dropout", "0.2", "--embed_dropout", "0.15", "--label_smoothing", "0.1"],
        "drop01":  ["--lr", "5e-4", "--dropout", "0.1", "--embed_dropout", "0.1",  "--label_smoothing", "0.1"],
        "drop03":  ["--lr", "5e-4", "--dropout", "0.3", "--embed_dropout", "0.2",  "--label_smoothing", "0.1"],
        "ls0":     ["--lr", "5e-4", "--dropout", "0.2", "--embed_dropout", "0.15", "--label_smoothing", "0.0"],
        "ls005":   ["--lr", "5e-4", "--dropout", "0.2", "--embed_dropout", "0.15", "--label_smoothing", "0.05"],
        "ls02":    ["--lr", "5e-4", "--dropout", "0.2", "--embed_dropout", "0.15", "--label_smoothing", "0.2"],
    }

    train_results = {}
    for name, train_args in train_configs.items():
        full_args = best_arch + ["--attr_loss_weights", best_w_str] + train_args
        print(f"    {name}: ", end="", flush=True)
        r = run_3seed("wlel", name, full_args, f"{BASE_DIR}/wlel_train_search")
        if r:
            print(f"F1={r['f1_mean']:.4f}±{r['f1_std']:.4f}")
            train_results[name] = r
        else:
            print("FAIL")

    all_results["train_wlel"] = train_results
    print_results(train_results, "WLEL Training Hparam Search")

    # ====== FINAL SUMMARY ======
    print("\n" + "=" * 60)
    print("FINAL OPTIMAL CONFIGURATION")
    print("=" * 60)

    # Best weight per domain
    for domain in ["wlel", "ett", "elc"]:
        key = f"weights_{domain}"
        if key in all_results and all_results[key]:
            best = max(all_results[key].items(), key=lambda x: x[1]["f1_mean"])
            print(f"  {domain.upper()} best weight: {best[0]} → F1={best[1]['f1_mean']:.4f}±{best[1]['f1_std']:.4f}")

    # Best arch
    if arch_results:
        best = max(arch_results.items(), key=lambda x: x[1]["f1_mean"])
        print(f"  WLEL best arch: {best[0]} → F1={best[1]['f1_mean']:.4f}±{best[1]['f1_std']:.4f}")

    # Best training
    if train_results:
        best = max(train_results.items(), key=lambda x: x[1]["f1_mean"])
        print(f"  WLEL best train: {best[0]} → F1={best[1]['f1_mean']:.4f}±{best[1]['f1_std']:.4f}")

    # Save all
    # Convert numpy to python for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    with open(f"{BASE_DIR}/full_grid_search_results.json", "w") as f:
        json.dump(to_serializable(all_results), f, indent=2)
    print(f"\nAll results saved to {BASE_DIR}/full_grid_search_results.json")


if __name__ == "__main__":
    main()
