"""
Grid search for Phase 2 architecture and training hyperparameters.
Run on WLEL s42 only for fast screening.
Uses the best attr_loss_weights from the weight grid search (or default 4.0,0.5,4.0,0.5).
"""
import sys
import subprocess
import os
import json
import time

PYTHON = sys.executable
TRAIN_SCRIPT = "patchevent/phase2/train.py"

ENCODER = "patchevent/phase1/checkpoints/wlel/s42/best_model.pth"
SERIES = "dataset/wlel/event_v1/data/wlel_event_series_v1.csv"
EVENTS = "dataset/wlel/event_v1/data/wlel_events_v1.jsonl"

BASE_COMMON = [
    "--encoder_ckpt", ENCODER,
    "--series_path", SERIES,
    "--events_path", EVENTS,
    "--encoder_mode", "frozen",
    "--unfreeze_last_n", "1",
    "--use_memory_pos",
    "--use_self_attn_agg",
    "--batch_size", "32",
    "--train_epochs", "50",
    "--seed", "42",
    "--pred_len", "96",
    "--seq_len", "96",
]

# Best attr weights (update after weight grid search if needed)
BEST_ATTR_WEIGHTS = "4.0,0.5,4.0,0.5"


def run_one(name, extra_args, out_dir):
    summary_path = os.path.join(out_dir, "test_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            d = json.load(f)
        print(f"  SKIP {name}: F1={d['test_event_f1']:.4f}")
        return d

    cmd = [PYTHON, "-u", TRAIN_SCRIPT] + BASE_COMMON + extra_args + ["--output_dir", out_dir]

    print(f"  RUN {name} ...", end="", flush=True)
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
    base_dir = "patchevent/phase2/checkpoints/wlel_hparam_search"

    # ====== GROUP A: Architecture ======
    print("=" * 60)
    print("GROUP A: Architecture Search (WLEL s42)")
    print("=" * 60)

    arch_configs = {
        # Decoder depth
        "nlayers2": ["--n_layers", "2", "--d_model", "128", "--d_ff", "256",
                      "--dropout", "0.2", "--embed_dropout", "0.15",
                      "--label_smoothing", "0.1", "--lr", "5e-4",
                      "--attr_loss_weights", BEST_ATTR_WEIGHTS],
        "nlayers3": ["--n_layers", "3", "--d_model", "128", "--d_ff", "256",
                      "--dropout", "0.2", "--embed_dropout", "0.15",
                      "--label_smoothing", "0.1", "--lr", "5e-4",
                      "--attr_loss_weights", BEST_ATTR_WEIGHTS],  # current default
        "nlayers4": ["--n_layers", "4", "--d_model", "128", "--d_ff", "256",
                      "--dropout", "0.2", "--embed_dropout", "0.15",
                      "--label_smoothing", "0.1", "--lr", "5e-4",
                      "--attr_loss_weights", BEST_ATTR_WEIGHTS],
        # Model dim (decoder only, encoder is frozen at 128)
        "dmodel64": ["--n_layers", "3", "--d_model", "64", "--d_ff", "128",
                      "--dropout", "0.2", "--embed_dropout", "0.15",
                      "--label_smoothing", "0.1", "--lr", "5e-4",
                      "--attr_loss_weights", BEST_ATTR_WEIGHTS],
        "dmodel256": ["--n_layers", "3", "--d_model", "256", "--d_ff", "512",
                       "--dropout", "0.2", "--embed_dropout", "0.15",
                       "--label_smoothing", "0.1", "--lr", "5e-4",
                       "--attr_loss_weights", BEST_ATTR_WEIGHTS],
        # d_ff variations (with d_model=128)
        "dff128": ["--n_layers", "3", "--d_model", "128", "--d_ff", "128",
                    "--dropout", "0.2", "--embed_dropout", "0.15",
                    "--label_smoothing", "0.1", "--lr", "5e-4",
                    "--attr_loss_weights", BEST_ATTR_WEIGHTS],
        "dff512": ["--n_layers", "3", "--d_model", "128", "--d_ff", "512",
                    "--dropout", "0.2", "--embed_dropout", "0.15",
                    "--label_smoothing", "0.1", "--lr", "5e-4",
                    "--attr_loss_weights", BEST_ATTR_WEIGHTS],
    }

    arch_results = []
    for name, args in arch_configs.items():
        out = f"{base_dir}/arch_{name}"
        d = run_one(f"arch/{name}", args, out)
        if d:
            arch_results.append({"name": name, **{k: d[k] for k in
                ["test_event_f1", "test_onset_mae", "test_apex_mae"]}})

    print(f"\n  {'Config':>12} | {'F1':>7} | {'Onset':>7} | {'Apex':>7}")
    print(f"  {'-'*42}")
    for r in sorted(arch_results, key=lambda x: -x["test_event_f1"]):
        print(f"  {r['name']:>12} | {r['test_event_f1']:.4f} | {r['test_onset_mae']:.4f} | {r['test_apex_mae']:.4f}")

    # ====== GROUP B: Training Hyperparameters ======
    print()
    print("=" * 60)
    print("GROUP B: Training Hparam Search (WLEL s42)")
    print("=" * 60)

    train_configs = {
        # Learning rate
        "lr1e-4": ["--n_layers", "3", "--d_model", "128", "--d_ff", "256",
                    "--dropout", "0.2", "--embed_dropout", "0.15",
                    "--label_smoothing", "0.1", "--lr", "1e-4",
                    "--attr_loss_weights", BEST_ATTR_WEIGHTS],
        "lr3e-4": ["--n_layers", "3", "--d_model", "128", "--d_ff", "256",
                    "--dropout", "0.2", "--embed_dropout", "0.15",
                    "--label_smoothing", "0.1", "--lr", "3e-4",
                    "--attr_loss_weights", BEST_ATTR_WEIGHTS],
        "lr1e-3": ["--n_layers", "3", "--d_model", "128", "--d_ff", "256",
                    "--dropout", "0.2", "--embed_dropout", "0.15",
                    "--label_smoothing", "0.1", "--lr", "1e-3",
                    "--attr_loss_weights", BEST_ATTR_WEIGHTS],
        # Dropout
        "drop01": ["--n_layers", "3", "--d_model", "128", "--d_ff", "256",
                    "--dropout", "0.1", "--embed_dropout", "0.1",
                    "--label_smoothing", "0.1", "--lr", "5e-4",
                    "--attr_loss_weights", BEST_ATTR_WEIGHTS],
        "drop03": ["--n_layers", "3", "--d_model", "128", "--d_ff", "256",
                    "--dropout", "0.3", "--embed_dropout", "0.2",
                    "--label_smoothing", "0.1", "--lr", "5e-4",
                    "--attr_loss_weights", BEST_ATTR_WEIGHTS],
        # Label smoothing
        "ls0": ["--n_layers", "3", "--d_model", "128", "--d_ff", "256",
                "--dropout", "0.2", "--embed_dropout", "0.15",
                "--label_smoothing", "0.0", "--lr", "5e-4",
                "--attr_loss_weights", BEST_ATTR_WEIGHTS],
        "ls005": ["--n_layers", "3", "--d_model", "128", "--d_ff", "256",
                  "--dropout", "0.2", "--embed_dropout", "0.15",
                  "--label_smoothing", "0.05", "--lr", "5e-4",
                  "--attr_loss_weights", BEST_ATTR_WEIGHTS],
        "ls02": ["--n_layers", "3", "--d_model", "128", "--d_ff", "256",
                 "--dropout", "0.2", "--embed_dropout", "0.15",
                 "--label_smoothing", "0.2", "--lr", "5e-4",
                 "--attr_loss_weights", BEST_ATTR_WEIGHTS],
    }

    train_results = []
    for name, args in train_configs.items():
        out = f"{base_dir}/train_{name}"
        d = run_one(f"train/{name}", args, out)
        if d:
            train_results.append({"name": name, **{k: d[k] for k in
                ["test_event_f1", "test_onset_mae", "test_apex_mae"]}})

    print(f"\n  {'Config':>12} | {'F1':>7} | {'Onset':>7} | {'Apex':>7}")
    print(f"  {'-'*42}")
    for r in sorted(train_results, key=lambda x: -x["test_event_f1"]):
        print(f"  {r['name']:>12} | {r['test_event_f1']:.4f} | {r['test_onset_mae']:.4f} | {r['test_apex_mae']:.4f}")

    # Save all
    with open(f"{base_dir}/search_results.json", "w") as f:
        json.dump({"arch": arch_results, "train": train_results}, f, indent=2)
    print(f"\nAll results saved to {base_dir}/search_results.json")


if __name__ == "__main__":
    main()
