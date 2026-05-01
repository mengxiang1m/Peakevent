"""
Fix and re-run TimeMixer + Seq2Peak for WLEL pred168/336.

TimeMixer fix: add --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg
Seq2Peak fix: investigate and handle tensor size mismatch in peak_Autoformer

Usage:
    python baselines/scripts/run_wlel_timemixer_seq2peak.py
"""
import subprocess
import sys
import os

PYTHON = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

DATASET = "wlel"
DATA_PATH = f"{DATASET}.csv"
CONFIG_PATH = "./data/dataset_configs.json"

PRED_LENS = [168, 336]
SEEDS = [42, 123, 456]

def run_cmd(cmd, desc=""):
    print(f"\n{'='*60}\n  {desc}\n{'='*60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"  [WARN] Exit code {result.returncode}")
    return result.returncode

# ---- TimeMixer with correct down_sampling params ----
print("\n[TimeMixer] Running with down_sampling fix...")
for pred_len in PRED_LENS:
    for seed in SEEDS:
        model_id = f"{DATASET}_p{pred_len}_s{seed}_TimeMixer"
        desc = f"TimeMixer pl={pred_len} s={seed}"
        cmd = [
            PYTHON, "./tslib/run.py",
            "--task_name", "long_term_forecast",
            "--is_training", "1",
            "--model_id", model_id,
            "--model", "TimeMixer",
            "--data", "stsep",
            "--root_path", "./data",
            "--data_path", DATA_PATH,
            "--dataset_config", CONFIG_PATH,
            "--dataset_name", DATASET,
            "--window_stride", "4",
            "--features", "S",
            "--target", "OT",
            "--seq_len", "96",
            "--label_len", "48",
            "--pred_len", str(pred_len),
            "--enc_in", "1",
            "--dec_in", "1",
            "--c_out", "1",
            "--learning_rate", "1e-4",
            "--batch_size", "32",
            "--num_workers", "0",
            "--train_epochs", "10",
            "--patience", "5",
            "--itr", "1",
            "--seed", str(seed),
            "--inverse",
            # Fix: TimeMixer requires explicit down_sampling params
            "--down_sampling_layers", "3",
            "--down_sampling_window", "2",
            "--down_sampling_method", "avg",
        ]
        run_cmd(cmd, desc)

# ---- Seq2Peak ----
print("\n[Seq2Peak] Running...")
for pred_len in PRED_LENS:
    for seed in SEEDS:
        model_id = f"{DATASET}_p{pred_len}_s{seed}_peak_Autoformer"
        desc = f"Seq2Peak pl={pred_len} s={seed}"
        cmd = [
            PYTHON, "./seq2peak/run.py",
            "--is_training", "1",
            "--model_id", model_id,
            "--model", "peak_Autoformer",
            "--data", "stsep",
            "--root_path", "./data",
            "--data_path", DATA_PATH,
            "--dataset_config", CONFIG_PATH,
            "--dataset_name", DATASET,
            "--window_stride", "4",
            "--features", "S",
            "--target", "OT",
            "--seq_len", "96",
            "--label_len", "48",
            "--pred_len", str(pred_len),
            "--enc_in", "1",
            "--dec_in", "1",
            "--c_out", "1",
            "--learning_rate", "1e-4",
            "--batch_size", "32",
            "--num_workers", "0",
            "--train_epochs", "10",
            "--patience", "5",
            "--itr", "1",
            "--seed", str(seed),
            "--inverse",
        ]
        run_cmd(cmd, desc)

# ---- Evaluate ----
print("\n[Eval] Post-hoc evaluation...")
for pred_len in PRED_LENS:
    cmd = [
        PYTHON, "./posthoc_evaluate.py",
        "--dataset", DATASET,
        "--pred_len", str(pred_len),
        "--event_mode", "gradient_width",
        "--rate_frac", "0.4",
    ]
    run_cmd(cmd, f"eval {DATASET} pl={pred_len}")

subprocess.run([PYTHON, "./collect_results.py"], cwd=ROOT)
print("\nDone!")
