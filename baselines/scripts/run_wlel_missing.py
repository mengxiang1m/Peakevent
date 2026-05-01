"""
Run WLEL baseline experiments for pred168 and pred336 ONLY (pred96 already done).
Trains 5 models x 2 pred_lens x 3 seeds = 30 runs, then evaluates and collects.

Usage:
    python baselines/scripts/run_wlel_missing.py
"""
import subprocess
import sys
import os

PYTHON = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # baselines/
os.chdir(ROOT)

DATASET = "wlel"
DATA_PATH = f"{DATASET}.csv"
CONFIG_PATH = "./data/dataset_configs.json"

PRED_LENS = [168, 336]
SEEDS = [42, 123, 456]
TSLIB_MODELS = ["PatchTST", "iTransformer", "TimeMixer", "DLinear"]

EPOCHS = 10
BATCH_SIZE = 32
LR = "1e-4"

def run_cmd(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(cmd[:6])}...")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"  [WARN] Exit code {result.returncode} for {desc}")
    return result.returncode

# ---- Step 1: TSLib baselines ----
total_tslib = len(TSLIB_MODELS) * len(PRED_LENS) * len(SEEDS)
print(f"\n[Step 1/3] Training TSLib baselines: {total_tslib} runs")

count = 0
for model in TSLIB_MODELS:
    for pred_len in PRED_LENS:
        for seed in SEEDS:
            count += 1
            model_id = f"{DATASET}_p{pred_len}_s{seed}_{model}"
            desc = f"[{count}/{total_tslib}] TSLib {model} pl={pred_len} s={seed}"
            cmd = [
                PYTHON, "./tslib/run.py",
                "--task_name", "long_term_forecast",
                "--is_training", "1",
                "--model_id", model_id,
                "--model", model,
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
                "--learning_rate", LR,
                "--batch_size", str(BATCH_SIZE),
                "--num_workers", "0",
                "--train_epochs", str(EPOCHS),
                "--patience", "5",
                "--itr", "1",
                "--seed", str(seed),
                "--inverse",
            ]
            run_cmd(cmd, desc)

# ---- Step 2: Seq2Peak baseline ----
total_seq2peak = len(PRED_LENS) * len(SEEDS)
print(f"\n[Step 2/3] Training Seq2Peak: {total_seq2peak} runs")

count = 0
for pred_len in PRED_LENS:
    for seed in SEEDS:
        count += 1
        model_id = f"{DATASET}_p{pred_len}_s{seed}_peak_Autoformer"
        desc = f"[{count}/{total_seq2peak}] Seq2Peak pl={pred_len} s={seed}"
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
            "--learning_rate", LR,
            "--batch_size", str(BATCH_SIZE),
            "--num_workers", "0",
            "--train_epochs", str(EPOCHS),
            "--patience", "5",
            "--itr", "1",
            "--seed", str(seed),
            "--inverse",
        ]
        run_cmd(cmd, desc)

# ---- Step 3: Evaluate ----
print(f"\n[Step 3/3] Post-hoc evaluation")

RATE_FRAC = "0.4"
for pred_len in PRED_LENS:
    desc = f"[eval] {DATASET} pred_len={pred_len}"
    cmd = [
        PYTHON, "./posthoc_evaluate.py",
        "--dataset", DATASET,
        "--pred_len", str(pred_len),
        "--event_mode", "gradient_width",
        "--rate_frac", RATE_FRAC,
    ]
    run_cmd(cmd, desc)

# Collect all results
print(f"\n[Collect] Aggregating results...")
subprocess.run([PYTHON, "./collect_results.py"], cwd=ROOT)

print("\n" + "="*60)
print("  DONE! Check:")
print("    results/posthoc/wlel/pred168/")
print("    results/posthoc/wlel/pred336/")
print("    results/posthoc/summary_agg.csv")
print("="*60)
