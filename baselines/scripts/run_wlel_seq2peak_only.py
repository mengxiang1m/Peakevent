"""Run Seq2Peak WLEL pred168/336 x 3 seeds + eval."""
import subprocess, sys, os

PYTHON = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

PRED_LENS = [168, 336]
SEEDS = [42, 123, 456]

def run(cmd, desc):
    print(f"\n{'='*60}\n  {desc}\n{'='*60}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"  [WARN] exit {r.returncode}")
    return r.returncode

for pl in PRED_LENS:
    for s in SEEDS:
        run([PYTHON, "./seq2peak/run.py",
            "--is_training", "1",
            "--model_id", f"wlel_p{pl}_s{s}_peak_Autoformer",
            "--model", "peak_Autoformer",
            "--data", "stsep", "--root_path", "./data", "--data_path", "wlel.csv",
            "--dataset_config", "./data/dataset_configs.json", "--dataset_name", "wlel",
            "--window_stride", "4", "--features", "S", "--target", "OT",
            "--seq_len", "96", "--label_len", "48", "--pred_len", str(pl),
            "--enc_in", "1", "--dec_in", "1", "--c_out", "1",
            "--learning_rate", "1e-4", "--batch_size", "32", "--num_workers", "0",
            "--train_epochs", "10", "--patience", "5", "--itr", "1",
            "--seed", str(s), "--inverse"],
            f"Seq2Peak pl={pl} s={s}")

for pl in PRED_LENS:
    run([PYTHON, "./posthoc_evaluate.py",
        "--dataset", "wlel", "--pred_len", str(pl),
        "--event_mode", "gradient_width", "--rate_frac", "0.4"],
        f"eval wlel pl={pl}")

subprocess.run([PYTHON, "./collect_results.py"], cwd=ROOT)
print("\nDone!")
