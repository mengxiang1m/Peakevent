"""
Organize results/, checkpoints/, visualization/output/predictions/ directories.

Moves "冗余" (won't appear in the final paper) dirs into a `pre/`
subdirectory inside each folder. Keeps only runs that are either
(a) correct 244-weight paper-needed results, or (b) will be produced
by upcoming experiments.

DRY RUN by default. Pass --execute to actually move.
"""
import os, re, shutil, argparse
from pathlib import Path

CODE = str(Path(__file__).resolve().parent.parent)

# Patterns to MOVE to pre/ (冗余):
MOVE_PATTERNS = [
    # NOTE: weather 实验是论文附录 §weather_ablation 需要的，不能移！
    re.compile(r"DLinear"),              # excluded baseline
    re.compile(r"MetaEformer"),          # excluded baseline
    re.compile(r"_test_|_test$"),        # test-only runs
    re.compile(r"etth"),                 # wrong dataset

    # 424-weight runs that will be replaced by 244 re-runs:
    re.compile(r"peak_detect_ltf_electricity_mixed_proposed_model_maxIn_maxOut_MSE_244_23_168_(336|720)_\d+$"),
    re.compile(r"peak_detect_ltf_basic_electricity_mixed_proposed_model_maxIn_maxOut_MSE_244_23_168_(336|720)_\d+$"),
    re.compile(r"ablation_mask_hard_336"),
    re.compile(r"ablation_nscales_\d"),
]

# Patterns to KEEP (override MOVE if matched):
# These are explicitly needed even though they match a MOVE pattern.
KEEP_PATTERNS = [
    # Keep nothing extra for now — all MOVE patterns are definitive.
]

def should_move(dirname):
    for kp in KEEP_PATTERNS:
        if kp.search(dirname):
            return False
    for mp in MOVE_PATTERNS:
        if mp.search(dirname):
            return True
    return False

def organize_folder(folder, execute=False):
    if not os.path.isdir(folder):
        print(f"  [skip] {folder} does not exist")
        return 0, 0
    pre = os.path.join(folder, "pre")
    items = [d for d in os.listdir(folder) if d != "pre" and os.path.isdir(os.path.join(folder, d))]
    # Also handle stray .txt files (mean_metrics_*)
    txt_files = [f for f in os.listdir(folder) if f.startswith("mean_metrics_") and f.endswith(".txt")]

    moved = 0
    kept = 0
    for d in items:
        if should_move(d):
            if execute:
                os.makedirs(pre, exist_ok=True)
                src = os.path.join(folder, d)
                dst = os.path.join(pre, d)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            moved += 1
        else:
            kept += 1

    for f in txt_files:
        if execute:
            os.makedirs(pre, exist_ok=True)
            src = os.path.join(folder, f)
            dst = os.path.join(pre, f)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        moved += 1

    return moved, kept

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true",
                    help="Actually move dirs. Without this flag, dry-run only.")
    args = ap.parse_args()

    folders = [
        os.path.join(CODE, "results"),
        os.path.join(CODE, "checkpoints"),
        os.path.join(CODE, "visualization/output/predictions"),
    ]

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"=== {mode} ===\n")

    for folder in folders:
        print(f"Folder: {folder}")
        moved, kept = organize_folder(folder, execute=args.execute)
        print(f"  → {moved} dirs to move to pre/, {kept} dirs to keep\n")

if __name__ == "__main__":
    main()
