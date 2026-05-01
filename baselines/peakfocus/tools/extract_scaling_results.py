"""
Filter the full parsed_logs.csv for scaling-experiment runs and aggregate
mean±std across the 5 seeds. Output a tidy CSV matching the schema expected
by viz/common.load_results().

Usage:
    python tools/extract_scaling_results.py \
        --parsed paper_data/parsed_logs.csv \
        --out paper_data/results.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


# Tags injected by our shell scripts into model_id:
#   maxIn_maxOut_MSE_244_23_dmodel_<D>
#   maxIn_maxOut_MSE_244_23_elayers_<E>
#   maxIn_maxOut_MSE_244_23_dff_<F>
#   maxIn_maxOut_MSE_244_23_fair          (PatchTST fair re-run)
#   maxIn_maxOut_MSE_244_23               (original default, already-existing runs)
TAG_RX = {
    "d_model":   re.compile(r"dmodel_(\d+)"),
    "e_layers":  re.compile(r"elayers_(\d+)"),
    "d_ff":      re.compile(r"dff_(\d+)"),
}

DATASET_MAP = {
    "load_data_mixed":   "WLEL",
    "electricity_mixed": "ELC",
}


def classify(row) -> tuple[str, float | None]:
    """Return (experiment_type, value). experiment_type ∈ {d_model, e_layers, d_ff, main_default, fair, other}."""
    mid = row.get("model_id", "") or ""
    model = row.get("model", "") or ""
    for exp, rx in TAG_RX.items():
        m = rx.search(mid)
        if m:
            return exp, float(m.group(1))
    if mid.endswith("_fair"):
        return "fair", None
    if mid == "maxIn_maxOut_MSE_244_23":
        return "main_default", None
    return "other", None


def aggregate(parsed_csv: Path, out_csv: Path):
    df = pd.read_csv(parsed_csv)
    df = df[df["completed"] == True].copy()  # noqa: E712

    # Classify each run
    exp_type = []
    exp_val = []
    for _, row in df.iterrows():
        t, v = classify(row)
        exp_type.append(t)
        exp_val.append(v)
    df["experiment"] = exp_type
    df["value"] = exp_val

    # Map data → dataset label
    df["dataset"] = df["data"].map(DATASET_MAP).fillna(df["data"])

    metric_cols = ["MSE", "MAE", "F1", "BPE", "PIM", "TP_MSE", "TP_MAE", "Precision", "Recall"]
    support_cols = ["num_params", "avg_epoch_s", "total_train_s",
                    "peak_mem_mb", "infer_ms", "wall_clock_s"]
    group_keys = ["experiment", "model", "dataset", "pred_len", "value", "model_id"]

    # Aggregate: mean/std for metrics, first(num_params)/mean(times/memory) for support
    agg_dict = {m: ["mean", "std"] for m in metric_cols}
    for c in support_cols:
        agg_dict[c] = "mean"

    grouped = df.groupby(group_keys, dropna=False).agg(agg_dict)
    grouped.columns = [
        f"{c}_{s}" if s in ("mean", "std") else c
        for c, s in grouped.columns.to_flat_index()
    ]
    grouped = grouped.reset_index()

    # Horizon column to match load_results() schema
    grouped["horizon"] = grouped["pred_len"]

    # Final column order
    cols = ["experiment", "model", "dataset", "horizon", "value", "model_id",
            "num_params", "avg_epoch_s", "total_train_s",
            "peak_mem_mb", "infer_ms", "wall_clock_s",
            "MSE_mean", "MSE_std",
            "MAE_mean", "MAE_std",
            "F1_mean", "F1_std",
            "BPE_mean", "BPE_std",
            "TP_MSE_mean", "TP_MSE_std",
            "PIM_mean", "PIM_std",
            "Precision_mean", "Recall_mean"]
    grouped = grouped[[c for c in cols if c in grouped.columns]]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_csv, index=False)
    print(f"Aggregated {len(df)} runs into {len(grouped)} rows → {out_csv}")

    # Print a summary
    print("\nPer-experiment counts:")
    print(df.groupby(["experiment", "dataset", "pred_len"]).size())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default="paper_data/parsed_logs.csv")
    ap.add_argument("--out", default="paper_data/results.csv")
    args = ap.parse_args()
    aggregate(Path(args.parsed), Path(args.out))


if __name__ == "__main__":
    main()
