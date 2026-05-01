"""
Build the single master table that every paper figure reads from.

Output: paper_data/figure_data.csv

This script replaces the ad-hoc mock data in viz/ with real numbers aggregated
from the parsed experiment logs. It understands the three data sources that
feed each figure:

    1. Old runs (already on disk before this project started) — default-config
       baselines, main ablations, K-sweep (nscales), λ-sweep (lossw), hard-mask.
       These live in results/*/experiment_log.txt and are discovered by
       parse_epoch_time.py. They usually LACK the new [PARAMS]/[MEMORY]/[INFER]
       markers.

    2. New runs launched for this paper — the PeakFocus d_model/e_layers/d_ff
       sweeps. These are tagged with custom model_id suffixes like
       "_dmodel_128", "_elayers_2", "_dff_512".
       They include the new instrumentation (num_params/peak_mem/infer_ms).

    3. Baseline num_params — computed offline by count_baseline_params.py and
       merged in. Old baseline logs don't have them, but we can compute them
       analytically since the model definitions are deterministic.

The output is a wide table with a `figure` discriminator column so every plot
can simply do:

    df = pd.read_csv("paper_data/figure_data.csv")
    main_df = df[df.figure == "main"]
    ablation_df = df[df.figure == "ablation"]

Usage:
    python tools/build_figure_data.py \
        --parsed paper_data/parsed_logs.csv \
        --baseline_params paper_data/baseline_params.csv \
        --out paper_data/figure_data.csv

If --parsed doesn't exist yet, the script will call parse_epoch_time.py first.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_CODE = HERE.parent  # Code/


DATASET_MAP = {
    "load_data_mixed":   "WLEL",
    "electricity_mixed": "ELC",
}

# Baselines we deliberately exclude from every figure.
# MetaEformer: too large, not part of the baseline list.
# DLinear: not in the user's baseline list either.
EXCLUDED_MODELS = {"MetaEformer", "DLinear"}

# The 8 baseline models the paper reports. Seq2Peak is a peak-aware baseline
# that lives under its own task (peak_detect_ltf_seq2peak), the rest are
# vanilla long-term forecasting baselines evaluated under peak_detect_ltf_basic.
WANTED_BASELINES = {
    "CycleNet", "PatchTST", "SegRNN", "STID",
    "TimeMixer", "Transformer", "Informer", "Seq2Peak",
}


# ===========================================================================
# Classification of a single parsed-log row into a figure category
# ===========================================================================
#
# Each classifier returns either None (this row does not belong to the figure)
# or a dict of extra columns to attach (variant, value, sweep_var, etc.).
# A single row can contribute to multiple figures (e.g. the default baseline
# feeds both the main table AND the generality chart as "Vanilla").
# ===========================================================================

def _is_default(model_id: str) -> bool:
    """True for the canonical default run that every baseline has. We accept
    the few name variants that live across 'proposed', 'basic' and 'seq2peak'
    subdirectories (different old scripts dropped the MSE_ suffix)."""
    return model_id in (
        "maxIn_maxOut_MSE_244_23",
        "maxIn_maxOut_244_23",              # Seq2Peak WLEL variant
        "model_maxIn_maxOut_MSE_244_23",
        "electricity_mixed_peak_Transformer_maxIn_maxOut_MSE_244_23",
        "load_data_mixed_peak_Transformer_maxIn_maxOut_244_23",
    )


def _is_proposed(model: str) -> bool:
    # In parsed_logs the proposed model is saved as "proposed"
    return model.strip().lower() in ("proposed", "peakfocus", "proposed_model")


# --- per-figure classifiers ------------------------------------------------

def classify_main(row) -> dict | None:
    """Main result table.
    All baselines (including PatchTST) use the default model_id.
    PatchTST now uses d_model=256, d_ff=512 as its standard config.
    """
    if not _is_default(row["model_id"]):
        return None
    task = row.get("task_name", "")
    model = row["model"]

    if _is_proposed(model) and task == "peak_detect_ltf":
        return dict(figure="main", variant="PeakFocus", model="PeakFocus",
                    sweep_var=None, value=None)

    if model == "Seq2Peak" and task == "peak_detect_ltf_seq2peak":
        return dict(figure="main", variant="Seq2Peak", model="Seq2Peak",
                    sweep_var=None, value=None)

    if model in WANTED_BASELINES and model != "Seq2Peak" \
            and task == "peak_detect_ltf":
        return dict(figure="main", variant=model, model=model,
                    sweep_var=None, value=None)

    return None


def classify_ablation(row) -> dict | None:
    """Five-variant ablation. All variants must come from the
    `peak_detect_ltf` task (the full PeakFocus pipeline) — the same
    `model_maxIn_maxOut_MSE_244_23` model_id also exists under the
    peak_detect_ltf_basic task, which is a pure MSE-loss baseline and must
    not be conflated with the full PeakFocus.
    """
    mid = row["model_id"]
    if not _is_proposed(row["model"]):
        return None
    if row.get("task_name", "") != "peak_detect_ltf":
        return None
    variant = None
    if mid == "model_maxIn_maxOut_MSE_244_23":
        variant = "PeakFocus"
    elif mid == "model_maxIn_maxOut_MSE_244_23_wo_msm_pl":
        variant = "w/o MSM-PL"
    elif mid == "model_maxIn_maxOut_MSE_244_23_wo_lad":
        variant = "w/o LAD"
    elif mid == "model_maxIn_maxOut_MSE_244_23_wo_upap":
        # Canonical w/o UPAP: keep MSM-PL+LAD arch, zero peak+tp_mse losses.
        # New runs use this model_id via ablation_wo_upap_{wlel,elc}.sh.
        # Takes priority over the old nscales_0 proxy below.
        variant = "w/o UPAP"
    elif mid == "model_ablation_nscales_0":
        # Legacy proxy (K=0 collapses MSM-PL); kept only as fallback until
        # the canonical w/o UPAP runs land. Delete once figure_data covers
        # both datasets with the canonical model_id.
        variant = "w/o UPAP"
    elif mid == "model_ablation_mask_hard_336":
        variant = "Hard mask"
    if variant is None:
        return None
    return dict(figure="ablation", variant=variant, model="PeakFocus",
                sweep_var=None, value=None)


def classify_param_K(row) -> dict | None:
    """MSM-PL depth sweep K ∈ {0,1,2,3}."""
    m = re.match(r"model_ablation_nscales_(\d+)", str(row["model_id"]))
    if not m:
        return None
    return dict(figure="param_K", variant="PeakFocus", model="PeakFocus",
                sweep_var="K", value=float(m.group(1)))


def classify_param_lambda(row) -> dict | None:
    """Loss weight λ_peak sweep.

    Old logs encode the three weights as `lossw_{V}_{P}_{T}`. The paper
    sweep: this is not a single-dim axis, it is a tuple (λ₁, λ₂, λ₃) over a
    fixed list of configurations. We encode it as `value = 1..N` (config
    index, stable ordering = sorted by model_id) and store the full triple
    string in `lambda_triple` so plots can use it as the x-tick label.

    Mapping: shell-level (V, P, T) → paper (λ₁ Global, λ₃ Peak BCE,
    λ₂ TP-MSE). So the string we emit is "(λ₁,λ₂,λ₃)" = "(V,T,P)".
    """
    m = re.match(r"model_ablation_lossw_([\d.]+)_([\d.]+)_([\d.]+)",
                 str(row["model_id"]))
    if not m:
        return None
    V, P, T = float(m.group(1)), float(m.group(2)), float(m.group(3))
    # Stable ordering: sort the four known configs by (V, P, T) tuple so
    # that the plot x-axis is deterministic regardless of read order.
    CONFIG_ORDER = [
        (0.2, 0.4, 0.4),   # = WLEL default-aligned config (λ₁,λ₃,λ₂ shell order)
        (0.2, 0.6, 0.2),   # heavy peak classification
        (0.2, 0.2, 0.6),   # heavy TP-MSE
        (0.33, 0.33, 0.33),
        (0.6, 0.2, 0.2),
    ]
    try:
        idx = CONFIG_ORDER.index((V, P, T)) + 1
    except ValueError:
        return None
    # paper label ordering is (λ₁, λ₂, λ₃) = (V, T, P)
    triple_label = f"({V:g}, {T:g}, {P:g})"
    return dict(figure="param_lambda", variant="PeakFocus", model="PeakFocus",
                sweep_var="lambda_config", value=float(idx),
                loss_V=V, loss_P=P, loss_T=T, lambda_triple=triple_label)


def classify_param_dmodel(row) -> dict | None:
    """Hidden dim sweep. Default=256 comes from the plain default run;
    64/128/512 come from new sweep runs with _dmodel_* suffix.

    Must restrict to `peak_detect_ltf` task: the same default model_id also
    exists under peak_detect_ltf_basic (pure MSE, no UPAP), which would
    pollute the sweep point with non-PeakFocus numbers.
    """
    if row.get("task_name", "") != "peak_detect_ltf":
        return None
    mid = str(row["model_id"])
    if _is_default(mid) and _is_proposed(row["model"]):
        return dict(figure="param_dmodel", variant="PeakFocus",
                    model="PeakFocus", sweep_var="d_model", value=256.0)
    m = re.search(r"dmodel_(\d+)", mid)
    if m and _is_proposed(row["model"]):
        return dict(figure="param_dmodel", variant="PeakFocus",
                    model="PeakFocus", sweep_var="d_model",
                    value=float(m.group(1)))
    return None


def classify_param_elayers(row) -> dict | None:
    """Encoder depth sweep. Default=1. Restricted to peak_detect_ltf."""
    if row.get("task_name", "") != "peak_detect_ltf":
        return None
    mid = str(row["model_id"])
    if _is_default(mid) and _is_proposed(row["model"]):
        return dict(figure="param_elayers", variant="PeakFocus",
                    model="PeakFocus", sweep_var="e_layers", value=1.0)
    m = re.search(r"elayers_(\d+)", mid)
    if m and _is_proposed(row["model"]):
        return dict(figure="param_elayers", variant="PeakFocus",
                    model="PeakFocus", sweep_var="e_layers",
                    value=float(m.group(1)))
    return None


def classify_param_dff(row) -> dict | None:
    """Feed-forward dim sweep. Default=256. Restricted to peak_detect_ltf."""
    if row.get("task_name", "") != "peak_detect_ltf":
        return None
    mid = str(row["model_id"])
    if _is_default(mid) and _is_proposed(row["model"]):
        return dict(figure="param_dff", variant="PeakFocus",
                    model="PeakFocus", sweep_var="d_ff", value=256.0)
    m = re.search(r"dff_(\d+)", mid)
    if m and _is_proposed(row["model"]):
        return dict(figure="param_dff", variant="PeakFocus",
                    model="PeakFocus", sweep_var="d_ff",
                    value=float(m.group(1)))
    return None




def classify_generality(row) -> dict | None:
    """Generality bar: for each vanilla backbone, compare Vanilla vs +UPAP.

        Vanilla  = task_name peak_detect_ltf_basic   (pure MSE training)
        +UPAP    = task_name peak_detect_ltf         (baseline + peak_head +
                   UPAP soft-mask + peak loss, i.e. UPAP plug-and-play only)

    Note: LAD (Location-Aware Decoder) is an internal module of
    proposed_model, NOT a plug-and-play component — it is never injected
    into baselines. So the old `peak_detect_ltf` task only adds UPAP to the
    baseline, not UPAP+LAD.

    Seq2Peak is a peak-aware model by design, so it has no "vanilla vs UPAP"
    split — we exclude it from the generality chart.
    """
    if _is_proposed(row["model"]):
        return None
    if row["model"] not in WANTED_BASELINES or row["model"] == "Seq2Peak":
        return None
    if not _is_default(row["model_id"]):
        return None

    task = row.get("task_name", "")
    if task == "peak_detect_ltf_basic":
        variant = "Vanilla"
    elif task == "peak_detect_ltf":
        variant = "+UPAP"
    else:
        return None
    return dict(figure="generality", variant=variant, model=row["model"],
                sweep_var=None, value=None)


CLASSIFIERS = [
    classify_main,
    classify_ablation,
    classify_param_K,
    classify_param_lambda,
    classify_param_dmodel,
    classify_param_elayers,
    classify_param_dff,
    classify_generality,
]


# ===========================================================================
# Aggregation
# ===========================================================================

METRIC_COLS = ["MSE", "MAE", "F1", "BPE", "PIM", "TP_MSE", "TP_MAE",
               "Precision", "Recall"]
SUPPORT_COLS = ["num_params", "avg_epoch_s", "total_train_s",
                "peak_mem_mb", "infer_ms", "wall_clock_s"]


def aggregate_group(sub: pd.DataFrame, tag_cols: dict) -> dict:
    """Aggregate one (figure, dataset, horizon, variant, value) group into a
    single wide row of mean±std statistics."""
    row: dict[str, Any] = dict(tag_cols)
    row["num_seeds"] = int(sub.shape[0])

    for m in METRIC_COLS:
        vals = sub[m].dropna().values
        row[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
        row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    for c in SUPPORT_COLS:
        vals = sub[c].dropna().values
        row[c] = float(np.mean(vals)) if len(vals) else np.nan

    return row


def build(parsed_csv: Path, baseline_params_csv: Path | None,
          out_csv: Path) -> pd.DataFrame:
    if not parsed_csv.exists():
        print(f"[info] {parsed_csv} not found — running parse_epoch_time.py first")
        subprocess.run(
            [sys.executable,
             str(HERE / "parse_epoch_time.py"),
             "--results_root", "results",
             "--out", str(parsed_csv)],
            cwd=REPO_CODE, check=True,
        )

    df = pd.read_csv(parsed_csv)
    df = df[df["completed"] == True].copy()  # noqa: E712
    df = df[~df["model"].isin(EXCLUDED_MODELS)].copy()
    df["dataset"] = df["data"].map(DATASET_MAP).fillna(df["data"])
    df["horizon"] = df["pred_len"]

    # Merge in profiled baseline inference latency (ms/sample) if present.
    # Same alias rule as num_params: old logs use "proposed", the csv uses
    # "PeakFocus".
    infer_csv = REPO_CODE / "paper_data" / "baseline_infer.csv"
    if infer_csv.exists():
        bi = pd.read_csv(infer_csv)
        merge_key = df["model"].replace(
            {"proposed": "PeakFocus", "proposed_model": "PeakFocus"}
        )
        df["_merge_model"] = merge_key
        df = df.merge(
            bi[["model", "dataset", "horizon", "infer_ms"]].rename(
                columns={"model": "_merge_model",
                         "infer_ms": "infer_ms_calc"}),
            on=["_merge_model", "dataset", "horizon"], how="left",
        )
        df = df.drop(columns=["_merge_model"])
        df["infer_ms"] = df["infer_ms"].fillna(df["infer_ms_calc"])
        df = df.drop(columns=["infer_ms_calc"])

    # Merge in analytically-computed baseline num_params if provided. This only
    # fills NaN cells so that real [PARAMS] readings from new runs win.
    if baseline_params_csv is not None and baseline_params_csv.exists():
        bp = pd.read_csv(baseline_params_csv)
        # baseline_params.csv may contain multiple rows per (model, horizon)
        # when a single baseline has several hand-picked configs — PatchTST has
        # a "fair" and an "unfair" row. We deduplicate by keeping the
        # "default" config (or the first one) so the merge doesn't blow up
        # the dataframe with a Cartesian product. The `_fair` variant carries
        # its own num_params via the NEW logs ([PARAMS] marker), which is why
        # we don't need PatchTST_fair here.
        bp = bp.drop_duplicates(subset=["model", "horizon"], keep="first")
        bp = bp.drop_duplicates(subset=["model", "horizon"], keep="first")

        # The log model column uses "proposed" while baseline_params uses the
        # display name "PeakFocus". Bridge the two via an alias lookup.
        merge_key = df["model"].replace(
            {"proposed": "PeakFocus", "proposed_model": "PeakFocus"}
        )
        df["_merge_model"] = merge_key
        df = df.merge(
            bp[["model", "horizon", "num_params"]].rename(
                columns={"model": "_merge_model",
                         "num_params": "num_params_calc"}),
            on=["_merge_model", "horizon"], how="left",
        )
        df = df.drop(columns=["_merge_model"])
        df["num_params"] = df["num_params"].fillna(df["num_params_calc"])
        df = df.drop(columns=["num_params_calc"])

    # Classify every row into zero or more figure categories
    figure_rows: list[dict] = []
    for _, row in df.iterrows():
        for cls in CLASSIFIERS:
            tag = cls(row)
            if tag is None:
                continue
            entry = dict(row)
            entry.update(tag)
            figure_rows.append(entry)

    if not figure_rows:
        print("[warn] no rows matched any figure classifier — nothing to write")
        return pd.DataFrame()

    tag = pd.DataFrame(figure_rows)

    # Group keys differ per figure type, so we do it figure-by-figure
    out_rows: list[dict] = []
    group_key_by_fig = {
        "main":          ["figure", "dataset", "horizon", "variant", "model"],
        "ablation":      ["figure", "dataset", "horizon", "variant", "model"],
        "param_K":       ["figure", "dataset", "horizon", "sweep_var", "value"],
        "param_lambda":  ["figure", "dataset", "horizon", "sweep_var", "value"],
        "param_dmodel":  ["figure", "dataset", "horizon", "sweep_var", "value"],
        "param_elayers": ["figure", "dataset", "horizon", "sweep_var", "value"],
        "param_dff":     ["figure", "dataset", "horizon", "sweep_var", "value"],
        "generality":    ["figure", "dataset", "horizon", "variant", "model"],
    }
    for fig_name, keys in group_key_by_fig.items():
        sub = tag[tag["figure"] == fig_name]
        if sub.empty:
            continue
        for grp_vals, grp_df in sub.groupby(keys, dropna=False):
            tag_cols = dict(zip(keys, grp_vals))
            # carry variant/sweep_var/value even if not a group key, so the
            # output schema is uniform
            for c in ("variant", "sweep_var", "value", "model", "lambda_triple"):
                if c not in tag_cols and c in grp_df.columns:
                    tag_cols[c] = grp_df[c].iloc[0]
            out_rows.append(aggregate_group(grp_df, tag_cols))

    result = pd.DataFrame(out_rows)

    # Canonical column order
    leading = ["figure", "dataset", "horizon", "model", "variant",
               "sweep_var", "value", "lambda_triple", "num_seeds"]
    metric_cols = sum(
        ([f"{m}_mean", f"{m}_std"] for m in METRIC_COLS), []
    )
    tail = metric_cols + SUPPORT_COLS
    ordered = leading + [c for c in tail if c in result.columns]
    result = result[[c for c in ordered if c in result.columns]]
    result = result.sort_values(
        ["figure", "dataset", "horizon", "variant", "value"],
        na_position="last",
    ).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    print(f"[ok] wrote {len(result)} rows → {out_csv}")

    # Print a figure-by-figure summary
    print("\n=== Figure coverage ===")
    counts = result.groupby(["figure", "dataset", "horizon"]).size().unstack(
        fill_value=0)
    print(counts)

    # Flag missing support columns so we know what still needs to be filled
    print("\n=== Support-column fill rate ===")
    for c in SUPPORT_COLS:
        if c in result.columns:
            filled = result[c].notna().sum()
            print(f"  {c:<15}  {filled:3}/{len(result)}")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default=str(REPO_CODE / "paper_data" / "parsed_logs.csv"))
    ap.add_argument("--baseline_params",
                    default=str(REPO_CODE / "paper_data" / "baseline_params.csv"),
                    help="Optional CSV with analytically-computed num_params "
                         "for baseline models (filled when log data lacks it).")
    ap.add_argument("--out", default=str(REPO_CODE / "paper_data" / "figure_data.csv"))
    args = ap.parse_args()

    build(Path(args.parsed),
          Path(args.baseline_params) if args.baseline_params else None,
          Path(args.out))


if __name__ == "__main__":
    main()
