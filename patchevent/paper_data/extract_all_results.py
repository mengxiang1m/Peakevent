#!/usr/bin/env python3
"""Extract all experiment results into paper-ready CSV files.

Usage:
    python patchevent/paper_data/extract_all_results.py

Outputs paper_data/*.csv, one per paper table.
Mirrors PeakFocus's paper_data/ convention.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
P2_CKPT = ROOT / "patchevent" / "phase2" / "checkpoints"
BL_SUMMARY = ROOT / "baselines" / "results" / "posthoc" / "summary_agg.csv"
OUT = pathlib.Path(__file__).resolve().parent

SEEDS = [42, 2024, 7]
DOMAINS = ["wlel", "ett", "elc"]
METRICS = ["test_event_f1", "test_onset_mae", "test_apex_mae",
           "test_duration_mae", "test_intensity_mape"]
SHORT = {"test_event_f1": "F1", "test_onset_mae": "OnsetMAE",
         "test_apex_mae": "ApexMAE", "test_duration_mae": "DurMAE",
         "test_intensity_mape": "IntMAPE"}


# ── helpers ──────────────────────────────────────────────────────────

def _load_summaries(ckpt_dir: pathlib.Path, pattern: str) -> list[dict]:
    """Load all test_summary.json under ckpt_dir matching a glob pattern."""
    rows = []
    for p in sorted(ckpt_dir.glob(pattern)):
        if p.name != "test_summary.json":
            continue
        with open(p) as f:
            d = json.load(f)
        # parse path: {domain}_{exp}/{variant}_{seed}/test_summary.json
        parts = p.relative_to(ckpt_dir).parts
        exp_dir = parts[0]          # e.g. wlel_arch_ablation_v2
        run_dir = parts[1]          # e.g. G0_full_s42
        # extract domain from exp_dir prefix
        for dom in DOMAINS:
            if exp_dir.startswith(dom + "_"):
                domain = dom
                exp_name = exp_dir[len(dom) + 1:]
                break
        else:
            domain = "unknown"
            exp_name = exp_dir
        # extract seed from run_dir suffix: pattern is {variant}_s{seed}
        import re
        seed_match = re.search(r'_s(\d+)$', run_dir)
        if seed_match:
            seed = int(seed_match.group(1))
            variant = run_dir[:seed_match.start()]
        else:
            seed = None
            variant = run_dir

        row = {"domain": domain, "experiment": exp_name, "variant": variant,
               "seed": seed, **{SHORT.get(k, k): v for k, v in d.items()
                                if k in METRICS}}
        rows.append(row)
    return rows


def _agg(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate over seeds: mean ± std."""
    metric_cols = [c for c in df.columns if c in SHORT.values()]
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
    # flatten multi-level columns
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def _fmt(mean, std, prec=3):
    """Format as mean±std string."""
    if pd.isna(mean):
        return ""
    return f"{mean:.{prec}f}±{std:.{prec}f}"


# ── extractors ───────────────────────────────────────────────────────

def extract_parsed_logs():
    """Master table: every single run across all experiments."""
    rows = _load_summaries(P2_CKPT, "*/*/test_summary.json")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "parsed_logs.csv", index=False)
    print(f"  parsed_logs.csv: {len(df)} runs")
    return df


def extract_table_main(df: pd.DataFrame):
    """tab:main — WLEL component study (arch_ablation_v2)."""
    sub = df[df["experiment"] == "arch_ablation_v2"].copy()
    agg = _agg(sub, ["domain", "variant"])
    agg.to_csv(OUT / "table_main_arch_ablation.csv", index=False)
    print(f"  table_main_arch_ablation.csv: {len(agg)} rows")


def extract_table_hybrid_ablation(df: pd.DataFrame):
    """tab:hybrid_abl — Hybrid DETR-AR internal ablation."""
    sub = df[df["experiment"] == "hybrid_ablation"].copy()
    agg = _agg(sub, ["domain", "variant"])
    agg.to_csv(OUT / "table_hybrid_ablation.csv", index=False)
    print(f"  table_hybrid_ablation.csv: {len(agg)} rows")


def extract_table_baseline():
    """tab:baseline — External baseline comparison."""
    if not BL_SUMMARY.exists():
        print("  SKIP table_baseline.csv (no baselines summary)")
        return
    bl = pd.read_csv(BL_SUMMARY)
    bl.rename(columns={
        "event_f1_mean": "F1_mean", "event_f1_std": "F1_std",
        "onset_mae_mean": "OnsetMAE_mean", "onset_mae_std": "OnsetMAE_std",
        "apex_mae_mean": "ApexMAE_mean", "apex_mae_std": "ApexMAE_std",
        "duration_mae_mean": "DurMAE_mean", "duration_mae_std": "DurMAE_std",
        "intensity_mape_mean": "IntMAPE_mean", "intensity_mape_std": "IntMAPE_std",
    }, inplace=True)
    bl.to_csv(OUT / "table_baseline.csv", index=False)
    print(f"  table_baseline.csv: {len(bl)} rows")


def extract_table_horizon(df: pd.DataFrame):
    """tab:horizon — Horizon scalability (pred96/168/336)."""
    rows = []
    for dom in DOMAINS:
        for h, exp in [(96, "arch_ablation_v2"), (168, "pred168"), (336, "pred336")]:
            sub = df[(df["domain"] == dom) & (df["experiment"] == exp)]
            if exp == "arch_ablation_v2":
                sub = sub[sub["variant"] == "G0_full"]
            if len(sub) == 0:
                continue
            row = {"domain": dom, "horizon": h,
                   "F1_mean": sub["F1"].mean(), "F1_std": sub["F1"].std(),
                   "OnsetMAE_mean": sub["OnsetMAE"].mean(),
                   "OnsetMAE_std": sub["OnsetMAE"].std()}
            rows.append(row)
    hdf = pd.DataFrame(rows)
    hdf.to_csv(OUT / "table_horizon.csv", index=False)
    print(f"  table_horizon.csv: {len(hdf)} rows")


def extract_table_hw(df: pd.DataFrame):
    """tab:hw_arch + tab:hw_ar — Stronger weight ablations."""
    for exp, name in [("hw_ablation", "hw_arch"), ("hw_ablation", "hw_ar")]:
        sub = df[(df["experiment"] == exp)].copy()
        if len(sub) == 0:
            # try exact match
            sub = df[df["experiment"].str.contains("hw")].copy()
        agg = _agg(sub, ["domain", "variant"])
        agg.to_csv(OUT / f"table_{name}.csv", index=False)
        print(f"  table_{name}.csv: {len(agg)} rows")


def extract_table_grid_search(df: pd.DataFrame):
    """Hyperparameter grid search results."""
    sub = df[df["experiment"].str.contains("grid_search|hparam_search")].copy()
    if len(sub) == 0:
        print("  SKIP table_grid_search.csv (no data)")
        return
    agg = _agg(sub, ["domain", "experiment", "variant"])
    agg.to_csv(OUT / "table_grid_search.csv", index=False)
    print(f"  table_grid_search.csv: {len(agg)} rows")


def extract_table_cross_domain(df: pd.DataFrame):
    """tab:cross — Cross-domain validation."""
    sub = df[df["experiment"] == "arch_ablation_v2"].copy()
    # Select key variants for cross-domain comparison
    key_variants = ["H0_hybrid", "HQ8_queries", "HQ12_queries", "HR2_refine2"]
    sub = sub[sub["variant"].isin(key_variants)]
    agg = _agg(sub, ["domain", "variant"])
    agg.to_csv(OUT / "table_cross_domain.csv", index=False)
    print(f"  table_cross_domain.csv: {len(agg)} rows")


def generate_readme(df: pd.DataFrame):
    """Generate README.md for paper_data/."""
    n_runs = len(df)
    n_experiments = df["experiment"].nunique()
    n_domains = df["domain"].nunique()
    csvs = sorted(OUT.glob("*.csv"))

    lines = [
        "# Paper Data",
        "",
        f"Auto-generated experiment results for paper tables. "
        f"Total: {n_runs} runs across {n_experiments} experiments × {n_domains} domains.",
        "",
        "## Files",
        "",
        "| File | Description | Rows |",
        "|------|-------------|------|",
    ]
    desc_map = {
        "parsed_logs.csv": "Master table: every single run",
        "table_main_arch_ablation.csv": "tab:main — Component study (arch ablation)",
        "table_hybrid_ablation.csv": "tab:hybrid_abl — Hybrid DETR-AR internal ablation",
        "table_baseline.csv": "tab:baseline — External baseline comparison",
        "table_horizon.csv": "tab:horizon — Horizon scalability",
        "table_hw_arch.csv": "tab:hw_arch — Stronger weight arch ablation",
        "table_hw_ar.csv": "legacy — Stronger weight AR ablation",
        "table_grid_search.csv": "Hyperparameter grid search",
        "table_cross_domain.csv": "tab:cross — Cross-domain validation",
    }
    for csv in csvs:
        if csv.name == "extract_all_results.py":
            continue
        try:
            n = len(pd.read_csv(csv))
        except Exception:
            n = "?"
        desc = desc_map.get(csv.name, csv.name)
        lines.append(f"| `{csv.name}` | {desc} | {n} |")

    lines += [
        "",
        "## Regenerate",
        "",
        "```bash",
        "python patchevent/paper_data/extract_all_results.py",
        "```",
        "",
        "## Paper Table ↔ CSV Mapping",
        "",
        "| Paper Table | CSV File | Experiment Source |",
        "|-------------|----------|-------------------|",
        "| tab:main (WLEL component) | table_main_arch_ablation.csv | phase2/checkpoints/*_arch_ablation_v2 |",
        "| tab:baseline (External) | table_baseline.csv | baselines/results/posthoc/summary_agg.csv |",
        "| tab:hybrid_abl (Hybrid internals) | table_hybrid_ablation.csv | phase2/checkpoints/*_hybrid_ablation |",
        "| tab:cross (Cross-domain) | table_cross_domain.csv | phase2/checkpoints/*_arch_ablation_v2 |",
        "| tab:horizon (Horizon) | table_horizon.csv | phase2/checkpoints/*_pred{168,336} |",
        "| tab:hw_arch/tab:hw_ar | table_hw_*.csv | phase2/checkpoints/*_hw_ablation |",
    ]

    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"  README.md generated")


# ── main ─────────────────────────────────────────────────────────────

def main():
    print("Extracting all experiment results to paper_data/...\n")

    df = extract_parsed_logs()

    extract_table_main(df)
    extract_table_hybrid_ablation(df)
    extract_table_baseline()
    extract_table_horizon(df)
    extract_table_hw(df)
    extract_table_grid_search(df)
    extract_table_cross_domain(df)
    generate_readme(df)

    print(f"\nDone. Files in: {OUT}")


if __name__ == "__main__":
    main()
