from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect post-hoc evaluation results")
    parser.add_argument("--posthoc_root", type=str, default="results/posthoc")
    return parser.parse_args()


def resolve_path(base_dir: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _fmt(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    root = resolve_path(here, args.posthoc_root)
    if not root.exists():
        print(f"[done] no posthoc directory found: {root}")
        return

    rows = []
    for json_path in root.glob("*/*/*.json"):
        if json_path.name == "summary_runs.json":
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

    if not rows:
        print("[done] no posthoc json results found")
        return

    df = pd.DataFrame(rows)
    raw_csv = root / "summary_all_runs.csv"
    df.to_csv(raw_csv, index=False)

    agg = (
        df.groupby(["dataset", "pred_len", "method"], dropna=False)
        .agg(
            n_runs=("run_name", "count"),
            event_f1_mean=("event_f1", "mean"),
            event_f1_std=("event_f1", "std"),
            onset_mae_mean=("onset_mae", "mean"),
            onset_mae_std=("onset_mae", "std"),
            apex_mae_mean=("apex_mae", "mean"),
            apex_mae_std=("apex_mae", "std"),
            duration_mae_mean=("duration_mae", "mean"),
            duration_mae_std=("duration_mae", "std"),
            intensity_mape_mean=("intensity_mape", "mean"),
            intensity_mape_std=("intensity_mape", "std"),
        )
        .reset_index()
        .sort_values(["dataset", "pred_len", "method"])
    )
    agg = agg.fillna(0.0)

    agg_csv = root / "summary_agg.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"[ok] saved {raw_csv}")
    print(f"[ok] saved {agg_csv}")

    for dataset in sorted(agg["dataset"].unique()):
        sub = agg[agg["dataset"] == dataset].copy()
        sub["F1"] = sub.apply(lambda r: _fmt(r["event_f1_mean"], r["event_f1_std"], 3), axis=1)
        sub["Onset MAE"] = sub.apply(lambda r: _fmt(r["onset_mae_mean"], r["onset_mae_std"], 3), axis=1)
        sub["Apex MAE"] = sub.apply(lambda r: _fmt(r["apex_mae_mean"], r["apex_mae_std"], 3), axis=1)
        sub["Dur MAE"] = sub.apply(lambda r: _fmt(r["duration_mae_mean"], r["duration_mae_std"], 3), axis=1)
        sub["Int. MAPE"] = sub.apply(lambda r: _fmt(r["intensity_mape_mean"], r["intensity_mape_std"], 3), axis=1)
        show = sub[["method", "pred_len", "n_runs", "F1", "Onset MAE", "Apex MAE", "Dur MAE", "Int. MAPE"]].rename(
            columns={"method": "Method", "pred_len": "PredLen", "n_runs": "Runs"}
        )
        tex_path = root / f"{dataset}_table.tex"
        show.to_latex(tex_path, index=False, escape=False)
        print(f"[ok] saved {tex_path}")


if __name__ == "__main__":
    main()
