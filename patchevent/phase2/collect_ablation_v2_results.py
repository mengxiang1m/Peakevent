"""Collect Hybrid DETR-AR ablation results.

Phase-1 encoder metrics and standalone AR ablations are intentionally excluded
from the new mainline summary.
"""

from __future__ import annotations

import json
import math
import os
from statistics import mean, stdev


SEEDS = [42, 123, 456]
DOMAINS = ["wlel", "ett", "elc"]

HYBRID_CONFIGS = [
    "H0_hybrid",
    "HQ8_queries",
    "HQ12_queries",
    "HR0_no_refine",
    "HR2_refine2",
    "HM_no_intensity_cost",
    "HP_no_proposal_aux",
]

HYBRID_BASE = "patchevent/phase2/checkpoints/{domain}_hybrid_ablation"
OUT_JSON = "patchevent/phase2/checkpoints/hybrid_ablation_summary.json"
OUT_MD = "patchevent/phase2/checkpoints/hybrid_ablation_summary.md"
METRICS = ["event_f1", "onset_mae", "apex_mae", "duration_mae", "intensity_mape"]


def _read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metrics_path(base: str, name: str, seed: int) -> str:
    return os.path.join(base, f"{name}_s{seed}", "test_eval", "eval_metrics.json")


def _agg(rows: list[dict], key: str):
    vals = []
    for row in rows:
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if not (math.isnan(value) or math.isinf(value)):
            vals.append(value)
    if not vals:
        return {"mean": None, "std": None}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0}
    return {"mean": mean(vals), "std": stdev(vals)}


def _collect_group(base_tmpl: str, names: list[str]) -> dict:
    out = {}
    for domain in DOMAINS:
        base = base_tmpl.format(domain=domain)
        out[domain] = {}
        for name in names:
            rows = []
            per_seed = {}
            for seed in SEEDS:
                metrics = _read_json(_metrics_path(base, name, seed))
                if metrics is None:
                    continue
                per_seed[str(seed)] = metrics
                rows.append(metrics)
            out[domain][name] = {
                "n_done": len(rows),
                "per_seed": per_seed,
                "summary": {metric: _agg(rows, metric) for metric in METRICS},
            }
    return out


def _fmt(value):
    if value is None:
        return "NA"
    return f"{value:.4f}"


def _render_table(title: str, data: dict, names: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    lines.append("| Domain | Config | n | F1 (mean+/-std) | OnMAE | ApMAE | DurMAE | IntMAPE |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for domain in DOMAINS:
        for name in names:
            row = data.get(domain, {}).get(name, {})
            summary = row.get("summary", {})
            f1 = summary.get("event_f1", {})
            f1_text = "NA" if f1.get("mean") is None else f"{f1['mean']:.4f}+/-{(f1.get('std') or 0.0):.4f}"
            lines.append(
                f"| {domain} | {name} | {row.get('n_done', 0)} | {f1_text} | "
                f"{_fmt(summary.get('onset_mae', {}).get('mean'))} | "
                f"{_fmt(summary.get('apex_mae', {}).get('mean'))} | "
                f"{_fmt(summary.get('duration_mae', {}).get('mean'))} | "
                f"{_fmt(summary.get('intensity_mape', {}).get('mean'))} |"
            )
    lines.append("")
    return lines


def main():
    hybrid = _collect_group(HYBRID_BASE, HYBRID_CONFIGS)
    summary = {
        "seeds": SEEDS,
        "domains": DOMAINS,
        "note": "Hybrid DETR-AR mainline summary. Phase-1 metrics and standalone AR ablations are excluded.",
        "hybrid_ablation": hybrid,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {OUT_JSON}")

    lines = ["# Hybrid DETR-AR Ablation Summary", ""]
    lines.extend(_render_table("Hybrid Internal Ablation", hybrid, HYBRID_CONFIGS))
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[SAVE] {OUT_MD}")


if __name__ == "__main__":
    main()
