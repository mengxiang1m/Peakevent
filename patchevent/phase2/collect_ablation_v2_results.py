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
    "HO_query_order",
    "HC_no_causal_refine",
    "HM_no_intensity_cost",
    "HP_no_proposal_aux",
    "HN_noobj05",
    "HF_focal2",
    "HT_count_head",
    "HS_structure",
]

HYBRID_V21_CONFIGS = [
    "H0_v21_hybrid",
    "HQ8_queries",
    "HQ12_queries",
    "HNC_no_count_head",
    "HCD_count_train_only",
    "HID_independent_time",
    "HID_independent_time_nocount",
    "HR0_no_refine",
    "HR2_refine2",
    "HO_query_order",
    "HC_no_causal_refine",
    "HM_no_intensity_cost",
    "HP_no_proposal_aux",
    "HN_noobj05",
    "HF_focal2",
    "HS_structure",
]

HYBRID_V22_CONFIGS = [
    "H0_v22_hybrid",
    "HQ8_queries",
    "HQ12_queries",
    "HCT_count_aux_only",
    "HCD_count_decode",
    "HID_independent_time_nocount",
    "HR0_no_refine",
    "HR2_refine2",
    "HO_query_order",
    "HMO_matched_order",
    "HC_no_causal_refine",
    "HM_no_intensity_cost",
    "HP_no_proposal_aux",
    "HN_noobj05",
    "HF_focal2",
    "HS_structure",
]

HYBRID_BASE = "patchevent/phase2/checkpoints/{domain}_hybrid_ablation"
HYBRID_V21_BASE = "patchevent/phase2/checkpoints/{domain}_hybrid_v21_ablation"
HYBRID_V22_BASE = "patchevent/phase2/checkpoints/{domain}_hybrid_v22_ablation"
OUT_JSON = "patchevent/phase2/checkpoints/hybrid_ablation_summary.json"
OUT_MD = "patchevent/phase2/checkpoints/hybrid_ablation_summary.md"
OUT_CAL_JSON = "patchevent/phase2/checkpoints/hybrid_calibrated_summary.json"
OUT_CAL_MD = "patchevent/phase2/checkpoints/hybrid_calibrated_summary.md"
OUT_V21_JSON = "patchevent/phase2/checkpoints/hybrid_v21_ablation_summary.json"
OUT_V21_MD = "patchevent/phase2/checkpoints/hybrid_v21_ablation_summary.md"
OUT_V21_CAL_JSON = "patchevent/phase2/checkpoints/hybrid_v21_calibrated_summary.json"
OUT_V21_CAL_MD = "patchevent/phase2/checkpoints/hybrid_v21_calibrated_summary.md"
OUT_V22_JSON = "patchevent/phase2/checkpoints/hybrid_v22_ablation_summary.json"
OUT_V22_MD = "patchevent/phase2/checkpoints/hybrid_v22_ablation_summary.md"
OUT_V22_CAL_JSON = "patchevent/phase2/checkpoints/hybrid_v22_calibrated_summary.json"
OUT_V22_CAL_MD = "patchevent/phase2/checkpoints/hybrid_v22_calibrated_summary.md"
METRICS = ["event_f1", "onset_mae", "apex_mae", "duration_mae", "intensity_mape"]


def _read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metrics_path(base: str, name: str, seed: int) -> str:
    return os.path.join(base, f"{name}_s{seed}", "test_eval", "eval_metrics.json")


def _calibrated_metrics_path(base: str, name: str, seed: int) -> str:
    return os.path.join(
        base,
        f"{name}_s{seed}",
        "calibrated_eval",
        "threshold_sweep_test.json",
    )


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


def _collect_calibrated_group(base_tmpl: str, names: list[str]) -> dict:
    out = {}
    for domain in DOMAINS:
        base = base_tmpl.format(domain=domain)
        out[domain] = {}
        for name in names:
            rows = []
            per_seed = {}
            selections = {}
            for seed in SEEDS:
                payload = _read_json(_calibrated_metrics_path(base, name, seed))
                if payload is None:
                    continue
                metrics = payload.get("metrics") or {}
                selection = payload.get("selected_from_val") or {}
                per_seed[str(seed)] = metrics
                selections[str(seed)] = {
                    "exist_threshold": selection.get("exist_threshold"),
                    "hybrid_nms_onset_radius": selection.get("hybrid_nms_onset_radius"),
                }
                rows.append(metrics)
            out[domain][name] = {
                "n_done": len(rows),
                "per_seed": per_seed,
                "selected_from_val": selections,
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
    calibrated = _collect_calibrated_group(HYBRID_BASE, HYBRID_CONFIGS)
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

    cal_summary = {
        "seeds": SEEDS,
        "domains": DOMAINS,
        "note": "Calibrated Hybrid DETR-AR summary using validation-selected exist_threshold and onset NMS radius.",
        "hybrid_calibrated": calibrated,
    }
    with open(OUT_CAL_JSON, "w", encoding="utf-8") as f:
        json.dump(cal_summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {OUT_CAL_JSON}")

    cal_lines = ["# Calibrated Hybrid DETR-AR Summary", ""]
    cal_lines.extend(_render_table("Validation-Calibrated Hybrid Internal Ablation", calibrated, HYBRID_CONFIGS))
    with open(OUT_CAL_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(cal_lines))
    print(f"[SAVE] {OUT_CAL_MD}")

    hybrid_v21 = _collect_group(HYBRID_V21_BASE, HYBRID_V21_CONFIGS)
    calibrated_v21 = _collect_calibrated_group(HYBRID_V21_BASE, HYBRID_V21_CONFIGS)
    v21_summary = {
        "seeds": SEEDS,
        "domains": DOMAINS,
        "note": "Hybrid DETR-AR v2.1 summary: structured temporal head, matched-order refinement, and optional count-controlled decoding.",
        "hybrid_v21_ablation": hybrid_v21,
    }
    with open(OUT_V21_JSON, "w", encoding="utf-8") as f:
        json.dump(v21_summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {OUT_V21_JSON}")

    v21_lines = ["# Hybrid DETR-AR v2.1 Ablation Summary", ""]
    v21_lines.extend(_render_table("Hybrid v2.1 Internal Ablation", hybrid_v21, HYBRID_V21_CONFIGS))
    with open(OUT_V21_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(v21_lines))
    print(f"[SAVE] {OUT_V21_MD}")

    v21_cal_summary = {
        "seeds": SEEDS,
        "domains": DOMAINS,
        "note": "Calibrated Hybrid DETR-AR v2.1 summary using validation-selected exist_threshold and onset NMS radius.",
        "hybrid_v21_calibrated": calibrated_v21,
    }
    with open(OUT_V21_CAL_JSON, "w", encoding="utf-8") as f:
        json.dump(v21_cal_summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {OUT_V21_CAL_JSON}")

    v21_cal_lines = ["# Calibrated Hybrid DETR-AR v2.1 Summary", ""]
    v21_cal_lines.extend(_render_table("Validation-Calibrated Hybrid v2.1 Internal Ablation", calibrated_v21, HYBRID_V21_CONFIGS))
    with open(OUT_V21_CAL_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(v21_cal_lines))
    print(f"[SAVE] {OUT_V21_CAL_MD}")

    hybrid_v22 = _collect_group(HYBRID_V22_BASE, HYBRID_V22_CONFIGS)
    calibrated_v22 = _collect_calibrated_group(HYBRID_V22_BASE, HYBRID_V22_CONFIGS)
    v22_summary = {
        "seeds": SEEDS,
        "domains": DOMAINS,
        "note": "Hybrid DETR-AR v2.2 summary: structured temporal head with count top-K and matched-order teacher forcing disabled by default.",
        "hybrid_v22_ablation": hybrid_v22,
    }
    with open(OUT_V22_JSON, "w", encoding="utf-8") as f:
        json.dump(v22_summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {OUT_V22_JSON}")

    v22_lines = ["# Hybrid DETR-AR v2.2 Ablation Summary", ""]
    v22_lines.extend(_render_table("Hybrid v2.2 Internal Ablation", hybrid_v22, HYBRID_V22_CONFIGS))
    with open(OUT_V22_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(v22_lines))
    print(f"[SAVE] {OUT_V22_MD}")

    v22_cal_summary = {
        "seeds": SEEDS,
        "domains": DOMAINS,
        "note": "Calibrated Hybrid DETR-AR v2.2 summary using validation-selected exist_threshold and onset NMS radius.",
        "hybrid_v22_calibrated": calibrated_v22,
    }
    with open(OUT_V22_CAL_JSON, "w", encoding="utf-8") as f:
        json.dump(v22_cal_summary, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {OUT_V22_CAL_JSON}")

    v22_cal_lines = ["# Calibrated Hybrid DETR-AR v2.2 Summary", ""]
    v22_cal_lines.extend(_render_table("Validation-Calibrated Hybrid v2.2 Internal Ablation", calibrated_v22, HYBRID_V22_CONFIGS))
    with open(OUT_V22_CAL_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(v22_cal_lines))
    print(f"[SAVE] {OUT_V22_CAL_MD}")


if __name__ == "__main__":
    main()
