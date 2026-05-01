import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "patchevent", "phase2", "checkpoints")
OUT_DIR = os.path.join(ROOT, "patchevent", "phase2", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

DOMAIN_TO_SUBDIR = {
    "wlel": "wlel_arch_ablation_v2",
    "ett": "ett_arch_ablation_v2",
    "elc": "elc_arch_ablation_v2",
}
SERIES = {
    "wlel": os.path.join(ROOT, "dataset", "wlel", "event_v1", "data", "wlel_event_series_v1.csv"),
    "ett": os.path.join(ROOT, "dataset", "ett", "event_v1", "data", "ett_event_series_v1.csv"),
    "elc": os.path.join(ROOT, "dataset", "electricity", "event_v1", "data", "elc_event_series_v1.csv"),
}


def _fallback_load_series(domain):
    df = pd.read_csv(SERIES[domain])
    return df["value"].to_numpy()


def _fallback_load_outputs(domain, group, seed):
    subdir = DOMAIN_TO_SUBDIR[domain]
    path = os.path.join(BASE, subdir, f"{group}_s{seed}", "test_eval", "eval_outputs.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


try:
    from visualize_results import load_series as _vr_load_series
    from visualize_results import load_outputs as _vr_load_outputs

    def load_series(domain):
        return _vr_load_series(domain)

    def load_outputs(domain, group, seed):
        return _vr_load_outputs(domain, group, seed)

except Exception:
    load_series = _fallback_load_series
    load_outputs = _fallback_load_outputs


def _event_intensity(ev):
    try:
        v = float(ev.get("apex_intensity"))
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _match_events(pred_evs, gt_evs, tol=3):
    matched_pred, matched_gt = set(), set()
    if not pred_evs or not gt_evs:
        return matched_pred, matched_gt

    pred_apex = np.array([e["apex_idx"] for e in pred_evs], dtype=float)
    gt_apex = np.array([e["apex_idx"] for e in gt_evs], dtype=float)
    cost = np.abs(pred_apex[:, None] - gt_apex[None, :])

    try:
        from scipy.optimize import linear_sum_assignment

        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if cost[r, c] <= tol:
                matched_pred.add(int(r))
                matched_gt.add(int(c))
    except Exception:
        candidates = []
        for i in range(cost.shape[0]):
            for j in range(cost.shape[1]):
                d = float(cost[i, j])
                if d <= tol:
                    candidates.append((d, i, j))
        candidates.sort(key=lambda x: x[0])
        used_pred, used_gt = set(), set()
        for _, i, j in candidates:
            if i not in used_pred and j not in used_gt:
                used_pred.add(i)
                used_gt.add(j)
                matched_pred.add(i)
                matched_gt.add(j)
    return matched_pred, matched_gt

def _select_good_hard(records):
    valid = [r for r in records if (r.get("tp", 0) + r.get("fn", 0)) > 0]
    if not valid:
        raise RuntimeError("没有可用样本。")

    perfect = sorted(
        [r for r in valid if r.get("fp", 0) == 0 and r.get("fn", 0) == 0],
        key=lambda x: x.get("sample_idx", 0),
    )
    if perfect:
        good = perfect[len(perfect) // 3]
    else:
        good = max(valid, key=lambda r: r.get("tp", 0) - r.get("fp", 0) - r.get("fn", 0))

    hard_pool = [r for r in valid if r.get("fp", 0) > 0 or r.get("fn", 0) > 0]
    if hard_pool:
        hard = max(hard_pool, key=lambda r: (r.get("fp", 0) + r.get("fn", 0), r.get("fn", 0), r.get("sample_idx", 0)))
    else:
        hard = valid[-1]

    return good, hard


def _marker_y(series, ev):
    y = _event_intensity(ev)
    if y is not None:
        return y
    apex = int(ev.get("apex_idx", -1))
    if 0 <= apex < len(series):
        return float(series[apex])
    return np.nan


def _plot_sample(ax, series, record, title, pred_len=96, ctx_len=48):
    pred_start = int(record["pred_start"])
    pred_end = pred_start + pred_len
    hist_start = max(0, pred_start - ctx_len)
    window_end = min(pred_end, len(series))

    x_hist = np.arange(hist_start, pred_start)
    x_pred = np.arange(pred_start, window_end)
    if len(x_hist) > 0:
        ax.plot(x_hist, series[x_hist], color="#9a9a9a", lw=1.0)
    if len(x_pred) > 0:
        ax.plot(x_pred, series[x_pred], color="#1f1f1f", lw=1.2)
    ax.axvline(pred_start, color="#666666", lw=0.9, ls=":")

    pred_evs = record.get("pred_events", [])
    gt_evs = record.get("gt_events", [])
    matched_pred, _ = _match_events(pred_evs, gt_evs, tol=3)

    y_pool = list(series[pred_start:window_end])
    y_pool += [y for y in (_marker_y(series, ev) for ev in gt_evs) if np.isfinite(y)]
    y_pool += [y for y in (_marker_y(series, ev) for ev in pred_evs) if np.isfinite(y)]
    if not y_pool:
        y_pool = [0.0, 1.0]
    ymin, ymax = float(np.min(y_pool)), float(np.max(y_pool))
    span = max(ymax - ymin, 1e-6)

    ax.set_xlim(hist_start, max(hist_start + 1, window_end - 1))
    ax.set_ylim(ymin - span * 0.1, ymax + span * 0.22)

    for ev in gt_evs:
        s, e = int(ev["onset_idx"]), int(ev["end_idx"])
        ax.axvspan(s, e, facecolor="#4C78A8", edgecolor="#4C78A8", alpha=0.20, lw=0.8, zorder=1)
        my = _marker_y(series, ev)
        if np.isfinite(my):
            ax.plot(
                int(ev["apex_idx"]),
                my,
                marker="v",
                color="#D62728",
                linestyle="",
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.55,
                zorder=6,
            )

    rect_y = ymin - span * 0.05
    rect_h = span * 1.15
    for i, ev in enumerate(pred_evs):
        s, e = int(ev["onset_idx"]), int(ev["end_idx"])
        ax.add_patch(
            mpatches.Rectangle(
                (s, rect_y),
                max(e - s, 1),
                rect_h,
                fill=False,
                edgecolor="#1F77B4",
                linewidth=1.0,
                linestyle="--",
                zorder=4,
            )
        )
        if i not in matched_pred:
            ax.axvspan(s, e, facecolor="#FF7F0E", edgecolor="none", alpha=0.24, zorder=2)
        my = _marker_y(series, ev)
        if np.isfinite(my):
            ax.plot(
                int(ev["apex_idx"]),
                my,
                marker="^",
                color="#1F77B4",
                linestyle="",
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=0.55,
                zorder=7,
            )

    ax.grid(axis="y", ls="--", alpha=0.25)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(title, fontsize=10)

def _status_text(r):
    tp, fp, fn = int(r.get("tp", 0)), int(r.get("fp", 0)), int(r.get("fn", 0))
    n_gt = tp + fn
    return f"TP={tp}/{n_gt}, FP={fp}, FN={fn}"


def main():
    domain_rows = [("wlel", "WLEL"), ("ett", "ETT"), ("elc", "ELC")]
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 10),
        gridspec_kw={"hspace": 0.35, "wspace": 0.08},
    )

    axes[0, 0].text(0.5, 1.13, "Good Cases", transform=axes[0, 0].transAxes, ha="center", fontsize=11)
    axes[0, 1].text(0.5, 1.13, "Hard/Failure Cases", transform=axes[0, 1].transAxes, ha="center", fontsize=11)

    for row, (domain, domain_label) in enumerate(domain_rows):
        series = load_series(domain)
        records = load_outputs(domain, "G0_full", 42)
        good, hard = _select_good_hard(records)

        _plot_sample(axes[row, 0], series, good, f"Good: {_status_text(good)}")
        _plot_sample(axes[row, 1], series, hard, f"Hard: {_status_text(hard)}")

        axes[row, 0].set_ylabel(domain_label, fontsize=11, fontweight="bold")
        axes[row, 0].set_xlabel(f"sample #{good['sample_idx']}", fontsize=8)
        axes[row, 1].set_xlabel(f"sample #{hard['sample_idx']}", fontsize=8)

    handles = [
        plt.Line2D([0], [0], color="#9a9a9a", lw=1.0, label="History"),
        plt.Line2D([0], [0], color="#1f1f1f", lw=1.2, label="Prediction window"),
        mpatches.Patch(facecolor="#4C78A8", edgecolor="#4C78A8", alpha=0.20, label="GT span"),
        mpatches.Patch(facecolor="none", edgecolor="#1F77B4", lw=1.0, linestyle="--", label="Pred span"),
        plt.Line2D([0], [0], marker="v", color="#D62728", linestyle="", markersize=8, label="GT apex"),
        plt.Line2D([0], [0], marker="^", color="#1F77B4", linestyle="", markersize=8, label="Pred apex"),
        mpatches.Patch(facecolor="#FF7F0E", edgecolor="none", alpha=0.24, label="FP span"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.12, hspace=0.35, wspace=0.08)
    out_path = os.path.join(OUT_DIR, "fig6_sample_overview_v2.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()


