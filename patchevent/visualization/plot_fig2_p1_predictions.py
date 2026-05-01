import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, _ROOT)

from patchevent.visualization.visualize_results import (  # noqa: E402
    DOMAINS,
    load_outputs,
    load_series,
    pick_samples,
    plot_sample,
)

OUT_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "papers"
    / "69c8ebe9537191b1512a6125"
    / "figs"
    / "p1_predictions.png"
)

plt.rcParams.update({
    "font.family": "Comic Sans MS",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
})


def fix_xticks(ax, step=24):
    lo, hi = ax.get_xlim()
    lo, hi = int(lo), int(hi)
    base = lo
    ticks = np.arange(lo, hi + 1, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t - base) for t in ticks], fontsize=9)
    ax.set_xlabel("Hour", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)


def restyle_event_spans(ax):
    gt_fill = "#AED6F1"
    gt_edge = "#1565C0"
    pred_edge = "#2ca02c"
    for patch in ax.patches:
        if patch is ax.patch:
            continue
        if isinstance(patch, mpatches.Rectangle) and not patch.get_fill():
            patch.set_edgecolor(pred_edge)
            patch.set_linestyle("--")
            patch.set_linewidth(1.0)
            continue
        if patch.get_fill():
            patch.set_facecolor(gt_fill)
            patch.set_edgecolor(gt_edge)
            patch.set_alpha(0.30)
            patch.set_linewidth(1.0)


def enforce_linewidths(ax):
    for line in ax.lines:
        marker = line.get_marker()
        linestyle = line.get_linestyle()
        if linestyle == ":":
            line.set_linewidth(1.0)
        elif marker in (None, "", "None", " "):
            line.set_linewidth(1.5)
        else:
            line.set_linewidth(1.0)


def main():
    os.makedirs(OUT_PATH.parent, exist_ok=True)

    domain = "wlel"
    series = load_series(domain)
    records = load_outputs(domain, "G0_full", 42)
    samples = pick_samples(records, n_good=2, n_hard=1, stride=20)
    dlabel = DOMAINS[domain]["label"]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
    titles = ["(a) Perfect", "(b) Perfect", "(c) Failure"]
    for i, (ax, sample) in enumerate(zip(axes, samples)):
        plot_sample(
            ax,
            series,
            sample,
            dlabel,
            ctx_len=48,
            show_intensity_labels=False,
            show_xticks=True,
            show_axis_labels=(i == 0),
            show_mini_legend=False,
        )
        enforce_linewidths(ax)
        restyle_event_spans(ax)
        ax.set_title(titles[i] if i < len(titles) else "", fontsize=9, fontweight="bold", pad=4)
        ax.grid(axis="y", linewidth=0.3, alpha=0.30, linestyle="--")
        fix_xticks(ax, step=24)

    handles = [
        Line2D([0], [0], color="#aaa", linewidth=1.5, label="History"),
        Line2D([0], [0], color="#222", linewidth=1.5, label="Pred window"),
        mpatches.Patch(facecolor="#AED6F1", edgecolor="#1565C0", alpha=0.30, linewidth=1.0, label="GT span"),
        mpatches.Patch(facecolor="none", edgecolor="#2ca02c", linewidth=1.0, linestyle="--", label="Pred span"),
        Line2D([0], [0], marker="v", color="#1565C0", linestyle="", markersize=6, label="GT apex"),
        Line2D([0], [0], marker="^", color="#2ca02c", linestyle="", markersize=6, label="Pred apex"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        framealpha=0.95,
    )
    fig.subplots_adjust(left=0.06, right=0.995, top=0.88, bottom=0.30, wspace=0.22)
    fig.savefig(str(OUT_PATH), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
