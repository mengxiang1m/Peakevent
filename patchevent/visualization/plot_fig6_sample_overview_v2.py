"""
Fig.6 Sample overview (paper layout):
3 rows x 2 columns for WLEL/ETT/ELC with good and hard cases.
"""

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
    / "sample_overview.png"
)


def configure_style():
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
    for patch in ax.patches:
        if patch is ax.patch:
            continue
        patch.set_linewidth(1.2)


def rel_hour_ticks(ax, step=24):
    lo, hi = int(ax.get_xlim()[0]), int(ax.get_xlim()[1])
    ticks = np.arange(lo, hi + 1, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t - lo) for t in ticks], fontsize=9)


def main():
    configure_style()
    os.makedirs(OUT_PATH.parent, exist_ok=True)

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(3.5, 6.8),
        sharex=False,
        sharey="row",
    )

    col_titles = ["Good", "Hard"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold", pad=6)

    for row, domain in enumerate(["wlel", "ett", "elc"]):
        series = load_series(domain)
        records = load_outputs(domain, "G0_full", 42)
        samples = pick_samples(records, n_good=1, n_hard=1, stride=20)
        dlabel = DOMAINS[domain]["label"]

        for col in range(2):
            ax = axes[row, col]
            if col < len(samples):
                plot_sample(
                    ax,
                    series,
                    samples[col],
                    dlabel,
                    ctx_len=48,
                    show_intensity_labels=False,
                    show_xticks=True,
                    show_axis_labels=(col == 0),
                    highlight_pred_window=True,
                    show_mini_legend=False,
                )
                enforce_linewidths(ax)
                rel_hour_ticks(ax, step=24)
                ax.grid(axis="y", linewidth=0.3, alpha=0.30, linestyle="--")
                ax.tick_params(axis="both", labelsize=9)
                if col == 0:
                    ax.set_ylabel(f"{dlabel}\nLoad", fontsize=10, fontweight="bold")
                    ax.set_xlabel("Hour", fontsize=10)
                else:
                    ax.set_ylabel("")
                    ax.set_xlabel("")
            else:
                ax.axis("off")

    handles = [
        Line2D([0], [0], color="#aaa", linewidth=1.5, label="History"),
        Line2D([0], [0], color="#222", linewidth=1.5, label="Pred window"),
        mpatches.Patch(facecolor="#AED6F1", edgecolor="#1565C0", alpha=0.30, linewidth=1.0, label="GT span"),
        mpatches.Patch(facecolor="none", edgecolor="#2ca02c", linewidth=1.0, linestyle="--", label="Pred span"),
        Line2D([0], [0], marker="v", color="#1565C0", linestyle="", markersize=5, label="GT apex"),
        Line2D([0], [0], marker="^", color="#2ca02c", linestyle="", markersize=5, label="Pred apex"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=9,
    )

    fig.subplots_adjust(left=0.20, right=0.99, top=0.96, bottom=0.14, hspace=0.45, wspace=0.35)
    fig.savefig(str(OUT_PATH), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
