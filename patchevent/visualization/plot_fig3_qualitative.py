"""
Fig.3 Qualitative examples (combined figure):
2 rows x 1 column for ELC (good and hard cases).
"""

import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, _ROOT)

from patchevent.visualization.visualize_results import (  # noqa: E402
    DOMAINS,
    load_outputs,
    load_series,
    make_legend_handles,
    pick_samples,
    plot_sample,
)

OUT_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "papers"
    / "69c8ebe9537191b1512a6125"
    / "figs"
    / "qual_combined.png"
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


def fix_xticks(ax, step=24):
    import numpy as np

    lo, hi = int(ax.get_xlim()[0]), int(ax.get_xlim()[1])
    ticks = np.arange(lo, hi + 1, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t - lo) for t in ticks], fontsize=9)
    ax.set_xlabel("Hour", fontsize=10)


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


def main():
    configure_style()
    os.makedirs(OUT_PATH.parent, exist_ok=True)

    domain = "elc"
    series = load_series(domain)
    records = load_outputs(domain, "G0_full", 42)
    samples = pick_samples(records, n_good=1, n_hard=1, stride=20)
    dlabel = DOMAINS[domain]["label"]

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.5), sharex=False)
    titles = [
        "(a) ELC: Perfect (TP=4/4)",
        "(b) ELC: Hard case (TP=3/4, +1FP/-1FN)",
    ]

    for ax, sample, title in zip(axes, samples, titles):
        plot_sample(
            ax,
            series,
            sample,
            dlabel,
            ctx_len=48,
            show_intensity_labels=False,
            show_xticks=True,
            show_axis_labels=True,
            highlight_pred_window=True,
        )
        enforce_linewidths(ax)
        ax.set_title(title, fontsize=10, pad=4)
        ax.tick_params(axis="both", labelsize=9)
        ax.yaxis.label.set_size(10)
        ax.grid(axis="y", linewidth=0.3, alpha=0.30, linestyle="--")
        fix_xticks(ax, step=24)

    handles = make_legend_handles(fontsize=9)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.00),
        borderpad=0.35,
        handlelength=1.4,
        columnspacing=0.8,
    )

    fig.subplots_adjust(left=0.16, right=0.99, top=0.97, bottom=0.14, hspace=0.38)
    fig.savefig(str(OUT_PATH), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
