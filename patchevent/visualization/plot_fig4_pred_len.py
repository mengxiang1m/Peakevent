import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = (
    PROJECT_ROOT
    / "papers"
    / "69c8ebe9537191b1512a6125"
    / "figs"
    / "pred_len.png"
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
        "axes.prop_cycle": cycler(color=["#1f77b4", "#ff7f0e", "#2ca02c"]),
    })


def main():
    configure_style()

    data = {
        # Each value:
        # (F1_mean, F1_std, OnsetMAE_mean, OnsetMAE_std,
        #  ApexMAE_mean, ApexMAE_std, DurationMAE_mean, DurationMAE_std)
        "WLEL": {
            96:  (0.8407205939395020, 0.0045631057692239, 0.7469076556602356, 0.0216505979149057, 0.7073246732220649, 0.0153835915225347, 0.4396698905972875, 0.0112475783096264),
            168: (0.8304956717769373, 0.0021040499303183, 0.7235958114540469, 0.0038574880847920, 0.6955919033438561, 0.0129660079915532, 0.4299286456562203, 0.0093540489385096),
            336: (0.7943443987857096, 0.0012154104376239, 0.7793998415613915, 0.0519511937366535, 0.7669370585425671, 0.0433240366059172, 0.4430980443397403, 0.0055491595342956),
        },
        "ETT": {
            96:  (0.8547580283167768, 0.0002640039907634, 0.7035835297661174, 0.0104065354045723, 0.6520686619876516, 0.0083906381997077, 0.2127946376056683, 0.0004765257015306),
            168: (0.8391734049870087, 0.0105223809914129, 0.7324394368819908, 0.0385607944760391, 0.6894678274397852, 0.0372421551058190, 0.2104030101639242, 0.0013124038037995),
            336: (0.7503075238851027, 0.0070916638124630, 0.7614935794899776, 0.0407625831231907, 0.7178518809688482, 0.0378552968462400, 0.2219190813641132, 0.0013711515473379),
        },
        "ELC": {
            96:  (0.7943065785990787, 0.0127738039115354, 1.1162346604351008, 0.0518569809974979, 1.0909445980422860, 0.0476553559195013, 0.5411868022364676, 0.0251084917332914),
            168: (0.7693750812889362, 0.0297774202466966, 1.1060929540408801, 0.0389015798090392, 1.0962064765785940, 0.0375045011034787, 0.5642837571495166, 0.0252824746592817),
            336: (0.7354640694025786, 0.0295198348841679, 1.1158209003111532, 0.0621870887680428, 1.0655065393179683, 0.0844844530629846, 0.5810807695090940, 0.0137362235292176),
        },
    }

    styles = {
        "WLEL": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        "ETT": {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},
        "ELC": {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},
    }

    x = np.array([96, 168, 336])
    xlabels = ['H=96', 'H=168', 'H=336']

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
    axes = axes.flatten()

    panel_specs = [
        ("(a)", "Event F1", 0, 1, (0.70, 0.88), "lower left"),
        ("(b)", "Onset MAE (h)", 2, 3, (0.64, 1.20), "upper left"),
        ("(c)", "Apex MAE (h)", 4, 5, (0.60, 1.22), "upper left"),
        ("(d)", "Duration MAE (h)", 6, 7, (0.16, 0.64), "upper left"),
    ]

    for ax, (panel_tag, ylabel, mean_idx, std_idx, ylim, legend_loc) in zip(axes, panel_specs):
        for name in ['WLEL', 'ETT', 'ELC']:
            y_mean = np.array([data[name][h][mean_idx] for h in x])
            y_std = np.array([data[name][h][std_idx] for h in x])
            style = styles[name]

            ax.plot(
                x,
                y_mean,
                label=name,
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                linewidth=1.5,
                markersize=4.8,
                markeredgewidth=1.0,
            )
            ax.fill_between(
                x,
                y_mean - y_std,
                y_mean + y_std,
                color=style['color'],
                alpha=0.15,
                linewidth=0,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="y", linewidth=0.3, alpha=0.35, linestyle="--")
        ax.legend(loc=legend_loc, frameon=True, fontsize=8)
        ax.text(
            0.02,
            0.98,
            panel_tag,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
        )

    out_path = str(OUT_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.10, wspace=0.32, hspace=0.34)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
