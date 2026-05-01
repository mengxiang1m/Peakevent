from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


OUTPUT_PATH = Path(
    str(Path(__file__).resolve().parent / "fig1_architecture.png")
)


def draw_round_box(ax, cx, cy, w, h, text, *, edgecolor="black", linestyle="-", lw=0.9, fontsize=7.2):
    x0 = cx - w / 2
    y0 = cy - h / 2
    box = FancyBboxPatch(
        (x0, y0),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.8",
        facecolor="white",
        edgecolor=edgecolor,
        linestyle=linestyle,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, zorder=4)
    return {
        "left": (x0, cy),
        "right": (x0 + w, cy),
        "top": (cx, y0 + h),
        "bottom": (cx, y0),
        "center": (cx, cy),
    }


def draw_arrow(ax, p1, p2, *, dashed=False, color="black", lw=0.9):
    ax.annotate(
        "",
        xy=p2,
        xytext=p1,
        arrowprops=dict(
            arrowstyle="-|>",
            lw=lw,
            color=color,
            linestyle="--" if dashed else "-",
            mutation_scale=8,
            shrinkA=2,
            shrinkB=2,
        ),
        zorder=2,
    )


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = ["Comic Sans MS", "cursive", "DejaVu Sans"]
    plt.rcParams["font.size"] = 7.5
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=300)
    ax.set_xlim(0, 132)
    ax.set_ylim(0, 58)
    ax.axis("off")

    # Region backgrounds
    ax.add_patch(Rectangle((2, 16), 66, 38, facecolor="#E3F2FD", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((68, 16), 16, 38, facecolor="#E8F5E9", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((84, 16), 46, 38, facecolor="#FFF3E0", edgecolor="none", zorder=0))

    ax.text(4, 52.5, "Phase 1 Encoder", fontsize=8, weight="bold", ha="left", va="center")
    ax.text(70, 52.5, "Memory Bridge", fontsize=8, weight="bold", ha="left", va="center")
    ax.text(86, 52.5, "Phase 2 Decoder", fontsize=8, weight="bold", ha="left", va="center")

    y_main = 38
    box_w, box_h = 13, 8

    b_input = draw_round_box(ax, 9, y_main, box_w, box_h, "Input\\n$x \\in \\mathbb{R}^{96}$")
    b_norm = draw_round_box(ax, 23, y_main, box_w, box_h, "z-score\nnorm")
    b_unfold = draw_round_box(ax, 37, y_main, box_w, box_h, "Patch Unfold\nP=8, s=4, N=23")
    b_embed = draw_round_box(ax, 51, y_main, box_w, box_h, "Linear + PE\n+ LN")
    b_enc = draw_round_box(ax, 64, y_main, box_w, box_h, "2-layer\nTransformer\nEncoder")

    b_bridge = draw_round_box(
        ax,
        76,
        y_main,
        14,
        10,
        "Projection\n-> MemPos\n(learnable)\n-> SA Agg.",
        fontsize=6.9,
    )
    b_dec = draw_round_box(
        ax,
        93,
        y_main,
        16,
        10,
        "3-layer AR Decoder\ncausal self-attn\n-> cross-attn\n-> FFN",
        fontsize=6.8,
    )
    b_lm = draw_round_box(ax, 108, y_main, 11, 8, "LM Head")
    b_mask = draw_round_box(
        ax,
        119,
        y_main,
        11,
        8,
        "PosValidMask",
        edgecolor="#D32F2F",
        linestyle="--",
        lw=1.1,
        fontsize=6.9,
    )
    b_out = draw_round_box(ax, 127, y_main, 10, 8, "Output\n[BOS,...,EOS]", fontsize=6.7)

    # Main flow arrows
    draw_arrow(ax, b_input["right"], b_norm["left"])
    draw_arrow(ax, b_norm["right"], b_unfold["left"])
    draw_arrow(ax, b_unfold["right"], b_embed["left"])
    draw_arrow(ax, b_embed["right"], b_enc["left"])

    # Freeze transition
    draw_arrow(ax, b_enc["right"], b_bridge["left"], dashed=True)
    ax.text(70.4, 44.5, "Freeze encoder", fontsize=7, ha="center", va="center")
    ax.plot([69, 69], [33, 43], linestyle="--", linewidth=0.9, color="black", zorder=1)

    draw_arrow(ax, b_bridge["right"], b_dec["left"])
    draw_arrow(ax, b_dec["right"], b_lm["left"])
    draw_arrow(ax, b_lm["right"], b_mask["left"])
    draw_arrow(ax, b_mask["right"], b_out["left"])

    # Phase 1 heads branching down
    head_labels = ["has_apex", "d_to_apex", "phase_dist", "apex_offset"]
    head_x = [50, 58, 66, 74]
    head_boxes = []
    for x, label in zip(head_x, head_labels):
        hb = draw_round_box(ax, x, 22.5, 10, 5, label, fontsize=6.6)
        head_boxes.append(hb)
        draw_arrow(ax, b_enc["bottom"], hb["top"])

    ax.text(45.5, 28.0, "Phase 1 heads", fontsize=6.8, ha="left", va="center")

    # Token sequence at the bottom
    ax.text(4, 8.8, "Token sequence:", fontsize=7.2, ha="left", va="center")
    token_specs = [
        ("BOS", "#E0E0E0"),
        ("onset", "#90CAF9"),
        ("dur", "#A5D6A7"),
        ("apex", "#FFCC80"),
        ("int", "#F8BBD0"),
        ("...", "#FFFFFF"),
        ("EOS", "#E0E0E0"),
    ]

    x = 25
    y = 8.8
    for token, color in token_specs:
        w = 7 if token in {"BOS", "EOS", "..."} else 9
        rect = FancyBboxPatch(
            (x, y - 2.2),
            w,
            4.4,
            boxstyle="round,pad=0.02,rounding_size=0.6",
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y, token, fontsize=6.7, ha="center", va="center", zorder=4)
        x += w + 1.5

    ax.text(106, 8.8, "[BOS, o1, d1, a1, v1, ..., EOS]", fontsize=6.8, ha="left", va="center")

    fig.tight_layout(pad=0.2)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

