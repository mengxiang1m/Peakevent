"""visualize_p1_results.py — Phase-1 可视化 (Phase-2 审美风格)
============================================================
输出到 patchevent/phase1/plots/
  p1_fig1_training_curves.png   — 训练收敛曲线 (3 seeds)
  p1_fig2_predictions.png       — 预测总览 (3×3 grid)
  p1_fig3_subtask_detail.png    — 子任务详图 (2 samples × 4 tasks)
  samples/good/good_s{idx}.png  — 好样本个体图
  samples/hard/hard_s{idx}.png  — 难样本个体图
"""

from __future__ import annotations
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pandas as pd

# ── 路径 ──────────────────────────────────────────────────────────────────────
_pkg  = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_pkg)
for p in [_pkg, _root]:
    if p not in sys.path:
        sys.path.insert(0, p)  # TODO(R-future): migrate to patchevent package import

from patchevent.phase1.dataset import build_dataloaders
from patchevent.phase1.model   import PatchEncoder

OUT  = os.path.join(_pkg, "plots")
os.makedirs(OUT, exist_ok=True)

CKPT = os.path.join(_pkg, "checkpoints", "wlel", "s42", "best_model.pth")
SERIES_PATH = os.path.join(_root, "dataset", "wlel", "event_v1", "data",
                            "wlel_event_series_v1.csv")
LABELS_PATH = os.path.join(_root, "dataset", "wlel", "event_v1", "data",
                            "wlel_patch_labels_v1.csv")

# ── 配色 (Okabe-Ito, 与 Phase-2 一致) ─────────────────────────────────────────
C_GT   = "#0072B2"     # 深蓝 — GT
C_PRED = "#D55E00"     # 橙红 — Pred
C_TS   = "#555555"     # 深灰 — 时序信号
C_GRID = "#DDDDDD"     # 浅灰 — patch 边界

# Phase-2 统一色彩语言: 蓝=正确/TP, 琥珀-橙=漏检-误报/FN-FP
C_GT_TP_FILL = "#AED6F1"   # 浅蓝 — GT patch TP 填充
C_GT_FN_FILL = "#FAD7A0"   # 浅琥 — GT patch FN 填充
C_GT_TP_EDGE = "#1565C0"   # 深蓝 — GT patch TP 边框
C_GT_FN_EDGE = "#E65100"   # 深橙 — GT patch FN 边框
C_PR_TP      = "#0072B2"   # 深蓝 — Pred TP
C_PR_FP      = "#D55E00"   # 橙红 — Pred FP

PHASE_NAMES  = ["BG", "Rising", "Apex", "Falling"]
PHASE_COLORS = ["#999999", "#56B4E9", "#0072B2", "#E69F00"]
SEED_COLORS  = {42: "#0072B2", 123: "#D55E00", 456: "#E69F00"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载 & 推理
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_data():
    ckpt   = torch.load(CKPT, map_location="cpu", weights_only=False)
    args   = ckpt["args"]
    seq_len   = args.get("seq_len",   96)
    patch_len = args.get("patch_len",  8)
    stride    = args.get("stride",     4)
    log1p_d   = bool(args.get("log1p_d", 1))

    _, _, test_loader, mean, std = build_dataloaders(
        series_path=SERIES_PATH,
        patch_labels_path=LABELS_PATH,
        seq_len=seq_len, patch_len=patch_len, stride=stride,
        batch_size=64, num_workers=0, log1p_d=log1p_d,
    )
    model = PatchEncoder(
        seq_len=seq_len, patch_len=patch_len, stride=stride,
        d_model=args.get("d_model", 128), n_heads=args.get("n_heads", 4),
        e_layers=args.get("e_layers", 2), d_ff=args.get("d_ff", 256),
        dropout=args.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model, test_loader, mean, std, seq_len, patch_len, stride, log1p_d


def run_inference(model, test_loader):
    """推理全部测试集，返回 list of sample dicts。"""
    records = []
    sample_idx = 0
    with torch.no_grad():
        for batch in test_loader:
            x   = batch["x"]
            ha  = batch["has_apex"]
            ao  = batch["apex_offset"]
            md  = batch["min_d"]
            pd_ = batch["phase_dist"]
            ai  = batch["apex_int"]
            out = model(x)

            B = x.shape[0]
            for b in range(B):
                gt_bin = (ha[b].numpy() > 0.5).astype(float)
                pr_bin = (out["has_apex"][b].numpy() > 0.5).astype(float)
                tp = int((gt_bin * pr_bin).sum())
                fp = int((pr_bin * (1 - gt_bin)).sum())
                fn = int(((1 - pr_bin) * gt_bin).sum())
                records.append({
                    "sample_idx":  sample_idx + b,
                    "x":           x[b].numpy(),
                    "gt_has_apex": ha[b].numpy(),
                    "gt_offset":   ao[b].numpy(),
                    "gt_min_d":    md[b].numpy(),
                    "gt_phase":    pd_[b].numpy(),
                    "pr_has_apex": out["has_apex"][b].numpy(),
                    "pr_d":        out["d_to_apex"][b].numpy(),
                    "pr_phase":    out["phase_dist"][b].numpy(),
                    "pr_offset_p": torch.softmax(out["apex_offset"][b], dim=-1).numpy(),
                    "pr_offset":   out["apex_offset"][b].argmax(-1).numpy(),
                    "tp": tp, "fp": fp, "fn": fn,
                })
            sample_idx += B
    return records


def pick_samples(records, n_good=4, n_hard=4, stride=6):
    """挑选有峰值的样本: good (Perfect) + hard (有误差)。"""
    good, hard = [], []
    for r in records:
        if r["gt_has_apex"].max() < 0.5:
            continue
        if r["fp"] == 0 and r["fn"] == 0:
            good.append(r)
        else:
            hard.append(r)
    return good[::stride][:n_good] + hard[::stride][:n_hard]


# ─────────────────────────────────────────────────────────────────────────────
# 共享图例 — Phase-2 同构
# ─────────────────────────────────────────────────────────────────────────────

def make_p1_legend_handles():
    return [
        mpatches.Patch(fc=C_GT_TP_FILL, ec=C_GT_TP_EDGE, lw=1, alpha=0.8,
                       label="GT apex patch (TP)"),
        mpatches.Patch(fc=C_GT_FN_FILL, ec=C_GT_FN_EDGE, lw=1, alpha=0.8,
                       label="GT apex patch (FN)"),
        mpatches.Patch(fc="none", ec=C_PR_TP, lw=2, ls="--",
                       label="Pred patch (TP)"),
        mpatches.Patch(fc="none", ec=C_PR_FP, lw=2, ls="--",
                       label="Pred patch (FP)"),
        plt.Line2D([0],[0], marker="v", color=C_GT_TP_EDGE, ls="",
                   ms=9, label="GT apex (TP)"),
        plt.Line2D([0],[0], marker="v", color=C_GT_FN_EDGE, ls="",
                   ms=9, label="GT apex (FN)"),
        plt.Line2D([0],[0], marker="^", color=C_PR_TP, ls="",
                   ms=8, label="Pred apex (TP)"),
        plt.Line2D([0],[0], marker="^", color=C_PR_FP, ls="",
                   ms=8, label="Pred apex (FP)"),
        plt.Line2D([0],[0], color=C_TS, lw=1.2, label="Input signal"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 核心: 单 Axes 时序 + GT filled + Pred dashed — Phase-2 plot_sample 同构
# ─────────────────────────────────────────────────────────────────────────────

def plot_p1_sample(ax, rec, patch_len, stride, show_mini_legend=False):
    x = rec["x"]
    N = rec["gt_has_apex"].shape[0]
    gt_ha, gt_off = rec["gt_has_apex"], rec["gt_offset"]
    pr_ha, pr_off = rec["pr_has_apex"], rec["pr_offset"]

    t = np.arange(len(x))
    ax.plot(t, x, color=C_TS, linewidth=1.1, zorder=3)

    ymin, ymax = float(x.min()), float(x.max())
    span = max(ymax - ymin, 1e-6)
    ax.set_ylim(ymin - span * 0.10, ymax + span * 0.22)

    for i in range(N + 1):
        ax.axvline(i * stride, color=C_GRID, linewidth=0.4, zorder=1)

    gt_bin = (gt_ha > 0.5)
    pr_bin = (pr_ha > 0.5)

    # GT apex patches: filled spans
    for i in range(N):
        if not gt_bin[i]:
            continue
        ps, pe = i * stride, i * stride + patch_len
        is_tp = bool(pr_bin[i])
        fc = C_GT_TP_FILL if is_tp else C_GT_FN_FILL
        ec = C_GT_TP_EDGE if is_tp else C_GT_FN_EDGE
        ax.axvspan(ps, pe, alpha=0.55, facecolor=fc, edgecolor=ec,
                   linewidth=1.0, zorder=2)
        if gt_off[i] >= 0:
            apex_t = ps + int(gt_off[i])
            if apex_t < len(x):
                ax.plot(apex_t, x[apex_t], "v",
                        color=C_GT_TP_EDGE if is_tp else C_GT_FN_EDGE,
                        markersize=9, markeredgecolor="white",
                        markeredgewidth=0.6, zorder=6)

    # Pred apex patches: dashed outlines
    for i in range(N):
        if not pr_bin[i]:
            continue
        ps, pe = i * stride, i * stride + patch_len
        is_tp = bool(gt_bin[i])
        ec = C_PR_TP if is_tp else C_PR_FP
        ax.add_patch(mpatches.Rectangle(
            (ps, ymin - span * 0.05), pe - ps, (ymax - ymin) * 1.15,
            facecolor="none", edgecolor=ec, linewidth=2.0,
            linestyle="--", zorder=4,
        ))
        pred_apex_t = ps + int(pr_off[i])
        if pred_apex_t < len(x):
            ax.plot(pred_apex_t, x[pred_apex_t], "^", color=ec, markersize=8,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=7)

    tp, fp, fn = rec["tp"], rec["fp"], rec["fn"]
    n_gt = tp + fn
    status = "Perfect" if fp == 0 and fn == 0 else (
             f"+{fp}FP" if fn == 0 else (
             f"-{fn}FN" if fp == 0 else f"+{fp}FP/-{fn}FN"))
    ax.set_title(f"TP={tp}/{n_gt}  {status}", fontsize=9, pad=3,
                 color=C_PR_TP if fp == 0 and fn == 0 else C_PR_FP)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=7.5)

    if show_mini_legend:
        mini = [
            mpatches.Patch(fc=C_GT_TP_FILL, ec=C_GT_TP_EDGE, lw=1,
                           alpha=0.8, label="GT-TP"),
            mpatches.Patch(fc=C_GT_FN_FILL, ec=C_GT_FN_EDGE, lw=1,
                           alpha=0.8, label="GT-FN"),
            mpatches.Patch(fc="none", ec=C_PR_TP, lw=2, ls="--",
                           label="Pred-TP"),
            mpatches.Patch(fc="none", ec=C_PR_FP, lw=2, ls="--",
                           label="Pred-FP"),
        ]
        ax.legend(handles=mini, fontsize=6.5, loc="upper left",
                  framealpha=0.85, ncol=2, handlelength=1.2,
                  borderpad=0.4, labelspacing=0.25)


# ─────────────────────────────────────────────────────────────────────────────
# 子任务详图行: 4 panels (has_apex / d_to_apex / phase_dist / apex_offset)
# ─────────────────────────────────────────────────────────────────────────────

def plot_subtask_row(axes, rec, patch_len, stride, log1p_d):
    ax_ha, ax_d, ax_ph, ax_off = axes
    N = rec["gt_has_apex"].shape[0]
    xi = np.arange(N)

    gt_ha, gt_off = rec["gt_has_apex"], rec["gt_offset"]
    gt_d, gt_ph   = rec["gt_min_d"],   rec["gt_phase"]
    pr_ha, pr_d   = rec["pr_has_apex"], rec["pr_d"]
    pr_ph         = rec["pr_phase"]
    pr_off        = rec["pr_offset"]
    pr_off_p      = rec["pr_offset_p"]

    gt_d_raw = np.expm1(gt_d) if log1p_d else gt_d
    pr_d_raw = np.expm1(pr_d) if log1p_d else pr_d

    # has_apex
    bw = 0.35
    ax_ha.bar(xi - bw/2, gt_ha, bw, color=C_GT, alpha=0.80, label="GT")
    ax_ha.bar(xi + bw/2, pr_ha, bw, color=C_PRED, alpha=0.80, label="Pred")
    ax_ha.axhline(0.5, color="#888", lw=0.8, ls="--", alpha=0.6, label="Thr")
    ax_ha.set_ylim(-0.05, 1.22)
    ax_ha.set_ylabel("Prob", fontsize=7.5)
    ax_ha.set_title("has_apex", fontsize=8, fontweight="bold")
    ax_ha.legend(fontsize=6, loc="upper right", ncol=3, framealpha=0.8)
    ax_ha.tick_params(labelsize=6.5)

    # d_to_apex
    ax_d.plot(xi, gt_d_raw, "o-", color=C_GT, lw=1.4, ms=3.5, label="GT")
    ax_d.plot(xi, pr_d_raw, "s--", color=C_PRED, lw=1.4, ms=3.5, label="Pred")
    ax_d.set_ylabel("Dist (h)", fontsize=7.5)
    ax_d.set_title("d_to_apex", fontsize=8, fontweight="bold")
    ax_d.legend(fontsize=6, loc="upper right", framealpha=0.8)
    ax_d.yaxis.grid(True, alpha=0.3, ls="--")
    ax_d.tick_params(labelsize=6.5)

    # phase_dist
    bot = np.zeros(N)
    for k in range(4):
        ax_ph.fill_between(xi, bot, bot + gt_ph[:, k],
                           color=PHASE_COLORS[k], alpha=0.70)
        bot += gt_ph[:, k]
    for k in range(4):
        ax_ph.plot(xi, pr_ph[:, k], ls="--", color=PHASE_COLORS[k],
                   lw=1.4, alpha=0.95)
    ax_ph.set_ylim(-0.05, 1.05)
    ax_ph.set_ylabel("Phase", fontsize=7.5)
    ax_ph.set_title("phase_dist (fill=GT, dash=Pred)", fontsize=8,
                    fontweight="bold")
    ph_h = [mpatches.Patch(color=PHASE_COLORS[k], alpha=0.7,
                           label=PHASE_NAMES[k]) for k in range(4)]
    ax_ph.legend(handles=ph_h, fontsize=5.5, ncol=4, loc="upper right",
                 handlelength=0.7, framealpha=0.8)
    ax_ph.tick_params(labelsize=6.5)

    # apex_offset
    apex_patches = [(i, int(gt_off[i]), int(pr_off[i]))
                    for i in range(N) if gt_ha[i] > 0.5]
    if apex_patches:
        pi  = np.array([a[0] for a in apex_patches])
        gto = np.array([a[1] for a in apex_patches])
        pro = np.array([a[2] for a in apex_patches])
        heat = np.stack([pr_off_p[i] for i in pi], axis=0)
        im = ax_off.imshow(heat.T, aspect="auto", cmap="viridis", alpha=0.55,
                           origin="lower",
                           extent=[-0.5, len(pi)-0.5, -0.5, 7.5])
        plt.colorbar(im, ax=ax_off, pad=0.02, shrink=0.85)
        ax_off.scatter(np.arange(len(pi)), gto, c=C_GT, s=70, marker="o",
                       edgecolors="white", linewidths=0.8, zorder=5,
                       label="GT")
        ax_off.scatter(np.arange(len(pi)), pro, c=C_PRED, s=70, marker="^",
                       edgecolors="white", linewidths=0.8, zorder=5,
                       label="Pred")
        ax_off.set_xticks(np.arange(len(pi)))
        ax_off.set_xticklabels([f"p{i}" for i in pi], fontsize=6.5)
        ax_off.set_yticks(np.arange(8))
        ax_off.set_ylabel("Offset", fontsize=7.5)
        ax_off.legend(fontsize=6, loc="upper right", framealpha=0.8)
    else:
        ax_off.text(0.5, 0.5, "No apex patches", ha="center", va="center",
                    transform=ax_off.transAxes, fontsize=8, alpha=0.5)
    ax_off.set_title("apex_offset (bg=softmax)", fontsize=8, fontweight="bold")
    ax_off.tick_params(labelsize=6.5)

    for a in axes:
        a.set_xlabel("Patch idx", fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: 训练收敛曲线 (3 seeds)
# ─────────────────────────────────────────────────────────────────────────────

def fig1_training_curves():
    CKPT_BASE = os.path.join(_pkg, "checkpoints", "wlel")
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase-1 Training Convergence  |  WLEL  |  3 seeds",
                 fontsize=12, fontweight="bold")

    for seed in [42, 123, 456]:
        hp = os.path.join(CKPT_BASE, f"s{seed}", "training_history.json")
        if not os.path.exists(hp):
            print(f"  [SKIP] {hp}"); continue
        with open(hp) as f:
            h = json.load(f)
        ep = np.arange(1, len(h["train_loss"]) + 1)
        c  = SEED_COLORS[seed]
        best = ep[int(np.argmin(h["val_loss"]))]

        ax_loss.plot(ep, h["train_loss"], color=c, lw=1.6, alpha=0.85,
                     label=f"s{seed} train")
        ax_loss.plot(ep, h["val_loss"],   color=c, lw=1.8, ls="--",
                     alpha=0.95, label=f"s{seed} val")
        ax_loss.axvline(best, color=c, lw=0.8, ls=":", alpha=0.5)

        ax_f1.plot(ep, h["val_f1"], color=c, lw=1.8, label=f"s{seed} val F1")
        ax_f1.axvline(best, color=c, lw=0.8, ls=":", alpha=0.5)

    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Total Loss")
    ax_loss.set_title("Train / Val Loss (solid=train, dashed=val)",
                      fontsize=9, fontweight="bold")
    ax_loss.yaxis.grid(True, alpha=0.3, ls="--"); ax_loss.set_axisbelow(True)
    lh = []
    for s, c in SEED_COLORS.items():
        lh.append(plt.Line2D([0],[0], color=c, lw=1.6, label=f"s{s} train"))
        lh.append(plt.Line2D([0],[0], color=c, lw=1.8, ls="--",
                              label=f"s{s} val"))
    ax_loss.legend(handles=lh, fontsize=7.5, ncol=3, loc="upper right")

    ax_f1.set_xlabel("Epoch"); ax_f1.set_ylabel("has_apex Val F1")
    ax_f1.set_title("Val has_apex F1", fontsize=9, fontweight="bold")
    ax_f1.yaxis.grid(True, alpha=0.3, ls="--"); ax_f1.set_axisbelow(True)
    ax_f1.legend(fontsize=7.5, loc="lower right")

    plt.tight_layout()
    out = os.path.join(OUT, "p1_fig1_training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[fig1] saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: 预测总览 (3×3 grid) — Phase-2 fig2 同构
# ─────────────────────────────────────────────────────────────────────────────

def fig2_predictions_overview(records, patch_len, stride):
    samples = pick_samples(records, n_good=6, n_hard=3, stride=4)
    n = min(9, len(samples))
    n_cols, n_rows = 3, 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 18),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.08})
    fig.suptitle(
        "Phase-1  |  WLEL  |  Patch Prediction Overview  (seed=42)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    for idx in range(n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        if idx < n:
            plot_p1_sample(ax, samples[idx], patch_len, stride,
                           show_mini_legend=(idx == 0))
            ax.set_xlabel(f"sample #{samples[idx]['sample_idx']}", fontsize=9)
        else:
            ax.axis("off")
    for r in range(n_rows):
        axes[r, 0].set_ylabel("Load (norm)", fontsize=10, fontweight="bold")

    handles = make_p1_legend_handles()
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.04), frameon=True,
               title="Legend (applies to all subplots)",
               title_fontsize=10, framealpha=0.95)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(OUT, "p1_fig2_predictions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[fig2] saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: 子任务详图 (2 samples × (时序 + 4 subtask panels))
# ─────────────────────────────────────────────────────────────────────────────

def fig3_subtask_detail(records, patch_len, stride, log1p_d):
    samples = pick_samples(records, n_good=1, n_hard=1, stride=1)
    n = min(2, len(samples))

    fig = plt.figure(figsize=(24, 6 * n))
    outer = gridspec.GridSpec(n, 1, figure=fig, hspace=0.55)

    for i in range(n):
        rec = samples[i]
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 4, subplot_spec=outer[i],
            height_ratios=[1.2, 1.0], hspace=0.45, wspace=0.35,
        )
        ax_ts = fig.add_subplot(inner[0, :])
        plot_p1_sample(ax_ts, rec, patch_len, stride, show_mini_legend=True)
        ax_ts.set_xlabel(f"sample #{rec['sample_idx']}", fontsize=9)

        ax_row = [fig.add_subplot(inner[1, c]) for c in range(4)]
        plot_subtask_row(ax_row, rec, patch_len, stride, log1p_d)

    fig.suptitle("Phase-1  |  WLEL  |  Subtask Detail  (good + hard)",
                 fontsize=13, fontweight="bold", y=1.02)
    out = os.path.join(OUT, "p1_fig3_subtask_detail.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[fig3] saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 单样本个体图 (时序 + 4 subtask), samples/good/ & samples/hard/
# ─────────────────────────────────────────────────────────────────────────────

def _save_individual(rec, patch_len, stride, log1p_d, tag):
    sub_dir = os.path.join(OUT, "samples", tag)
    os.makedirs(sub_dir, exist_ok=True)

    fig = plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           height_ratios=[1.3, 1.0], hspace=0.45, wspace=0.35)
    ax_ts = fig.add_subplot(gs[0, :])
    plot_p1_sample(ax_ts, rec, patch_len, stride, show_mini_legend=True)
    ax_ts.set_xlabel("Time (hours)", fontsize=9)

    ax_row = [fig.add_subplot(gs[1, c]) for c in range(4)]
    plot_subtask_row(ax_row, rec, patch_len, stride, log1p_d)

    sidx = rec["sample_idx"]
    fig.suptitle(f"Phase-1  |  WLEL  |  Sample #{sidx}  ({tag})",
                 fontsize=13, fontweight="bold", y=1.01)
    fname = f"{tag}_s{sidx:04d}.png"
    out = os.path.join(sub_dir, fname)
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    return fname


def fig4_individual_samples(records, patch_len, stride, log1p_d,
                            n_good=12, n_hard=8):
    samples = pick_samples(records, n_good=n_good, n_hard=n_hard, stride=3)
    good = [r for r in samples if r["fp"] == 0 and r["fn"] == 0][:n_good]
    hard = [r for r in samples if r["fp"] > 0 or r["fn"] > 0][:n_hard]

    print(f"  [WLEL] good={len(good)}, hard={len(hard)}")
    for r in good:
        print(f"    {_save_individual(r, patch_len, stride, log1p_d, 'good')}")
    for r in hard:
        print(f"    {_save_individual(r, patch_len, stride, log1p_d, 'hard')}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase-1 Visualization (Phase-2 aesthetic)")
    print(f"Checkpoint: {CKPT}")
    print(f"Output dir: {OUT}")
    print("=" * 60)

    print("\nLoading model & data ...")
    model, test_loader, mean, std, seq_len, patch_len, stride, log1p_d = \
        load_model_and_data()

    print("Running inference on test set ...")
    records = run_inference(model, test_loader)
    print(f"  Total test samples: {len(records)}")

    print("\n[1/4] Fig1: training convergence curves ...")
    fig1_training_curves()

    print("\n[2/4] Fig2: predictions overview (3x3 grid) ...")
    fig2_predictions_overview(records, patch_len, stride)

    print("\n[3/4] Fig3: subtask detail (good + hard) ...")
    fig3_subtask_detail(records, patch_len, stride, log1p_d)

    print("\n[4/4] Individual sample PNGs ...")
    fig4_individual_samples(records, patch_len, stride, log1p_d)

    print(f"\nDone. All figures saved to: {OUT}")
