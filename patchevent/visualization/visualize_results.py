"""
visualize_results.py  —  Phase-2 结果可视化 (v2, 6张图)
=====================================================
读取已有 checkpoint 输出，无需重新推理。
使用 v2 消融命名（G0/GS1-GS5）及 checkpoint 目录结构。

输出到 patchevent/phase2/plots/
  fig1_ablation_bar.png       — 三域架构消融柱状图 (v2, 9配置)
  fig2_sample_predictions.png  — G0_full 预测样本展示 (3域×3样本)
  fig3_error_dist.png          — TP误差分布直方图
  fig4_training_curves.png     — 训练收敛曲线
  fig6_intensity_scatter.png   — Pred vs GT 强度散点图
  plots/samples/{domain}/      — 每样本单独 PNG (good/hard)
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT = str(Path(__file__).resolve().parents[2])
BASE = os.path.join(ROOT, "patchevent", "phase2", "checkpoints")
OUT  = os.path.join(ROOT, "patchevent", "phase2", "plots")
os.makedirs(OUT, exist_ok=True)

SERIES = {
    "wlel": os.path.join(ROOT, "dataset", "wlel",        "event_v1", "data", "wlel_event_series_v1.csv"),
    "ett":  os.path.join(ROOT, "dataset", "ett",         "event_v1", "data", "ett_event_series_v1.csv"),
    "elc":  os.path.join(ROOT, "dataset", "electricity", "event_v1", "data", "elc_event_series_v1.csv"),
}

DOMAINS = {
    "wlel": {"label": "WLEL", "subdir": "wlel_arch_ablation_v2", "color": "#2196F3"},
    "ett":  {"label": "ETT",  "subdir": "ett_arch_ablation_v2",  "color": "#4CAF50"},
    "elc":  {"label": "ELC",  "subdir": "elc_arch_ablation_v2",  "color": "#FF9800"},
}
SEEDS   = [42, 123, 456]
CONFIGS = ["G0_full", "GS1_wo_mempos", "GS2_wo_sa", "GS3_bare",
           "GS4_cnn", "GS4_lstm", "GS4_mlp", "GS4_scratch", "GS5_detr"]
CFG_LABELS = ["G0\nFull", "GS1\nw/o\nMemPos", "GS2\nw/o SA", "GS3\nBare",
              "GS4\nCNN", "GS4\nLSTM", "GS4\nMLP", "GS4\nScratch", "GS5\nDETR"]

# 配色方案：Okabe-Ito 色盲友好色板，无红绿直接对比
# 蓝色系 = 正确检测 (TP)，琥珀/橙 = 错误/漏检 (FP/FN)
C_GT_TP_FILL   = "#AED6F1"  # 浅蓝  — GT span TP 填充
C_GT_FN_FILL   = "#FAD7A0"  # 浅琥  — GT span FN 填充
C_GT_TP_EDGE   = "#1565C0"  # 深蓝  — GT span TP 边框和标记
C_GT_FN_EDGE   = "#E65100"  # 深橙  — GT span FN 边框和标记
C_PR_TP        = "#2E7D32"  # 深绿 — Pred span/apex TP (区别于 GT 蓝色)
C_PR_FP        = "#D55E00"  # Okabe-Ito 橙红 — Pred span/apex FP (不再用深红)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_metrics(domain, config, seed):
    """读取单个 checkpoint 的 eval_metrics.json。"""
    subdir = DOMAINS[domain]["subdir"]
    p = os.path.join(BASE, subdir, f"{config}_s{seed}", "test_eval", "eval_metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def load_outputs(domain, config, seed):
    """读取 eval_outputs.jsonl，返回 list[dict]。"""
    subdir = DOMAINS[domain]["subdir"]
    p = os.path.join(BASE, subdir, f"{config}_s{seed}", "test_eval", "eval_outputs.jsonl")
    if not os.path.exists(p):
        return []
    recs = []
    with open(p) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def load_train_log(domain, seed):
    """读取 G0_full train_log.json。"""
    subdir = DOMAINS[domain]["subdir"]
    p = os.path.join(BASE, subdir, f"G0_full_s{seed}", "train_log.json")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return json.load(f)


def get_mean_std(domain, config, metric):
    vals = []
    for s in SEEDS:
        m = load_metrics(domain, config, s)
        if m and metric in m and not np.isnan(float(m[metric])):
            vals.append(float(m[metric]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def load_series(domain):
    """加载时序 CSV，返回 numpy array of 'value' 列。"""
    df = pd.read_csv(SERIES[domain])
    return df["value"].values


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: 三域架构消融柱状图
# ─────────────────────────────────────────────────────────────────────────────

def fig1_ablation_bar():
    metrics = [
        ("event_f1",        "Event F1",       False),
        ("onset_mae",       "Onset MAE (h)",  True),
        ("apex_mae",        "Apex MAE (h)",   True),
        ("intensity_mape",  "Intensity MAPE", True),
    ]
    n_cfg = len(CONFIGS)
    x     = np.arange(n_cfg)
    width = 0.6

    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    fig.suptitle("Architecture Ablation Results (v2) — 3 Domains × 4 Metrics (3-seed mean±std, 9 configs)",
                 fontsize=13, fontweight="bold", y=1.01)

    domain_colors = {
        "wlel": ["#BBDEFB","#90CAF9","#64B5F6","#42A5F5","#2196F3","#1E88E5","#1976D2","#1565C0","#0D47A1"],
        "ett":  ["#C8E6C9","#A5D6A7","#81C784","#66BB6A","#4CAF50","#43A047","#388E3C","#2E7D32","#1B5E20"],
        "elc":  ["#FFE0B2","#FFCC80","#FFB74D","#FFA726","#FF9800","#FB8C00","#F57C00","#E65100","#BF360C"],
    }

    for row, (metric_key, metric_label, lower_is_better) in enumerate(metrics):
        for col, domain in enumerate(["wlel", "ett", "elc"]):
            ax = axes[row, col]
            vals, errs = [], []
            for cfg in CONFIGS:
                m, s = get_mean_std(domain, cfg, metric_key)
                vals.append(m)
                errs.append(s)
            vals = np.array(vals)
            errs = np.array(errs)

            colors = domain_colors[domain]
            bars = ax.bar(x, vals, width, yerr=errs,
                          color=colors, edgecolor="white", linewidth=0.5,
                          capsize=3, error_kw={"linewidth":1.2, "ecolor":"#555"})

            # 标注数值
            for i, (v, e) in enumerate(zip(vals, errs)):
                if not np.isnan(v):
                    ax.text(i, v + e + (max(vals[~np.isnan(vals)]) * 0.015),
                            f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color="#333")

            # 最优值高亮
            valid = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
            if valid:
                best_i = min(valid, key=lambda t: t[1] if lower_is_better else -t[1])[0]
                bars[best_i].set_edgecolor("#c62828")
                bars[best_i].set_linewidth(2.2)

            ax.set_xticks(x)
            ax.set_xticklabels(CFG_LABELS, fontsize=8)
            ax.set_ylabel(metric_label, fontsize=9)
            if row == 0:
                ax.set_title(f"{DOMAINS[domain]['label']}", fontsize=11, fontweight="bold")
            ax.yaxis.grid(True, alpha=0.4, linestyle="--")
            ax.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(OUT, "fig1_ablation_bar.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fig1] saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: G0_full 预测样本展示 (3域 × 3样本)
# ─────────────────────────────────────────────────────────────────────────────

def pick_samples(records, n_good=3, n_hard=2, stride=1):
    """
    挑选有代表性的样本：
      - good: tp == n_gt (全命中, fp==0, fn==0)
      - hard: fp>0 or fn>0 (有误差)
    stride: 从 good/hard 中每隔 stride 取一个，避免时序上过于相邻的样本
    """
    good, hard = [], []
    for r in records:
        n_gt = r["tp"] + r["fn"]
        if n_gt == 0:
            continue
        if r["tp"] == n_gt and r["fp"] == 0:
            good.append(r)
        else:
            hard.append(r)
    # 间隔采样使样本更分散
    good_sel = good[::stride][:n_good]
    hard_sel = hard[::stride][:n_hard]
    chosen = good_sel + hard_sel
    # 补不足
    all_valid = [r for r in records if (r["tp"]+r["fn"]) > 0 and r not in chosen]
    while len(chosen) < (n_good + n_hard) and all_valid:
        chosen.append(all_valid.pop(0))
    return chosen[:n_good + n_hard]


# ── 共享图例元素 (在 make_legend_handles 中构建) ──────────────────────────────
def make_legend_handles(fontsize=9):
    return [
        mpatches.Patch(facecolor=C_GT_TP_FILL, edgecolor=C_GT_TP_EDGE, linewidth=1,
                       alpha=0.8, label="GT span (TP matched)"),
        mpatches.Patch(facecolor=C_GT_FN_FILL, edgecolor=C_GT_FN_EDGE, linewidth=1,
                       alpha=0.8, label="GT span (FN missed)"),
        mpatches.Patch(facecolor="none",        edgecolor=C_PR_TP, linewidth=2,
                       linestyle="--",          label="Pred span (TP)"),
        mpatches.Patch(facecolor="none",        edgecolor=C_PR_FP, linewidth=2,
                       linestyle="--",          label="Pred span (FP)"),
        plt.Line2D([0],[0], marker="v", color=C_GT_TP_EDGE, linestyle="",
                   markersize=9, label="GT apex (TP)"),
        plt.Line2D([0],[0], marker="v", color=C_GT_FN_EDGE, linestyle="",
                   markersize=9, label="GT apex (FN)"),
        plt.Line2D([0],[0], marker="^", color=C_PR_TP, linestyle="",
                   markersize=9, label="Pred apex (TP)"),
        plt.Line2D([0],[0], marker="^", color=C_PR_FP, linestyle="",
                   markersize=9, label="Pred apex (FP)"),
        plt.Line2D([0],[0], color="#aaa", linewidth=1.2, label="History (input)"),
        plt.Line2D([0],[0], color="#222", linewidth=1.2, label="Pred window (target)"),
    ]


def _match_events(pred_evs, gt_evs, tol=3):
    """匈牙利匹配，返回 matched_pred, matched_gt (index sets), pred_to_gt (dict)。"""
    from scipy.optimize import linear_sum_assignment
    matched_pred, matched_gt = set(), set()
    pred_to_gt = {}  # pred_idx -> gt_idx for TP pairs
    if pred_evs and gt_evs:
        P = np.array([e["apex_idx"] for e in pred_evs], dtype=float)
        G = np.array([e["apex_idx"] for e in gt_evs],   dtype=float)
        cost = np.abs(P[:, None] - G[None, :])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if cost[r, c] <= tol:
                matched_pred.add(r)
                matched_gt.add(c)
                pred_to_gt[r] = c
    return matched_pred, matched_gt, pred_to_gt


def _event_intensity(ev):
    try:
        v = float(ev.get("apex_intensity", float("nan")))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v) or v <= 0:
        return None
    return v


def plot_sample(ax, series, record, domain_label, pred_len=96, ctx_len=96,
               show_mini_legend=False, show_axis_labels=False,
               show_xticks=False, show_intensity_labels=False,
               highlight_pred_window=False):
    """
    在 ax 上绘制单个样本：时序 + GT事件(彩色填充) + Pred事件(虚线边框)。

    图例色彩规范
    ─────────────────────────────────────────────────────
    蓝色填充区 (GT span TP)   : GT事件且被预测命中
    橙色填充区 (GT span FN)   : GT事件但被漏检
    绿色虚线框 (Pred span TP) : 预测事件且成功命中GT
    红色虚线框 (Pred span FP) : 预测事件但无对应GT
    ▼ 蓝色  : GT apex (TP)
    ▼ 橙色  : GT apex (FN)
    ▲ 绿色  : Pred apex (TP)
    ▲ 红色  : Pred apex (FP)
    """
    pred_start = record["pred_start"]
    hist_start = max(0, pred_start - ctx_len)
    pred_end   = pred_start + pred_len
    window_right = min(pred_end, len(series))

    seg_hist = np.arange(hist_start, pred_start)
    seg_pred = np.arange(pred_start, window_right)

    if highlight_pred_window and window_right > pred_start:
        ax.axvspan(pred_start, window_right, facecolor="#F5F5F5", alpha=0.65, zorder=0)
    if len(seg_hist) > 0:
        ax.plot(seg_hist, series[seg_hist], color="#aaa", linewidth=1.0)
    if len(seg_pred) > 0:
        ax.plot(seg_pred, series[seg_pred], color="#222", linewidth=1.2)

    # 竖线分隔历史/预测窗口
    ax.axvline(pred_start, color="#555", linewidth=1.0, linestyle=":", alpha=0.8)

    pred_evs = record["pred_events"]
    gt_evs   = record["gt_events"]
    matched_pred, matched_gt, pred_to_gt = _match_events(pred_evs, gt_evs)

    # y 范围 (包含 apex_intensity 值以免标记超出范围)
    win_vals = series[pred_start : window_right]
    if len(win_vals) == 0:
        return
    all_ys = list(win_vals)
    for ev in gt_evs:
        v = _event_intensity(ev)
        if v is not None:
            all_ys.append(v)
    for ev in pred_evs:
        v = _event_intensity(ev)
        if v is not None:
            all_ys.append(v)
    ymin, ymax = np.min(all_ys), np.max(all_ys)
    span = max(ymax - ymin, 1e-6)
    ax.set_ylim(ymin - span * 0.10, ymax + span * 0.22)
    ax.set_xlim(hist_start, max(hist_start + 1, window_right - 1))

    # ── GT 事件：填充区域 ──────────────────────────────────────────────────────
    for i, ev in enumerate(gt_evs):
        s, e, a = ev["onset_idx"], ev["end_idx"], ev["apex_idx"]
        fc = C_GT_TP_FILL if i in matched_gt else C_GT_FN_FILL
        ec = C_GT_TP_EDGE if i in matched_gt else C_GT_FN_EDGE
        ax.axvspan(s, e, alpha=0.20, facecolor=fc, edgecolor=ec,
                   linewidth=0.8, zorder=2)
        if pred_start <= a < pred_end and a < len(series):
            mc = C_GT_TP_EDGE if i in matched_gt else C_GT_FN_EDGE
            gt_int = _event_intensity(ev)
            marker_y = gt_int if gt_int is not None else series[a]
            ax.plot(a, marker_y, "v", color=mc, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.4, zorder=6)
            if show_intensity_labels and gt_int is not None:
                ax.text(a, marker_y - span * 0.055, f"{gt_int:.2f}",
                        ha="center", va="top", fontsize=7.5, color=mc,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8),
                        zorder=8)

    # ── Pred 事件：虚线轮廓 ───────────────────────────────────────────────────
    for i, ev in enumerate(pred_evs):
        s, e, a = ev["onset_idx"], ev["end_idx"], ev["apex_idx"]
        ec = C_PR_TP if i in matched_pred else C_PR_FP
        ax.add_patch(mpatches.Rectangle(
            (s, ymin - span * 0.05), e - s, (ymax - ymin) * 1.15,
            facecolor="none", edgecolor=ec, linewidth=1.2,
            linestyle="--", zorder=4
        ))
        if pred_start <= a < pred_end and a < len(series):
            mc = C_PR_TP if i in matched_pred else C_PR_FP
            pred_int = _event_intensity(ev)
            if pred_int is not None:
                ax.plot(a, pred_int, "^", color=mc, markersize=5,
                        markeredgecolor="white", markeredgewidth=0.4, zorder=7)
                if show_intensity_labels:
                    ax.text(a, pred_int + span * 0.055, f"{pred_int:.2f}",
                            ha="center", va="bottom", fontsize=7.5, color=mc,
                            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8),
                            zorder=8)

    tp, fp, fn = record["tp"], record["fp"], record["fn"]
    n_gt = tp + fn
    status = "Perfect" if fp == 0 and fn == 0 else (
             f"+{fp}FP" if fn == 0 else (
             f"-{fn}FN" if fp == 0 else f"+{fp}FP/-{fn}FN"))
    ax.set_title(f"TP={tp}/{n_gt}  {status}", fontsize=9, pad=3,
                 color=C_PR_TP if fp == 0 and fn == 0 else C_PR_FP)
    if show_xticks:
        tick_step = 24 if pred_len >= 168 else 12
        ticks = np.arange(hist_start, window_right, tick_step)
        if pred_start not in ticks:
            ticks = np.sort(np.unique(np.append(ticks, pred_start)))
        if len(ticks) > 0:
            ax.set_xticks(ticks)
        ax.tick_params(axis="x", labelsize=8)
    else:
        ax.set_xticks([])
    if show_axis_labels:
        ax.set_xlabel("Time index", fontsize=10)
        ax.set_ylabel("Load", fontsize=10)
        ax.yaxis.grid(True, alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9 if show_axis_labels else 7.5)

    if show_mini_legend:
        mini = [
            mpatches.Patch(fc=C_GT_TP_FILL, ec=C_GT_TP_EDGE, lw=1, alpha=0.8, label="GT-TP"),
            mpatches.Patch(fc=C_GT_FN_FILL, ec=C_GT_FN_EDGE, lw=1, alpha=0.8, label="GT-FN"),
            mpatches.Patch(fc="none",        ec=C_PR_TP,      lw=2, ls="--",   label="Pred-TP"),
            mpatches.Patch(fc="none",        ec=C_PR_FP,      lw=2, ls="--",   label="Pred-FP"),
        ]
        ax.legend(handles=mini, fontsize=6.5, loc="upper left",
                  framealpha=0.85, ncol=2, handlelength=1.2,
                  borderpad=0.4, labelspacing=0.25)


def fig2_sample_predictions():
    """3域 × 3样本总览图，底部大图例，每格有mini图例。"""
    n_cols = 3
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(36, 24),
                             gridspec_kw={"hspace": 0.38, "wspace": 0.10})
    fig.suptitle(
        "G0_Full Prediction Samples Overview  (seed=42, 3 domains × 3 samples)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for row, domain in enumerate(["wlel", "ett", "elc"]):
        series  = load_series(domain)
        records = load_outputs(domain, "G0_full", 42)
        samples = pick_samples(records, n_good=2, n_hard=1, stride=12)

        axes[row, 0].set_ylabel(
            f"{DOMAINS[domain]['label']}\n(value)", fontsize=12, fontweight="bold",
        )
        for col in range(n_cols):
            ax = axes[row, col]
            if col < len(samples):
                plot_sample(ax, series, samples[col], DOMAINS[domain]["label"],
                            show_mini_legend=(col == 0), ctx_len=48)
                ax.set_xlabel(f"sample #{samples[col]['sample_idx']}", fontsize=9)
            else:
                ax.axis("off")

    # ── 底部大图例 ─────────────────────────────────────────────────────────────
    handles = make_legend_handles(fontsize=10)
    fig.legend(
        handles=handles, loc="lower center", ncol=5, fontsize=10,
        bbox_to_anchor=(0.5, -0.06), frameon=True,
        title="Legend (applies to all subplots)",
        title_fontsize=10, framealpha=0.95,
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out = os.path.join(OUT, "fig2_sample_predictions.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fig2] saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: TP 误差分布直方图
# ─────────────────────────────────────────────────────────────────────────────

def fig3_error_dist():
    from scipy.stats import gaussian_kde

    metrics_info = [
        ("onset_errors",    "Onset Error (h)"),
        ("apex_errors",     "Apex Error (h)"),
        ("duration_errors", "Duration Error (h)"),
        ("intensity_apes",  "Intensity APE"),
    ]
    n_rows = len(metrics_info)

    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 13))
    fig.suptitle("TP Event Error Distributions — G0_Full (seed=42)\n"
                 "x-axis: absolute error (hours/APE);  vertical line = mean",
                 fontsize=12, fontweight="bold")

    domain_colors = {"wlel": "#2196F3", "ett": "#4CAF50", "elc": "#FF9800"}

    for row, (err_key, err_label) in enumerate(metrics_info):
        for col, domain in enumerate(["wlel", "ett", "elc"]):
            ax = axes[row, col]
            records = load_outputs(domain, "G0_full", 42)

            # collect all TP pair errors
            errors = []
            for rec in records:
                pred_evs = rec["pred_events"]
                gt_evs   = rec["gt_events"]
                if not pred_evs or not gt_evs:
                    continue
                from scipy.optimize import linear_sum_assignment
                P = np.array([e["apex_idx"] for e in pred_evs], dtype=float)
                G = np.array([e["apex_idx"] for e in gt_evs],   dtype=float)
                cost = np.abs(P[:, None] - G[None, :])
                ri, ci = linear_sum_assignment(cost)
                for r, c in zip(ri, ci):
                    if cost[r, c] <= 3:
                        if err_key == "onset_errors":
                            errors.append(abs(pred_evs[r]["onset_idx"] - gt_evs[c]["onset_idx"]))
                        elif err_key == "apex_errors":
                            errors.append(abs(pred_evs[r]["apex_idx"] - gt_evs[c]["apex_idx"]))
                        elif err_key == "duration_errors":
                            errors.append(abs(pred_evs[r]["duration"] - gt_evs[c]["duration"]))
                        elif err_key == "intensity_apes":
                            gt_int = float(gt_evs[c].get("apex_intensity", 0))
                            if gt_int > 0:
                                ape = abs(float(pred_evs[r].get("apex_intensity", 0)) - gt_int) / gt_int
                                errors.append(ape)

            errors = np.array(errors)
            color  = domain_colors[domain]
            is_ape = (err_key == "intensity_apes")

            if len(errors) < 5:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            max_e = min(np.percentile(errors, 98), errors.max())
            n_bins = 35
            bins  = np.linspace(0, max_e + (0.02 if is_ape else 0.5), n_bins)
            ax.hist(errors, bins=bins, color=color, alpha=0.7, edgecolor="white",
                    linewidth=0.5, density=True)

            # KDE
            if len(np.unique(errors)) > 2:
                kde = gaussian_kde(errors, bw_method=0.3)
                xs  = np.linspace(0, max_e, 300)
                ax.plot(xs, kde(xs), color=color, linewidth=2.0, alpha=0.9)

            mean_e = errors.mean()
            med_e  = np.median(errors)
            unit = "" if is_ape else "h"
            ax.axvline(mean_e, color="#c62828", linewidth=1.8, linestyle="--",
                       label=f"mean={mean_e:.3f}{unit}" if is_ape else f"mean={mean_e:.2f}{unit}")
            ax.axvline(med_e,  color="#333",    linewidth=1.2, linestyle=":",
                       label=f"median={med_e:.3f}{unit}" if is_ape else f"median={med_e:.2f}{unit}")

            ax.set_xlabel(err_label, fontsize=9)
            ax.set_ylabel("Density", fontsize=8)
            ax.legend(fontsize=8, loc="upper right")

            if row == 0:
                ax.set_title(f"{DOMAINS[domain]['label']}  (n={len(errors)} TP pairs)",
                             fontsize=10, fontweight="bold")

            if is_ape:
                pct_le10 = (errors <= 0.10).mean() * 100
                pct_le20 = (errors <= 0.20).mean() * 100
                ax.text(0.97, 0.97,
                        f"≤10%: {pct_le10:.0f}%\n≤20%: {pct_le20:.0f}%",
                        ha="right", va="top", fontsize=8.5,
                        transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            else:
                pct_zero = (errors == 0).mean() * 100
                pct_le1  = (errors <= 1).mean() * 100
                ax.text(0.97, 0.97,
                        f"0h: {pct_zero:.0f}%\n≤1h: {pct_le1:.0f}%",
                        ha="right", va="top", fontsize=8.5,
                        transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()
    out = os.path.join(OUT, "fig3_error_dist.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fig3] saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: 训练收敛曲线 (3域 × 3seeds)
# ─────────────────────────────────────────────────────────────────────────────

def fig4_training_curves():
    seed_colors = {42: "#1f77b4", 123: "#ff7f0e", 456: "#2ca02c"}
    seed_train_styles = {42: "-", 123: "--", 456: ":"}

    with plt.rc_context({
        "font.family": "Comic Sans MS",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }):
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
        fig.suptitle("Training convergence curves (G0_Full, 3 seeds)", fontsize=10, y=0.98)

        for col, domain in enumerate(["wlel", "ett", "elc"]):
            ax = axes[col]
            for seed in SEEDS:
                log = load_train_log(domain, seed)
                if not log:
                    continue
                epochs = [d["epoch"] for d in log]
                t_loss = [d["train_loss"] for d in log]
                v_loss = [d["val_loss"] for d in log]
                c = seed_colors[seed]
                train_ls = seed_train_styles[seed]

                ax.plot(
                    epochs,
                    t_loss,
                    color=c,
                    linewidth=1.5,
                    linestyle=train_ls,
                    alpha=0.9,
                )
                ax.plot(
                    epochs,
                    v_loss,
                    color=c,
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.85,
                )

                best_ep = epochs[int(np.argmin(v_loss))]
                ax.axvline(best_ep, color=c, linewidth=1.0, linestyle=":", alpha=0.35)

            ax.set_xlabel("Epoch")
            if col == 0:
                ax.set_ylabel("Cross-Entropy Loss")
            else:
                ax.set_ylabel("")
            ax.set_title(f"{DOMAINS[domain]['label']}", fontsize=9, fontweight="bold")
            ax.yaxis.grid(True, alpha=0.35, linestyle="--", linewidth=0.3)
            ax.set_axisbelow(True)

        handles = []
        for seed in SEEDS:
            c = seed_colors[seed]
            handles.append(
                plt.Line2D([0], [0], color=c, linewidth=1.5, linestyle=seed_train_styles[seed], label=f"s{seed} train")
            )
            handles.append(
                plt.Line2D([0], [0], color=c, linewidth=1.5, linestyle="--", label=f"s{seed} val")
            )
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=3,
            frameon=True,
            framealpha=0.95,
            bbox_to_anchor=(0.5, -0.02),
            fontsize=9,
        )
        fig.subplots_adjust(left=0.07, right=0.995, top=0.80, bottom=0.32, wspace=0.28)

        out = os.path.join(
            ROOT,
            "papers",
            "69c8ebe9537191b1512a6125",
            "figs",
            "fig_training_curves.png",
        )
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
    print(f"[fig4] saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: 每个样本单独一张 PNG
# ─────────────────────────────────────────────────────────────────────────────

def _draw_single_sample(rec, series, domain, tag, out_dir, pred_len=96):
    """将单个样本绘制为独立 PNG，带完整图例。"""
    dlabel = DOMAINS[domain]["label"]
    sidx   = rec["sample_idx"]
    tp, fp, fn = rec["tp"], rec["fp"], rec["fn"]
    n_gt = tp + fn
    status = "Perfect" if fp == 0 and fn == 0 else (
             f"+{fp}FP" if fn == 0 else (
             f"-{fn}FN" if fp == 0 else f"+{fp}FP / -{fn}FN"))
    cfg_name = _PRED_LEN_CONFIG[pred_len]

    fig, ax = plt.subplots(figsize=(18, 7.8))
    fig.suptitle(
        f"{dlabel}  —  {cfg_name}  pred_len={pred_len}  sample #{sidx}  "
        f"[{tag}]   TP={tp}/{n_gt}  {status}  (seed=42)",
        fontsize=13, fontweight="bold",
    )

    plot_sample(ax, series, rec, dlabel, pred_len=pred_len, ctx_len=48,
                show_mini_legend=False, show_axis_labels=True,
                show_xticks=True, show_intensity_labels=True,
                highlight_pred_window=True)

    handles = make_legend_handles(fontsize=11)
    ax.legend(
        handles=handles, loc="upper right", ncol=2, fontsize=10,
        framealpha=0.92, handlelength=1.5, borderpad=0.6,
        title="Legend", title_fontsize=10,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f"{tag}_s{sidx:04d}.png"
    out   = os.path.join(out_dir, fname)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return out


def fig5_domain_samples(domain, n_good_total=12, n_hard_total=8, pred_len=96):
    """
    每域每个样本单独保存为一张 PNG。
    输出到 plots/samples/{domain}_pred{pred_len}/good/ 和 .../hard/
    pred_len=96  → 使用 G0_full (arch_ablation_v2)
    pred_len=168/336 → 使用 E5_full (pred168/pred336 目录)
    """
    dlabel  = DOMAINS[domain]["label"]
    series  = load_series(domain)
    if pred_len == 96:
        records = load_outputs(domain, "G0_full", 42)
    else:
        records = load_pred_len_outputs(domain, pred_len, 42)

    good, hard = [], []
    for r in records:
        n_gt = r["tp"] + r["fn"]
        if n_gt == 0:
            continue
        if r["tp"] == n_gt and r["fp"] == 0:
            good.append(r)
        else:
            hard.append(r)

    stride_g = max(1, len(good) // (n_good_total * 2))
    stride_h = max(1, len(hard) // (n_hard_total * 2))
    good_sel = good[::stride_g][:n_good_total]
    hard_sel = hard[::stride_h][:n_hard_total]

    print(f"  [{dlabel}|pred={pred_len}] good={len(good_sel)}, hard={len(hard_sel)}")

    subdir   = f"{domain}_pred{pred_len}"
    good_dir = os.path.join(OUT, "samples", subdir, "good")
    hard_dir = os.path.join(OUT, "samples", subdir, "hard")
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(hard_dir, exist_ok=True)

    for rec in good_sel:
        p = _draw_single_sample(rec, series, domain, "good", good_dir, pred_len=pred_len)
        print(f"    {os.path.basename(p)}")
    for rec in hard_sel:
        p = _draw_single_sample(rec, series, domain, "hard", hard_dir, pred_len=pred_len)
        print(f"    {os.path.basename(p)}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Pred vs GT 强度散点图 (3域, G0_full, seed=42)
# ─────────────────────────────────────────────────────────────────────────────

def fig6_intensity_scatter():
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import pearsonr

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Predicted vs GT Apex Intensity — G0_Full (seed=42, TP pairs only)",
                 fontsize=13, fontweight="bold")

    for col, domain in enumerate(["wlel", "ett", "elc"]):
        ax = axes[col]
        color = DOMAINS[domain]["color"]
        records = load_outputs(domain, "G0_full", 42)

        pred_ints, gt_ints = [], []
        for rec in records:
            pred_evs = rec["pred_events"]
            gt_evs   = rec["gt_events"]
            if not pred_evs or not gt_evs:
                continue
            P = np.array([e["apex_idx"] for e in pred_evs], dtype=float)
            G = np.array([e["apex_idx"] for e in gt_evs],   dtype=float)
            cost = np.abs(P[:, None] - G[None, :])
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if cost[r, c] <= 3:
                    gt_int   = float(gt_evs[c].get("apex_intensity", 0))
                    pred_int = float(pred_evs[r].get("apex_intensity", 0))
                    if gt_int > 0:
                        pred_ints.append(pred_int)
                        gt_ints.append(gt_int)

        pred_ints = np.array(pred_ints)
        gt_ints   = np.array(gt_ints)

        ax.scatter(gt_ints, pred_ints, c=color, alpha=0.35, s=12, linewidths=0)

        # y=x diagonal
        lo = min(gt_ints.min(), pred_ints.min())
        hi = max(gt_ints.max(), pred_ints.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, alpha=0.6, label="y = x")

        # stats
        r_val, _ = pearsonr(gt_ints, pred_ints)
        mape_val = float(np.mean(np.abs(pred_ints - gt_ints) / gt_ints))
        ax.text(0.04, 0.96,
                f"r = {r_val:.3f}\nMAPE = {mape_val:.3f}\nn = {len(gt_ints)}",
                ha="left", va="top", fontsize=10, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

        ax.set_xlabel("GT apex intensity", fontsize=10)
        ax.set_ylabel("Pred apex intensity", fontsize=10)
        ax.set_title(f"{DOMAINS[domain]['label']}", fontsize=12, fontweight="bold")
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.xaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT, "fig6_intensity_scatter.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fig6] saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: AR 内部消融 (GD1-GD4, 3域 F1 柱状图)
# ─────────────────────────────────────────────────────────────────────────────

SUMMARY_JSON = os.path.join(BASE, "ablation_v2_summary.json")

def fig7_ar_ablation():
    """从 ablation_v2_summary.json 读取 AR 内部消融结果，绘制 3域 F1 柱状图。"""
    if not os.path.exists(SUMMARY_JSON):
        print(f"  [SKIP] {SUMMARY_JSON} not found")
        return
    with open(SUMMARY_JSON) as f:
        data = json.load(f)

    ar = data.get("ar_v2", {})
    domains = ["wlel", "ett", "elc"]
    # Okabe-Ito 色盲友好 4 色（无红绿对）
    cfg_colors = {"GD1_no_pos_mask": "#0072B2", "GD2_no_pos_smooth": "#56B4E9",
                  "GD3_no_attr_weight": "#E69F00", "GD4_no_causal": "#D55E00"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle("AR Internal Ablation (v2)  —  Event F1 (3-seed mean±std)",
                 fontsize=12, fontweight="bold")

    for i, d in enumerate(domains):
        rows = ar.get(d, {})
        names, means, stds = [], [], []
        for cfg, rec in rows.items():
            s = rec.get("summary", {}).get("event_f1", {})
            names.append(cfg)
            means.append(s.get("mean", 0))
            stds.append(s.get("std", 0))
        colors = [cfg_colors.get(n, "#999999") for n in names]
        xi = np.arange(len(names))
        axes[i].bar(xi, means, yerr=stds, color=colors, alpha=0.85,
                    capsize=4, edgecolor="white", linewidth=0.6)
        axes[i].set_xticks(xi)
        axes[i].set_xticklabels([n.replace("_", "\n") for n in names],
                                fontsize=7.5, ha="center")
        axes[i].set_title(f"{d.upper()}", fontsize=10, fontweight="bold")
        axes[i].set_ylabel("Event F1" if i == 0 else "", fontsize=9)
        axes[i].yaxis.grid(True, alpha=0.3, linestyle="--")
        axes[i].set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(OUT, "fig7_ar_ablation.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fig7] saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Phase1→Phase2 子任务消融 (WLEL, F1 + IntMAPE 双轴)
# ─────────────────────────────────────────────────────────────────────────────

def fig8_p1_to_p2():
    """Phase1 子任务消融对 Phase2 下游效果的影响。"""
    if not os.path.exists(SUMMARY_JSON):
        print(f"  [SKIP] {SUMMARY_JSON} not found")
        return
    with open(SUMMARY_JSON) as f:
        data = json.load(f)

    rows = data.get("gp1_wlel", {})
    if not rows:
        print("  [SKIP] gp1_wlel not found in summary")
        return

    pairs = []
    for cfg, rec in rows.items():
        s = rec.get("summary", {})
        f1   = s.get("event_f1", {}).get("mean", float("nan"))
        mape = s.get("intensity_mape", {}).get("mean", float("nan"))
        pairs.append((cfg, f1, mape))
    pairs.sort(key=lambda x: x[1], reverse=True)

    names = [p[0] for p in pairs]
    f1    = np.array([p[1] for p in pairs], dtype=float)
    mape  = np.array([p[2] for p in pairs], dtype=float)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle("Phase1 Subtask Ablation → Phase2 Downstream (WLEL, 3-seed mean)",
                 fontsize=12, fontweight="bold")
    x = np.arange(len(names))
    w = 0.36

    # F1 bars (Okabe-Ito blue)
    ax1.bar(x - w/2, f1, width=w, color="#0072B2", alpha=0.85, label="P2 Event F1")
    ax1.set_ylabel("P2 Event F1", fontsize=9, color="#0072B2")
    ax1.set_ylim(max(0, np.nanmin(f1) - 0.05), min(1.0, np.nanmax(f1) + 0.03))
    ax1.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    # IntMAPE bars (Okabe-Ito orange)
    ax2 = ax1.twinx()
    ax2.bar(x + w/2, mape, width=w, color="#D55E00", alpha=0.85, label="P2 Int MAPE")
    ax2.set_ylabel("P2 Intensity MAPE", fontsize=9, color="#D55E00")
    ax2.set_ylim(max(0, np.nanmin(mape) - 0.01), np.nanmax(mape) + 0.01)

    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace("_", "\n") for n in names],
                        fontsize=8, ha="center")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right", framealpha=0.88)

    plt.tight_layout()
    out = os.path.join(OUT, "fig8_p1_to_p2.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fig8] saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9 & 10: 不同预测长度 (pred_len=96/168/336) 对比
# ─────────────────────────────────────────────────────────────────────────────

PRED_LENS = [96, 168, 336]

# pred_len=96 复用消融目录 G0_full；168/336 使用专用目录
_PRED_LEN_SUBDIR = {
    96:  "{domain}_arch_ablation_v2",
    168: "{domain}_pred168",
    336: "{domain}_pred336",
}
_PRED_LEN_CONFIG = {
    96:  "G0_full",
    168: "E5_full",
    336: "E5_full",
}


def load_pred_len_summary(domain, pred_len, seed):
    """读取 test_summary.json (test_* 键名)。"""
    subdir  = _PRED_LEN_SUBDIR[pred_len].format(domain=domain)
    config  = _PRED_LEN_CONFIG[pred_len]
    p = os.path.join(BASE, subdir, f"{config}_s{seed}", "test_summary.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def load_pred_len_outputs(domain, pred_len, seed):
    """读取 eval_outputs.jsonl，返回 list[dict]。"""
    subdir  = _PRED_LEN_SUBDIR[pred_len].format(domain=domain)
    config  = _PRED_LEN_CONFIG[pred_len]
    p = os.path.join(BASE, subdir, f"{config}_s{seed}",
                     "test_eval", "eval_outputs.jsonl")
    if not os.path.exists(p):
        return []
    recs = []
    with open(p) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def get_pred_len_mean_std(domain, pred_len, metric_key):
    """计算 3-seed mean±std，metric_key 为 test_summary.json 中的键名。"""
    vals = []
    for s in SEEDS:
        m = load_pred_len_summary(domain, pred_len, s)
        if m and metric_key in m:
            v = float(m[metric_key])
            if not np.isnan(v):
                vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def fig9_pred_len_comparison():
    """
    Fig9: pred_len=96/168/336 性能对比
    3行 (F1 / OnsetMAE / ApexMAE) × 3列 (WLEL / ETT / ELC)
    每列内 3组柱 (96/168/336)
    """
    metrics = [
        ("test_event_f1",   "Event F1",      False),
        ("test_onset_mae",  "Onset MAE (h)", True),
        ("test_apex_mae",   "Apex MAE (h)",  True),
    ]
    pl_colors  = {96: "#2196F3", 168: "#4CAF50", 336: "#FF9800"}
    pl_labels  = {96: "96→96", 168: "96→168", 336: "96→336"}
    n_pl   = len(PRED_LENS)
    x      = np.arange(n_pl)
    width  = 0.55

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle(
        "Prediction Length Scalability  (seq_len=96, pred_len ∈ {96, 168, 336})\n"
        "E5-Full configuration, 3-seed mean±std",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for row, (mkey, mlabel, lower) in enumerate(metrics):
        for col, domain in enumerate(["wlel", "ett", "elc"]):
            ax = axes[row, col]
            means, stds, colors = [], [], []
            for pl in PRED_LENS:
                mn, sd = get_pred_len_mean_std(domain, pl, mkey)
                means.append(mn)
                stds.append(sd)
                colors.append(pl_colors[pl])

            means = np.array(means)
            stds  = np.array(stds)
            bars = ax.bar(x, means, width, yerr=stds, color=colors,
                          edgecolor="white", linewidth=0.6,
                          capsize=4, error_kw={"linewidth": 1.2, "ecolor": "#555"})

            # 数值标注
            for i, (v, e) in enumerate(zip(means, stds)):
                if not np.isnan(v):
                    top = v + e + (np.nanmax(means) * 0.018)
                    ax.text(i, top, f"{v:.3f}", ha="center", va="bottom",
                            fontsize=8, color="#333")

            # 最优高亮
            valid = [(i, v) for i, v in enumerate(means) if not np.isnan(v)]
            if valid:
                best_i = min(valid, key=lambda t: t[1] if lower else -t[1])[0]
                bars[best_i].set_edgecolor("#c62828")
                bars[best_i].set_linewidth(2.2)

            ax.set_xticks(x)
            ax.set_xticklabels([pl_labels[pl] for pl in PRED_LENS], fontsize=9)
            ax.set_ylabel(mlabel, fontsize=9)
            if row == 0:
                dlabel = {"wlel": "WLEL", "ett": "ETT", "elc": "ELC"}[domain]
                ax.set_title(dlabel, fontsize=11, fontweight="bold")
            ax.yaxis.grid(True, alpha=0.4, linestyle="--")
            ax.set_axisbelow(True)

    # 图例
    handles = [mpatches.Patch(facecolor=pl_colors[pl], label=pl_labels[pl])
               for pl in PRED_LENS]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.03), frameon=True, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(OUT, "fig9_pred_len_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fig9] saved → {out}")


def fig10_pred_len_samples():
    """
    Fig10: pred_len=168/336 样本预测展示
    2行 (168 / 336) × 3列 (WLEL / ETT / ELC), seed=42, 各取1个样本
    """
    fig, axes = plt.subplots(2, 3, figsize=(30, 14),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.10})
    fig.suptitle(
        "Prediction Samples — Extended pred_len  (seed=42, G0/E5-Full)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    pl_row = {168: 0, 336: 1}
    for pred_len, row in pl_row.items():
        for col, domain in enumerate(["wlel", "ett", "elc"]):
            ax = axes[row, col]
            series  = load_series(domain)
            records = load_pred_len_outputs(domain, pred_len, 42)
            if not records:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.axis("off")
                continue
            samples = pick_samples(records, n_good=1, n_hard=0, stride=20)
            if not samples:
                samples = records[:1]

            dlabel = {"wlel": "WLEL", "ett": "ETT", "elc": "ELC"}[domain]
            if row == 0:
                ax.set_title(dlabel, fontsize=11, fontweight="bold")
            plot_sample(ax, series, samples[0], dlabel,
                        pred_len=pred_len, ctx_len=96,
                        show_mini_legend=(col == 0))
            ax.set_xlabel(f"sample #{samples[0]['sample_idx']}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"pred_len={pred_len}\n(value)",
                              fontsize=11, fontweight="bold")

    handles = make_legend_handles(fontsize=10)
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.06), frameon=True,
               title="Legend", title_fontsize=10, framealpha=0.95)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out = os.path.join(OUT, "fig10_pred_len_samples.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[fig10] saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase-2 Visualization")
    print(f"Output dir: {OUT}")
    print("=" * 60)

    print("\n[ 1/10] Fig1: ablation bar chart (4 metrics) ...")
    fig1_ablation_bar()

    print("\n[ 2/10] Fig2: sample predictions overview ...")
    fig2_sample_predictions()

    print("\n[ 3/10] Fig3: error distributions (4 rows incl. intensity APE) ...")
    fig3_error_dist()

    print("\n[ 4/10] Fig4: training curves ...")
    fig4_training_curves()

    print("\n[ 5/10] Fig6: intensity scatter plot ...")
    fig6_intensity_scatter()

    print("\n[ 6/10] Fig7: AR internal ablation ...")
    fig7_ar_ablation()

    print("\n[ 7/10] Fig8: P1→P2 subtask ablation ...")
    fig8_p1_to_p2()

    step = 8
    for pl in PRED_LENS:
        for dom in ["wlel", "ett", "elc"]:
            print(f"\n[{step:2d}] Fig5: domain samples — {dom.upper()} pred_len={pl} ...")
            fig5_domain_samples(dom, n_good_total=12, n_hard_total=8, pred_len=pl)
            step += 1

    print(f"\n[{step}] Fig9: pred_len comparison (96/168/336) ...")
    fig9_pred_len_comparison()
    step += 1

    print(f"\n[{step}] Fig10: pred_len sample predictions (168/336) ...")
    fig10_pred_len_samples()

    print("\nDone. All figures saved to:", OUT)
