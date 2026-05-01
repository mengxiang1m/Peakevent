"""
可视化对比 ETT/ELC 新旧事件标注 (rate_frac=0.40 vs 0.020)
同时展示新数据的 duration 分布和 events/window 分布
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT_DIR = "patchevent/phase2/checkpoints"

DATASETS = {
    "ett": {
        "series_old": "dataset/ett/event_v1_old_ratefrac040/data/ett_event_series_v1.csv",
        "events_old": "dataset/ett/event_v1_old_ratefrac040/data/ett_events_v1.jsonl",
        "series_new": "dataset/ett/event_v1/data/ett_event_series_v1.csv",
        "events_new": "dataset/ett/event_v1/data/ett_events_v1.jsonl",
        "title": "ETT (ETTh2)",
    },
    "elc": {
        "series_old": "dataset/electricity/event_v1_old_ratefrac040/data/elc_event_series_v1.csv",
        "events_old": "dataset/electricity/event_v1_old_ratefrac040/data/elc_events_v1.jsonl",
        "series_new": "dataset/electricity/event_v1/data/elc_event_series_v1.csv",
        "events_new": "dataset/electricity/event_v1/data/elc_events_v1.jsonl",
        "title": "ELC (Electricity)",
    },
}


def load_events(path):
    events = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def plot_event_overlay(ax, series_vals, events, start, end, color, label, alpha=0.3, y_offset=0):
    """在时序上叠加事件区间"""
    for ev in events:
        s = int(ev["onset_idx"])
        e = int(ev["end_idx"])
        a = int(ev["apex_idx"])
        if e < start or s > end:
            continue
        s_clip = max(s, start) - start
        e_clip = min(e, end) - start
        a_clip = a - start
        ax.axvspan(s_clip, e_clip, alpha=alpha, color=color, label=label if ev == events[0] else None)
        if start <= a <= end:
            ax.axvline(a_clip, color=color, alpha=0.7, linewidth=1.5, linestyle='--')


fig, axes = plt.subplots(4, 2, figsize=(18, 16))
fig.suptitle("Event Annotation Comparison: rate_frac=0.40 (OLD) vs rate_frac=0.020 (NEW)", fontsize=14, fontweight='bold')

for col, (domain, cfg) in enumerate(DATASETS.items()):
    series_old = pd.read_csv(cfg["series_old"], parse_dates=["timestamp"])
    series_new = pd.read_csv(cfg["series_new"], parse_dates=["timestamp"])
    events_old = load_events(cfg["events_old"])
    events_new = load_events(cfg["events_new"])

    vals_old = series_old["value"].values
    vals_new = series_new["value"].values

    # --- Row 0: 时序 + 事件overlay (旧 vs 新) --- 选一段有代表性的区间
    # 选训练集中间一段 (约200小时)
    n = len(vals_old)
    mid = int(n * 0.35)
    window = 200
    start, end = mid, mid + window

    ax = axes[0, col]
    x_range = np.arange(window)
    ax.plot(x_range, vals_old[start:end], 'k-', linewidth=0.8, alpha=0.8)
    for ev in events_old:
        s = int(ev["onset_idx"])
        e = int(ev["end_idx"])
        a = int(ev["apex_idx"])
        if e < start or s > end:
            continue
        s_c = max(s, start) - start
        e_c = min(e, end) - start
        ax.axvspan(s_c, e_c, alpha=0.25, color='red')
        if start <= a <= end:
            ax.plot(a - start, vals_old[a], 'rv', markersize=6)
    ax.set_title(f"{cfg['title']} — OLD (rate_frac=0.40)", fontsize=11)
    ax.set_ylabel("Value")
    # 添加自定义图例
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor='red', alpha=0.25), plt.Line2D([0],[0], marker='v', color='red', linestyle='')],
              ['Event span', 'Apex'], loc='upper right', fontsize=8)

    ax = axes[1, col]
    ax.plot(x_range, vals_new[start:end], 'k-', linewidth=0.8, alpha=0.8)
    for ev in events_new:
        s = int(ev["onset_idx"])
        e = int(ev["end_idx"])
        a = int(ev["apex_idx"])
        if e < start or s > end:
            continue
        s_c = max(s, start) - start
        e_c = min(e, end) - start
        ax.axvspan(s_c, e_c, alpha=0.25, color='blue')
        if start <= a <= end:
            ax.plot(a - start, vals_new[a], 'bv', markersize=6)
    ax.set_title(f"{cfg['title']} — NEW (rate_frac=0.020)", fontsize=11)
    ax.set_ylabel("Value")
    ax.legend([Patch(facecolor='blue', alpha=0.25), plt.Line2D([0],[0], marker='v', color='blue', linestyle='')],
              ['Event span', 'Apex'], loc='upper right', fontsize=8)

    # --- Row 2: Duration 分布对比 ---
    ax = axes[2, col]
    dur_old = [int(e["duration"]) for e in events_old]
    dur_new = [int(e["duration"]) for e in events_new]

    bins = np.arange(2.5, 8.5, 1)
    ax.hist(dur_old, bins=bins, alpha=0.6, color='red', label=f'OLD (mean={np.mean(dur_old):.2f})', edgecolor='darkred')
    ax.hist(dur_new, bins=bins, alpha=0.6, color='blue', label=f'NEW (mean={np.mean(dur_new):.2f})', edgecolor='darkblue')
    ax.set_title(f"{cfg['title']} — Duration Distribution", fontsize=11)
    ax.set_xlabel("Duration (hours)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)

    # --- Row 3: Events/window 分布 ---
    ax = axes[3, col]
    SEQ_LEN, PRED_LEN, STRIDE = 96, 96, 4

    for version, evts, series, color, lbl in [
        ("old", events_old, series_old, "red", "OLD"),
        ("new", events_new, series_new, "blue", "NEW"),
    ]:
        splits = series["split"].values
        sp_idx = np.where(splits == "test")[0]
        g_start = int(sp_idx[0])
        g_end = int(sp_idx[-1])
        window_end = g_end - SEQ_LEN - PRED_LEN + 1
        offset = (STRIDE - g_start % STRIDE) % STRIDE
        first_aligned = g_start + offset
        if window_end >= first_aligned:
            starts = np.arange(first_aligned, window_end + 1, STRIDE)
            onset_arr = np.array([e["onset_idx"] for e in evts])
            ev_counts = []
            for s in starts:
                ps = s + SEQ_LEN
                pe = ps + PRED_LEN
                n_ev = int(np.sum((onset_arr >= ps) & (onset_arr < pe)))
                ev_counts.append(n_ev)
            ev_counts = np.array(ev_counts)
            unique, counts = np.unique(ev_counts, return_counts=True)
            ax.bar(unique - 0.15 if version == "old" else unique + 0.15,
                   counts / len(ev_counts) * 100,
                   width=0.3, color=color, alpha=0.7,
                   label=f'{lbl} (mean={ev_counts.mean():.2f})')

    ax.set_title(f"{cfg['title']} — Test Window Events/Window Distribution", fontsize=11)
    ax.set_xlabel("Events per window")
    ax.set_ylabel("Percentage (%)")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/_viz_event_comparison.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR}/_viz_event_comparison.png")
plt.close()
