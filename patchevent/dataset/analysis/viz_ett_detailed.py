"""
ETT la=3 数据详细可视化:
1. 全局时序 + 事件标注 (train/val/test分段)
2. 放大展示几个典型事件 (onset-apex-end)
3. Duration分布
4. Peak间距分布
5. Events/window分布
6. 消融MAE对比柱状图
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# ── Data ──
SERIES = "dataset/ett/event_v1/data/ett_event_series_v1.csv"
EVENTS = "dataset/ett/event_v1/data/ett_events_v1.jsonl"

series = pd.read_csv(SERIES, parse_dates=["timestamp"])
vals = series["value"].values
splits = series["split"].values

events = []
with open(EVENTS) as f:
    for line in f:
        if line.strip():
            events.append(json.loads(line))

# ── Ablation results ──
ABLATION = {
    "S0_full":      {"F1": 0.8511, "ApMAE": 0.631, "OnMAE": 0.680},
    "S1_wo_mempos": {"F1": 0.8534, "ApMAE": 0.642, "OnMAE": 0.688},
    "S2_wo_sa":     {"F1": 0.8535, "ApMAE": 0.683, "OnMAE": 0.731},
    "S3_bare":      {"F1": 0.8532, "ApMAE": 0.664, "OnMAE": 0.719},
    "S4_cnn":       {"F1": 0.8548, "ApMAE": 0.741, "OnMAE": 0.802},
}

# ── Figure ──
fig = plt.figure(figsize=(22, 24))
gs = gridspec.GridSpec(5, 2, height_ratios=[1.2, 1.2, 1, 1, 1], hspace=0.35, wspace=0.3)

SPLIT_COLORS = {"train": "#2196F3", "val": "#FF9800", "test": "#E91E63"}

# ============================================================
# Row 0: Full time series with split regions + apex markers
# ============================================================
ax0 = fig.add_subplot(gs[0, :])
n = len(vals)
ax0.plot(range(n), vals, 'k-', linewidth=0.3, alpha=0.7)

# Split background + boundary lines
for sp, color in SPLIT_COLORS.items():
    mask = splits == sp
    idx = np.where(mask)[0]
    if len(idx) > 0:
        ax0.axvspan(idx[0], idx[-1], alpha=0.06, color=color, label=f"{sp} ({len(idx)}h)")
        ax0.axvline(idx[0], color=color, linewidth=1.5, linestyle='--', alpha=0.6)

# Mark apexes only (not full spans — too dense for global view)
apex_idx_arr = np.array([int(e["apex_idx"]) for e in events])
ax0.scatter(apex_idx_arr, vals[apex_idx_arr], c='red', s=6, zorder=3, alpha=0.5, label=f'Apexes ({len(events)})')

ax0.set_title("ETT (ETTh2) Full Series with Peak Locations (la=3, rf=0.020)", fontsize=13, fontweight='bold')
ax0.set_xlabel("Hour Index")
ax0.set_ylabel("OT Value")
ax0.legend(loc='upper right', fontsize=9)
ax0.set_xlim(0, n)

# Stats annotation
n_train_ev = sum(1 for e in events if splits[int(e["apex_idx"])] == "train")
n_val_ev = sum(1 for e in events if splits[int(e["apex_idx"])] == "val")
n_test_ev = sum(1 for e in events if splits[int(e["apex_idx"])] == "test")
ax0.text(0.01, 0.95, f"Total: {len(events)} events  (train:{n_train_ev}  val:{n_val_ev}  test:{n_test_ev})",
         transform=ax0.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================
# Row 1: Zoomed views of 3 representative event regions
# ============================================================
# Pick 2 regions: train and test, each ~120h to show clear event structure
regions = []
for target_split in ["train", "test"]:
    sp_events = [e for e in events if splits[int(e["apex_idx"])] == target_split]
    if sp_events:
        # Pick a region with ~3-4 events visible
        mid_ev = sp_events[len(sp_events) // 3]
        center = int(mid_ev["apex_idx"])
        start = max(0, center - 60)
        end = min(n, center + 60)
        regions.append((target_split, start, end))

for i, (sp, rstart, rend) in enumerate(regions[:2]):
    ax = fig.add_subplot(gs[1, i])
    win = rend - rstart
    x = np.arange(win)
    ax.plot(x, vals[rstart:rend], 'k-', linewidth=1.2)

    ev_in_view = []
    for ev in events:
        s, e, a = int(ev["onset_idx"]), int(ev["end_idx"]), int(ev["apex_idx"])
        if e < rstart or s > rend:
            continue
        ev_in_view.append(ev)
        s_c = max(s, rstart) - rstart
        e_c = min(e, rend) - rstart
        a_c = a - rstart
        ax.axvspan(s_c, e_c, alpha=0.18, color='coral')
        if rstart <= a < rend:
            ax.plot(a_c, vals[a], 'rv', markersize=10, zorder=5)
        if rstart <= s < rend:
            ax.plot(s - rstart, vals[s], 'g>', markersize=8, zorder=5)
        if rstart <= e < rend:
            ax.plot(e - rstart, vals[e], 'b<', markersize=8, zorder=5)

    ax.set_title(f"Zoomed: {sp} split (hour {rstart}-{rend}, {len(ev_in_view)} events)", fontsize=11)
    ax.set_xlabel("Relative Hour")
    ax.set_ylabel("OT Value")
    ax.legend([Patch(facecolor='coral', alpha=0.2),
               plt.Line2D([0],[0], marker='v', color='red', linestyle='', markersize=8),
               plt.Line2D([0],[0], marker='>', color='green', linestyle='', markersize=8),
               plt.Line2D([0],[0], marker='<', color='blue', linestyle='', markersize=8)],
              ['Event span', 'Apex', 'Onset', 'End'], loc='upper right', fontsize=8)

# ============================================================
# Row 2 left: Duration distribution
# ============================================================
ax = fig.add_subplot(gs[2, 0])
durations = [int(e["duration"]) for e in events]
bins = np.arange(min(durations) - 0.5, max(durations) + 1.5, 1)
counts, _, bars = ax.hist(durations, bins=bins, color='steelblue', edgecolor='navy', alpha=0.8)
ax.set_title("Event Duration Distribution", fontsize=11, fontweight='bold')
ax.set_xlabel("Duration (hours)")
ax.set_ylabel("Count")
ax.axvline(np.mean(durations), color='red', linestyle='--', linewidth=1.5,
           label=f'mean={np.mean(durations):.2f}h')
ax.axvline(np.median(durations), color='orange', linestyle=':', linewidth=1.5,
           label=f'median={np.median(durations):.1f}h')
ax.legend(fontsize=9)
# Add count labels on bars
for bar, cnt in zip(bars, counts):
    if cnt > 0:
        ax.text(bar.get_x() + bar.get_width()/2, cnt + 2, f'{int(cnt)}',
               ha='center', fontsize=8)

# ============================================================
# Row 2 right: Peak spacing distribution
# ============================================================
ax = fig.add_subplot(gs[2, 1])
apex_indices = sorted([int(e["apex_idx"]) for e in events])
spacings = np.diff(apex_indices)
ax.hist(spacings, bins=30, color='#66BB6A', edgecolor='#2E7D32', alpha=0.8)
ax.axvline(np.mean(spacings), color='red', linestyle='--', linewidth=1.5,
           label=f'mean={np.mean(spacings):.1f}h')
ax.axvline(24, color='orange', linestyle=':', linewidth=1.5, label='24h (daily)')
ax.set_title("Inter-Peak Spacing Distribution", fontsize=11, fontweight='bold')
ax.set_xlabel("Hours between consecutive peaks")
ax.set_ylabel("Count")
ax.legend(fontsize=9)

# ============================================================
# Row 3 left: Events per 96h prediction window (test split)
# ============================================================
ax = fig.add_subplot(gs[3, 0])
SEQ_LEN, PRED_LEN, STRIDE = 96, 96, 4
sp_idx = np.where(splits == "test")[0]
g_start, g_end = int(sp_idx[0]), int(sp_idx[-1])
offset = (STRIDE - g_start % STRIDE) % STRIDE
first_aligned = g_start + offset
window_end = g_end - SEQ_LEN - PRED_LEN + 1
onset_arr = np.array([int(e["onset_idx"]) for e in events])

if window_end >= first_aligned:
    starts = np.arange(first_aligned, window_end + 1, STRIDE)
    ev_counts = []
    for s in starts:
        ps = s + SEQ_LEN
        pe = ps + PRED_LEN
        nc = int(np.sum((onset_arr >= ps) & (onset_arr < pe)))
        ev_counts.append(nc)
    ev_counts = np.array(ev_counts)
    unique, cnts = np.unique(ev_counts, return_counts=True)
    bars = ax.bar(unique, cnts, color='#AB47BC', edgecolor='#6A1B9A', alpha=0.8)
    for bar, cnt in zip(bars, cnts):
        ax.text(bar.get_x() + bar.get_width()/2, cnt + 1, f'{cnt}',
               ha='center', fontsize=8)
    ax.set_title(f"Events per Prediction Window (test, n={len(starts)})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Number of events in 96h window")
    ax.set_ylabel("Count")
    ax.text(0.98, 0.95, f'mean={ev_counts.mean():.2f}\nstd={ev_counts.std():.2f}',
            transform=ax.transAxes, fontsize=9, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))

# ============================================================
# Row 3 right: Apex intensity distribution
# ============================================================
ax = fig.add_subplot(gs[3, 1])
intensities = [float(e["apex_intensity"]) for e in events]
ax.hist(intensities, bins=30, color='#FF7043', edgecolor='#BF360C', alpha=0.8)
ax.axvline(np.mean(intensities), color='blue', linestyle='--', linewidth=1.5,
           label=f'mean={np.mean(intensities):.1f}')
ax.set_title("Apex Intensity Distribution", fontsize=11, fontweight='bold')
ax.set_xlabel("Apex Value (OT)")
ax.set_ylabel("Count")
ax.legend(fontsize=9)

# ============================================================
# Row 4: Ablation comparison (ApexMAE + OnsetMAE grouped bar)
# ============================================================
ax = fig.add_subplot(gs[4, 0])
names = list(ABLATION.keys())
labels = ["S0\nFull", "S1\n-MemPos", "S2\n-SA", "S3\nBare", "S4\nCNN"]
ap_mae = [ABLATION[n]["ApMAE"] for n in names]
on_mae = [ABLATION[n]["OnMAE"] for n in names]
x = np.arange(len(names))
w = 0.35
bars1 = ax.bar(x - w/2, ap_mae, w, color='#1976D2', alpha=0.85, label='Apex MAE')
bars2 = ax.bar(x + w/2, on_mae, w, color='#F57C00', alpha=0.85, label='Onset MAE')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("MAE (hours)")
ax.set_title("Ablation: Apex & Onset MAE (lower=better)", fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
           f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#1976D2')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
           f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#F57C00')
ax.set_ylim(0, max(on_mae) * 1.2)

# Row 4 right: F1 bar
ax = fig.add_subplot(gs[4, 1])
f1s = [ABLATION[n]["F1"] for n in names]
colors = ['#4CAF50' if n == 'S0_full' else '#78909C' for n in names]
bars = ax.bar(x, f1s, 0.6, color=colors, edgecolor='#37474F', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Event F1")
ax.set_title("Ablation: Event F1 (tolerance=3)", fontsize=11, fontweight='bold')
ax.set_ylim(0.84, 0.86)
for bar, f1 in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
           f'{f1:.4f}', ha='center', fontsize=9, fontweight='bold')

fig.suptitle("ETT Dataset (ETTh2) — Event Data & Ablation Analysis\nlookahead=3, rate_frac=0.020, 798 events",
             fontsize=15, fontweight='bold', y=0.995)
plt.savefig("_viz_ett_la3_detailed.png", dpi=150, bbox_inches='tight')
print("Saved: _viz_ett_la3_detailed.png")
plt.close()
