"""
ELC Dataset Detailed Visualization (same style as ETT)
- Full series + peak locations
- Zoomed train/test views
- Duration, spacing, events/window, intensity distributions
- Ablation MAE + F1 bar charts (3-seed mean)
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

SERIES = "dataset/electricity/event_v1/data/elc_event_series_v1.csv"
EVENTS = "dataset/electricity/event_v1/data/elc_events_v1.jsonl"

series = pd.read_csv(SERIES, parse_dates=["timestamp"])
vals = series["value"].values
splits = series["split"].values

events = []
with open(EVENTS) as f:
    for line in f:
        if line.strip():
            events.append(json.loads(line))

# Load ablation results (3-seed mean, S0-S4)
BASE = "patchevent/phase2/checkpoints/elc"
CONFIGS = ["S0_full", "S1_wo_mempos", "S2_wo_sa", "S3_bare", "S4_cnn"]
SEEDS = [42, 123, 456]
ABLATION = {}
for cfg in CONFIGS:
    f1s, ap, on = [], [], []
    for s in SEEDS:
        p = os.path.join(BASE, f"{cfg}_s{s}", "test_summary.json")
        if os.path.exists(p):
            r = json.load(open(p))
            f1s.append(r["test_event_f1"])
            ap.append(r["test_apex_mae"])
            on.append(r["test_onset_mae"])
    if f1s:
        ABLATION[cfg] = {
            "F1": np.mean(f1s), "F1_std": np.std(f1s),
            "ApMAE": np.mean(ap), "ApMAE_std": np.std(ap),
            "OnMAE": np.mean(on), "OnMAE_std": np.std(on),
        }

fig = plt.figure(figsize=(22, 24))
gs = gridspec.GridSpec(5, 2, height_ratios=[1.2, 1.2, 1, 1, 1],
                       hspace=0.35, wspace=0.3)
SPLIT_COLORS = {"train": "#2196F3", "val": "#FF9800", "test": "#E91E63"}

# Row 0: Full series + apex markers
ax0 = fig.add_subplot(gs[0, :])
n = len(vals)
ax0.plot(range(n), vals, 'k-', linewidth=0.3, alpha=0.7)
for sp, color in SPLIT_COLORS.items():
    mask = splits == sp
    idx = np.where(mask)[0]
    if len(idx) > 0:
        ax0.axvspan(idx[0], idx[-1], alpha=0.06, color=color,
                    label=f"{sp} ({len(idx)}h)")
        ax0.axvline(idx[0], color=color, linewidth=1.5,
                    linestyle='--', alpha=0.6)
apex_arr = np.array([int(e["apex_idx"]) for e in events])
ax0.scatter(apex_arr, vals[apex_arr], c='red', s=6, zorder=3,
            alpha=0.5, label=f'Apexes ({len(events)})')
ax0.set_title("ELC (Electricity) Full Series with Peak Locations "
              "(la=5, rf=0.020)", fontsize=13, fontweight='bold')
ax0.set_xlabel("Hour Index")
ax0.set_ylabel("Value")
ax0.legend(loc='upper right', fontsize=9)
ax0.set_xlim(0, n)
nt = sum(1 for e in events if splits[int(e["apex_idx"])] == "train")
nv = sum(1 for e in events if splits[int(e["apex_idx"])] == "val")
ne = sum(1 for e in events if splits[int(e["apex_idx"])] == "test")
ax0.text(0.01, 0.95,
         f"Total: {len(events)} events  (train:{nt}  val:{nv}  test:{ne})",
         transform=ax0.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Row 1: Zoomed views (train + test)
for col, target_split in enumerate(["train", "test"]):
    sp_events = [e for e in events
                 if splits[int(e["apex_idx"])] == target_split]
    if not sp_events:
        continue
    mid_ev = sp_events[len(sp_events) // 3]
    center = int(mid_ev["apex_idx"])
    rstart = max(0, center - 60)
    rend = min(n, center + 60)
    ax = fig.add_subplot(gs[1, col])
    win = rend - rstart
    ax.plot(np.arange(win), vals[rstart:rend], 'k-', linewidth=1.2)
    ev_cnt = 0
    for ev in events:
        s, e2, a = int(ev["onset_idx"]), int(ev["end_idx"]), int(ev["apex_idx"])
        if e2 < rstart or s > rend:
            continue
        ev_cnt += 1
        sc = max(s, rstart) - rstart
        ec = min(e2, rend) - rstart
        ac = a - rstart
        ax.axvspan(sc, ec, alpha=0.18, color='coral')
        if rstart <= a < rend:
            ax.plot(ac, vals[a], 'rv', markersize=10, zorder=5)
        if rstart <= s < rend:
            ax.plot(s - rstart, vals[s], 'g>', markersize=8, zorder=5)
        if rstart <= e2 < rend:
            ax.plot(e2 - rstart, vals[e2], 'b<', markersize=8, zorder=5)
    ax.set_title(f"Zoomed: {target_split} split "
                 f"(hour {rstart}-{rend}, {ev_cnt} events)", fontsize=11)
    ax.set_xlabel("Relative Hour")
    ax.set_ylabel("Value")
    ax.legend([Patch(facecolor='coral', alpha=0.2),
               plt.Line2D([0],[0], marker='v', color='red', ls='', ms=8),
               plt.Line2D([0],[0], marker='>', color='green', ls='', ms=8),
               plt.Line2D([0],[0], marker='<', color='blue', ls='', ms=8)],
              ['Event span', 'Apex', 'Onset', 'End'],
              loc='upper right', fontsize=8)

# Row 2 left: Duration distribution
ax = fig.add_subplot(gs[2, 0])
durations = [int(e["duration"]) for e in events]
bins = np.arange(min(durations)-0.5, max(durations)+1.5, 1)
counts, _, bars = ax.hist(durations, bins=bins, color='steelblue',
                          edgecolor='navy', alpha=0.8)
ax.axvline(np.mean(durations), color='red', ls='--', lw=1.5,
           label=f'mean={np.mean(durations):.2f}h')
ax.axvline(np.median(durations), color='orange', ls=':', lw=1.5,
           label=f'median={np.median(durations):.1f}h')
ax.set_title("Event Duration Distribution", fontsize=11, fontweight='bold')
ax.set_xlabel("Duration (hours)")
ax.set_ylabel("Count")
ax.legend(fontsize=9)
for bar, c in zip(bars, counts):
    if c > 0:
        ax.text(bar.get_x()+bar.get_width()/2, c+2, f'{int(c)}',
                ha='center', fontsize=8)

# Row 2 right: Peak spacing
ax = fig.add_subplot(gs[2, 1])
spacings = np.diff(sorted([int(e["apex_idx"]) for e in events]))
ax.hist(spacings, bins=30, color='#66BB6A', edgecolor='#2E7D32', alpha=0.8)
ax.axvline(np.mean(spacings), color='red', ls='--', lw=1.5,
           label=f'mean={np.mean(spacings):.1f}h')
ax.axvline(24, color='orange', ls=':', lw=1.5, label='24h (daily)')
ax.set_title("Inter-Peak Spacing Distribution", fontsize=11, fontweight='bold')
ax.set_xlabel("Hours between consecutive peaks")
ax.set_ylabel("Count")
ax.legend(fontsize=9)

# Row 3 left: Events per prediction window (test)
ax = fig.add_subplot(gs[3, 0])
sp_idx = np.where(splits == "test")[0]
g_start, g_end = int(sp_idx[0]), int(sp_idx[-1])
offset = (4 - g_start % 4) % 4
first_aligned = g_start + offset
window_end = g_end - 96 - 96 + 1
onset_arr = np.array([int(e["onset_idx"]) for e in events])
if window_end >= first_aligned:
    starts = np.arange(first_aligned, window_end + 1, 4)
    ev_counts = np.array([int(np.sum((onset_arr >= s+96) & (onset_arr < s+192)))
                          for s in starts])
    unique, cnts = np.unique(ev_counts, return_counts=True)
    brs = ax.bar(unique, cnts, color='#AB47BC', edgecolor='#6A1B9A', alpha=0.8)
    for bar, c in zip(brs, cnts):
        ax.text(bar.get_x()+bar.get_width()/2, c+1, f'{c}',
                ha='center', fontsize=8)
    ax.set_title(f"Events per Prediction Window (test, n={len(starts)})",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("Number of events in 96h window")
    ax.set_ylabel("Count")
    ax.text(0.98, 0.95,
            f'mean={ev_counts.mean():.2f}\nstd={ev_counts.std():.2f}',
            transform=ax.transAxes, fontsize=9, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))

# Row 3 right: Apex intensity
ax = fig.add_subplot(gs[3, 1])
intensities = [float(e["apex_intensity"]) for e in events]
ax.hist(intensities, bins=30, color='#FF7043', edgecolor='#BF360C', alpha=0.8)
ax.axvline(np.mean(intensities), color='blue', ls='--', lw=1.5,
           label=f'mean={np.mean(intensities):.0f}')
ax.set_title("Apex Intensity Distribution", fontsize=11, fontweight='bold')
ax.set_xlabel("Apex Value")
ax.set_ylabel("Count")
ax.legend(fontsize=9)

# Row 4 left: Ablation MAE (3-seed mean + std error bars)
ax = fig.add_subplot(gs[4, 0])
names = [c for c in CONFIGS if c in ABLATION]
labels = ["S0\nFull", "S1\n-MemPos", "S2\n-SA", "S3\nBare", "S4\nCNN"][:len(names)]
ap_mae = [ABLATION[n]["ApMAE"] for n in names]
ap_std = [ABLATION[n]["ApMAE_std"] for n in names]
on_mae = [ABLATION[n]["OnMAE"] for n in names]
on_std = [ABLATION[n]["OnMAE_std"] for n in names]
x = np.arange(len(names))
w = 0.35
b1 = ax.bar(x-w/2, ap_mae, w, yerr=ap_std, capsize=4,
            color='#1976D2', alpha=0.85, label='Apex MAE')
b2 = ax.bar(x+w/2, on_mae, w, yerr=on_std, capsize=4,
            color='#F57C00', alpha=0.85, label='Onset MAE')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("MAE (hours)")
ax.set_title("Ablation: Apex & Onset MAE (3-seed mean, lower=better)",
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.04,
            f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#1976D2')
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.04,
            f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#F57C00')

# Row 4 right: Ablation F1 (3-seed mean + std)
ax = fig.add_subplot(gs[4, 1])
f1s = [ABLATION[n]["F1"] for n in names]
f1_std = [ABLATION[n]["F1_std"] for n in names]
colors = ['#4CAF50' if n == 'S0_full' else '#78909C' for n in names]
brs = ax.bar(x, f1s, 0.6, yerr=f1_std, capsize=5,
             color=colors, edgecolor='#37474F', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Event F1")
ax.set_title("Ablation: Event F1 (3-seed mean, tolerance=3)",
             fontsize=11, fontweight='bold')
ax.set_ylim(0.55, 0.90)
for bar, f1, std in zip(brs, f1s, f1_std):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+std+0.005,
            f'{f1:.3f}', ha='center', fontsize=9, fontweight='bold')

fig.suptitle("ELC Dataset (Electricity) - Event Data & Ablation Analysis\n"
             "lookahead=5, rate_frac=0.020, "
             f"{len(events)} events",
             fontsize=15, fontweight='bold', y=0.995)
plt.savefig("_viz_elc_detailed.png", dpi=150, bbox_inches='tight')
print("Saved: _viz_elc_detailed.png")
plt.close()
