"""
诊断三域peak检测质量差异：
1. peak密度对比
2. peak是否真正位于局部最大值
3. peak间距分布
4. 可视化peak位置准确性
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASETS = {
    "wlel": {
        "series": "dataset/wlel/event_v1/data/wlel_event_series_v1.csv",
        "events": "dataset/wlel/event_v1/data/wlel_events_v1.jsonl",
        "intermediate": "dataset/wlel/intermediate/hf_load_data_20210101-20251127_mixed_with_peaks_lookahead_3.csv",
        "lookahead": 3,
    },
    "ett": {
        "series": "dataset/ett/event_v1/data/ett_event_series_v1.csv",
        "events": "dataset/ett/event_v1/data/ett_events_v1.jsonl",
        "intermediate": "dataset/ett/intermediate/ETTh2_mixed_with_peaks_lookahead_5.csv",
        "lookahead": 5,
    },
    "elc": {
        "series": "dataset/electricity/event_v1/data/elc_event_series_v1.csv",
        "events": "dataset/electricity/event_v1/data/elc_events_v1.jsonl",
        "intermediate": "dataset/electricity/intermediate/electricity_mixed_with_peaks_lookahead_5.csv",
        "lookahead": 5,
    },
}


def load_events(path):
    events = []
    with open(path) as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def check_local_max(values, peak_idx, window=3):
    """检查peak_idx是否是[-window, +window]范围内的局部最大值"""
    n = len(values)
    lo = max(0, peak_idx - window)
    hi = min(n - 1, peak_idx + window)
    local_max_idx = lo + np.argmax(values[lo:hi+1])
    return local_max_idx == peak_idx, int(local_max_idx), abs(local_max_idx - peak_idx)


print("=" * 90)
print("三域 Peak 检测质量诊断")
print("=" * 90)

fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle("Peak Detection Quality Diagnosis", fontsize=14, fontweight='bold')

for row, (domain, cfg) in enumerate(DATASETS.items()):
    series = pd.read_csv(cfg["series"], parse_dates=["timestamp"])
    vals = series["value"].values
    is_peak = series["is_peak"].values
    events = load_events(cfg["events"])

    peak_indices = np.where(is_peak == 1)[0]
    n_peaks = len(peak_indices)
    n_points = len(vals)

    # 1. peak密度
    peak_density = n_peaks / n_points * 1000
    
    # 2. peak间距
    if len(peak_indices) > 1:
        spacings = np.diff(peak_indices)
        spacing_mean = np.mean(spacings)
        spacing_std = np.std(spacings)
        spacing_min = np.min(spacings)
        spacing_max = np.max(spacings)
    else:
        spacing_mean = spacing_std = spacing_min = spacing_max = 0

    # 3. 检查peak是否真正是局部最大值
    offsets = []
    is_true_max_3 = 0
    is_true_max_5 = 0
    is_true_max_1 = 0
    for pi in peak_indices:
        ok1, _, off1 = check_local_max(vals, pi, window=1)
        ok3, _, off3 = check_local_max(vals, pi, window=3)
        ok5, real_idx, off5 = check_local_max(vals, pi, window=5)
        if ok1:
            is_true_max_1 += 1
        if ok3:
            is_true_max_3 += 1
        if ok5:
            is_true_max_5 += 1
        offsets.append(off5)

    pct_true_1 = is_true_max_1 / n_peaks * 100
    pct_true_3 = is_true_max_3 / n_peaks * 100
    pct_true_5 = is_true_max_5 / n_peaks * 100

    # 4. peak prominence (与周围的高度差)
    prominences = []
    for pi in peak_indices:
        lo = max(0, pi - 12)
        hi = min(n_points, pi + 12)
        local_min = np.min(vals[lo:hi])
        prominences.append(vals[pi] - local_min)

    print(f"\n{'='*60}")
    print(f"  {domain.upper()} (lookahead={cfg['lookahead']})")
    print(f"{'='*60}")
    print(f"  Total points: {n_points}, Peaks: {n_peaks}, Density: {peak_density:.1f}/1k")
    print(f"  Peak spacing: mean={spacing_mean:.1f}h, std={spacing_std:.1f}, "
          f"min={spacing_min}, max={spacing_max}")
    print(f"  Is TRUE local max (w=1): {pct_true_1:.1f}%")
    print(f"  Is TRUE local max (w=3): {pct_true_3:.1f}%")
    print(f"  Is TRUE local max (w=5): {pct_true_5:.1f}%")
    print(f"  Peak-to-true-max offset: mean={np.mean(offsets):.2f}, "
          f"max={np.max(offsets)}, >0: {sum(1 for o in offsets if o > 0)}/{n_peaks} "
          f"({sum(1 for o in offsets if o > 0)/n_peaks*100:.1f}%)")
    print(f"  Prominence: mean={np.mean(prominences):.2f}, "
          f"p10={np.percentile(prominences, 10):.2f}, p50={np.percentile(prominences, 50):.2f}")

    # --- Plot 1: 时序+peaks示例 ---
    ax = axes[row, 0]
    mid = int(n_points * 0.35)
    win = 200
    s, e = mid, mid + win
    ax.plot(range(win), vals[s:e], 'k-', linewidth=0.8)
    for pi in peak_indices:
        if s <= pi < e:
            # 检查是否是真正的局部最大值
            ok, real, off = check_local_max(vals, pi, window=3)
            color = 'green' if ok else 'red'
            ax.plot(pi - s, vals[pi], 'v', color=color, markersize=8)
            if not ok and s <= real < e:
                ax.plot(real - s, vals[real], '^', color='blue', markersize=6)
                ax.annotate('', xy=(real-s, vals[real]), xytext=(pi-s, vals[pi]),
                           arrowprops=dict(arrowstyle='->', color='red', lw=1))
    ax.set_title(f"{domain.upper()} peaks (green=correct, red=offset)", fontsize=10)
    ax.set_ylabel("Value")

    # --- Plot 2: peak间距分布 ---
    ax = axes[row, 1]
    if len(spacings) > 0:
        ax.hist(spacings, bins=30, color='steelblue', edgecolor='navy', alpha=0.7)
        ax.axvline(spacing_mean, color='red', linestyle='--', label=f'mean={spacing_mean:.1f}h')
        ax.axvline(24, color='orange', linestyle=':', label='24h (daily)')
        ax.legend(fontsize=8)
    ax.set_title(f"{domain.upper()} peak spacing distribution", fontsize=10)
    ax.set_xlabel("Hours between peaks")

    # --- Plot 3: prominence分布 ---
    ax = axes[row, 2]
    ax.hist(prominences, bins=30, color='coral', edgecolor='darkred', alpha=0.7)
    ax.axvline(np.mean(prominences), color='blue', linestyle='--', 
               label=f'mean={np.mean(prominences):.1f}')
    ax.axvline(np.percentile(prominences, 10), color='gray', linestyle=':', 
               label=f'p10={np.percentile(prominences, 10):.1f}')
    ax.legend(fontsize=8)
    ax.set_title(f"{domain.upper()} peak prominence", fontsize=10)
    ax.set_xlabel("Prominence (peak - local min)")

plt.tight_layout()
plt.savefig("_diagnose_peaks.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: _diagnose_peaks.png")
plt.close()
