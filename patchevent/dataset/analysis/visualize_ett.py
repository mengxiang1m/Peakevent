"""ETT Dataset Visualization: data overview, ablation results, error analysis"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import os

plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

OUT = Path('phase2_small_decoder/plots_ett')
OUT.mkdir(exist_ok=True)

# ============================================================
# Fig 1: ETT Time Series Overview with Events
# ============================================================
print('=== Fig 1: Time Series Overview ===')
df = pd.read_csv('dataset/ETT-small/ett_event_v1/data/ett_event_series_v1.csv', parse_dates=['timestamp'])
events = []
with open('dataset/ETT-small/ett_event_v1/data/ett_events_v1.jsonl') as f:
    for line in f:
        events.append(json.loads(line))

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=False)

# Full series
ax = axes[0]
ax.plot(df['timestamp'], df['value'], linewidth=0.5, color='steelblue', alpha=0.8)
# Mark train/val/test splits
for split, color in [('train', '#2196F3'), ('val', '#FF9800'), ('test', '#F44336')]:
    mask = df['split'] == split
    if mask.any():
        start = df.loc[mask, 'timestamp'].iloc[0]
        end = df.loc[mask, 'timestamp'].iloc[-1]
        ax.axvspan(start, end, alpha=0.08, color=color, label=split)
ax.set_ylabel('Temperature (°C)')
ax.set_title('ETT Time Series - Full Overview')
ax.legend(loc='upper right', fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Zoom: 2 weeks of train data with events
ax = axes[1]
zoom_start = pd.Timestamp('2016-07-01')
zoom_end = pd.Timestamp('2016-07-15')
mask_zoom = (df['timestamp'] >= zoom_start) & (df['timestamp'] <= zoom_end)
ax.plot(df.loc[mask_zoom, 'timestamp'], df.loc[mask_zoom, 'value'], linewidth=1, color='steelblue')
for ev in events:
    onset_t = pd.Timestamp(ev['onset_time'])
    apex_t = pd.Timestamp(ev['apex_time'])
    if zoom_start <= onset_t <= zoom_end:
        ax.axvline(onset_t, color='red', alpha=0.3, linewidth=0.8)
        ax.plot(apex_t, ev['apex_intensity'], 'v', color='red', markersize=6, alpha=0.7)
ax.set_ylabel('Temperature (°C)')
ax.set_title('Zoomed View (2 weeks) - Red markers = events')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# Zoom: 4 days with detailed event annotations
ax = axes[2]
zoom_start2 = pd.Timestamp('2016-07-05')
zoom_end2 = pd.Timestamp('2016-07-09')
mask_zoom2 = (df['timestamp'] >= zoom_start2) & (df['timestamp'] <= zoom_end2)
ax.plot(df.loc[mask_zoom2, 'timestamp'], df.loc[mask_zoom2, 'value'], linewidth=1.2, color='steelblue', marker='o', markersize=2)
for ev in events:
    onset_t = pd.Timestamp(ev['onset_time'])
    apex_t = pd.Timestamp(ev['apex_time'])
    if zoom_start2 <= onset_t <= zoom_end2:
        end_t = onset_t + pd.Timedelta(hours=ev['duration'])
        ax.axvspan(onset_t, end_t, alpha=0.15, color='red')
        ax.plot(apex_t, ev['apex_intensity'], 'v', color='red', markersize=8)
        ax.annotate(f'E{ev["event_id"]}', (apex_t, ev['apex_intensity']),
                   textcoords="offset points", xytext=(5, 8), fontsize=8, color='red')
ax.set_ylabel('Temperature (°C)')
ax.set_xlabel('Time')
ax.set_title('Detailed View (4 days) - Event spans highlighted')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

plt.tight_layout()
plt.savefig(OUT / 'fig1_ett_overview.png')
plt.close()
print(f'  Saved: {OUT}/fig1_ett_overview.png')

# ============================================================
# Fig 2: Event Statistics
# ============================================================
print('=== Fig 2: Event Statistics ===')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2a: Event duration distribution
durations = [ev['duration'] for ev in events]
ax = axes[0, 0]
ax.hist(durations, bins=range(1, max(durations)+2), color='steelblue', edgecolor='white', alpha=0.8, align='left')
ax.set_xlabel('Duration (hours)')
ax.set_ylabel('Count')
ax.set_title(f'Event Duration Distribution (n={len(events)})')
ax.axvline(np.mean(durations), color='red', linestyle='--', label=f'mean={np.mean(durations):.1f}h')
ax.legend()

# 2b: Apex intensity distribution
intensities = [ev['apex_intensity'] for ev in events]
ax = axes[0, 1]
ax.hist(intensities, bins=30, color='coral', edgecolor='white', alpha=0.8)
ax.set_xlabel('Apex Intensity (°C)')
ax.set_ylabel('Count')
ax.set_title('Apex Intensity Distribution')
ax.axvline(np.mean(intensities), color='red', linestyle='--', label=f'mean={np.mean(intensities):.1f}°C')
ax.legend()

# 2c: Events per window (from eval outputs)
eval_path = 'phase2_small_decoder/checkpoints_ett/ablation/E4_full_s42/test_eval/eval_outputs.jsonl'
if os.path.exists(eval_path):
    outputs = []
    with open(eval_path) as f:
        for line in f:
            outputs.append(json.loads(line))
    gt_counts = [len(o.get('gt_events', [])) for o in outputs]
    pred_counts = [len(o.get('pred_events', [])) for o in outputs]
    
    ax = axes[1, 0]
    bins = range(0, max(max(gt_counts), max(pred_counts)) + 2)
    ax.hist(gt_counts, bins=bins, alpha=0.6, color='steelblue', label='Ground Truth', align='left', edgecolor='white')
    ax.hist(pred_counts, bins=bins, alpha=0.6, color='coral', label='Predicted', align='left', edgecolor='white')
    ax.set_xlabel('Events per Window')
    ax.set_ylabel('Count')
    ax.set_title('Event Count Distribution (Test Set)')
    ax.legend()

# 2d: Train/Val/Test split sizes
ax = axes[1, 1]
split_counts = df['split'].value_counts()
colors = ['#2196F3', '#FF9800', '#F44336']
bars = ax.bar(split_counts.index, split_counts.values, color=colors, edgecolor='white')
for bar, val in zip(bars, split_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            str(val), ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Samples (hours)')
ax.set_title('Dataset Split Sizes')

plt.tight_layout()
plt.savefig(OUT / 'fig2_ett_statistics.png')
plt.close()
print(f'  Saved: {OUT}/fig2_ett_statistics.png')

# ============================================================
# Fig 3: Ablation Results Comparison
# ============================================================
print('=== Fig 3: Ablation Results ===')
ablation_data = {
    'E0\nBaseline': {'F1': 0.877, 'F1_std': 0.005, 'OnsetMAE': 0.888, 'ApexMAE': 0.630},
    'E1\nSA+MemPos': {'F1': 0.895, 'F1_std': 0.003, 'OnsetMAE': 0.882, 'ApexMAE': 0.624},
    'E2\nSA+Unf': {'F1': 0.892, 'F1_std': 0.003, 'OnsetMAE': 0.881, 'ApexMAE': 0.621},
    'E3\nMemPos+Unf': {'F1': 0.895, 'F1_std': 0.002, 'OnsetMAE': 0.884, 'ApexMAE': 0.625},
    'E4\nFull': {'F1': 0.895, 'F1_std': 0.003, 'OnsetMAE': 0.884, 'ApexMAE': 0.620},
    'E4+\nxPatch': {'F1': 0.897, 'F1_std': 0.001, 'OnsetMAE': 0.882, 'ApexMAE': 0.624},
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
names = list(ablation_data.keys())
x = np.arange(len(names))

# F1
ax = axes[0]
f1s = [ablation_data[n]['F1'] for n in names]
f1_stds = [ablation_data[n]['F1_std'] for n in names]
colors_f1 = ['#90CAF9'] * 4 + ['#1565C0', '#FF7043']
bars = ax.bar(x, f1s, yerr=f1_stds, capsize=3, color=colors_f1, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('Event F1')
ax.set_title('Event F1 (↑ better)')
ax.set_ylim(0.86, 0.91)
for i, (bar, v) in enumerate(zip(bars, f1s)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + f1_stds[i] + 0.001,
            f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

# OnsetMAE
ax = axes[1]
onsets = [ablation_data[n]['OnsetMAE'] for n in names]
colors_mae = ['#FFCC80'] * 4 + ['#E65100', '#FF7043']
bars = ax.bar(x, onsets, color=colors_mae, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('Onset MAE (hours)')
ax.set_title('Onset MAE (↓ better)')
ax.set_ylim(0.86, 0.90)
for bar, v in zip(bars, onsets):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

# ApexMAE
ax = axes[2]
apexes = [ablation_data[n]['ApexMAE'] for n in names]
colors_apex = ['#C8E6C9'] * 4 + ['#2E7D32', '#FF7043']
bars = ax.bar(x, apexes, color=colors_apex, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('Apex MAE (hours)')
ax.set_title('Apex MAE (↓ better)')
ax.set_ylim(0.61, 0.64)
for bar, v in zip(bars, apexes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

plt.suptitle('ETT Ablation Results (3-seed average)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'fig3_ett_ablation.png')
plt.close()
print(f'  Saved: {OUT}/fig3_ett_ablation.png')

# ============================================================
# Fig 4: Error Analysis - Count Prediction Problem
# ============================================================
print('=== Fig 4: Count Prediction Analysis ===')
if os.path.exists(eval_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    gt_counts_arr = np.array(gt_counts)
    pred_counts_arr = np.array(pred_counts)
    
    # 4a: GT vs Pred count scatter/confusion
    ax = axes[0, 0]
    from collections import Counter
    pairs = list(zip(gt_counts, pred_counts))
    pair_counts = Counter(pairs)
    for (g, p), cnt in pair_counts.items():
        ax.scatter(g, p, s=cnt*3, color='steelblue', alpha=0.7)
        if cnt > 5:
            ax.annotate(str(cnt), (g, p), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.plot([1, 7], [1, 7], 'k--', alpha=0.3)
    ax.set_xlabel('GT Event Count')
    ax.set_ylabel('Predicted Event Count')
    ax.set_title('GT vs Predicted Event Count')
    ax.set_xlim(1.5, 6.5)
    ax.set_ylim(2.5, 5.5)
    
    # 4b: F1 by GT event count
    ax = axes[0, 1]
    gt_unique = sorted(set(gt_counts))
    f1_by_count = []
    for n in gt_unique:
        mask = gt_counts_arr == n
        total_gt = gt_counts_arr[mask].sum()
        fn_sum = sum(o.get('fn', 0) for o, g in zip(outputs, gt_counts) if g == n)
        fp_sum = sum(o.get('fp', 0) for o, g in zip(outputs, gt_counts) if g == n)
        tp = total_gt - fn_sum
        p = tp / (tp + fp_sum) if (tp + fp_sum) > 0 else 0
        r = tp / (tp + fn_sum) if (tp + fn_sum) > 0 else 0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
        f1_by_count.append(f1)
    
    colors_count = ['#F44336' if abs(n-4) > 0 else '#4CAF50' for n in gt_unique]
    bars = ax.bar(gt_unique, f1_by_count, color=colors_count, edgecolor='white', alpha=0.8)
    for bar, v, n in zip(bars, f1_by_count, gt_unique):
        samples = (gt_counts_arr == n).sum()
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}\n(n={samples})', ha='center', fontsize=8)
    ax.set_xlabel('GT Event Count')
    ax.set_ylabel('Event F1')
    ax.set_title('F1 by Ground Truth Event Count')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.895, color='gray', linestyle='--', alpha=0.5, label='Overall F1')
    ax.legend()
    
    # 4c: Onset error distribution - compute by greedy matching
    onset_errors = []
    apex_errors = []
    for o in outputs:
        gts = sorted(o.get('gt_events', []), key=lambda e: e['onset_idx'])
        preds = sorted(o.get('pred_events', []), key=lambda e: e['onset_idx'])
        used = set()
        for g in gts:
            best_j, best_d = -1, 999
            for j, p in enumerate(preds):
                if j in used:
                    continue
                d = abs(g['onset_idx'] - p['onset_idx'])
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0 and best_d <= 6:
                used.add(best_j)
                p = preds[best_j]
                onset_errors.append(abs(g['onset_idx'] - p['onset_idx']))
                apex_errors.append(abs(g['apex_idx'] - p['apex_idx']))
    
    ax = axes[1, 0]
    if onset_errors:
        ax.hist(onset_errors, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(np.mean(onset_errors), color='red', linestyle='--',
                   label=f'mean={np.mean(onset_errors):.3f}h')
        ax.axvline(np.median(onset_errors), color='orange', linestyle='--',
                   label=f'median={np.median(onset_errors):.3f}h')
        ax.set_xlabel('|Onset Error| (hours)')
        ax.set_ylabel('Count')
        ax.set_title('Onset Error Distribution (matched events)')
        ax.legend()
    
    # 4d: Apex error distribution
    ax = axes[1, 1]
    if apex_errors:
        ax.hist(apex_errors, bins=50, color='coral', edgecolor='white', alpha=0.8)
        ax.axvline(np.mean(apex_errors), color='red', linestyle='--',
                   label=f'mean={np.mean(apex_errors):.3f}h')
        ax.axvline(np.median(apex_errors), color='orange', linestyle='--',
                   label=f'median={np.median(apex_errors):.3f}h')
        ax.set_xlabel('|Apex Error| (hours)')
        ax.set_ylabel('Count')
        ax.set_title('Apex Error Distribution (matched events)')
        ax.legend()
    
    plt.suptitle('ETT Error Analysis - Event Count Prediction Problem', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT / 'fig4_ett_error_analysis.png')
    plt.close()
    print(f'  Saved: {OUT}/fig4_ett_error_analysis.png')

# ============================================================
# Fig 5: Cross-dataset Comparison (WLEL vs ETT)
# ============================================================
print('=== Fig 5: Cross-dataset Comparison ===')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# WLEL results (from memory)
wlel = {'E0': 0.835, 'E1': 0.842, 'E2': 0.826, 'E3': 0.841, 'E4': 0.846}
ett = {'E0': 0.877, 'E1': 0.895, 'E2': 0.892, 'E3': 0.895, 'E4': 0.895}

configs = ['E0', 'E1', 'E2', 'E3', 'E4']
x = np.arange(len(configs))
w = 0.35

ax = axes[0]
bars1 = ax.bar(x - w/2, [wlel[c] for c in configs], w, label='WLEL', color='#1976D2', alpha=0.8)
bars2 = ax.bar(x + w/2, [ett[c] for c in configs], w, label='ETT', color='#FF7043', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylabel('Event F1')
ax.set_title('F1 Comparison: WLEL vs ETT')
ax.legend()
ax.set_ylim(0.80, 0.92)
for bar, v in zip(bars1, [wlel[c] for c in configs]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{v:.3f}', ha='center', fontsize=7)
for bar, v in zip(bars2, [ett[c] for c in configs]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{v:.3f}', ha='center', fontsize=7)

# Module contribution comparison
ax = axes[1]
wlel_contrib = {
    'MemPos': wlel['E4'] - wlel['E2'],
    'SA': wlel['E4'] - wlel['E3'],
    'Unfreeze': wlel['E4'] - wlel['E1'],
}
ett_contrib = {
    'MemPos': ett['E4'] - ett['E2'],
    'SA': ett['E4'] - ett['E3'],
    'Unfreeze': ett['E4'] - ett['E1'],
}
modules = ['MemPos', 'SA', 'Unfreeze']
x2 = np.arange(len(modules))
bars1 = ax.bar(x2 - w/2, [wlel_contrib[m]*100 for m in modules], w, label='WLEL', color='#1976D2', alpha=0.8)
bars2 = ax.bar(x2 + w/2, [ett_contrib[m]*100 for m in modules], w, label='ETT', color='#FF7043', alpha=0.8)
ax.set_xticks(x2)
ax.set_xticklabels(modules)
ax.set_ylabel('ΔF1 (%)')
ax.set_title('Module Contribution (ΔF1 from removal)')
ax.legend()
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
for bar, v in zip(bars1, [wlel_contrib[m]*100 for m in modules]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{v:.1f}%', ha='center', fontsize=8)
for bar, v in zip(bars2, [ett_contrib[m]*100 for m in modules]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{v:.1f}%', ha='center', fontsize=8)

plt.suptitle('Cross-Dataset Comparison: WLEL vs ETT', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'fig5_cross_dataset.png')
plt.close()
print(f'  Saved: {OUT}/fig5_cross_dataset.png')

# ============================================================
# Fig 6: Sample Predictions Visualization
# ============================================================
print('=== Fig 6: Sample Predictions ===')
if os.path.exists(eval_path):
    # Pick samples with different GT counts
    samples_by_count = {}
    for i, o in enumerate(outputs):
        n = len(o.get('gt_events', []))
        if n not in samples_by_count:
            samples_by_count[n] = i
    
    sample_indices = []
    for n in [2, 3, 4, 5]:
        if n in samples_by_count:
            sample_indices.append(samples_by_count[n])
    
    if len(sample_indices) >= 4:
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
        for ax_idx, si in enumerate(sample_indices[:4]):
            o = outputs[si]
            gt_evts = o.get('gt_events', [])
            pred_evts = o.get('pred_events', [])
            pred_start = o.get('pred_start', 0)
            
            ax = axes[ax_idx]
            # Show timeline
            timeline = np.arange(96)
            ax.plot(timeline, np.zeros(96), 'k-', alpha=0.1)
            
            # GT events (use relative position)
            for j, ev in enumerate(gt_evts):
                onset_rel = ev.get('onset_idx', 0) - pred_start
                dur = ev.get('duration', 1)
                ax.barh(0.3, dur, left=onset_rel, height=0.2, color='steelblue', alpha=0.6,
                       edgecolor='steelblue')
                ax.text(onset_rel + dur/2, 0.45, f'GT{j+1}', ha='center', fontsize=7, color='steelblue')
            
            # Pred events (use relative position)
            for j, ev in enumerate(pred_evts):
                onset_rel = ev.get('onset_idx', 0) - pred_start
                dur = ev.get('duration', 1)
                ax.barh(-0.3, dur, left=onset_rel, height=0.2, color='coral', alpha=0.6,
                       edgecolor='coral')
                ax.text(onset_rel + dur/2, -0.15, f'P{j+1}', ha='center', fontsize=7, color='coral')
            
            n_gt = len(gt_evts)
            n_pred = len(pred_evts)
            status = 'match' if n_gt == n_pred else ('over' if n_pred > n_gt else 'under')
            ax.set_title(f'Sample {si}: GT={n_gt} events, Pred={n_pred} events ({status})',
                        fontsize=10)
            ax.set_xlim(-2, 98)
            ax.set_ylim(-0.7, 0.7)
            ax.set_ylabel('GT / Pred', fontsize=8)
            ax.set_yticks([])
            if ax_idx == 3:
                ax.set_xlabel('Time Step (hours)')
        
        plt.suptitle('ETT Sample Predictions: GT (blue) vs Predicted (coral)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUT / 'fig6_sample_predictions.png')
        plt.close()
        print(f'  Saved: {OUT}/fig6_sample_predictions.png')

print(f'\n=== All figures saved to {OUT}/ ===')
