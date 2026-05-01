"""
用 lookahead=3 重新检测 ETT/ELC 的peaks, 对比 lookahead=5 的差异
"""
import os, sys
import numpy as np
import pandas as pd

# 动态导入peak检测函数
import importlib.util
detect_peak_file = r"d:\ywz_experiment_papers\ywz_7_electric\Time-Series-Library\peak_detect_example\detect_peak.py"
spec = importlib.util.spec_from_file_location("detect_peak", detect_peak_file)
detect_peak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detect_peak)
find_peaks_findpeaks = detect_peak.find_peaks_findpeaks


DATASETS = {
    "ett": {
        "raw_csv": "d:/ywz_experiment_papers/ywz_7_electric/Time-Series-Library/dataset/ETT-small/ETTh2.csv",
        "value_col": "OT",
        "out_dir": "dataset/ett/intermediate",
    },
    "elc": {
        "raw_csv": "d:/ywz_experiment_papers/ywz_7_electric/Time-Series-Library/dataset/electricity/electricity.csv",
        "value_col": "OT",
        "out_dir": "dataset/electricity/intermediate",
    },
}


for domain, cfg in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"  {domain.upper()}: Peak detection comparison")
    print(f"{'='*60}")

    df = pd.read_csv(cfg["raw_csv"])
    values = df[cfg["value_col"]].values

    for la in [5, 3]:
        peaks, _ = find_peaks_findpeaks(values, method='peakdetect', lookahead=la)
        n_peaks = len(peaks)
        if n_peaks > 1:
            spacings = np.diff(sorted(peaks))
            sp_mean = np.mean(spacings)
        else:
            sp_mean = 0

        # 每96h窗口平均event数
        n = len(values)
        window_counts = []
        for start in range(0, n - 96, 4):
            count = sum(1 for p in peaks if start <= p < start + 96)
            window_counts.append(count)
        mean_per_window = np.mean(window_counts) if window_counts else 0

        print(f"  lookahead={la}: peaks={n_peaks}, density={n_peaks/n*1000:.1f}/1k, "
              f"spacing_mean={sp_mean:.1f}h, events/window={mean_per_window:.1f}")

    # 生成 lookahead=3 和 lookahead=4 版本的 intermediate CSV
    for target_la in [3, 4]:
        peaks_target, _ = find_peaks_findpeaks(values, method='peakdetect', lookahead=target_la)

    df_out = pd.DataFrame()
    df_out['date_60min'] = pd.to_datetime(df['date'])
    df_out['value_60min'] = df[cfg['value_col']]
    df_out['date_max'] = df_out['date_60min']
    df_out['value_max'] = df_out['value_60min']
    df_out['is_peak'] = 0
    df_out.loc[peaks_target, 'is_peak'] = 1

    base_name = os.path.splitext(os.path.basename(cfg["raw_csv"]))[0]
    out_path = os.path.join(cfg["out_dir"], f"{base_name}_mixed_with_peaks_lookahead_3.csv")
    df_out.to_csv(out_path, index=False)
    print(f"  Saved lookahead=3 intermediate: {out_path}")
    print(f"  Peaks: {int(df_out['is_peak'].sum())}")
