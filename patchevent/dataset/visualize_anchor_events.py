"""
可视化 gradient_width 事件数据集（10张图增强版）。
"""

import json
import math
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from .visualize_utils import VisualizationConfig, get_visualizer
except Exception:
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))  # TODO(R-future): migrate to patchevent package import
    from visualize_utils import VisualizationConfig, get_visualizer

config = VisualizationConfig()
visualizer = get_visualizer(config)

_CN_HOLIDAY_PERIODS = [
    ("2021-02-11", "2021-02-17"), ("2022-01-31", "2022-02-06"),
    ("2023-01-21", "2023-01-27"), ("2024-02-10", "2024-02-17"),
    ("2025-01-28", "2025-02-04"), ("2021-05-01", "2021-05-05"),
    ("2022-04-30", "2022-05-04"), ("2023-04-29", "2023-05-03"),
    ("2024-05-01", "2024-05-05"), ("2025-05-01", "2025-05-05"),
    ("2021-10-01", "2021-10-07"), ("2022-10-01", "2022-10-07"),
    ("2023-10-01", "2023-10-07"), ("2024-10-01", "2024-10-07"),
    ("2025-10-01", "2025-10-07"),
]


def _build_holiday_set():
    dates = set()
    for start, end in _CN_HOLIDAY_PERIODS:
        d = pd.Timestamp(start)
        end_d = pd.Timestamp(end)
        while d <= end_d:
            dates.add(d.date())
            d += pd.Timedelta(days=1)
    return dates


_HOLIDAY_DATES = _build_holiday_set()


def classify_day(d):
    date_obj = pd.Timestamp(d).date()
    if date_obj in _HOLIDAY_DATES:
        return "法定假日"
    if pd.Timestamp(d).weekday() >= 5:
        return "周末"
    return "工作日"


def _season_name(ts):
    m = pd.Timestamp(ts).month
    if m in (3, 4, 5):
        return "春季"
    if m in (6, 7, 8):
        return "夏季"
    if m in (9, 10, 11):
        return "秋季"
    return "冬季"


def load_data(data_dir: Path):
    df = pd.read_csv(data_dir / "wlel_event_series_v1.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    events_list = []
    with open(data_dir / "wlel_events_v1.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events_list.append(json.loads(line))

    events_arr = np.array(
        [[e["onset_idx"], e["end_idx"], e["apex_idx"], e["duration"]] for e in events_list],
        dtype=int,
    )
    return df, events_arr, events_list


def build_events_df(df, events_arr, events_list):
    rows = []
    n = len(df)
    for i, row in enumerate(events_arr):
        onset_idx, end_idx, apex_idx, duration = map(int, row)
        if not (0 <= onset_idx < n and 0 <= end_idx < n and 0 <= apex_idx < n):
            continue
        e = events_list[i] if i < len(events_list) else {}
        onset_time = df.iloc[onset_idx]["timestamp"]
        end_time = df.iloc[end_idx]["timestamp"]
        apex_time = df.iloc[apex_idx]["timestamp"]
        apex_value = float(df.iloc[apex_idx]["value"])
        rows.append({
            "event_id": i,
            "onset_idx": onset_idx,
            "end_idx": end_idx,
            "apex_idx": apex_idx,
            "duration": duration,
            "left": apex_idx - onset_idx,
            "right": end_idx - apex_idx,
            "onset_time": onset_time,
            "end_time": end_time,
            "apex_time": apex_time,
            "apex_date": pd.Timestamp(apex_time).date(),
            "onset_hour": pd.Timestamp(onset_time).hour,
            "apex_value": apex_value,
            "apex_intensity": float(e.get("apex_intensity", apex_value)),
            "season": _season_name(apex_time),
            "day_type": classify_day(apex_time),
        })
    return pd.DataFrame(rows)


def compute_event_statistics(events_arr):
    if len(events_arr) == 0:
        return {}
    durs = events_arr[:, 3]
    lefts = events_arr[:, 2] - events_arr[:, 0]
    rights = events_arr[:, 1] - events_arr[:, 2]
    return {
        "total_events": int(len(events_arr)),
        "duration_mean": float(np.mean(durs)),
        "duration_median": float(np.median(durs)),
        "duration_std": float(np.std(durs)),
        "duration_min": float(np.min(durs)),
        "duration_max": float(np.max(durs)),
        "left_mean": float(np.mean(lefts)),
        "right_mean": float(np.mean(rights)),
        "asymmetry_pct": float(100 * np.mean(lefts != rights)),
        "left_right_corr": float(np.corrcoef(lefts, rights)[0, 1]) if len(lefts) > 1 else 0.0,
    }


def _kde_numpy(x, x_grid):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.zeros_like(x_grid, dtype=float)
    std = float(np.std(x))
    if std < 1e-12:
        std = 1.0
    h = max(1.06 * std * (n ** (-1 / 5)), 0.1)
    z = (x_grid[:, None] - x[None, :]) / h
    kern = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    return kern.mean(axis=1) / h


def _pick_season_windows(events_df, days=7):
    windows = []
    half = days // 2
    for season in ["春季", "夏季", "秋季", "冬季"]:
        sub = events_df[events_df["season"] == season]
        if len(sub) == 0:
            continue
        e = sub.sort_values("apex_intensity", ascending=False).iloc[0]
        center = pd.Timestamp(e["apex_time"])
        start = center - pd.Timedelta(days=half)
        end = start + pd.Timedelta(days=days - 1)
        windows.append((start, end, f"{season}代表周（{center:%Y-%m}）"))
    hsub = events_df[events_df["day_type"] == "法定假日"]
    if len(hsub) > 0:
        c = pd.to_datetime(hsub["apex_time"]).dt.date.value_counts().idxmax()
        center = pd.Timestamp(c)
        start = center - pd.Timedelta(days=half)
        end = start + pd.Timedelta(days=days - 1)
        windows.append((start, end, f"节假日代表周（{center:%Y-%m-%d}）"))
    uniq, seen = [], set()
    for s, e, t in windows:
        key = (s.date(), e.date())
        if key not in seen:
            uniq.append((s, e, t))
            seen.add(key)
    return uniq[:5]


def _pick_sample_days(events_df):
    def pick_one(mask, fallback_mask, label):
        sub = events_df[mask]
        if len(sub) == 0:
            sub = events_df[fallback_mask]
        if len(sub) == 0:
            return None
        day = pd.to_datetime(sub["apex_time"]).dt.date.value_counts().idxmax()
        return (str(day), label)

    samples = []
    samples.append(pick_one((events_df["day_type"] == "工作日") & (pd.to_datetime(events_df["apex_time"]).dt.month.isin([6, 7, 8])), (events_df["day_type"] == "工作日"), "工作日（夏季）"))
    samples.append(pick_one((events_df["day_type"] == "周末") & (pd.to_datetime(events_df["apex_time"]).dt.month.isin([9, 10, 11])), (events_df["day_type"] == "周末"), "周末（秋季）"))
    samples.append(pick_one((events_df["day_type"] == "法定假日"), pd.Series(True, index=events_df.index), "法定假日"))
    national = events_df[(pd.to_datetime(events_df["apex_time"]).dt.month == 10) & (pd.to_datetime(events_df["apex_time"]).dt.day.between(1, 7))]
    if len(national) > 0:
        day = pd.to_datetime(national["apex_time"]).dt.date.value_counts().idxmax()
        samples.append((str(day), "国庆窗口"))
    else:
        samples.append(pick_one((events_df["day_type"] == "法定假日"), pd.Series(True, index=events_df.index), "节假日窗口"))
    out, seen = [], set()
    for it in samples:
        if it is None or it[0] in seen:
            continue
        out.append(it)
        seen.add(it[0])
    return out[:4]


def _save_close(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def create_fig1_sample_weeks(df, events_arr, events_df, output_dir):
    sample_windows = _pick_season_windows(events_df, days=7)
    if len(sample_windows) == 0:
        sample_windows = [(df["timestamp"].min(), df["timestamp"].min() + pd.Timedelta(days=6), "默认窗口")]
    fig, axes = config.create_figure(nrows=len(sample_windows), ncols=1, figsize=(16, 3.2 * len(sample_windows)))
    if len(sample_windows) == 1:
        axes = [axes]
    for ax, (start_time, end_time, title) in zip(axes, sample_windows):
        visualizer.plot_time_series_with_events(ax, df, events_arr, (pd.Timestamp(start_time), pd.Timestamp(end_time)), title=title, show_peaks=True, show_event_regions=True)
        mask = (events_df["apex_time"] >= pd.Timestamp(start_time)) & (events_df["apex_time"] <= pd.Timestamp(end_time))
        win = events_df[mask]
        stats = {
            "事件数": int(len(win)),
            "平均持续": f"{win['duration'].mean():.2f}h" if len(win) else "0",
            "平均强度": f"{win['apex_intensity'].mean():.2f}" if len(win) else "0",
        }
        config.add_stat_annotation(ax, stats, x=0.01, y=0.97)
    fig.suptitle("Fig1 典型周时序（四季 + 节假日）", fontsize=14, fontweight="bold")
    out = output_dir / "fig1_sample_weeks.png"
    _save_close(fig, out, dpi=150)
    return out


def create_fig2_duration_distribution(df, events_arr, events_df, output_dir):
    durs = events_arr[:, 3]
    lefts = events_arr[:, 2] - events_arr[:, 0]
    rights = events_arr[:, 1] - events_arr[:, 2]
    fig, axes = config.create_figure(nrows=1, ncols=3, figsize=(17, 5.5))
    ax_a, ax_b, ax_c = axes

    bins = np.arange(durs.min() - 0.5, durs.max() + 1.5, 1)
    n, _, patches = ax_a.hist(durs, bins=bins, color=config.get_color("cmaps", "sequential")(0.62), edgecolor="white", linewidth=0.5, alpha=0.78)
    for patch, cnt in zip(patches, n):
        if cnt > 0:
            ax_a.text(patch.get_x() + patch.get_width() / 2, cnt + 0.4, str(int(cnt)), ha="center", fontsize=8)

    mean_v, med_v = float(np.mean(durs)), float(np.median(durs))
    q25, q75 = float(np.percentile(durs, 25)), float(np.percentile(durs, 75))
    ax_a.axvline(mean_v, ls="--", lw=1.6, color="#E67E22", label=f"均值={mean_v:.2f}")
    ax_a.axvline(med_v, ls="--", lw=1.6, color="#E74C3C", label=f"中位数={med_v:.2f}")
    ax_a.axvline(q25, ls=":", lw=1.2, color="#7F8C8D", label=f"Q1={q25:.1f}")
    ax_a.axvline(q75, ls=":", lw=1.2, color="#7F8C8D", label=f"Q3={q75:.1f}")

    grid = np.linspace(float(durs.min()) - 0.5, float(durs.max()) + 0.5, 300)
    kde = _kde_numpy(durs, grid)
    ax2 = ax_a.twinx()
    ax2.plot(grid, kde, color="#8E44AD", lw=2.0, label="KDE")
    ax2.set_ylabel("密度")
    ax2.tick_params(axis="y", labelsize=8)

    h1, l1 = ax_a.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax_a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax_a.set_title("总持续时间分布 + KDE")
    ax_a.set_xlabel("Duration (h)")
    ax_a.set_ylabel("事件数")
    config.apply_style(ax_a)

    half_bins = np.arange(-0.5, max(lefts.max(), rights.max()) + 1.5, 1)
    ax_b.hist(lefts, bins=half_bins, alpha=0.65, color="#3498DB", label=f"左侧 mean={lefts.mean():.2f}")
    ax_b.hist(rights, bins=half_bins, alpha=0.65, color="#F39C12", label=f"右侧 mean={rights.mean():.2f}")
    ax_b.set_title("左右半宽分布")
    ax_b.set_xlabel("半宽 (h)")
    ax_b.set_ylabel("事件数")
    ax_b.legend(fontsize=9)
    config.apply_style(ax_b)

    lim = max(float(lefts.max()), float(rights.max())) + 0.6
    ax_c.scatter(lefts, rights, s=14, alpha=0.25, color="#16A085")
    ax_c.plot([0, lim], [0, lim], "r--", lw=1.4, label="对称线")
    asym = 100.0 * np.mean(lefts != rights)
    ax_c.text(0.03, 0.97, f"不对称率: {asym:.1f}%\n左>右: {int((lefts > rights).sum())}\n右>左: {int((rights > lefts).sum())}", transform=ax_c.transAxes, va="top", fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax_c.set_xlim(-0.2, lim)
    ax_c.set_ylim(-0.2, lim)
    ax_c.set_title("左右不对称散点")
    ax_c.set_xlabel("左侧半宽 (h)")
    ax_c.set_ylabel("右侧半宽 (h)")
    ax_c.legend(fontsize=9)
    config.apply_style(ax_c)

    fig.suptitle("Fig2 持续时间与不对称统计", fontsize=14, fontweight="bold")
    out = output_dir / "fig2_duration_dist.png"
    _save_close(fig, out, dpi=150)
    return out


def create_fig3_peaks_per_day(df, events_arr, events_df, output_dir):
    peak_col = "is_peak_event" if "is_peak_event" in df.columns else ("is_peak" if "is_peak" in df.columns else None)
    if peak_col is None:
        raise ValueError("数据中缺少 is_peak_event / is_peak 列")
    tmp = df[["timestamp", peak_col]].copy()
    tmp["date"] = tmp["timestamp"].dt.floor("D")
    daily_peak = tmp[tmp[peak_col] == 1].groupby("date").size().sort_index()
    full_days = pd.date_range(tmp["date"].min(), tmp["date"].max(), freq="D")
    daily_peak = daily_peak.reindex(full_days, fill_value=0)

    day_dist = daily_peak.value_counts().sort_index()
    monthly_mean = daily_peak.resample("MS").mean()
    monthly_median = daily_peak.resample("MS").median()

    fig, (ax1, ax2) = config.create_figure(nrows=2, ncols=1, figsize=(15, 8.5))
    colors = [config.get_color("cmaps", "sequential")(i / max(1, len(day_dist) - 1)) for i in range(len(day_dist))]
    bars = ax1.bar(day_dist.index.astype(str), day_dist.values, color=colors, edgecolor="white", alpha=0.85)
    for b, v in zip(bars, day_dist.values):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.6, str(int(v)), ha="center", fontsize=8)
    stats = {"总天数": int(len(daily_peak)), "平均每天": f"{daily_peak.mean():.2f}", "最多一天": int(daily_peak.max()), "零峰值天数": int((daily_peak == 0).sum())}
    ax1.set_title("每天峰值数分布")
    ax1.set_xlabel("单日峰值数量")
    ax1.set_ylabel("天数")
    config.add_stat_annotation(ax1, stats, x=0.01, y=0.97)
    config.apply_style(ax1)

    x = monthly_mean.index
    ax2.bar(x, monthly_mean.values, width=20, color="#5DADE2", alpha=0.85, label="月度日均")
    ax2.plot(x, monthly_mean.rolling(window=3, min_periods=1, center=True).mean(), color="#E74C3C", lw=2.2, marker="o", ms=3, label="3月移动平均")
    ax2.plot(x, monthly_median.values, color="#2E86C1", lw=1.8, ls="--", label="月度中位数")
    ax2.axhline(monthly_mean.mean(), color="#E67E22", ls=":", lw=1.6, label=f"整体均值={monthly_mean.mean():.2f}")
    ax2.set_title("月度日均峰值数（含中位数）")
    ax2.set_ylabel("峰值数")
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(fontsize=9)
    config.apply_style(ax2)

    fig.suptitle("Fig3 日/月峰值密度", fontsize=14, fontweight="bold")
    out = output_dir / "fig3_peaks_per_day.png"
    _save_close(fig, out, dpi=150)
    return out


def create_fig4_single_day_zoom(df, events_arr, events_df, output_dir):
    sample_days = _pick_sample_days(events_df)
    if len(sample_days) == 0:
        sample_days = [(str(df["timestamp"].dt.date.iloc[0]), "默认样本")]

    n, ncols = len(sample_days), 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = config.create_figure(nrows=nrows, ncols=ncols, figsize=(16, 4.5 * nrows))
    axes = list(axes) if hasattr(axes, "__len__") else [axes]

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue
        day, label = sample_days[i]
        day_date = pd.Timestamp(day).date()
        sub_df = df[df["timestamp"].dt.date == day_date]
        if len(sub_df) == 0:
            ax.text(0.5, 0.5, f"无数据: {day}", ha="center", va="center", transform=ax.transAxes)
            continue

        sub_min_idx, sub_max_idx = int(sub_df.index[0]), int(sub_df.index[-1])
        ax.plot(sub_df["timestamp"], sub_df["value"], color=config.color_palettes["line_color"], lw=1.8, label="负荷曲线", zorder=3)

        day_events = events_df[(events_df["onset_idx"] <= sub_max_idx) & (events_df["end_idx"] >= sub_min_idx)].copy()
        for ec, (_, ev) in enumerate(day_events.iterrows()):
            s_idx = max(int(ev["onset_idx"]), sub_min_idx)
            e_idx = min(int(ev["end_idx"]), sub_max_idx)
            t_start = df.iloc[s_idx]["timestamp"]
            t_end = df.iloc[e_idx]["timestamp"]
            ax.axvspan(t_start, t_end, alpha=0.22, color=config.get_color("original", index=ec), zorder=1)

        peak_col = "is_peak_event" if "is_peak_event" in sub_df.columns else ("is_peak" if "is_peak" in sub_df.columns else None)
        if peak_col is not None:
            apexes = sub_df[sub_df[peak_col] == 1]
            ax.scatter(apexes["timestamp"], apexes["value"], s=40, color=config.color_palettes["peak_color"], zorder=5, label="峰值点")
            offsets = [12, 24, 36, 18, 30]
            for j, (_, r) in enumerate(apexes.iterrows()):
                dy = offsets[j % len(offsets)]
                ax.annotate(f"{r['value']:.0f}", xy=(r["timestamp"], r["value"]), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=8, color=config.color_palettes["peak_color"], arrowprops=dict(arrowstyle="->", color=config.color_palettes["peak_color"], lw=0.8, alpha=0.6))

        ax.set_title(f"{day} {label}")
        ax.set_ylabel("负荷 (MW)")
        config.format_time_axis(ax, "hour")
        config.apply_style(ax)
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(h[:4], l[:4], loc="lower right", fontsize=8)

    fig.suptitle("Fig4 单日精细对比（4类样本）", fontsize=14, fontweight="bold")
    out = output_dir / "fig4_single_day_zoom.png"
    _save_close(fig, out, dpi=150)
    return out


def create_fig5_global_coverage(df, events_arr, events_df, output_dir):
    fig, ax = config.create_figure(nrows=1, ncols=1, figsize=(20, 5.5))
    if "phase" in df.columns:
        event_mask = df["phase"] != "background"
        background_mask = ~event_mask
    else:
        event_mask = pd.Series(False, index=df.index)
        for row in events_arr:
            onset, end = int(row[0]), int(row[1])
            event_mask.iloc[max(0, onset): min(len(df), end + 1)] = True
        background_mask = ~event_mask

    ax.fill_between(df["timestamp"], df["value"], where=background_mask.values, color="#D5D8DC", alpha=0.35, label="非事件区")
    ax.fill_between(df["timestamp"], df["value"], where=event_mask.values, color="#F1948A", alpha=0.55, label="事件区")
    ax.plot(df["timestamp"], df["value"], color="#2C3E50", lw=0.45, alpha=0.85, label="负荷时序")

    y_top = float(df["value"].max()) * 1.02
    start_year, end_year = int(df["timestamp"].min().year), int(df["timestamp"].max().year)
    seasonal_md = [(3, 20, "春分", config.get_color("season", "spring")), (6, 21, "夏至", config.get_color("season", "summer")), (9, 22, "秋分", config.get_color("season", "autumn")), (12, 21, "冬至", config.get_color("season", "winter"))]
    for y in range(start_year, end_year + 1):
        for m, d, lbl, c in seasonal_md:
            t = pd.Timestamp(year=y, month=m, day=d)
            if df["timestamp"].min() <= t <= df["timestamp"].max():
                ax.axvline(t, color=c, ls=":", lw=0.8, alpha=0.65)
                ax.text(t, y_top, lbl, rotation=90, va="top", ha="center", color=c, fontsize=7, alpha=0.85)

    if "split" in df.columns:
        for split_label, color in [("val", "#E67E22"), ("test", "#E74C3C")]:
            sub = df[df["split"] == split_label]
            if len(sub) > 0:
                ax.axvline(sub["timestamp"].min(), color=color, ls="--", lw=1.4, alpha=0.85, label=f"{split_label} 起始")

    month_events = events_df.set_index(pd.to_datetime(events_df["apex_time"])).resample("MS").size()
    ax2 = ax.twinx()
    ax2.plot(month_events.index, month_events.values, color="#8E44AD", lw=1.6, alpha=0.75, label="月度事件数")
    ax2.set_ylabel("月度事件数", color="#8E44AD")
    ax2.tick_params(axis="y", labelcolor="#8E44AD")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax.set_xlabel("时间")
    ax.set_ylabel("负荷 (MW)")
    ax.set_title("全局时序覆盖（动态季节线 + 事件密度）")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)
    config.apply_style(ax)

    fig.suptitle("Fig5 全局覆盖图", fontsize=14, fontweight="bold")
    out = output_dir / "fig5_global_coverage.png"
    _save_close(fig, out, dpi=130)
    return out

def create_fig6_daytype_boxplot(df, events_arr, events_df, output_dir):
    all_days = pd.date_range(df["timestamp"].dt.floor("D").min(), df["timestamp"].dt.floor("D").max(), freq="D")
    daily_events = events_df.groupby(pd.to_datetime(events_df["apex_time"]).dt.floor("D")).size()
    daily_events = daily_events.reindex(all_days, fill_value=0)
    daytype = pd.Series([classify_day(d) for d in all_days], index=all_days)
    daily_df = pd.DataFrame({"date": all_days, "count": daily_events.values, "day_type": daytype.values})

    order = ["工作日", "周末", "法定假日"]
    vals = [daily_df[daily_df["day_type"] == k]["count"].values for k in order]
    fig, ax = config.create_figure(nrows=1, ncols=1, figsize=(10, 5.5))
    box = ax.boxplot(vals, labels=order, patch_artist=True, showfliers=True)
    colors = ["#5DADE2", "#F5B041", "#EC7063"]
    for patch, c in zip(box["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    means = [np.mean(v) if len(v) else 0 for v in vals]
    for i, m in enumerate(means, start=1):
        ax.scatter([i], [m], marker="D", color="#2C3E50", s=30, zorder=4)
        ax.text(i + 0.05, m + 0.05, f"{m:.2f}", fontsize=8)
    ax.set_title("工作日/周末/法定假日 事件数对比")
    ax.set_ylabel("单日事件数")
    config.apply_style(ax)
    out = output_dir / "fig6_daytype_boxplot.png"
    _save_close(fig, out, dpi=150)
    return out


def create_fig7_onset_polar(df, events_arr, events_df, output_dir):
    hours = events_df["onset_hour"].astype(int).values
    counts = np.bincount(hours, minlength=24)
    theta = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    width = 2 * np.pi / 24

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    bars = ax.bar(theta, counts, width=width * 0.92, bottom=0.0, alpha=0.8)
    for i, b in enumerate(bars):
        b.set_facecolor(plt.cm.plasma(i / 23 if 23 else 0.5))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(theta)
    ax.set_xticklabels([str(i) for i in range(24)], fontsize=8)
    ax.set_title("事件起始时刻 24h 分布（极坐标）", pad=20)
    out = output_dir / "fig7_onset_polar.png"
    _save_close(fig, out, dpi=150)
    return out


def create_fig8_intensity_duration_hexbin(df, events_arr, events_df, output_dir):
    x = events_df["duration"].values
    y = events_df["apex_intensity"].values
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4], hspace=0.08, wspace=0.08)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    ax_main = fig.add_subplot(gs[1, 0])

    hb = ax_main.hexbin(x, y, gridsize=20, cmap="viridis", mincnt=1)
    cb = fig.colorbar(hb, ax=ax_main)
    cb.set_label("计数")
    ax_top.hist(x, bins=np.arange(x.min() - 0.5, x.max() + 1.5, 1), color="#5DADE2", alpha=0.85)
    ax_right.hist(y, bins=25, orientation="horizontal", color="#F5B041", alpha=0.85)
    ax_main.set_xlabel("Duration (h)")
    ax_main.set_ylabel("Apex Intensity")
    ax_main.set_title("峰值强度 vs 持续时间（Hexbin）")
    ax_top.set_xticklabels([])
    ax_top.set_ylabel("频次")
    ax_right.set_yticklabels([])
    ax_right.set_xlabel("频次")

    m = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 50)
    ax_main.plot(xx, m[0] * xx + m[1], "r--", lw=1.5, label=f"线性趋势 slope={m[0]:.3f}")
    ax_main.legend(fontsize=8)
    out = output_dir / "fig8_intensity_duration_hexbin.png"
    _save_close(fig, out, dpi=150)
    return out


def create_fig9_aligned_events(df, events_arr, events_df, output_dir, window=12):
    aligned = []
    n = len(df)
    for row in events_arr:
        apex = int(row[2])
        if apex - window < 0 or apex + window >= n:
            continue
        seg = df.iloc[apex - window: apex + window + 1]["value"].to_numpy(dtype=float)
        if len(seg) != 2 * window + 1:
            continue
        seg = (seg - seg.min()) / (seg.max() - seg.min() + 1e-8)
        aligned.append(seg)

    aligned = np.array(aligned)
    if len(aligned) == 0:
        raise ValueError("无可对齐事件")

    x = np.arange(-window, window + 1)
    mean = aligned.mean(axis=0)
    std = aligned.std(axis=0)

    fig, ax = config.create_figure(nrows=1, ncols=1, figsize=(11, 5.5))
    max_draw = min(len(aligned), 1000)
    for trace in aligned[:max_draw]:
        ax.plot(x, trace, color="#3498DB", alpha=0.035, lw=0.8)
    ax.plot(x, mean, color="#E74C3C", lw=2.4, label="均值轮廓")
    ax.fill_between(x, mean - std, mean + std, color="#E74C3C", alpha=0.22, label="±1σ")
    ax.axvline(0, color="#2C3E50", ls="--", lw=1.2, label="apex")
    ax.set_xlabel("相对 apex 时间 (h)")
    ax.set_ylabel("归一化负荷")
    ax.set_title(f"事件对齐叠加图（N={len(aligned)}）")
    ax.legend(fontsize=9)
    config.apply_style(ax)
    out = output_dir / "fig9_aligned_events.png"
    _save_close(fig, out, dpi=150)
    return out


def _find_spans(mask, ts):
    spans = []
    mask = np.asarray(mask, dtype=bool)
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if (not v) and start is not None:
            spans.append((ts.iloc[start], ts.iloc[i - 1]))
            start = None
    if start is not None:
        spans.append((ts.iloc[start], ts.iloc[len(mask) - 1]))
    return spans


def create_fig10_split_coverage(df, events_arr, events_df, output_dir):
    if "split" not in df.columns:
        raise ValueError("df 缺少 split 列，无法绘制数据划分图")

    fig, (ax1, ax2) = config.create_figure(nrows=2, ncols=1, figsize=(14, 6.8))
    ts = df["timestamp"].reset_index(drop=True)
    split_styles = {"train": ("#5DADE2", 2), "val": ("#F5B041", 1), "test": ("#EC7063", 0)}
    for split_name, (color, y) in split_styles.items():
        mask = (df["split"].values == split_name)
        spans = _find_spans(mask, ts)
        for s, e in spans:
            ax1.plot([s, e], [y, y], color=color, lw=10, solid_capstyle="butt", alpha=0.9)

    ax1.set_yticks([2, 1, 0])
    ax1.set_yticklabels(["train", "val", "test"])
    ax1.set_title("时间轴上的数据划分覆盖")
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.tick_params(axis="x", rotation=45)
    config.apply_style(ax1)

    counts = df["split"].value_counts().reindex(["train", "val", "test"], fill_value=0)
    ratios = counts / counts.sum() * 100
    bars = ax2.bar(counts.index, counts.values, color=[split_styles[k][0] for k in counts.index], alpha=0.85)
    for b, r in zip(bars, ratios.values):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01, f"{r:.1f}%", ha="center", fontsize=9)
    ax2.set_title("各划分样本占比")
    ax2.set_ylabel("样本点数")
    config.apply_style(ax2)

    out = output_dir / "fig10_split_coverage.png"
    _save_close(fig, out, dpi=150)
    return out


def main(data_dir=None, output_dir=None):
    data_dir = Path("dataset/load_data/hf_load_data/wlel_event_v1_gw/data") if data_dir is None else Path(data_dir)
    output_dir = (data_dir.parent / "plots") if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"数据目录: {data_dir.resolve()}")
    print(f"输出目录: {output_dir.resolve()}")
    print("-" * 60)

    print("加载数据...")
    df, events_arr, events_list = load_data(data_dir)
    events_df = build_events_df(df, events_arr, events_list)

    stats = compute_event_statistics(events_arr)
    print(f"加载完成: {len(df)} 个数据点, {len(events_arr)} 个事件")
    if stats:
        print(f"事件统计: 平均持续时间 {stats['duration_mean']:.2f}h, 不对称率 {stats['asymmetry_pct']:.2f}%")

    plot_jobs = [
        ("fig1_sample_weeks", create_fig1_sample_weeks),
        ("fig2_duration_dist", create_fig2_duration_distribution),
        ("fig3_peaks_per_day", create_fig3_peaks_per_day),
        ("fig4_single_day_zoom", create_fig4_single_day_zoom),
        ("fig5_global_coverage", create_fig5_global_coverage),
        ("fig6_daytype_boxplot", create_fig6_daytype_boxplot),
        ("fig7_onset_polar", create_fig7_onset_polar),
        ("fig8_intensity_duration_hexbin", create_fig8_intensity_duration_hexbin),
        ("fig9_aligned_events", create_fig9_aligned_events),
        ("fig10_split_coverage", create_fig10_split_coverage),
    ]

    manifest = []
    print("\n开始生成 10 张图...")
    for i, (name, fn) in enumerate(plot_jobs, start=1):
        try:
            print(f"[{i:02d}/10] {name} ...")
            out_path = fn(df, events_arr, events_df, output_dir)
            manifest.append({"plot": name, "status": "ok", "file": str(out_path), "error": ""})
            print(f"  [OK] {Path(out_path).name}")
        except Exception as e:
            manifest.append({"plot": name, "status": "failed", "file": "", "error": f"{type(e).__name__}: {e}"})
            print(f"  [FAILED] {name}: {e}")
            print("  " + traceback.format_exc().splitlines()[-1])

    pd.DataFrame([stats] if stats else []).to_csv(output_dir / "event_statistics.csv", index=False)
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(output_dir / "plot_manifest.csv", index=False)

    ok_count = int((manifest_df["status"] == "ok").sum()) if len(manifest_df) else 0
    print("\n" + "=" * 60)
    print(f"图表完成: {ok_count}/10")
    print(f"统计文件: {output_dir / 'event_statistics.csv'}")
    print(f"清单文件: {output_dir / 'plot_manifest.csv'}")
    print(f"输出目录: {output_dir.resolve()}")
    return df, events_arr, stats, manifest_df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        main()
