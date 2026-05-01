"""
可视化工具类：VisualizationConfig, EventVisualizer, get_visualizer
供 visualize_anchor_events.py 及其他可视化模块使用。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams


class VisualizationConfig:
    """统一的可视化配置类，管理字体、颜色、样式参数。"""

    def __init__(self):
        self.font_config = {
            'title_fontsize': 13,
            'label_fontsize': 11,
            'tick_fontsize': 9,
            'legend_fontsize': 10,
            'annotation_fontsize': 8,
        }

        self.color_palettes = {
            'line_color': '#2C3E50',
            'peak_color': '#E74C3C',
            'event_colors': [
                '#3498DB', '#E67E22', '#27AE60', '#9B59B6',
                '#F39C12', '#1ABC9C', '#E91E63', '#00BCD4',
            ],
            'background_color': '#BDC3C7',
            'grid_color': '#95A5A6',
            'text_color': '#2C3E50',
            'season_colors': {
                'spring': '#27AE60',
                'summer': '#E74C3C',
                'autumn': '#E67E22',
                'winter': '#3498DB',
            },
        }

        self.plot_styles = {
            'lines.linewidth': 1.5,
            'figure.dpi': 100,
        }

        self._setup_matplotlib()

    def _setup_matplotlib(self):
        """配置 matplotlib 以支持中文字体。"""
        rcParams['axes.unicode_minus'] = False
        # 尝试常见中文字体
        for font in ['SimHei', 'Microsoft YaHei', 'SimSun',
                     'WenQuanYi Micro Hei', 'Arial Unicode MS']:
            try:
                rcParams['font.sans-serif'] = [font] + list(rcParams.get('font.sans-serif', []))
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.set_title('测')
                fig.canvas.draw()
                plt.close(fig)
                break
            except Exception:
                plt.close('all')

    def get_color(self, category, key=None, index=None):
        """
        获取颜色。

        Args:
            category: 'season' | 'original' | 'cmaps'
            key: season名称 or cmap名称
            index: event_colors 索引（category='original' 时使用）
        """
        if category == 'season':
            return self.color_palettes['season_colors'].get(key, '#7F8C8D')
        elif category == 'original':
            colors = self.color_palettes['event_colors']
            if index is not None:
                return colors[index % len(colors)]
            return colors
        elif category == 'cmaps':
            cmaps = {
                'sequential': plt.cm.viridis,
                'sequential2': plt.cm.plasma,
                'qualitative': plt.cm.Set2,
            }
            return cmaps.get(key, plt.cm.viridis)
        return '#7F8C8D'

    def create_figure(self, nrows=1, ncols=1, figsize=(12, 6)):
        """创建图表，返回 (fig, axes_flat)。单子图时直接返回 (fig, ax)。"""
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize,
            facecolor='white',
        )
        fig.subplots_adjust(hspace=0.45, wspace=0.35)
        if nrows == 1 and ncols == 1:
            return fig, axes
        if hasattr(axes, 'flatten'):
            return fig, axes.flatten()
        return fig, axes

    def apply_style(self, ax):
        """应用统一样式：去除顶/右边框，添加网格。"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=self.font_config['tick_fontsize'])
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    def format_time_axis(self, ax, level='day'):
        """格式化时间轴刻度。level: 'hour' | 'day' | 'week' | 'month'"""
        if level == 'hour':
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif level == 'day':
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        elif level == 'week':
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        elif level == 'month':
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=30,
                       labelsize=self.font_config['tick_fontsize'])
        ax.set_xlabel('时间', fontsize=self.font_config['label_fontsize'])

    def add_stat_annotation(self, ax, stats_dict, x=0.02, y=0.98, fontsize=None):
        """在轴左上角添加统计文本框。"""
        text = '\n'.join([f'{k}: {v}' for k, v in stats_dict.items()])
        fs = fontsize or self.font_config['annotation_fontsize']
        ax.text(
            x, y, text, transform=ax.transAxes,
            fontsize=fs, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.85, edgecolor='#CCCCCC'),
            zorder=10,
        )


class EventVisualizer:
    """事件序列可视化辅助类，封装常用绘图操作。"""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def plot_time_series_with_events(
        self, ax, df, events_arr, time_window,
        title='', show_peaks=True, show_event_regions=True,
    ):
        """
        在给定时间窗口内绘制时间序列及事件高亮区域。

        Args:
            ax: matplotlib Axes
            df: 序列DataFrame，含 timestamp, value 列
            events_arr: (N,4) ndarray [onset_idx, end_idx, apex_idx, duration]
            time_window: (start_time, end_time) Timestamp tuple
            title: 子图标题
            show_peaks: 是否标注峰值点
            show_event_regions: 是否高亮事件区域
        """
        start_time, end_time = time_window
        mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
        sub_df = df[mask]

        if len(sub_df) == 0:
            ax.text(0.5, 0.5, '该时间段无数据', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(title, fontsize=self.config.font_config['title_fontsize'])
            return

        # 绘制时间序列主线
        ax.plot(
            sub_df['timestamp'], sub_df['value'],
            color=self.config.color_palettes['line_color'],
            lw=self.config.plot_styles['lines.linewidth'],
            zorder=3, label='负荷曲线',
        )

        # 高亮事件区域
        if show_event_regions and len(events_arr) > 0:
            sub_min_idx = int(sub_df.index[0])
            sub_max_idx = int(sub_df.index[-1])
            win_events = [
                row for row in events_arr
                if int(row[0]) <= sub_max_idx and int(row[1]) >= sub_min_idx
            ]
            for ec, row in enumerate(win_events):
                onset_idx = max(int(row[0]), sub_min_idx)
                end_idx = min(int(row[1]), sub_max_idx)
                if onset_idx not in df.index or end_idx not in df.index:
                    continue
                t_start = df.at[onset_idx, 'timestamp']
                t_end = df.at[end_idx, 'timestamp']
                color = self.config.get_color('original', index=ec)
                ax.axvspan(
                    t_start, t_end, alpha=0.25, color=color, zorder=1,
                    label=f'事件{ec + 1}({int(row[3])}h)',
                )

        # 标注峰值点
        if show_peaks:
            col = 'is_peak_event' if 'is_peak_event' in sub_df.columns else 'is_peak'
            if col in sub_df.columns:
                apexes = sub_df[sub_df[col] == 1]
                if len(apexes) > 0:
                    ax.scatter(
                        apexes['timestamp'], apexes['value'],
                        s=70, color=self.config.color_palettes['peak_color'],
                        zorder=5, label='峰值点', marker='*',
                    )

        ax.set_title(
            title, fontsize=self.config.font_config['title_fontsize'],
            fontweight='bold',
        )
        ax.set_ylabel('负荷 (MW)', fontsize=self.config.font_config['label_fontsize'])
        self.config.format_time_axis(ax, 'day')
        self.config.apply_style(ax)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles[:7], labels[:7],
                fontsize=self.config.font_config['legend_fontsize'] - 1,
                loc='upper right',
            )


def get_visualizer(config: VisualizationConfig) -> EventVisualizer:
    """工厂函数：创建 EventVisualizer 实例。"""
    return EventVisualizer(config)
