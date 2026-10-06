"""共享绘图函数"""
import os
import visualization.config  # noqa — 确保 matplotlib 后端和字体配置先加载
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from visualization.axis import format_xaxis


PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')


def _ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def plot_price_with_ma(df, title, filename, period_label='',
                       show_mas=None, ma_colors=None):
    """通用价格走势 + 均线图"""
    if show_mas is None:
        show_mas = [5, 30, 120]
    if ma_colors is None:
        ma_colors = ['#ff7f0e', '#2ca02c', '#d62728']

    num_days = len(df)
    width = max(40, num_days * 0.04)
    fig, ax = plt.subplots(figsize=(width, 8))

    ax.plot(df.index, df['close'], label='点位', color='#1f77b4', linewidth=0.8)

    for i, ma in enumerate(show_mas):
        col = f'ma{ma}'
        if col in df.columns:
            ax.plot(df.index, df[col], label=f'MA{ma}',
                    color=ma_colors[i % len(ma_colors)],
                    linestyle='--', alpha=0.6, linewidth=0.7)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("点位", fontsize=12)
    ax.set_xlabel("日期", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=10)
    format_xaxis(ax, df)
    plt.tight_layout()

    _ensure_plot_dir()
    filepath = os.path.join(PLOT_DIR, filename)
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    print(f"   [√] 图表已保存: {filepath}")
    plt.close()


def plot_valuation(df, title, filename, period_label=''):
    """估值分析图（PE-TTM + 分位线）"""
    pe_series = df['pe_ttm'].dropna()
    if len(pe_series) < 2:
        print("   [!] 估值数据不足，跳过估值图表")
        return

    num_days = len(df)
    fig, ax = plt.subplots(figsize=(max(40, num_days * 0.04), 6))

    ax.plot(pe_series.index, pe_series, label='PE-TTM', color='#d62728', linewidth=1)

    stats = pe_series.describe(percentiles=[0.2, 0.5, 0.8])
    ax.axhline(stats['50%'], color='gray', linestyle='--', alpha=0.6, label='中位数', linewidth=1)
    ax.axhline(stats['20%'], color='green', linestyle=':', alpha=0.7, label='低估线(20%)', linewidth=1)
    ax.axhline(stats['80%'], color='red', linestyle=':', alpha=0.7, label='高估线(80%)', linewidth=1)

    ax.fill_between(pe_series.index, stats['20%'], stats['80%'],
                    color='gray', alpha=0.1, label='合理区间')
    ax.fill_between(pe_series.index, pe_series.min(), stats['20%'],
                    color='green', alpha=0.08)
    ax.fill_between(pe_series.index, stats['80%'], pe_series.max(),
                    color='red', alpha=0.08)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("PE (倍)", fontsize=12)
    ax.set_xlabel("日期", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=10)
    format_xaxis(ax, df)
    plt.tight_layout()

    _ensure_plot_dir()
    filepath = os.path.join(PLOT_DIR, filename)
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    print(f"   [√] 估值图表已保存: {filepath}")
    plt.close()


def plot_stock_analysis(df, title, filename, show_mas=None, ma_colors=None):
    """个股走势图"""
    if show_mas is None:
        show_mas = [5, 20, 60]
    if ma_colors is None:
        ma_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(df.index, df['close'], label='收盘价', color='#1f77b4', linewidth=1.5)

    for i, ma in enumerate(show_mas):
        col = f'ma{ma}'
        if col in df.columns:
            ax.plot(df.index, df[col], label=f'MA{ma}',
                    color=ma_colors[i % len(ma_colors)],
                    linestyle='--', alpha=0.7, linewidth=0.8)

    ax.set_title(title, fontsize=13)
    ax.set_ylabel("价格 (元)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()

    _ensure_plot_dir()
    filepath = os.path.join(PLOT_DIR, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    print(f"   [√] 图表已保存: {filepath}")
    plt.close()
