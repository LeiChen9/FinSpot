"""X 轴时间刻度格式化工具"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def format_xaxis(ax, df_plot):
    """根据数据跨度设置刻度密度"""
    num_days = len(df_plot)

    if num_days <= 30:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        minor_locator = mdates.DayLocator()
    elif num_days <= 90:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        minor_locator = mdates.DayLocator(interval=5)
    elif num_days <= 250:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        minor_locator = mdates.WeekdayLocator()
    elif num_days <= 750:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        minor_locator = mdates.WeekdayLocator(interval=4)
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        minor_locator = mdates.MonthLocator()

    ax.xaxis.set_minor_locator(minor_locator)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)

    ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
