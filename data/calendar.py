"""A 股财报日历规则 — 回测中判断某日期可用哪些财报"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional


REPORT_DEADLINES = {
    # (report_type, deadline_month, deadline_day)
    'Q1':      (4, 30),   # 一季报: 4月30日前
    'Q2':      (8, 31),   # 中报:   8月31日前
    'Q3':      (10, 31),  # 三季报: 10月31日前
    'Q4':      (4, 30),   # 年报:   次年4月30日前
}


@dataclass
class ReportPeriod:
    report_type: str    # 'Q1', 'Q2', 'Q3', 'Q4'
    year: int          # 财报所属年份
    report_date: date  # 报告期截止日
    deadline: date     # 最晚发布日期


def report_periods_for(as_of: date, n_years: int = 5) -> List[ReportPeriod]:
    """返回截至 as_of 日期, 最新可用的 N 年期次财报列表

    规则（A 股）:
      - Q1 报告期: 当年 3-31, 截止 当年 4-30
      - Q2 (中报): 当年 6-30, 截止 当年 8-31
      - Q3 报告期: 当年 9-30, 截止 当年 10-31
      - Q4 (年报): 上一年 12-31, 截止 当年 4-30
    """
    periods = []

    for year in range(as_of.year - n_years, as_of.year + 1):
        for rtype, (dead_m, dead_d) in REPORT_DEADLINES.items():
            if rtype == 'Q4':
                report_d = date(year - 1, 12, 31)
                deadline_d = date(year, dead_m, dead_d)
            elif rtype == 'Q1':
                report_d = date(year, 3, 31)
                deadline_d = date(year, dead_m, dead_d)
            elif rtype == 'Q2':
                report_d = date(year, 6, 30)
                deadline_d = date(year, dead_m, dead_d)
            else:  # Q3
                report_d = date(year, 9, 30)
                deadline_d = date(year, dead_m, dead_d)

            if deadline_d <= as_of:
                fiscal_year = year - 1 if rtype == 'Q4' else year
                periods.append(ReportPeriod(
                    report_type=rtype, year=fiscal_year,
                    report_date=report_d, deadline=deadline_d,
                ))

    return sorted(periods, key=lambda p: p.report_date, reverse=True)


def latest_n_complete_years(as_of: date, n: int = 3) -> List[ReportPeriod]:
    """获取截至 as_of 最近 N 个完整财年的年报

    例如: 若当前为 2026-07-15, 返回 2025, 2024, 2023 年报
    """
    periods = report_periods_for(as_of)
    annuals = [p for p in periods if p.report_type == 'Q4']
    return annuals[:n]


def latest_n_reports(as_of: date, n: int = 3) -> List[ReportPeriod]:
    """获取最新的 N 期财报（不限类型）"""
    periods = report_periods_for(as_of)
    return periods[:n]


def generate_rebalance_dates(
    start: Optional[date] = None,
    end: Optional[date] = None,
    years_back: int = 3,
    days_of_month: List[int] = None,
) -> List[date]:
    """生成调仓日期序列

    默认: 从三年前到今天, 每月 1 日和 15 日
    """
    if days_of_month is None:
        days_of_month = [1, 15]
    end = end or date.today()
    start = start or date(end.year - years_back, end.month, 1)

    dates = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        for d in days_of_month:
            try:
                dt = date(y, m, d)
                if start <= dt <= end:
                    dates.append(dt)
            except ValueError:
                pass
        m += 1
        if m > 12:
            m = 1
            y += 1

    return dates
