"""Reusable rebalance calendar and weight-history builders."""

from datetime import datetime
from typing import Iterable, Mapping, Sequence
import pandas as pd

def first_trading_day(trading_days: Iterable[object], year: int, month: int) -> datetime | None:
    target = pd.Timestamp(year=year, month=month, day=1)
    for day in sorted(pd.Timestamp(value).normalize() for value in trading_days):
        if day >= target:
            return day.to_pydatetime()
    return None

def scheduled_rebalance_dates(trading_days: Iterable[object], start: object, end: object,
                              months: Sequence[int] = (1, 4, 7, 10)) -> list[datetime]:
    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    days = sorted({pd.Timestamp(day).normalize() for day in trading_days})
    dates = []
    for year in range(start_ts.year, end_ts.year + 1):
        for month in months:
            day = first_trading_day(days, year, month)
            if day is not None and start_ts <= pd.Timestamp(day) <= end_ts:
                dates.append(day)
    return sorted(set(dates))

def fixed_weight_history(rebalance_dates: Iterable[object], weights: Mapping[str, float]) -> list[dict]:
    return [{'date': pd.Timestamp(day).to_pydatetime(), 'weights': dict(weights)} for day in rebalance_dates]
