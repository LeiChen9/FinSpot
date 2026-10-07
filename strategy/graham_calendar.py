"""Trading-calendar rules used by the Graham strategy."""

from typing import Callable, List

import pandas as pd


def trading_days(load_index: Callable) -> pd.DatetimeIndex:
    """Use the CSI 300 cache as the broad-market trading calendar."""
    frame = load_index("000300")
    if frame is None or frame.empty:
        return pd.DatetimeIndex([])
    return frame.index


def rebalance_dates(
    load_index: Callable,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[pd.Timestamp]:
    calendar = trading_days(load_index)
    dates = []
    current = pd.Timestamp(start)
    while current <= end:
        next_day = calendar[calendar >= current]
        if len(next_day):
            dates.append(pd.Timestamp(next_day[0]))
        current = pd.Timestamp(year=current.year, month=current.month, day=1)
        current += pd.DateOffset(months=3)
    return dates
