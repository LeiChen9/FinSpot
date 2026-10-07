"""Trading-calendar helpers for in-memory market data."""

from typing import Dict
import pandas as pd


def market_days(market_data: Dict[str, pd.DataFrame], start, end):
    days = sorted({day for frame in market_data.values() for day in frame.index})
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    return [day for day in days if start <= day <= end]
