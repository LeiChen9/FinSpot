"""数据层的规范化契约。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MARKET_COLUMNS = ("open", "high", "low", "close", "volume")


def read_market_csv(path: str | Path) -> pd.DataFrame:
    """读取一个本地行情文件并返回按日期排序的标准表。"""
    frame = pd.read_csv(path, index_col="date", parse_dates=True)
    if "close" not in frame:
        raise ValueError(f"market data must contain close: {path}")
    frame.index.name = "date"
    return frame.sort_index()
