"""回测应用共用的市场数据编排。"""

from __future__ import annotations

import pandas as pd


def benchmark_nav(
    market_data: dict[str, pd.DataFrame],
    code: str,
    start: object,
    end: object,
) -> pd.Series | None:
    """从已加载行情构造区间基准净值。"""
    frame = market_data[code].loc[pd.Timestamp(start):pd.Timestamp(end)]
    if frame.empty:
        return None
    return frame["close"] / frame["close"].iloc[0]
