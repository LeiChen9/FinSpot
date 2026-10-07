"""Market-history acquisition and normalization for the Graham dataset."""

from datetime import datetime
import os

import numpy as np
import pandas as pd

from dataload.sources import baostock, qq

from common.paths import DATA_DIR
K_START = "2022-11-01"


def _symbol_sina(code: str) -> str:
    exchange = "sh" if code[0] in ("6", "9") else "sz"
    if code[0] in ("4", "8") or code.startswith("92"):
        exchange = "bj"
    return f"{exchange}{code}"


def fetch_market_history(code, qfq: bool, expected_min):
    """Retry short histories until their start anchor and span are credible."""
    import time as _t

    for _ in range(4):
        frame = _fetch_market_sina(code, qfq=qfq)
        if frame is None or frame.empty:
            frame = _fetch_market_ak(code, qfq=qfq)
        if frame is None or frame.empty:
            frame = _fetch_market_local(code, qfq=qfq)
        if frame is None or frame.empty:
            return None
        if expected_min is None:
            return frame
        start_ok = frame.index.min() <= expected_min + pd.Timedelta("90D")
        span_ok = (frame.index.max() - frame.index.min()).days >= 100
        if start_ok and span_ok:
            return frame
        _t.sleep(2)
    return frame


def _fetch_market_sina(code, qfq: bool):
    """Fetch and normalize Sina daily bars."""
    import akshare as ak
    import time as _t

    for _ in range(3):
        try:
            frame = ak.stock_zh_a_daily(
                symbol=_symbol_sina(code),
                start_date=K_START.replace("-", ""),
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq" if qfq else "",
            )
            if frame is None or frame.empty:
                return None
            frame = frame.reset_index().rename(columns={
                "date": "date", "open": "open", "high": "high",
                "low": "low", "close": "close", "volume": "volume",
            })
            columns = ["date", "open", "high", "low", "close", "volume"]
            for column in columns[1:]:
                if column not in frame:
                    frame[column] = np.nan
            frame = frame[columns]
            frame["date"] = pd.to_datetime(frame["date"])
            frame = frame.set_index("date").sort_index()
            for column in columns[1:]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            return frame
        except Exception:
            _t.sleep(2)
    return None


def _fetch_market_ak(code, qfq: bool):
    import akshare as ak
    import time as _t

    for _ in range(3):
        try:
            frame = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=K_START.replace("-", ""),
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq" if qfq else "",
            )
            if frame is None or frame.empty:
                return None
            frame = frame.rename(columns={
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume",
            })
            frame = frame[["date", "open", "high", "low", "close", "volume"]]
            frame["date"] = pd.to_datetime(frame["date"])
            frame = frame.set_index("date").sort_index()
            for column in ("open", "high", "low", "close", "volume"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if frame.index[-1] - frame.index[0] < pd.Timedelta("100D"):
                _t.sleep(2)
                continue
            return frame
        except Exception:
            _t.sleep(2)
    return None


def _fetch_market_local(code, qfq: bool):
    """Use QQ and Baostock as source-specific fallback adapters."""
    start = pd.to_datetime(K_START)
    end = pd.Timestamp.now()
    if qfq:
        try:
            frame = qq.fetch(code, start, end)
            return frame if frame is not None and len(frame) else None
        except Exception:
            return None

    try:
        frame = baostock.fetch(code, start, end)
        if frame is not None and len(frame):
            return frame
        return qq.fetch(code, start, end)
    except Exception:
        return None
