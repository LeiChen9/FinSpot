"""高息成长池所需的本地行情、财务和分红缓存读取。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
FIN_DIR = DATA_DIR / "financial"
DIV_DIR = DATA_DIR / "dividend"

_qfq_cache: dict[str, pd.DataFrame | None] = {}
_raw_cache: dict[str, pd.DataFrame | None] = {}
_profit_cache: dict[str, pd.DataFrame | None] = {}
_balance_cache: dict[str, pd.DataFrame | None] = {}
_summary_cache: dict[str, pd.DataFrame | None] = {}
_dividend_cache: dict[str, pd.DataFrame | None] = {}


def _market(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def load_qfq(code: str) -> pd.DataFrame | None:
    if code not in _qfq_cache:
        _qfq_cache[code] = _market(DATA_DIR / f"{code}_qfq.csv")
        if _qfq_cache[code] is None:
            _qfq_cache[code] = _market(DATA_DIR / f"{code}_market.csv")
    return _qfq_cache[code]


def load_raw(code: str) -> pd.DataFrame | None:
    if code not in _raw_cache:
        _raw_cache[code] = _market(DATA_DIR / f"{code}_market.csv")
    return _raw_cache[code]


def _financial(code: str, kind: str, cache: dict) -> pd.DataFrame | None:
    if code in cache:
        return cache[code]
    path = FIN_DIR / f"{code}_{kind}.csv"
    if not path.exists():
        cache[code] = None
        return None
    frame = pd.read_csv(path, dtype={"报告日": str, "公告日期": str})
    for column in ("报告日", "公告日期"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    cache[code] = frame.dropna(subset=["报告日"]).sort_values("报告日")
    return cache[code]


def load_profit(code: str) -> pd.DataFrame | None:
    return _financial(code, "profit", _profit_cache)


def load_balance(code: str) -> pd.DataFrame | None:
    return _financial(code, "balance", _balance_cache)


def load_fin_summary(code: str) -> pd.DataFrame | None:
    if code not in _summary_cache:
        path = FIN_DIR / f"{code}_fin.csv"
        _summary_cache[code] = pd.read_csv(path) if path.exists() else None
    return _summary_cache[code]


def load_dividend(code: str) -> pd.DataFrame | None:
    if code not in _dividend_cache:
        path = DIV_DIR / f"{code}_dividend.csv"
        _dividend_cache[code] = pd.read_csv(path) if path.exists() else None
    return _dividend_cache[code]


__all__ = [
    "load_qfq", "load_raw", "load_profit", "load_balance",
    "load_fin_summary", "load_dividend",
]
