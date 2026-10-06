"""Graham 回测使用的本地缓存读取。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
FIN_DIR = DATA_DIR / "financial"
DIV_DIR = DATA_DIR / "dividend"
META_DIR = DATA_DIR / "meta"
_cache: dict[str, object] = {}


def _cached(key: str, loader):
    if key not in _cache:
        _cache[key] = loader()
    return _cache[key]


def load_universe() -> pd.DataFrame:
    return _cached("universe", lambda: pd.read_csv(META_DIR / "graham_universe.csv", dtype={"code": str}))


def _report(code: str, kind: str) -> pd.DataFrame | None:
    path = FIN_DIR / f"{code}_{kind}.csv"
    if not path.exists():
        return None
    def read():
        frame = pd.read_csv(path, dtype={"报告日": str, "公告日期": str})
        for column in ("报告日", "公告日期"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        return frame.dropna(subset=["报告日"]).sort_values("报告日")
    return _cached(f"{kind}_{code}", read)


def load_balance(code: str) -> pd.DataFrame | None:
    return _report(code, "balance")


def load_profit(code: str) -> pd.DataFrame | None:
    return _report(code, "profit")


def load_dividend(code: str) -> pd.DataFrame | None:
    path = DIV_DIR / f"{code}_dividend.csv"
    return _cached(f"dividend_{code}", lambda: pd.read_csv(path) if path.exists() else None)


def load_market(code: str, qfq: bool = False) -> pd.DataFrame | None:
    path = DATA_DIR / f"{code}_{'qfq' if qfq else 'market'}.csv"
    if not path.exists():
        return None
    def read():
        frame = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
        return frame[~frame.index.duplicated(keep="last")]
    return _cached(f"market_{qfq}_{code}", read)


def load_10y() -> pd.DataFrame:
    return _cached("10y", lambda: pd.read_csv(DATA_DIR / "zh_10y_treasury.csv", index_col="date", parse_dates=True))


def load_all_a_pe() -> pd.DataFrame:
    return _cached("allpe", lambda: pd.read_csv(META_DIR / "all_a_pe.csv", index_col="date", parse_dates=True))


def load_index(code: str) -> pd.DataFrame | None:
    return load_market(code)
