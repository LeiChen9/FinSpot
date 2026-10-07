"""本地缓存数据的统一读取器(只读, 进程内缓存)。

文件布局(全部位于 data/, 不入 git):
    {code}_market.csv / {code}_qfq.csv      日线(不复权/前复权)
    financial/{code}_{balance,profit}.csv   资产负债/利润报表(含公告日期)
    financial/{code}_fin.csv                同花顺财务摘要宽表
    dividend/{code}_dividend.csv            分红送配历史
    meta/graham_universe.csv                全A资产池
    meta/all_a_pe.csv                       全A平均PE
    zh_10y_treasury.csv                     中债10Y收益率
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from common.paths import DATA_DIR, DIV_DIR, FIN_DIR, META_DIR

_cache: dict[str, object] = {}


def _cached(key: str, loader: Callable):
    if key not in _cache:
        _cache[key] = loader()
    return _cache[key]


def _market(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
    return frame[~frame.index.duplicated(keep="last")]


# ── 资产池与元数据 ──

def load_universe() -> pd.DataFrame:
    return _cached("universe", lambda: pd.read_csv(META_DIR / "graham_universe.csv", dtype={"code": str}))


def load_all_a_pe() -> pd.DataFrame:
    return _cached("allpe", lambda: pd.read_csv(META_DIR / "all_a_pe.csv", index_col="date", parse_dates=True))


def load_10y() -> pd.DataFrame:
    return _cached("10y", lambda: pd.read_csv(DATA_DIR / "zh_10y_treasury.csv", index_col="date", parse_dates=True))


# ── 行情 ──

def load_market(code: str, qfq: bool = False) -> pd.DataFrame | None:
    return _cached(f"market_{qfq}_{code}", lambda: _market(DATA_DIR / f"{code}_{'qfq' if qfq else 'market'}.csv"))


def load_qfq(code: str) -> pd.DataFrame | None:
    """前复权行情; 无 qfq 文件时回退到不复权文件(如 688036 的 market 已复权)。"""
    frame = load_market(code, qfq=True)
    return frame if frame is not None else load_market(code, qfq=False)


def load_raw(code: str) -> pd.DataFrame | None:
    return load_market(code, qfq=False)


def load_index(code: str) -> pd.DataFrame | None:
    return load_market(code)


# ── 财务与分红 ──

def _report(code: str, kind: str) -> pd.DataFrame | None:
    path = FIN_DIR / f"{code}_{kind}.csv"
    if not path.exists():
        return None

    def read() -> pd.DataFrame:
        frame = pd.read_csv(path, dtype={"报告日": str, "公告日期": str})
        for column in ("报告日", "公告日期"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        return frame.dropna(subset=["报告日"]).sort_values("报告日")

    return _cached(f"{kind}_{code}", read)


def load_balance(code: str) -> pd.DataFrame | None:
    return _report(code, "balance")


def load_profit(code: str) -> pd.DataFrame | None:
    return _report(code, "profit")


def load_fin_summary(code: str) -> pd.DataFrame | None:
    def read():
        path = FIN_DIR / f"{code}_fin.csv"
        return pd.read_csv(path) if path.exists() else None
    return _cached(f"fin_{code}", read)


def load_dividend(code: str) -> pd.DataFrame | None:
    def read():
        path = DIV_DIR / f"{code}_dividend.csv"
        return pd.read_csv(path) if path.exists() else None
    return _cached(f"dividend_{code}", read)


# ── 批量加载(按日期截断) ──

def load_market_frames(
    codes: Iterable[str],
    data_dir: str | Path = DATA_DIR,
    min_rows: int = 1,
) -> dict[str, pd.DataFrame]:
    """按代码加载满足最小行数的本地不复权行情。"""
    root = Path(data_dir)
    frames: dict[str, pd.DataFrame] = {}
    for code in codes:
        key = str(code)
        path = root / f"{key}_market.csv"
        if path.exists():
            frame = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
            if len(frame) >= min_rows:
                frames[key] = frame
    return frames


def cached_market_loader(
    data_dir: str | Path = DATA_DIR, min_rows: int = 1
) -> Callable[[str, object], pd.DataFrame | None]:
    """构造按代码缓存、按日期截断的行情加载器。"""
    cache: dict[str, pd.DataFrame | None] = {}

    def load(code: str, as_of: object) -> pd.DataFrame | None:
        key = str(code)
        if key not in cache:
            cache[key] = load_market_frames([key], data_dir, min_rows).get(key)
        frame = cache[key]
        return None if frame is None else frame.loc[frame.index <= pd.Timestamp(as_of)].copy()

    return load
