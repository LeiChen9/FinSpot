"""本地市场数据的加载器。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from data.contracts import read_market_csv


def load_market_frames(
    codes: Iterable[str],
    data_dir: str | Path = "data",
    min_rows: int = 1,
) -> dict[str, pd.DataFrame]:
    """按代码加载满足最小长度的本地行情。"""
    root = Path(data_dir)
    frames: dict[str, pd.DataFrame] = {}
    for code in codes:
        key = str(code)
        path = root / f"{key}_market.csv"
        if path.exists():
            frame = read_market_csv(path)
            if len(frame) >= min_rows:
                frames[key] = frame
    return frames


def cached_market_loader(
    data_dir: str | Path = "data", min_rows: int = 1
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
