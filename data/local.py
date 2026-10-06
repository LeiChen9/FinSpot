"""本地 CSV 市场数据的读取与缓存接口。"""
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import pandas as pd


def load_market_frames(
    codes: Iterable[str],
    data_dir: str | Path = "data",
    min_rows: int = 1,
    required_column: str = "close",
) -> Dict[str, pd.DataFrame]:
    """读取多个 ``{code}_market.csv`` 文件并跳过不完整数据。"""
    root = Path(data_dir)
    frames: Dict[str, pd.DataFrame] = {}
    for code in codes:
        key = str(code)
        path = root / f"{key}_market.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
        if required_column in frame.columns and len(frame) >= min_rows:
            frames[key] = frame
    return frames


def cached_market_loader(
    data_dir: str | Path = "data",
    min_rows: int = 1,
) -> Callable[[str, object], Optional[pd.DataFrame]]:
    """返回适配策略回测器的 ``(code, as_of) -> DataFrame`` 缓存 loader。"""
    root = Path(data_dir)
    cache: Dict[str, Optional[pd.DataFrame]] = {}

    def load(code: str, as_of: object) -> Optional[pd.DataFrame]:
        key = str(code)
        if key not in cache:
            cache[key] = load_market_frames([key], root, min_rows=min_rows).get(key)
        frame = cache[key]
        if frame is None:
            return None
        return frame.loc[frame.index <= pd.Timestamp(as_of)].copy()

    return load

