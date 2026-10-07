"""Reusable local-cache and multi-source market data service."""

import os
from datetime import datetime, timedelta

import pandas as pd


class MarketCache:
    def __init__(self, data_dir: str, source_list: list, hk_sources: tuple):
        self.data_dir = data_dir
        self.source_list = source_list
        self.hk_sources = hk_sources
        os.makedirs(data_dir, exist_ok=True)

    def load_or_fetch(self, code: str, days: int, force: bool = False) -> pd.DataFrame:
        path = self.market_path(code)
        end = self.nearest_trading_day()
        start = end - timedelta(days=int(days * 1.5))
        if not force and os.path.exists(path):
            local = self._read(path)
            if local.index.max() >= end:
                print(f"   [√] 数据已最新: {path} ({len(local)} 条记录)")
                return self._tail(local, days)
            fresh = self.try_sources(code, local.index.max() + timedelta(days=1), end)
            if fresh is not None and not fresh.empty:
                combined = pd.concat([local, fresh])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                combined.to_csv(path)
                print(f"   [√] 数据已增量更新: {path} ({len(combined)} 条)")
                return self._tail(combined, days)
            return self._tail(local, days)
        print(f"   [...] 下载行情数据: {code}")
        frame = self.try_sources(code, start, end)
        if frame is not None and not frame.empty:
            frame.sort_index().to_csv(path)
            print(f"   [√] 行情已保存: {path} ({len(frame)} 条记录)")
            return self._tail(frame, days)
        if os.path.exists(path):
            local = self._read(path)
            print(f"   [√] 使用本地数据: {path} ({len(local)} 条记录)")
            return self._tail(local, days)
        raise RuntimeError(f"无法从任何源获取行情数据: {code}")

    def try_sources(self, code: str, start: datetime, end: datetime,
                    source_list: list | None = None) -> pd.DataFrame | None:
        errors = []
        for source in source_list or self.source_list:
            try:
                frame = source(code, start, end)
                if frame is not None and not frame.empty:
                    return frame
            except Exception as error:
                errors.append(f"{source.__module__}.{source.__name__}: {error}")
        if errors:
            raise RuntimeError(f"all market data sources failed for {code}: {'; '.join(errors)}")
        return None

    def fetch_hk(self, code: str, days: int) -> pd.DataFrame:
        end = self.nearest_trading_day()
        start = end - timedelta(days=int(days * 1.5))
        print(f"   [...] 下载港股数据: {code}")
        for source in self.hk_sources:
            frame = source(code, start, end)
            if frame is not None and not frame.empty:
                return self._tail(frame, days)
        raise RuntimeError(f"无法获取港股数据: {code}")

    def market_path(self, code: str) -> str:
        return os.path.join(self.data_dir, f"{code}_market.csv")

    @staticmethod
    def nearest_trading_day() -> datetime:
        now = datetime.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if now.hour > 15 or (now.hour == 15 and now.minute >= 30):
            if today.weekday() < 5:
                return today
        day = today - timedelta(days=1)
        while day.weekday() >= 5:
            day -= timedelta(days=1)
        return day

    @staticmethod
    def _read(path: str) -> pd.DataFrame:
        return pd.read_csv(path, index_col="date", parse_dates=True)

    @staticmethod
    def _tail(frame: pd.DataFrame, days: int) -> pd.DataFrame:
        return frame.tail(days) if len(frame) > days else frame
