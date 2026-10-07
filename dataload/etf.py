"""ETF/指数日线抓取 (QQ → akshare ETF → akshare 指数 回退链), 带本地缓存。"""
import os
from datetime import datetime
from typing import Dict

import pandas as pd

from common.paths import DATA_DIR


def _std(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        '日期': 'date', '开盘': 'open', '最高': 'high',
        '最低': 'low', '收盘': 'close', '成交量': 'volume',
    }
    df = df.rename(columns=rename)
    cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in cols if c in df.columns]]
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df


def _cache_path(code: str) -> str:
    return str(DATA_DIR / f'etf_{code}.csv')


def fetch_data(code: str, start_dt: datetime, end_dt: datetime):
    """QQ 优先, akshare 兜底; 结果缓存至 data/etf_{code}.csv。"""
    cache = _cache_path(code)
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col='date', parse_dates=True).sort_index()
        df = df[(df.index >= pd.Timestamp(start_dt)) & (df.index <= pd.Timestamp(end_dt))]
        if not df.empty:
            return df
    from dataload.sources import qq
    try:
        df = qq.fetch(code, start_dt, end_dt)
        if df is not None and not df.empty:
            _save_cache(cache, df)
            return df
    except Exception as e:
        print(f'  [QQ {code}] {e}')
    import akshare as ak
    try:
        s, e = start_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d')
        df = ak.fund_etf_hist_em(symbol=code, period='daily',
                                 start_date=s, end_date=e, adjust='qfq')
        if df is not None and not df.empty:
            df = _std(df)
            _save_cache(cache, df)
            return df
    except Exception as e:
        print(f'  [akshare {code}] {e}')
    try:
        s, e = start_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d')
        df = ak.index_zh_a_hist(symbol=code, period='daily',
                                start_date=s, end_date=e)
        if df is not None and not df.empty:
            df = _std(df)
            _save_cache(cache, df)
            return df
    except Exception as e:
        print(f'  [akshare idx {code}] {e}')
    return None


def _save_cache(cache: str, df: pd.DataFrame):
    os.makedirs(DATA_DIR, exist_ok=True)
    full = df[~df.index.duplicated(keep='last')].sort_index()
    full.to_csv(cache)
