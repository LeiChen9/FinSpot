"""ETF/指数日线抓取 (QQ → akshare ETF → akshare 指数 回退链), 带本地缓存。"""
import os
from datetime import datetime

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


def _cache_path(code: str, adjust: str) -> str:
    suffix = '' if adjust == 'qfq' else f'_{adjust}'
    return str(DATA_DIR / f'etf_{code}{suffix}.csv')


def _covers(df: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> bool:
    """缓存是否覆盖 [start, end]; end 允许因无未来行情/休市而留 10 天缓冲。"""
    if df.empty:
        return False
    required_end = min(pd.Timestamp(end_dt), pd.Timestamp.today().normalize())
    return (df.index.min() <= pd.Timestamp(start_dt)
            and df.index.max() >= required_end - pd.Timedelta(days=10))


def _sina(code: str, fetch, start_dt: datetime, end_dt: datetime) -> pd.DataFrame | None:
    """Sina 数据源无 sh/sz 前缀, 依次尝试两前缀并截取区间。"""
    for prefix in ('sh', 'sz'):
        try:
            df = fetch(f'{prefix}{code}')
            if df is None or df.empty:
                continue
            df = _std(df)
            return df[(df.index >= pd.Timestamp(start_dt)) & (df.index <= pd.Timestamp(end_dt))]
        except Exception:
            continue
    return None


def _fetch_remote(code: str, start_dt: datetime, end_dt: datetime, adjust: str) -> pd.DataFrame | None:
    import akshare as ak
    if adjust == '':
        return _sina(code, ak.fund_etf_hist_sina, start_dt, end_dt)
    s, e = start_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d')
    try:
        df = ak.fund_etf_hist_em(symbol=code, period='daily',
                                 start_date=s, end_date=e, adjust=adjust)
        if df is not None and not df.empty:
            return _std(df)
    except Exception as e:
        print(f'  [akshare {code}] {e}')
    from dataload.sources import qq
    try:
        df = qq.fetch(code, start_dt, end_dt)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        print(f'  [QQ {code}] {e}')
    return _sina(code, ak.stock_zh_index_daily, start_dt, end_dt)


def fetch_data(code: str, start_dt: datetime, end_dt: datetime, adjust: str = 'qfq'):
    """抓取 ETF/指数日线, 结果缓存至 data/etf_{code}[_adjust].csv。

    缓存仅覆盖请求区间时才复用; 否则按请求区间重新抓取, 避免命中截断的旧缓存。
    adjust='' 取不复权价 (现金分红类策略, Sina), 'qfq' 取前复权 (总收益类)。
    """
    cache = _cache_path(code, adjust)
    cached = None
    if os.path.exists(cache):
        cached = pd.read_csv(cache, index_col='date', parse_dates=True).sort_index()
        if _covers(cached, start_dt, end_dt):
            return cached[(cached.index >= pd.Timestamp(start_dt)) & (cached.index <= pd.Timestamp(end_dt))]
    df = _fetch_remote(code, start_dt, end_dt, adjust)
    if df is not None and not df.empty:
        _save_cache(cache, df)
        return df
    if cached is not None and not cached.empty:
        print(f'  [{code}] 在线抓取失败, 回退缓存 '
              f'({cached.index.min().date()}~{cached.index.max().date()}), 区间可能不完整')
        return cached[(cached.index >= pd.Timestamp(start_dt)) & (cached.index <= pd.Timestamp(end_dt))]
    return None


def fetch_etf_dividends(symbol: str) -> pd.Series:
    """ETF 每份现金分红 (按分红日期), symbol 如 'sh510880'。"""
    import akshare as ak
    df = ak.fund_etf_dividend_sina(symbol=symbol)
    dates = pd.to_datetime(df['日期'])
    order = dates.argsort()
    per_share = df['累计分红'].diff().fillna(df['累计分红']).iloc[order]
    return pd.Series(per_share.to_numpy(), index=dates.iloc[order]).sort_index()


def _save_cache(cache: str, df: pd.DataFrame):
    os.makedirs(DATA_DIR, exist_ok=True)
    full = df[~df.index.duplicated(keep='last')].sort_index()
    full.to_csv(cache)
