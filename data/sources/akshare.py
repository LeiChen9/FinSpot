"""Akshare 数据源（行情 / 基金净值 / 估值 fallback）"""
import pandas as pd
from datetime import datetime


def fetch(code: str, start: datetime, end: datetime) -> pd.DataFrame | None:
    """获取个股 / 指数日线行情（OHLCV），优先试股票再试指数"""
    import akshare as ak

    # 试个股
    try:
        df = ak.stock_zh_a_hist(
            symbol=code, period='daily',
            start_date=start.strftime('%Y%m%d'),
            end_date=end.strftime('%Y%m%d'),
        )
        if df is not None and not df.empty:
            return _standardize(df)
    except Exception:
        pass

    # 试指数
    try:
        df = ak.index_zh_a_hist(
            symbol=code, period='daily',
            start_date=start.strftime('%Y%m%d'),
            end_date=end.strftime('%Y%m%d'),
        )
        if df is not None and not df.empty:
            return _standardize(df)
    except Exception:
        pass

    # 试港股（同花顺接口可覆盖部分港股）
    try:
        df = ak.stock_zh_a_hist(
            symbol=code, period='daily',
            start_date=start.strftime('%Y%m%d'),
            end_date=end.strftime('%Y%m%d'),
        )
        if df is not None and not df.empty:
            return _standardize(df)
    except Exception:
        pass

    return None


def fetch_nav(code: str) -> pd.DataFrame | None:
    """获取基金净值数据"""
    import akshare as ak
    try:
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        if df is None or df.empty:
            return None
        _rename_map = {
            '净值日期': 'date', '单位净值': 'nav',
            '累计净值': 'acc_nav', '日增长率': 'daily_return',
            '日期': 'date',
        }
        df = df.rename(columns=_rename_map)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['nav', 'acc_nav', 'daily_return']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception:
        return None


def fetch_valuation(code: str) -> pd.DataFrame | None:
    """获取指数估值数据（PE / PB / 股息率），返回最新一行"""
    import akshare as ak

    try:
        df = ak.stock_zh_index_value_csindex(symbol=code)
        if df is not None and not df.empty:
            rename = {'日期': 'date', '市盈率2': 'pe_ttm', '市净率1': 'pb', '股息率1': 'dividend_yield'}
            cols = [c for c in rename if c in df.columns]
            if cols:
                df = df[cols].rename(columns=rename)
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                for col in ['pe_ttm', 'pb', 'dividend_yield']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                return df
    except Exception:
        pass

    try:
        df = ak.index_value_operate(symbol=code)
        if df is not None and not df.empty:
            rename = {
                '日期': 'date', '动态市盈率': 'pe_ttm', '市盈率': 'pe_ttm',
                '市净率': 'pb', '股息率': 'dividend_yield', '分红率': 'dividend_yield',
            }
            df = df.rename(columns=rename)
            cols = [c for c in ['date', 'pe_ttm', 'pb', 'dividend_yield'] if c in df.columns]
            if 'date' in cols:
                df = df[cols]
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                for col in ['pe_ttm', 'pb', 'dividend_yield']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                return df
    except Exception:
        pass

    return None


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """将 akshare 返回的列名标准化为英文"""
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
