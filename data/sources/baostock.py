"""Baostock 数据源"""
import pandas as pd
from datetime import datetime


def fetch(code: str, start: datetime, end: datetime) -> pd.DataFrame | None:
    """获取 A 股/指数日线行情（OHLCV），含 yfinance fallback"""
    import baostock as bs

    ticker = f"sh.{code}" if code.startswith('6') else f"sz.{code}"

    bs.login()
    try:
        rs = bs.query_history_k_data_plus(
            ticker, "date,code,open,high,low,close,volume",
            start_date=start.strftime('%Y%m%d'),
            end_date=end.strftime('%Y%m%d'),
        )
        if rs.error_code != '0':
            return None
        data = []
        while rs.next():
            data.append(rs.get_row_data())
        if not data:
            return None
        df = pd.DataFrame(data, columns=['date', 'code', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.set_index('date', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.sort_index(inplace=True)
        return df
    except Exception:
        return None
    finally:
        bs.logout()


def fetch_hk(code: str, start: datetime, end: datetime) -> pd.DataFrame | None:
    """获取港股日线行情（baostock + yfinance fallback）"""
    if not code.startswith('hk') and not code.endswith('.HK'):
        code = f"hk{code}" if not code.startswith('hk') else code
        code = code.replace('.HK', '')
    ticker = f"sh.{code}" if code.startswith('6') else f"sz.{code}"

    # baostock
    import baostock as bs
    bs.login()
    try:
        rs = bs.query_history_k_data_plus(
            ticker, "date,code,open,high,low,close,volume",
            start_date=start.strftime('%Y%m%d'),
            end_date=end.strftime('%Y%m%d'),
        )
        if rs.error_code == '0':
            data = []
            while rs.next():
                data.append(rs.get_row_data())
            if data:
                df = pd.DataFrame(data, columns=['date', 'code', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df.set_index('date', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df.sort_index(inplace=True)
                return df
    except Exception:
        pass
    finally:
        bs.logout()

    # yfinance fallback
    import yfinance as yf
    ticker_yf = f"{code}.HK"
    try:
        df = yf.download(ticker_yf, start=start, end=end, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume',
            })
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.index.name = 'date'
            return df
    except Exception:
        pass

    return None
