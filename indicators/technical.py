"""技术指标计算 — 纯函数，无状态"""
import pandas as pd
import numpy as np
from typing import List, Optional


def moving_averages(df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
    """计算收盘价的移动平均线，返回添加了 ma{period} 列的 DataFrame"""
    if periods is None:
        periods = [5, 30, 120]
    if 'close' not in df.columns:
        return df
    for p in periods:
        df = df.copy()
        df[f'ma{p}'] = df['close'].rolling(window=p).mean()
    return df


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """相对强弱指标 (RSI)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series,
         fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD 指标

    返回 DataFrame，包含: macd (DIF), signal (DEA), histogram (MACD 柱)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    histogram = 2 * (dif - dea)
    return pd.DataFrame({'macd': dif, 'signal': dea, 'histogram': histogram})


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """布林带 (Bollinger Bands)

    返回 DataFrame，包含: middle, upper, lower
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std(ddof=0)
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return pd.DataFrame({'middle': middle, 'upper': upper, 'lower': lower})


def atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """平均真实波幅 (ATR)

    参数:
        df: 包含 'high', 'low', 'close' 列的 DataFrame
        period: ATR 周期，默认 20

    返回:
        ATR 序列
    """
    high = df['high']
    low = df['low']
    prev_close = df['close'].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_series = tr.rolling(window=period).mean()
    return atr_series


def keltner_channel(df: pd.DataFrame, ma_period: int = 20, atr_period: int = 20, atr_multiplier: float = 2.0) -> pd.DataFrame:
    """Keltner 通道 (基于 ATR)

    参数:
        df: 包含 'high', 'low', 'close' 列的 DataFrame
        ma_period: 移动平均线周期，默认 20
        atr_period: ATR 周期，默认 20
        atr_multiplier: ATR 乘数，默认 2.0

    返回:
        DataFrame，包含: middle, upper, lower
    """
    middle = df['close'].rolling(window=ma_period).mean()
    atr_series = atr(df, period=atr_period)
    upper = middle + atr_multiplier * atr_series
    lower = middle - atr_multiplier * atr_series
    return pd.DataFrame({'middle': middle, 'upper': upper, 'lower': lower})
