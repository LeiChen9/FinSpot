"""Pure rules and calendar helpers for MA120 mean reversion."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

MA_WINDOW = 120
MA_VOL_LOOKBACK = 500

def target_weight(dev: float, cap: float = 0.10, buy_full: float = 0.10, sell_zero: float = 0.08) -> float:
    if dev <= -buy_full:
        return cap
    if dev < 0:
        return cap * (-dev) / buy_full
    if dev < sell_zero:
        return cap * (1.0 - dev / sell_zero)
    return 0.0

def ma120_lowvol_mask(df: pd.DataFrame, as_of: pd.Timestamp, lookback: int = MA_VOL_LOOKBACK) -> Optional[float]:
    hist = df[df.index <= as_of].tail(lookback + MA_WINDOW)
    if len(hist) < MA_WINDOW + 60:
        return None
    ma = hist['close'].rolling(MA_WINDOW).mean().dropna()
    if len(ma) < 60:
        return None
    return float(ma.pct_change().dropna().std() * np.sqrt(252))

def rolling_ma120_vol(stocks: Dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.Series:
    return pd.Series({code: vol for code, df in stocks.items() if (vol := ma120_lowvol_mask(df, as_of)) is not None and np.isfinite(vol) and vol > 0})

def make_rebalance_dates(trading_days: List, start: pd.Timestamp, end: pd.Timestamp, step: int = 15) -> List[pd.Timestamp]:
    days = [d for d in trading_days if start <= d <= end]
    return [days[i] for i in range(0, len(days), step)]
