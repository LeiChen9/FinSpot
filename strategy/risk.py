"""风险管理：止盈 / 止损 / 追踪止盈 / 趋势追踪"""
from typing import Optional
from backtest.portfolio import Holding


class RiskManager:
    """风险管理器 — 对每个持仓独立检查风控条件"""

    def __init__(
        self,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        max_hold_days: Optional[int] = None,
    ):
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trailing_stop = trailing_stop
        self.max_hold_days = max_hold_days

    def check(self, holding: Holding, current_date, buy_date=None) -> Optional[str]:
        if holding.shares <= 0 or holding.avg_cost <= 0:
            return None

        pnl_pct = holding.current_price / holding.avg_cost - 1

        if self.take_profit is not None and pnl_pct >= self.take_profit:
            return 'take_profit'

        if self.stop_loss is not None and pnl_pct <= -self.stop_loss:
            return 'stop_loss'

        if self.trailing_stop is not None and holding.highest_price > 0:
            drawdown = holding.current_price / holding.highest_price - 1
            if drawdown <= -self.trailing_stop:
                return 'trailing_stop'

        if self.max_hold_days is not None and buy_date is not None:
            if hasattr(current_date, 'date') and hasattr(buy_date, 'date'):
                delta = (current_date.date() - buy_date.date()).days
            else:
                delta = (current_date - buy_date).days
            if delta >= self.max_hold_days:
                return 'max_hold'

        return None

    def __repr__(self):
        parts = []
        if self.take_profit: parts.append(f"TP={self.take_profit:.0%}")
        if self.stop_loss: parts.append(f"SL={self.stop_loss:.0%}")
        if self.trailing_stop: parts.append(f"Trail={self.trailing_stop:.0%}")
        if self.max_hold_days: parts.append(f"MaxHold={self.max_hold_days}d")
        return f"RiskManager({'|'.join(parts)})" if parts else "RiskManager(off)"


class TrendTracker:
    """趋势追踪 — 基于技术指标判断趋势方向

    用于:
      - 确认趋势后才入场 (避免抄底)
      - 趋势走坏后离场
    """

    def __init__(self, ma_fast: int = 20, ma_slow: int = 60, rsi_period: int = 14):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.rsi_period = rsi_period

    def is_uptrend(self, df, as_of) -> bool:
        """判断 as_of 位置是否为上升趋势"""
        if as_of < self.ma_slow:
            return False
        close = df['close'].iloc[:as_of + 1]
        ma_f = close.rolling(self.ma_fast).mean().iloc[-1]
        ma_s = close.rolling(self.ma_slow).mean().iloc[-1]
        return ma_f > ma_s

    def trend_signal(self, df) -> str:
        """返回趋势方向: 'up' | 'down' | 'sideways'"""
        close = df['close']
        ma_s = close.rolling(self.ma_slow).mean()
        ma_f = close.rolling(self.ma_fast).mean()
        latest = ma_f.iloc[-1] > ma_s.iloc[-1]
        prev = ma_f.iloc[-2] > ma_s.iloc[-2] if len(ma_f) > 1 else latest
        if latest and not prev:
            return 'up'
        if not latest and prev:
            return 'down'
        return 'sideways'
