"""常用信号生成器"""
import pandas as pd
import numpy as np
from typing import Optional
from strategy.base import Signal
from indicators.technical import rsi, macd, keltner_channel


class MACrossSignal(Signal):
    """均线交叉信号: 短期均线上穿长期均线时买入，下穿时卖出"""

    def __init__(self, fast: int = 5, slow: int = 20):
        self.fast = fast
        self.slow = slow

    def generate(self, df: pd.DataFrame) -> pd.Series:
        fast_ma = df['close'].rolling(self.fast).mean()
        slow_ma = df['close'].rolling(self.slow).mean()

        signal = pd.Series(0, index=df.index)
        signal[(fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))] = 1
        signal[(fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))] = -1

        return signal


class RSISignal(Signal):
    """RSI 信号: 低于下限超卖买入，高于上限超买卖出"""

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate(self, df: pd.DataFrame) -> pd.Series:
        rsi_values = rsi(df['close'], self.period)

        signal = pd.Series(0, index=df.index)
        signal[rsi_values < self.oversold] = 1
        signal[rsi_values > self.overbought] = -1

        return signal


class ValuationSignal(Signal):
    """估值分位信号: 低估时买入，高估时卖出"""

    def __init__(self, indicator: str = 'pe_ttm',
                 lower_pct: float = 0.2, upper_pct: float = 0.8,
                 lookback_years: int = 5):
        self.indicator = indicator
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.lookback_years = lookback_years

    def generate(self, df: pd.DataFrame) -> pd.Series:
        if self.indicator not in df.columns:
            raise ValueError(f"df 缺少列: {self.indicator}")

        col = df[self.indicator].dropna()
        if len(col) < 20:
            raise ValueError("估值数据不足")

        signal = pd.Series(0, index=df.index)
        values = df[self.indicator]

        # 滚动分位数判断
        window = min(252 * self.lookback_years, len(values))
        for i in range(window, len(values)):
            hist = values.iloc[i - window:i].dropna()
            if len(hist) < 20:
                continue
            pct = (hist < values.iloc[i]).mean()
            if pct < self.lower_pct:
                signal.iloc[i] = 1
            elif pct > self.upper_pct:
                signal.iloc[i] = -1

        return signal


class ATRChannelBreakoutSignal(Signal):
    """ATR 通道突破信号: 价格突破通道上轨时买入，跌破通道下轨时卖出"""

    def __init__(self, ma_period: int = 20, atr_period: int = 20, atr_multiplier: float = 2.0):
        self.ma_period = ma_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def generate(self, df: pd.DataFrame) -> pd.Series:
        channel = keltner_channel(df, self.ma_period, self.atr_period, self.atr_multiplier)
        upper = channel['upper']
        lower = channel['lower']

        signal = pd.Series(0, index=df.index)

        # 买入信号: 价格从下方突破上轨
        buy_condition = (df['close'] > upper) & (df['close'].shift(1) <= upper.shift(1))
        signal[buy_condition] = 1

        # 卖出信号: 价格从上方跌破下轨
        sell_condition = (df['close'] < lower) & (df['close'].shift(1) >= lower.shift(1))
        signal[sell_condition] = -1

        return signal


class TurtleSignal(Signal):
    """海龟交易法则信号

    系统1（短期）:
      - 入场: 价格突破20日最高价买入
      - 出场: 价格跌破10日最低价卖出

    系统2（长期）:
      - 入场: 价格突破55日最高价买入
      - 出场: 价格跌破20日最低价卖出
    """

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
    ):
        """
        Args:
            entry_period: 入场突破周期，默认20日
            exit_period: 出场跌破周期，默认10日
        """
        self.entry_period = entry_period
        self.exit_period = exit_period

    def generate(self, df: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=df.index)

        # 计算N日最高价和最低价
        highest = df['high'].rolling(window=self.entry_period).max()
        lowest = df['low'].rolling(window=self.exit_period).min()

        # 买入信号: 价格突破N日最高价（当天收盘价 > 昨天的N日最高价）
        buy_condition = (df['close'] > highest.shift(1))
        signal[buy_condition] = 1

        # 卖出信号: 价格跌破M日最低价（当天收盘价 < 昨天的M日最低价）
        sell_condition = (df['close'] < lowest.shift(1))
        signal[sell_condition] = -1

        return signal
