"""MA120 均值回归择时组合回测

规则 (15 个交易日调仓, 调仓日收盘价成交):
  d = (close - MA120) / MA120          # >0 表示在 MA120 上方
  单股目标权重 (上限 cap=10%):
    d <= -buy_full  → cap                                    (深跌满仓: 低于MA ≥10%)
    -buy_full < d < 0 → cap * (-d) / buy_full                (下方线性加仓)
    0 <= d < sell_zero → cap * (1 - d / sell_zero)           (上方线性减仓)
    d >= sell_zero → 0                                       (清仓: 高于MA ≥8%)
  全组合总权重 > 1 时等比缩放到 1, 否则余下留现金
成本: 佣金双边万3 + 印花税卖出千0.5 + 滑点万5 (可关闭)

输入 market_data: {code: DataFrame(date,open,high,low,close,volume)}
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.engine import COMMISSION, SLIPPAGE, STAMP_TAX, market_days

MA_WINDOW = 120
MA_VOL_LOOKBACK = 500
MA_VOL_KEEP = 0.5              # 取低波动前 50%
LOT = 100


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


@dataclass
class WeightRecord:
    date: pd.Timestamp
    weights: Dict[str, float]
    cash_weight: float


@dataclass
class MeanReversionStrategy:
    market_data: Dict[str, pd.DataFrame]
    screener: Callable[[pd.Timestamp], List[str]]          # → 候选 codes
    initial_capital: float = 500_000.0
    cap: float = 0.10
    buy_full: float = 0.10
    sell_zero: float = 0.08
    rebalance_step: int = 15
    with_cost: bool = True
    use_timing: bool = True      # False = 池内等权满仓 (消融: 无择时)
    min_weight: float = 0.005

    weights_history: List[WeightRecord] = field(default_factory=list)
    trades: List[dict] = field(default_factory=list)

    # ── 主循环 ──
    def run(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        self.cash = self.initial_capital
        self.shares: Dict[str, float] = {}
        self.weights_history = []
        self.trades = []
        self._equity: List[dict] = []

        all_days = market_days(self.market_data, start, end)
        rb_dates = make_rebalance_dates(all_days, start, end, self.rebalance_step)
        rb_set = set(rb_dates)

        for day in all_days:
            if day in rb_set:
                self._rebalance(day)
            self._mark(day)
        eq = pd.DataFrame(self._equity).set_index('date')
        eq['exposure'] = eq['invested'] / eq['nav']
        return eq

    # ── 调仓 ──
    def _rebalance(self, day: pd.Timestamp):
        # 1. 候选池
        candidates = self.screener(day)
        if candidates:
            # 2. 目标权重
            weights = {}
            for code in candidates:
                df = self.market_data.get(code)
                if df is None:
                    continue
                hist = df[df.index <= day]
                if len(hist) < MA_WINDOW + 1:
                    continue
                close = float(hist['close'].iloc[-1])
                if self.use_timing:
                    ma120 = float(hist['close'].tail(MA_WINDOW).mean())
                    if ma120 <= 0:
                        continue
                    dev = (close - ma120) / ma120
                    w = target_weight(dev, self.cap, self.buy_full, self.sell_zero)
                else:
                    w = 1.0 / len(candidates)
                if w >= self.min_weight:
                    weights[code] = w
            # 3. 归一化
            total = sum(weights.values())
            if total > 1.0:
                weights = {c: w / total for c, w in weights.items()}
            self._apply_weights(day, weights)
            self.weights_history.append(WeightRecord(
                date=day, weights=weights.copy(),
                cash_weight=max(0.0, 1.0 - sum(weights.values())),
            ))

    def _apply_weights(self, day: pd.Timestamp, target: Dict[str, float]):
        nav = self.cash + sum(
            self.shares.get(c, 0) * float(self.market_data[c].loc[day, 'close'])
            for c in self.shares
            if self.market_data.get(c) is not None and day in self.market_data[c].index
        )
        if nav <= 0:
            return
        trade_plan = {}
        for code, w in target.items():
            df = self.market_data[code]
            if day not in df.index:
                continue
            price = float(df.loc[day, 'close'])
            want_val = nav * w
            want_shares = int(want_val / price // LOT) * LOT
            cur = int(self.shares.get(code, 0))
            trade_plan[code] = (cur, want_shares, price)
        # 卖出不在目标中的持仓
        for code in list(self.shares):
            if code not in trade_plan:
                df = self.market_data[code]
                if day not in df.index:
                    continue
                price = float(df.loc[day, 'close'])
                trade_plan[code] = (int(self.shares[code]), 0, price)
        self._execute(day, trade_plan)

    def _execute(self, day: pd.Timestamp, plan: Dict[str, tuple]):
        # 先卖后买, 保证资金满额使用
        for code, (cur, want, price) in plan.items():
            delta = want - cur
            if delta == 0:
                continue
            if delta < 0:  # 卖
                sell_price = price * (1 - SLIPPAGE) if self.with_cost else price
                proceeds = abs(delta) * sell_price
                if self.with_cost:
                    proceeds -= proceeds * COMMISSION + proceeds * STAMP_TAX
                self.cash += proceeds
                self.shares[code] = want
                self.trades.append({'date': day, 'code': code, 'action': 'sell',
                                    'shares': abs(delta), 'price': sell_price})
            elif self.cash >= price * (1 + SLIPPAGE) * delta:
                buy_price = price * (1 + SLIPPAGE) if self.with_cost else price
                cost = delta * buy_price
                if self.with_cost:
                    cost += cost * COMMISSION
                if cost <= self.cash:
                    self.cash -= cost
                    self.shares[code] = want
                    self.trades.append({'date': day, 'code': code, 'action': 'buy',
                                        'shares': delta, 'price': buy_price})
        # 清理零持仓
        self.shares = {c: s for c, s in self.shares.items() if s > 0}

    # ── 净值 ──
    def _mark(self, day: pd.Timestamp):
        invested = 0.0
        for code, sh in self.shares.items():
            df = self.market_data.get(code)
            if df is not None and day in df.index:
                invested += sh * float(df.loc[day, 'close'])
        self._equity.append({
            'date': day,
            'nav': self.cash + invested,
            'invested': invested,
        })


def run_backtest(market_data: Dict[str, pd.DataFrame],
                 screener: Callable[[pd.Timestamp], List[str]],
                 start: pd.Timestamp, end: pd.Timestamp,
                 with_cost: bool = True, **kw) -> pd.DataFrame:
    bt = MeanReversionStrategy(market_data, screener, with_cost=with_cost, **kw)
    return bt.run(start, end), bt


__all__ = ['target_weight', 'ma120_lowvol_mask', 'rolling_ma120_vol',
           'make_rebalance_dates', 'MeanReversionStrategy', 'run_backtest']
