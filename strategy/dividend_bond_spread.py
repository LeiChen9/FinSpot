"""红利低波 ETF 股息率-10Y国债利差阈值定投策略。

规则 (与用户确认稿一致):
  spread = TTM股息率(近4次分红/现价) − 10Y国债利率(百分点)
  spread > 3%   → 加仓当前持仓市值的 40%(现金)
  spread > 1.5% → 加仓当前持仓市值的 20%
  1%~1.5%       → 不动
  spread < 1%   → 清仓
  首次建仓等信号首次 >1.5%, 固定买入 FIRST_BUY
  分红日现金分红当日全部买回; 本金 10 万, 允许空仓
  收盘算信号, 次一交易日收盘成交

口径:
  价格用不复权收盘价 (现金分红另计, 不重复计息); 股息率 = 名义每份分红/现价。
  成本含佣金(最低5元)+滑点(双边)+卖出印花税, 100 股整手 (见 backtest.engine)。

可选项 (默认关闭, 用于出风险研究):
  cap_mode='signal'  → 以 NAV 目标上限加仓(>3% 满仓 cap_mult, 1.5~3% 80%)
  trend_ma           → 跌破 200 日均线时 cap_mult 降至 1/3
  vol_target         → 按目标波动率倒数缩放 cap_mult
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import pandas as pd

from backtest.engine import (
    affordable_shares, buy_cost, dividends_by_day, sell_cost, whole_lot_shares,
)
from dataload.etf import fetch_data, fetch_etf_dividends
from dataload.readers import load_10y

CODE = '159307'
SYMBOL = 'sz159307'
INITIAL_CAPITAL = 100_000.0
FIRST_BUY = 10_000.0
LOT = 100


@dataclass
class DividendBondSpreadStrategy:
    start: datetime
    end: datetime
    capital: float = INITIAL_CAPITAL
    price_df: Optional[pd.DataFrame] = None
    divs: Optional[pd.Series] = None
    r10: Optional[pd.Series] = None
    with_cost: bool = True
    cap_mode: str = 'legacy'
    trend_ma: bool = False
    vol_target: Optional[float] = None
    trades: List[dict] = field(default_factory=list)
    nav: List[dict] = field(default_factory=list)

    def run(self) -> pd.DataFrame:
        self.trades = []
        self.nav = []
        px = self.price_df if self.price_df is not None else \
            fetch_data(CODE, self.start, self.end, adjust='')
        if px is None or px.empty:
            raise ValueError(f'{CODE}: 无法获取 {self.start:%Y-%m-%d}~{self.end:%Y-%m-%d} 行情')
        divs = self.divs if self.divs is not None else fetch_etf_dividends(SYMBOL)
        r10 = self.r10 if self.r10 is not None else load_10y()['中国国债收益率10年']
        r10 = r10.reindex(px.index, method='ffill')

        per = divs.sort_index()
        ttm = per.rolling(4).sum().reindex(px.index, method='ffill')
        cnt = pd.Series(range(1, len(per) + 1), index=per.index).reindex(px.index, method='ffill')
        spread_series = (ttm.where(cnt >= 4) / px['close'] * 100) - r10

        div_on = dividends_by_day(px.index, per)

        b_cost = buy_cost if self.with_cost else (lambda a: 0.0)
        s_cost = sell_cost if self.with_cost else (lambda a: 0.0)
        lot_shares = affordable_shares if self.with_cost else whole_lot_shares
        closes = px['close']

        cash = self.capital
        shares = 0.0
        entered = False
        pending = None

        for i, (dt, row) in enumerate(px.iterrows()):
            close = float(row['close'])

            dividend = div_on.get(dt)
            if dividend and shares > 0:
                divcash = shares * dividend
                self.trades.append({'date': dt, 'side': 'dividend', 'amount': divcash})
                n = lot_shares(divcash, close, LOT)
                if n > 0:
                    cash += divcash - (n * close + b_cost(n * close))
                    shares += n
                    self.trades.append({'date': dt, 'side': 'buy(div)', 'shares': n, 'price': close})

            if pending is not None:
                side, amount = pending
                if side == 'sell' and shares > 0:
                    gross = shares * close
                    cash += gross - s_cost(gross)
                    self.trades.append({'date': dt, 'side': 'sell', 'shares': shares, 'price': close})
                    shares = 0.0
                elif side == 'buy':
                    n = lot_shares(min(amount, cash), close, LOT)
                    if n > 0:
                        cash -= n * close + b_cost(n * close)
                        shares += n
                        self.trades.append({'date': dt, 'side': 'buy', 'shares': n, 'price': close})
                pending = None

            nav_now = cash + shares * close

            cap_mult = 1.0
            if self.trend_ma and i >= 200 and close < closes.iloc[i - 200:i].mean():
                cap_mult *= 1 / 3
            if self.vol_target and i >= 63:
                rv = closes.iloc[i - 63:i].pct_change().std() * (252 ** 0.5)
                if rv > 0:
                    cap_mult *= min(self.vol_target / rv, 1.0)

            if cap_mult < 1 and shares > 0:
                target_val = nav_now * cap_mult
                if shares * close > target_val * 1.05:
                    n_trim = int((shares * close - target_val) / close // LOT) * LOT
                    if n_trim > 0:
                        gross = n_trim * close
                        cash += gross - s_cost(gross)
                        shares -= n_trim
                        self.trades.append({'date': dt, 'side': 'trim', 'shares': n_trim, 'price': close})

            spread = spread_series.loc[dt]
            if spread == spread:
                if spread < 1.0:
                    entered = False
                    if shares > 0:
                        pending = ('sell', None)
                elif spread > 3.0 and cash > 0 and entered:
                    if self.cap_mode == 'legacy':
                        amount = shares * close * 0.40 * cap_mult
                    else:
                        amount = max(nav_now * cap_mult - shares * close, 0.0)
                    if amount > 0:
                        pending = ('buy', min(amount, cash))
                elif spread > 1.5 and cash > 0:
                    if not entered:
                        pending = ('buy', min(FIRST_BUY, cash))
                        entered = True
                    else:
                        if self.cap_mode == 'legacy':
                            amount = shares * close * 0.20 * cap_mult
                        else:
                            amount = max(nav_now * cap_mult * 0.80 - shares * close, 0.0)
                        if amount > 0:
                            pending = ('buy', min(amount, cash))

            self.nav.append({'date': dt, 'nav': nav_now, 'cash': cash,
                             'shares': shares, 'spread': float(spread)})

        return pd.DataFrame(self.nav).set_index('date')


__all__ = ['DividendBondSpreadStrategy', 'INITIAL_CAPITAL', 'FIRST_BUY', 'LOT', 'CODE', 'SYMBOL']
