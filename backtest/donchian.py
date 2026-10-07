"""高息价值池 × 唐奇安 20/10 突破策略

规则:
  1. 每月首个交易日用 value_screener 构建股票池 (as_of 当日, point-in-time)
  2. 池内成员: 收盘价突破 20 日最高(m1 日) -> 买入至目标权重
  3. 持仓: 收盘价跌破 10 日最低(m1 日) -> 清仓 (唐奇安离场)
  4. 月度再平衡(当月首个交易日): 跌出池 -> 清仓; 权重 > 目标 -> 减持回目标
  5. 成本: 买卖佣金万3, 卖出印花税万5
"""
from typing import Dict, Optional

import pandas as pd

from backtest.portfolio import Portfolio


class DonchianValueStrategy:
    def __init__(
        self,
        screener,
        initial_capital: float = 1_000_000,
        value_pct: float = 0.05,
        entry_period: int = 20,
        exit_period: int = 10,
        commission: float = 0.0003,
        stamp_tax: float = 0.0005,
        stock_names: Optional[Dict[str, str]] = None,
    ):
        self.screener = screener
        self.value_pct = value_pct
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.stock_names = stock_names or {}

        self.portfolio = Portfolio(initial_cash=initial_capital)
        self.last_pool: set = set()
        self._dc: Dict[str, Dict[str, pd.Series]] = {}

    def _channels(self, code: str, df: pd.DataFrame):
        if code not in self._dc:
            self._dc[code] = {
                'dc_high': df['high'].rolling(self.entry_period).max().shift(1),
                'dc_low': df['low'].rolling(self.exit_period).min().shift(1),
            }
        return self._dc[code]

    def _price(self, df: pd.DataFrame, today: pd.Timestamp) -> Optional[float]:
        if df is None or today not in df.index:
            return None
        return float(df.loc[today, 'close'])

    def run(
        self,
        market_data: Dict[str, pd.DataFrame],
        start_date: str = '2023-01-01',
        end_date: str = '2026-08-22',
    ) -> pd.DataFrame:
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        all_dates = sorted({d for df in market_data.values() for d in df.index})
        trading_days = [d for d in all_dates if start <= d <= end]
        if not trading_days:
            raise ValueError('没有交易日数据')

        last_month = None
        for today in trading_days:
            month = (today.year, today.month)

            # 每月首个交易日: 更新池 + 月度再平衡
            is_monthly = month != last_month
            if is_monthly:
                last_month = month
                self.last_pool = {
                    c for c, d in self.screener(today).items()
                    if d['pass'] and c in market_data
                }

            # 更新持仓价格
            price_map = {}
            for code in list(self.portfolio.holdings.keys()):
                p = self._price(market_data.get(code), today)
                if p is not None:
                    price_map[code] = p
            self.portfolio.update_prices(price_map)
            self.portfolio.update_weights()

            # 唐奇安离场: 跌破 10 日最低
            for code in list(self.portfolio.holdings.keys()):
                df = market_data.get(code)
                p = self._price(df, today)
                if p is None:
                    continue
                dc_low = self._channels(code, df)['dc_low'].get(today)
                if dc_low is not None and p < dc_low:
                    self.portfolio.sell(code, p, date=today, reason='donchian_exit',
                                        commission=self.commission, stamp_tax=self.stamp_tax)

            # 月度再平衡: 跌出池清仓 + 权重回落至目标
            if is_monthly:
                for code in list(self.portfolio.holdings.keys()):
                    if code not in self.last_pool:
                        p = self._price(market_data.get(code), today)
                        if p is not None:
                            self.portfolio.sell(code, p, date=today, reason='pool_exit',
                                                commission=self.commission, stamp_tax=self.stamp_tax)

                nav = self.portfolio.value()
                for code in list(self.portfolio.holdings.keys()):
                    h = self.portfolio.holdings[code]
                    if h.market_value <= nav * self.value_pct * 1.001:
                        continue
                    target_val = self.value_pct * nav
                    delta = h.market_value - target_val
                    shares = int(delta / h.current_price / 100) * 100
                    if shares > 0:
                        self.portfolio.sell(code, h.current_price, shares, date=today,
                                            reason='rebalance_trim',
                                            commission=self.commission, stamp_tax=self.stamp_tax)

            # 唐奇安入场: 池内成员突破 20 日最高 -> 买入至目标权重
            for code in self.last_pool:
                if code in self.portfolio.holdings:
                    continue
                df = market_data.get(code)
                p = self._price(df, today)
                if p is None:
                    continue
                dc_high = self._channels(code, df)['dc_high'].get(today)
                if dc_high is None or p <= dc_high:
                    continue
                nav = self.portfolio.value()
                shares = int(nav * self.value_pct / p / 100) * 100
                if shares > 0:
                    self.portfolio.buy(code, p, shares, date=today,
                                       name=self.stock_names.get(code, f'股票{code}'),
                                       commission=self.commission)

            self.portfolio.nav_history.append(self.portfolio.snapshot(today))

        return pd.DataFrame(self.portfolio.nav_history).set_index('date')