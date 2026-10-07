"""成长股动量策略: 年收益 >10% 且价格处于近一年下半区的股票, 月初(1/4/7/10月)调仓。

仓位规则:
  - 每笔 10 万, 最多 10 只
  - 止损 -20% (按 0.999 卖出系数)
  - 个股翻倍后允许加仓一次 (每股最多 2 批)
  - 买入按 1.0003 成本系数 (佣金万3)
"""
import os
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from common.paths import DATA_DIR

POSITION_SIZE = 100_000.0
MAX_POSITIONS = 10
MAX_PER_STOCK = 2
STOP_LOSS = -0.20
TAKE_PROFIT = 1.00


def load_all_market_data(start_date="2020-01-01", end_date="2026-08-31", data_dir=DATA_DIR):
    """加载全 A (graham_universe) 的前复权行情, 至少 60 根 K。"""
    meta_path = os.path.join(str(data_dir), "meta", "graham_universe.csv")
    universe = pd.read_csv(meta_path, dtype={"code": str})
    codes = universe["code"].tolist()
    market_data = {}
    for code in codes:
        path = os.path.join(str(data_dir), f"{code}_qfq.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, index_col="date", parse_dates=True)
            if df.empty or len(df) < 60:
                continue
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            market_data[code] = df
        except Exception:
            continue
    print(f"加载 {len(market_data)} 只股票")
    return market_data


def precompute_indicators(market_data):
    """年收益 (252 日) 与近一年价格分位。"""
    indicators = {}
    for code, df in market_data.items():
        if len(df) < 60:
            continue
        df_ind = pd.DataFrame(index=df.index)
        df_ind["year_return"] = df["close"].pct_change(252)
        window = min(252, len(df))
        df_ind["price_rank"] = df["close"].rolling(window).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
        )
        indicators[code] = df_ind
    return indicators


def screen_stocks(as_of_date, market_data, indicators) -> List[str]:
    """年收益 >10% 且价格分位 ≤50% 的候选 (剔除代码 <1000 的指数)。"""
    as_of = pd.Timestamp(as_of_date)
    candidates = []
    for code, df in market_data.items():
        df_up_to = df[df.index <= as_of]
        if len(df_up_to) < 60:
            continue
        ind = indicators.get(code)
        if ind is None:
            continue
        ind_up_to = ind[ind.index <= as_of]
        if ind_up_to.empty:
            continue
        year_return = ind_up_to["year_return"].iloc[-1]
        if pd.isna(year_return) or year_return < 0.1:
            continue
        price_rank = ind_up_to["price_rank"].iloc[-1]
        if pd.isna(price_rank) or price_rank > 0.50:
            continue
        code_num = int(code)
        if code_num < 1000:
            continue
        candidates.append(code)
    return candidates


@dataclass
class Position:
    code: str
    buy_date: str
    buy_price: float
    shares: int
    shares_count: int = 1


@dataclass
class Trade:
    date: str
    code: str
    action: str
    price: float
    shares: int
    amount: float
    reason: str


class GrowthStockBacktest:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve = []

    def get_price(self, code, date, market_data):
        df = market_data.get(code)
        if df is None:
            return 0
        if date in df.index:
            return df.loc[date, "close"]
        mask = df.index <= date
        if mask.any():
            return df.loc[mask, "close"].iloc[-1]
        return 0

    def check_stop_loss(self, date, market_data):
        for code in list(self.positions.keys()):
            pos = self.positions[code]
            price = self.get_price(code, date, market_data)
            if price <= 0:
                continue
            ret = (price - pos.buy_price) / pos.buy_price
            if ret <= STOP_LOSS:
                proceeds = pos.shares * price * 0.999
                self.cash += proceeds
                self.trades.append(Trade(date, code, "sell", price, pos.shares, proceeds, "stop_loss"))
                del self.positions[code]

    def check_take_profit(self, date, market_data):
        for code in list(self.positions.keys()):
            pos = self.positions[code]
            if pos.shares_count >= MAX_PER_STOCK:
                continue
            price = self.get_price(code, date, market_data)
            if price <= 0:
                continue
            ret = (price - pos.buy_price) / pos.buy_price
            if ret >= TAKE_PROFIT:
                add_amount = min(POSITION_SIZE, self.cash)
                add_shares = int(add_amount / price / 100) * 100
                if add_shares <= 0:
                    continue
                cost = add_shares * price * 1.0003
                if cost > self.cash:
                    continue
                total_shares = pos.shares + add_shares
                avg_price = (pos.shares * pos.buy_price + add_shares * price) / total_shares
                pos.shares = total_shares
                pos.buy_price = avg_price
                pos.shares_count += 1
                self.cash -= cost
                self.trades.append(Trade(date, code, "buy", price, add_shares, cost, "add_position"))

    def buy_new(self, code, date, market_data):
        if code in self.positions:
            return
        if len(self.positions) >= MAX_POSITIONS:
            return
        price = self.get_price(code, date, market_data)
        if price <= 0:
            return
        buy_amount = min(POSITION_SIZE, self.cash)
        shares = int(buy_amount / price / 100) * 100
        if shares <= 0:
            return
        cost = shares * price * 1.0003
        if cost > self.cash:
            return
        self.positions[code] = Position(code, date, price, shares)
        self.cash -= cost
        self.trades.append(Trade(date, code, "buy", price, shares, cost, "new_position"))

    def rebalance(self, date, candidates, market_data):
        self.check_stop_loss(date, market_data)
        self.check_take_profit(date, market_data)
        for code in candidates:
            if code not in self.positions:
                self.buy_new(code, date, market_data)

    def calc_nav(self, date, market_data):
        pos_value = 0
        for code, pos in self.positions.items():
            price = self.get_price(code, date, market_data)
            pos_value += pos.shares * price
        nav = self.cash + pos_value
        self.equity_curve.append({"date": date, "nav": nav})
        return nav


def get_rebalance_dates(start="2020-01-01", end="2026-08-31") -> List[str]:
    """1/4/7/10 月首个工作日。"""
    dates = pd.date_range(start=start, end=end, freq="BMS")
    return [d.strftime("%Y-%m-%d") for d in dates if d.month in [1, 4, 7, 10]]
