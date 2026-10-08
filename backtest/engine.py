"""通用回测引擎机制: 组合记账、交易成本与交易日历。

策略(strategy/)基于这些原语编排各自的回测循环;
本模块不包含任何具体策略逻辑。
"""
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


# ── 交易成本 (A 股常规: 佣金万3双边、卖出印花税万5、滑点万5、单笔最低5元) ──
COMMISSION = 0.0003
STAMP_TAX = 0.0005
SLIPPAGE = 0.0005
MIN_FEE = 5.0


# ── 交易日历 ──

def market_days(market_data: Dict[str, pd.DataFrame], start, end):
    days = sorted({day for frame in market_data.values() for day in frame.index})
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    return [day for day in days if start <= day <= end]


# ── 交易成本计算 (佣金含最低 5 元, 加滑点与卖出印花税) ──

def buy_cost(amount: float) -> float:
    """买入 `amount`(股数×价格) 的交易成本。"""
    return amount * SLIPPAGE + max(amount * COMMISSION, MIN_FEE)


def sell_cost(amount: float) -> float:
    """卖出 `amount`(股数×价格) 的交易成本。"""
    return amount * SLIPPAGE + max(amount * COMMISSION, MIN_FEE) + amount * STAMP_TAX


# ── 仓位计算 ──

def whole_lot_shares(amount: float, price: float, lot: int = 100) -> int:
    return int(amount / price / lot) * lot if price > 0 else 0


def affordable_shares(cash: float, price: float, lot: int = 100) -> int:
    """在 cash 预算内, 扣除买入成本后能买的最大整手股数。"""
    if price <= 0:
        return 0
    shares = whole_lot_shares(cash, price, lot)
    while shares > 0 and shares * price + buy_cost(shares * price) > cash:
        shares -= lot
    return shares


# ── 分红与买入持有 ──

def dividends_by_day(index: pd.Index, divs: pd.Series) -> dict:
    """把每份分红 Series 对齐到交易日: {交易日: 当日每份分红金额}。"""
    out = {}
    for day, amount in divs.sort_index().items():
        pos = index.searchsorted(pd.Timestamp(day).normalize())
        if pos < len(index):
            key = index[pos]
            out[key] = out.get(key, 0.0) + float(amount)
    return out


def buy_and_hold_nav(close: pd.Series, dividends: dict = None,
                     initial_cash: float = 100_000.0) -> pd.Series:
    """首日全额买入并持有; 收到分红则当日现金再投 (金额口径同策略)。"""
    dividends = dividends or {}
    cash, shares, values = initial_cash, 0.0, []
    for dt, price in close.items():
        if shares > 0 and dividends.get(dt):
            divcash = shares * dividends[dt]
            n = affordable_shares(divcash, price)
            if n > 0:
                cash += divcash - (n * price + buy_cost(n * price))
                shares += n
        elif shares == 0:
            n = affordable_shares(cash, price)
            if n > 0:
                cash -= n * price + buy_cost(n * price)
                shares = n
        values.append(cash + shares * price)
    return pd.Series(values, index=close.index)


# ── 组合记账 ──

@dataclass
class Holding:
    code: str
    name: str = ''
    shares: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    weight: float = 0.0
    buy_date: object = None
    highest_price: float = 0.0


class Portfolio:
    """组合管理器，记录所有持仓和交易"""

    def __init__(self, initial_cash: float = 100_000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings: Dict[str, Holding] = {}
        self.trade_log: List[dict] = []
        self.nav_history: List[dict] = []

    def value(self) -> float:
        return self.cash + sum(h.market_value for h in self.holdings.values())

    def update_prices(self, price_map: Dict[str, float]):
        for code, price in price_map.items():
            if code in self.holdings:
                h = self.holdings[code]
                h.current_price = price
                h.market_value = h.shares * price
                h.unrealized_pnl = h.market_value - h.shares * h.avg_cost
                h.unrealized_pnl_pct = h.unrealized_pnl / (h.shares * h.avg_cost) if h.avg_cost > 0 else 0.0
                h.highest_price = max(h.highest_price, price)

    def update_weights(self):
        total = self.value()
        if total <= 0:
            return
        for h in self.holdings.values():
            h.weight = h.market_value / total

    def buy(self, code: str, price: float, shares: float, date=None, name='', commission=0.0003) -> bool:
        cost = shares * price * (1 + commission)
        if cost > self.cash:
            return False
        self.cash -= cost

        if code in self.holdings:
            h = self.holdings[code]
            total_shares = h.shares + shares
            total_cost = h.shares * h.avg_cost + shares * price
            h.avg_cost = total_cost / total_shares
            h.shares = total_shares
        else:
            self.holdings[code] = Holding(
                code=code, name=name, shares=shares,
                avg_cost=price, current_price=price,
                market_value=shares * price, buy_date=date,
                highest_price=price,
            )

        self.trade_log.append({
            'date': date, 'code': code, 'action': 'buy',
            'price': price, 'shares': shares, 'amount': cost,
            'fees': shares * price * commission,
            'reason': 'signal',
        })
        return True

    def sell(
        self,
        code: str,
        price: float,
        shares: float = None,
        date=None,
        reason='signal',
        commission=0.0003,
        stamp_tax=0.0,
    ) -> float:
        """卖出持仓，卖出费用包括佣金和印花税。"""
        if code not in self.holdings:
            return 0.0
        h = self.holdings[code]
        shares = shares or h.shares
        shares = min(shares, h.shares)
        gross_amount = shares * price
        fees = gross_amount * (commission + stamp_tax)
        proceeds = gross_amount - fees
        self.cash += proceeds

        h.shares -= shares
        h.current_price = price
        h.market_value = h.shares * price if h.shares > 0 else 0.0

        self.trade_log.append({
            'date': date, 'code': code, 'action': 'sell',
            'price': price, 'shares': shares, 'amount': proceeds,
            'fees': fees,
            'reason': reason,
        })

        if h.shares <= 0:
            del self.holdings[code]
        return proceeds

    def snapshot(self, date) -> dict:
        return {
            'date': date,
            'nav': self.value(),
            'cash': self.cash,
            'positions_value': sum(h.market_value for h in self.holdings.values()),
            'num_positions': len(self.holdings),
        }
