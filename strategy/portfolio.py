"""组合管理器 — 多标的持仓、权重、再平衡"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np


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
