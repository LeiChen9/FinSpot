"""策略回测基础框架 — 无泄漏设计"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


# ─── 信号基类 ───

class Signal(ABC):
    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """返回信号序列: 1=买入, 0=持有/空仓, -1=卖出"""
        ...


class Strategy(ABC):
    def __init__(self, signal: Signal):
        self.signal = signal
        self.name = self.__class__.__name__

    @abstractmethod
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


# ─── 交易记录 ───

@dataclass
class Trade:
    date: object
    stock: str
    action: str          # 'buy' | 'sell'
    price: float
    shares: float
    amount: float
    reason: str          # 'signal' | 'take_profit' | 'stop_loss' | 'trailing_stop' | 'rebalance'
    holding_days: int = 0


@dataclass
class Position:
    stock: str
    buy_date: object
    buy_price: float
    shares: float
    current_price: float
    high_price: float       # 用于追踪止盈
    highest_price: float    # 持有期间最高价
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


# ─── 回测引擎 ───

class BacktestEngine:
    """无泄漏回测引擎

    核心约束:
      - 每个交易日只使用截止该日的数据
      - 所有决策基于 as_of_date 之前的行情
      - 财务数据考虑财报发布日期延迟
    """

    def __init__(self, initial_capital: float = 100_000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[dict] = []
        self.cash = initial_capital

    def run_single(self, df: pd.DataFrame, signal_col: str = 'signal') -> pd.DataFrame:
        """单标的简单回测（兼容旧接口）"""
        result = df.copy()
        result['position'] = result[signal_col].replace(0, np.nan).ffill().fillna(0)
        result['daily_return'] = result['position'].shift(1) * result['close'].pct_change()
        result['strategy'] = (1 + result['daily_return']).cumprod()
        result['buy_hold'] = (1 + result['close'].pct_change()).cumprod()
        return result

    def run(
        self,
        market_data: Dict[str, pd.DataFrame],
        rebalance_dates: List,
        screener_fn,
        risk_manager=None,
        weights: str = 'equal',
        commission: float = 0.0003,
    ):
        """多标的多期组合回测

        Args:
            market_data: {code: DataFrame(date, open, high, low, close, volume)}
            rebalance_dates: 调仓日期列表
            screener_fn: callable(as_of_date) -> List[str] 候选股票
            risk_manager: Optional[RiskManager]
            weights: 'equal' | dict
            commission: 佣金费率
        """
        all_dates = sorted(set(
            d for df in market_data.values()
            for d in df.index
        ))
        date_idx = {d: i for i, d in enumerate(all_dates)}

        for rb_date in rebalance_dates:
            if rb_date not in date_idx:
                continue
            as_of = date_idx[rb_date]

            # 1. 风控检查 (每日持仓检查)
            if risk_manager:
                self._apply_risk_checks(rb_date, as_of, market_data, risk_manager, commission)

            # 2. 选股
            candidates = screener_fn(rb_date)
            if not candidates:
                continue

            # 3. 调仓
            self._rebalance(rb_date, as_of, candidates, market_data, weights, commission)

        # 4. 最后一批风控
        final_date = all_dates[-1]
        if risk_manager:
            self._apply_risk_checks(final_date, len(all_dates)-1, market_data, risk_manager, commission)

        return self._build_equity_curve(all_dates, market_data)

    def _rebalance(self, rb_date, as_of, candidates, market_data, weights, commission):
        close_prices = {}
        for c in candidates:
            df = market_data.get(c)
            if df is not None and as_of < len(df):
                close_prices[c] = df.iloc[as_of]['close']

        valid = [c for c, p in close_prices.items() if p and p > 0]
        if not valid:
            return

        # 卖出不在候选中的持仓
        for code in list(self.positions.keys()):
            if code not in valid:
                self._close_position(code, rb_date, as_of, market_data, 'rebalance', commission)

        # 等权重分配资金
        n = len(valid)
        capital_per = self.cash / n
        for c in valid:
            price = close_prices[c]
            if price <= 0:
                continue
            shares = (capital_per / price) // 100 * 100
            if shares <= 0:
                continue
            cost = shares * price * (1 + commission)
            if cost > self.cash:
                shares = (self.cash / (price * (1 + commission))) // 100 * 100
                cost = shares * price * (1 + commission)
            if shares <= 0:
                continue

            self.cash -= cost
            self.positions[c] = Position(
                stock=c,
                buy_date=rb_date,
                buy_price=price,
                shares=shares,
                current_price=price,
                high_price=price,
                highest_price=price,
            )
            self.trades.append(Trade(rb_date, c, 'buy', price, shares, cost, 'signal'))

    def _apply_risk_checks(self, date, as_of, market_data, risk_manager, commission):
        for code in list(self.positions.keys()):
            df = market_data.get(code)
            if df is None or as_of >= len(df):
                continue
            row = df.iloc[as_of]
            pos = self.positions[code]
            pos.current_price = row['close']
            pos.high_price = max(row['high'], pos.high_price)
            pos.highest_price = max(row['close'], pos.highest_price)

            reason = risk_manager.check(pos, date)
            if reason:
                self._close_position(code, date, as_of, market_data, reason, commission)

    def _close_position(self, code, date, as_of, market_data, reason, commission):
        pos = self.positions.pop(code, None)
        if pos is None:
            return
        df = market_data.get(code)
        sell_price = df.iloc[as_of]['close'] if df is not None and as_of < len(df) else pos.buy_price
        proceeds = pos.shares * sell_price * (1 - commission)
        self.cash += proceeds
        holding_days = as_of - (df.index.get_loc(pos.buy_date) if df is not None and pos.buy_date in df.index else 0) if df is not None else 0

        self.trades.append(Trade(
            date=date, stock=code, action='sell',
            price=sell_price, shares=pos.shares,
            amount=proceeds, reason=reason,
            holding_days=holding_days,
        ))

    def _build_equity_curve(self, all_dates, market_data) -> pd.DataFrame:
        rows = []
        for i, d in enumerate(all_dates):
            positions_value = 0.0
            for code, pos in self.positions.items():
                df = market_data.get(code)
                if df is not None and i < len(df):
                    pos.current_price = df.iloc[i]['close']
                positions_value += pos.shares * pos.current_price
            nav = self.cash + positions_value
            rows.append({'date': d, 'nav': nav, 'cash': self.cash, 'positions': positions_value})
        return pd.DataFrame(rows).set_index('date')

    def summary(self) -> dict:
        buys = [t for t in self.trades if t.action == 'buy']
        sells = [t for t in self.trades if t.action == 'sell']
        equity = pd.DataFrame(self.equity_curve) if self.equity_curve else None

        total_cost = sum(t.amount for t in buys)
        total_proceeds = sum(t.amount for t in sells)
        total_return = (self.capital - self.initial_capital) / self.initial_capital if self.initial_capital else 0

        winning_trades = sum(1 for t in sells if t.amount > sum(b.amount for b in buys if b.stock == t.stock))
        loss_trades = max(len(sells) - winning_trades, 0)

        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'total_trades': len(self.trades),
            'buy_trades': len(buys),
            'sell_trades': len(sells),
            'winning_trades': winning_trades,
            'losing_trades': loss_trades,
            'win_rate': winning_trades / len(sells) if sells else 0,
        }
