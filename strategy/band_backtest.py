"""高息成长池 MA120 区间触发回测 (每日收盘触发)

规则 (与用户确认):
  - 入场: 股息率>3% 且 PE_TTM<20 且 股价 ≤ MA120×(1−12%)  → 首买目标仓位 5%
  - 补仓: 相对首买成本价再跌 10%/20% (按池内 120 日波动率中位数分档)
          → 补 5%, 单票累计上限 10%, 只补一次
  - 卖出: 股价 ≥ MA120×(1+10%) → 清仓
  - 无硬止损; 基本面仅在入场时校验
  - 多只同时触发且总目标权重 > 100% → 等比缩放到 100%, 余下留现金
  - 每日收盘检查, 收盘价成交 (下一日延续则按当日价)

成本口径 (与 run_dividend_backtest 一致):
  佣金双边万 3 (每笔最低 5 元) + 卖出印花税千 0.5 + 双边滑点万 5, 100 股整手

价格口径:
  前复权价 (_qfq.csv) 用于 MA120/净值; 股息率/PE 用不复权价, point-in-time
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from strategy.trading_costs import COMMISSION, MIN_FEE, SLIPPAGE, STAMP_TAX

MA_WINDOW = 120
LOT = 100


@dataclass
class BandBacktest:
    """每日收盘触发的区间回测器。

    screener: Callable[[pd.Timestamp], Dict[str, Dict]] 返回
              {code: {'div_yield','pe_ttm','dev','add_thr','pass'}} (pass=满足入场)
    """
    market_data: Dict[str, pd.DataFrame]          # {code: qfq close frame}
    screener: Callable[[pd.Timestamp], Dict[str, Dict]]
    initial_capital: float = 500_000.0
    first_weight: float = 0.05
    add_weight: float = 0.05
    cap: float = 0.10
    dev_buy: float = -0.12
    dev_sell: float = 0.10
    with_cost: bool = True
    min_weight: float = 0.005

    cash: float = 0.0
    shares: Dict[str, float] = field(default_factory=dict)
    cost: Dict[str, float] = field(default_factory=dict)     # 首买成本价(前复权)
    added: Dict[str, bool] = field(default_factory=dict)     # 是否已补仓
    add_thr: Dict[str, float] = field(default_factory=dict)  # 补仓档位 (-10/-20cm)
    trades: List[dict] = field(default_factory=list)
    weights_history: List[dict] = field(default_factory=list)
    _equity: List[dict] = field(default_factory=list)

    # ── 主循环 ──
    def run(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        self.cash = self.initial_capital
        self.shares = {}
        self.cost = {}
        self.added = {}
        self.add_thr = {}
        self.trades = []
        self.weights_history = []
        self._equity = []

        all_days = sorted({d for df in self.market_data.values() for d in df.index})
        all_days = [d for d in all_days if start <= d <= end]

        for day in all_days:
            self._daily(day)
        eq = pd.DataFrame(self._equity).set_index('date')
        eq['exposure'] = eq['invested'] / eq['nav']
        return eq

    # ── 每日处理 (事件驱动: 仅触发日交易) ──
    def _daily(self, day: pd.Timestamp):
        signals = self.screener(day) if callable(self.screener) else {}
        nav = self._mark(day)

        # 1. 卖出: dev >= +10% → 清仓
        to_sell = [c for c in list(self.shares)
                   if (d := self._dev(c, day)) is not None and d >= self.dev_sell]

        # 2. 补仓: 未补 且 相对成本再跌阈值 → 追加到 cap
        to_add = []
        for code in list(self.shares):
            if code in to_sell or self.added.get(code, False):
                continue
            price = self._price(code, day)
            cost0 = self.cost.get(code)
            thr = self.add_thr.get(code, 0.20)
            if price is None or cost0 is None or cost0 <= 0:
                continue
            if price <= cost0 * (1 - thr):
                to_add.append(code)

        # 3. 入场: 未持有 且 满足三条件 → 买入 first_weight
        to_enter = []
        for code, sig in signals.items():
            if code in self.shares or not sig.get('pass', False):
                continue
            if self._price(code, day) is None:
                continue
            to_enter.append(code)

        if not (to_sell or to_add or to_enter):
            return

        # 4. 目标权重: 仅触发标的 (未触发的既有持仓保持不动)
        targets = {}
        for c in to_sell:
            targets[c] = 0.0
        for c in to_add:
            targets[c] = self.cap
        for c in to_enter:
            targets[c] = self.first_weight

        # 5. 新增部分 (入场 + 补仓增量) 等比缩放, 使总目标 ≤ 100%
        invested_w = sum(
            self.shares.get(c, 0) * self._price(c, day)
            for c in list(self.shares) if self._price(c, day) is not None
        ) / nav if nav > 0 else 0.0
        scale = 1.0
        new_budget = 1.0 - invested_w
        new_demand = sum(
            (targets[c] - (self._target_weight(c) if c in self.shares and c not in to_sell else 0.0))
            for c in to_add) + sum(self.first_weight for c in to_enter)
        if new_demand > new_budget > 0:
            scale = new_budget / new_demand
        if scale < 1.0:
            targets = {c: w * scale for c, w in targets.items()}
        elif new_demand <= 0:
            targets = {c: w for c, w in targets.items() if targets[c] == 0.0}

        reasons = {}
        for c in to_sell:
            reasons[c] = f'卖出 dev={self._dev(c, day):.1%}'
        for c in to_add:
            reasons[c] = f'补仓(-{self.add_thr.get(c, 0.20):.0%} 成本{self.cost[c]:.2f})'
            self.added[c] = True
        for c in to_enter:
            sig = signals[c]
            reasons[c] = (f'入场 股息率{sig.get("div_yield", 0):.1%} '
                          f'PE{sig.get("pe_ttm", 0):.1f} '
                          f'偏差{sig.get("dev", 0):.1%}')
            self.add_thr[c] = sig.get('add_thr', 0.20)
            self.cost[c] = self._price(c, day)
            self.added[c] = False

        self._apply_weights(day, targets, reasons)

    def _target_weight(self, code: str) -> float:
        """已持仓目标: 已补 → cap, 未补 → first"""
        return self.cap if self.added.get(code, False) else self.first_weight

    # ── 价格/偏差辅助 ──
    def _price(self, code: str, day: pd.Timestamp) -> Optional[float]:
        df = self.market_data.get(code)
        if df is None or day not in df.index:
            return None
        return float(df.loc[day, 'close'])

    def _dev(self, code: str, day: pd.Timestamp) -> Optional[float]:
        df = self.market_data.get(code)
        if df is None:
            return None
        hist = df[df.index <= day]
        if len(hist) < MA_WINDOW + 1:
            return None
        close = float(hist['close'].iloc[-1])
        ma120 = float(hist['close'].tail(MA_WINDOW).mean())
        if ma120 <= 0:
            return None
        return (close - ma120) / ma120

    # ── 调仓执行 ──
    def _apply_weights(self, day: pd.Timestamp, target: Dict[str, float],
                       reasons: Optional[Dict[str, str]] = None):
        reasons = reasons or {}
        nav = self.cash + sum(
            self.shares.get(c, 0) * self._price(c, day)
            for c in list(self.shares) if self._price(c, day) is not None
        )
        if nav <= 0:
            return
        trade_plan = {}
        for code, w in target.items():
            price = self._price(code, day)
            if price is None:
                continue
            want_val = nav * w
            want_shares = int(want_val / price // LOT) * LOT
            cur = int(self.shares.get(code, 0))
            trade_plan[code] = (cur, want_shares, price)
        self._execute(day, trade_plan, reasons)

    def _execute(self, day: pd.Timestamp, plan: Dict[str, tuple],
                 reasons: Optional[Dict[str, str]] = None):
        reasons = reasons or {}
        # 先卖后买
        for code, (cur, want, price) in plan.items():
            delta = want - cur
            if delta == 0:
                continue
            if delta < 0:
                sell_price = price * (1 - SLIPPAGE) if self.with_cost else price
                proceeds = abs(delta) * sell_price
                if self.with_cost:
                    fee = max(proceeds * COMMISSION, MIN_FEE) + proceeds * STAMP_TAX
                    proceeds -= fee
                self.cash += proceeds
                self.shares[code] = want
                self.trades.append({'date': day, 'code': code, 'action': 'sell',
                                    'shares': abs(delta), 'price': sell_price,
                                    'reason': reasons.get(code, '')})
                if want == 0:
                    self.shares.pop(code, None)
                    self.cost.pop(code, None)
                    self.added.pop(code, None)
                    self.add_thr.pop(code, None)
        for code, (cur, want, price) in plan.items():
            delta = want - cur
            if delta <= 0:
                continue
            buy_price = price * (1 + SLIPPAGE) if self.with_cost else price
            cost = delta * buy_price
            if self.with_cost:
                cost += max(cost * COMMISSION, MIN_FEE)
            if cost <= self.cash:
                self.cash -= cost
                self.shares[code] = want
                self.trades.append({'date': day, 'code': code, 'action': 'buy',
                                    'shares': delta, 'price': buy_price,
                                    'reason': reasons.get(code, '')})
        self.shares = {c: s for c, s in self.shares.items() if s > 0}

    # ── 净值 ──
    def _mark(self, day: pd.Timestamp) -> float:
        invested = 0.0
        for code, sh in self.shares.items():
            price = self._price(code, day)
            if price is not None:
                invested += sh * price
        nav = self.cash + invested
        self._equity.append({'date': day, 'nav': nav, 'invested': invested})
        return nav


__all__ = ['BandBacktest', 'MA_WINDOW', 'LOT', 'COMMISSION', 'STAMP_TAX', 'SLIPPAGE']
