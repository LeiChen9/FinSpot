"""双底形态突破策略"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from backtest.engine import Portfolio, whole_lot_shares, market_days


def find_pivots(close, left=3, right=3):
    high = np.zeros(len(close), dtype=bool)
    low = np.zeros(len(close), dtype=bool)
    for index in range(left, len(close) - right):
        window = close[index - left:index + right + 1]
        value = close[index]
        if value == window.max() and (window == value).sum() == 1:
            high[index] = True
        if value == window.min() and (window == value).sum() == 1:
            low[index] = True
    return high, low


def detect_double_bottom(close, pivot_left=3, pivot_right=3,
                         bottom_tolerance=0.03, min_rebound=0.05,
                         min_separation=5, max_separation=60,
                         breakout_buffer=0.005):
    close = np.asarray(close, dtype=float)
    highs, lows = find_pivots(close, left=pivot_left, right=pivot_right)
    high_points, low_points = np.where(highs)[0], np.where(lows)[0]
    signals = []
    for first_index, first in enumerate(low_points):
        for second in low_points[first_index + 1:]:
            distance = second - first
            if distance < min_separation:
                continue
            if distance > max_separation:
                break
            if abs(close[first] - close[second]) / close[first] > bottom_tolerance:
                continue
            peaks = high_points[(high_points > first) & (high_points < second)]
            if len(peaks) == 0:
                continue
            peak = peaks[np.argmax(close[peaks])]
            bottom = max(close[first], close[second])
            if (close[peak] - bottom) / bottom < min_rebound:
                continue
            neckline = close[peak]
            first_breakout = second + pivot_right + 1
            for breakout in range(first_breakout, len(close)):
                if close[breakout] > neckline * (1 + breakout_buffer):
                    signals.append({
                        "pattern_start": first, "bottom1": first,
                        "middle_peak": peak, "bottom2": second,
                        "neckline": neckline, "breakout": breakout,
                        "breakout_price": close[breakout],
                        "breakout_level": neckline * (1 + breakout_buffer),
                    })
                    break
    return pd.DataFrame(signals)


class DoubleBottomStrategy:
    """双底形态突破策略

    策略规则：
    1. 买入：价格突破双底形态颈线位
    2. 卖出：价格跌破颈线位
    3. 加仓：突破后每上涨 add_threshold_pct 加一次仓
    4. 最大持仓：每只股票最大持仓 10%
    5. 初始头寸：每只股票初始头寸 5%
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        initial_position_pct: float = 0.05,
        max_position_pct: float = 0.10,
        add_position_pct: float = 0.005,
        add_threshold_pct: float = 0.05,
        max_add_times: int = 3,
        pivot_left: int = 3,
        pivot_right: int = 3,
        bottom_tolerance: float = 0.03,
        min_rebound: float = 0.05,
        min_separation: int = 5,
        max_separation: int = 60,
        breakout_buffer: float = 0.005,
        commission: float = 0.0003,
        stamp_tax: float = 0.0005,
        stock_names: Optional[Dict[str, str]] = None,
    ):
        self.initial_capital = initial_capital
        self.initial_position_pct = initial_position_pct
        self.max_position_pct = max_position_pct
        self.add_position_pct = add_position_pct
        self.add_threshold_pct = add_threshold_pct
        self.max_add_times = max_add_times
        self.pivot_left = pivot_left
        self.pivot_right = pivot_right
        self.bottom_tolerance = bottom_tolerance
        self.min_rebound = min_rebound
        self.min_separation = min_separation
        self.max_separation = max_separation
        self.breakout_buffer = breakout_buffer
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.stock_names = stock_names or {}

        self.portfolio = Portfolio(initial_cash=initial_capital)

        # 每只股票的加仓信息: {code: {add_count, last_add_price, neckline}}
        self.position_info: Dict[str, dict] = {}

        # 预计算的信号: {code: DataFrame of patterns}
        self._signals: Dict[str, pd.DataFrame] = {}

    def _generate_signals(self, market_data: Dict[str, pd.DataFrame]):
        """为所有股票生成双底信号"""
        for code, df in market_data.items():
            if len(df) < 30:
                continue
            patterns = detect_double_bottom(
                df['close'].values,
                pivot_left=self.pivot_left,
                pivot_right=self.pivot_right,
                bottom_tolerance=self.bottom_tolerance,
                min_rebound=self.min_rebound,
                min_separation=self.min_separation,
                max_separation=self.max_separation,
                breakout_buffer=self.breakout_buffer,
            )
            if not patterns.empty:
                self._signals[code] = patterns

    def _calculate_position_size(self, price: float, current_nav: float) -> int:
        """计算买入股数（100股整数倍）"""
        amount = current_nav * self.initial_position_pct
        shares = whole_lot_shares(amount, price)
        max_shares = whole_lot_shares(current_nav * self.max_position_pct, price)
        return min(shares, max_shares)

    def _calculate_add_size(self, price: float, current_nav: float) -> int:
        """计算加仓股数"""
        amount = current_nav * self.initial_position_pct * self.add_position_pct
        return whole_lot_shares(amount, price)

    def _should_add(self, code: str, current_price: float) -> bool:
        """判断是否应该加仓"""
        info = self.position_info.get(code)
        if not info:
            return False

        if info['add_count'] >= self.max_add_times:
            return False

        if info['last_add_price'] <= 0:
            return False

        pct_increase = (current_price - info['last_add_price']) / info['last_add_price']
        return pct_increase >= self.add_threshold_pct

    def run(
        self,
        market_data: Dict[str, pd.DataFrame],
        start_date: str = '2023-01-01',
        end_date: str = '2026-08-22',
    ) -> pd.DataFrame:
        """运行回测"""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        trading_days = market_days(market_data, start, end)

        if not trading_days:
            raise ValueError("没有交易日数据")

        print(f"回测期间: {start_date} ~ {end_date}, 共 {len(trading_days)} 个交易日")

        self._generate_signals(market_data)

        for i, today in enumerate(trading_days):
            # 更新持仓价格
            price_map = {}
            for code in list(self.portfolio.holdings.keys()):
                df = market_data.get(code)
                if df is not None and today in df.index:
                    price_map[code] = df.loc[today, 'close']

            self.portfolio.update_prices(price_map)
            self.portfolio.update_weights()

            current_nav = self.portfolio.value()

            # 检查卖出（跌破 neckline）
            for code in list(self.portfolio.holdings.keys()):
                info = self.position_info.get(code)
                if not info:
                    continue

                df = market_data.get(code)
                if df is None or today not in df.index:
                    continue

                price = df.loc[today, 'close']
                neckline = info['neckline']

                if price < neckline:
                    self.portfolio.sell(code, price, date=today,
                                        reason='neckline_break',
                                        commission=self.commission,
                                        stamp_tax=self.stamp_tax)
                    del self.position_info[code]

            # 检查买入（突破颈线）
            for code, df in market_data.items():
                if code in self.portfolio.holdings:
                    continue

                patterns = self._signals.get(code)
                if patterns is None:
                    continue

                if today not in df.index:
                    continue

                # 找到今天突破的信号
                today_idx = df.index.get_loc(today)
                breakout_rows = patterns[patterns['breakout'] == today_idx]

                if breakout_rows.empty:
                    continue

                row = breakout_rows.iloc[0]
                price = df.loc[today, 'close']

                shares = self._calculate_position_size(price, current_nav)
                if shares > 0:
                    stock_name = self.stock_names.get(code, f'股票{code}')
                    self.portfolio.buy(code, price, shares, date=today,
                                        name=stock_name, commission=self.commission)

                    self.position_info[code] = {
                        'neckline': row['neckline'],
                        'add_count': 0,
                        'last_add_price': price,
                    }

            # 检查加仓
            for code in list(self.portfolio.holdings.keys()):
                current_holding = self.portfolio.holdings.get(code)
                if not current_holding:
                    continue

                df = market_data.get(code)
                if df is None or today not in df.index:
                    continue

                price = df.loc[today, 'close']

                if self._should_add(code, price):
                    add_shares = self._calculate_add_size(price, current_nav)
                    if add_shares > 0:
                        stock_name = self.stock_names.get(code, f'股票{code}')
                        self.portfolio.buy(code, price, add_shares, date=today,
                                            name=stock_name, commission=self.commission)

                        info = self.position_info.get(code, {})
                        info['add_count'] = info.get('add_count', 0) + 1
                        info['last_add_price'] = price
                        self.position_info[code] = info

            # 记录净值
            self.portfolio.nav_history.append(self.portfolio.snapshot(today))

            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{len(trading_days)}, 净值: {current_nav:,.0f}, "
                      f"持仓: {len(self.portfolio.holdings)}")

        print(f"回测完成! 最终净值: {self.portfolio.value():,.0f}, "
              f"总交易: {len(self.portfolio.trade_log)} 笔")

        return pd.DataFrame(self.portfolio.nav_history).set_index('date')

    def liquidate_all(self, market_data: Dict[str, pd.DataFrame], date):
        """按各标的截至 ``date`` 的最后收盘价清仓，返回清仓后的现金。"""
        for code in list(self.portfolio.holdings.keys()):
            df = market_data.get(code)
            if df is None:
                continue
            available = df.loc[:pd.Timestamp(date)]
            if available.empty:
                continue
            sell_date = available.index[-1]
            price = available.iloc[-1]['close']
            self.portfolio.sell(code, price, date=sell_date,
                                reason='segment_end',
                                commission=self.commission,
                                stamp_tax=self.stamp_tax)
        self.position_info.clear()
        return self.portfolio.cash
