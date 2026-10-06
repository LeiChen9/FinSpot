"""海龟交易法则策略"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from strategy.portfolio import Portfolio, Holding
from strategy.signals import TurtleSignal
from indicators.technical import atr


class TurtleStrategy:
    """海龟交易法则策略

    核心规则：
    1. 入场：价格突破N日最高价买入（默认20日）
    2. 出场：价格跌破M日最低价卖出（默认10日）
    3. 仓位管理：基于ATR计算头寸规模
       - 每单位风险 = 账户净值的1%
       - 头寸规模 = 每单位风险 / (N * ATR)
    4. 加仓：每上涨0.5个ATR加仓一个单位，最多加仓4次
    5. 止损：初始止损为2个ATR，加仓后调整止损
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 20,
        risk_per_trade: float = 0.01,  # 每笔交易风险比例（1%）
        max_units: int = 4,            # 最多加仓次数
        add_threshold: float = 0.5,    # 加仓阈值（ATR倍数）
        stop_loss_atr: float = 2.0,    # 止损ATR倍数
        commission: float = 0.0003,
        stamp_tax: float = 0.0005,
        stock_names: Optional[Dict[str, str]] = None,
    ):
        self.initial_capital = initial_capital
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade
        self.max_units = max_units
        self.add_threshold = add_threshold
        self.stop_loss_atr = stop_loss_atr
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.stock_names = stock_names or {}

        # 初始化组合管理器
        self.portfolio = Portfolio(initial_cash=initial_capital)

        # 信号生成器
        self.signal_generator = TurtleSignal(
            entry_period=entry_period,
            exit_period=exit_period,
        )

        # 记录每只股票的持仓信息
        self.position_info: Dict[str, dict] = {}

    def _calculate_unit_size(self, price: float, atr_value: float, current_nav: float) -> int:
        """计算每个交易单位的股数

        海龟交易法则：
        - 每单位风险 = 账户净值 * 风险比例
        - 每股风险 = N * ATR
        - 头寸规模 = 每单位风险 / 每股风险
        """
        if atr_value <= 0 or price <= 0:
            return 0

        # 每单位风险金额
        risk_amount = current_nav * self.risk_per_trade

        # 每股风险（使用ATR作为风险度量）
        risk_per_share = self.stop_loss_atr * atr_value

        # 计算股数（100股的整数倍）
        shares = int(risk_amount / risk_per_share / 100) * 100

        # 确保不超过最大持仓比例（10%）
        max_shares = int(current_nav * 0.10 / price / 100) * 100
        shares = min(shares, max_shares)

        return max(shares, 0)

    def _should_add_position(self, code: str, current_price: float, atr_value: float) -> bool:
        """判断是否应该加仓"""
        if atr_value <= 0:
            return False

        info = self.position_info.get(code)
        if not info:
            return False

        # 已达到最大加仓次数
        if info['units'] >= self.max_units:
            return False

        # 计算当前价格上涨了多少个ATR
        last_entry_price = info['last_entry_price']
        price_increase = current_price - last_entry_price
        atr_multiple = price_increase / atr_value

        # 每上涨0.5个ATR加仓一次
        if atr_multiple >= self.add_threshold:
            return True

        return False

    def _update_stop_loss(self, code: str, current_price: float, atr_value: float):
        """更新止损价"""
        info = self.position_info.get(code)
        if not info:
            return

        # 新的止损价 = 当前价格 - 2个ATR
        new_stop = current_price - self.stop_loss_atr * atr_value

        # 止损价只能上移，不能下移
        if new_stop > info['stop_loss']:
            info['stop_loss'] = new_stop

    def _check_stop_loss(self, code: str, current_price: float) -> bool:
        """检查是否触发止损"""
        info = self.position_info.get(code)
        if not info:
            return False

        return current_price <= info['stop_loss']

    def _generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """为所有股票生成信号"""
        signals = {}
        for code, df in market_data.items():
            if len(df) < max(self.entry_period, self.exit_period, self.atr_period):
                continue
            try:
                signal = self.signal_generator.generate(df)
                signals[code] = signal
            except Exception as e:
                print(f"生成 {code} 信号失败: {e}")
        return signals

    def run(
        self,
        market_data: Dict[str, pd.DataFrame],
        start_date: str = '2023-01-01',
        end_date: str = '2026-08-22',
    ) -> pd.DataFrame:
        """运行回测"""
        # 转换日期
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # 获取所有交易日
        all_dates = sorted(set(
            d for df in market_data.values()
            for d in df.index
        ))
        trading_days = [d for d in all_dates if start <= d <= end]

        if not trading_days:
            raise ValueError("没有交易日数据")

        print(f"回测期间: {start_date} ~ {end_date}, 共 {len(trading_days)} 个交易日")

        # 生成所有股票的信号
        signals = self._generate_signals(market_data)

        # 按日期回测
        for i, today in enumerate(trading_days):
            # 更新持仓价格
            price_map = {}
            for code in list(self.portfolio.holdings.keys()):
                df = market_data.get(code)
                if df is not None and today in df.index:
                    price_map[code] = df.loc[today, 'close']

            self.portfolio.update_prices(price_map)
            self.portfolio.update_weights()

            # 获取当前净值
            current_nav = self.portfolio.value()

            # 检查止损
            for code in list(self.portfolio.holdings.keys()):
                current_holding = self.portfolio.holdings.get(code)
                if not current_holding:
                    continue

                if self._check_stop_loss(code, current_holding.current_price):
                    # 触发止损，卖出全部
                    df = market_data.get(code)
                    if df is not None and today in df.index:
                        price = df.loc[today, 'close']
                        self.portfolio.sell(code, price, date=today, reason='stop_loss',
                                          commission=self.commission,
                                          stamp_tax=self.stamp_tax)
                        # 清除持仓信息
                        if code in self.position_info:
                            del self.position_info[code]

            # 检查卖出信号
            for code in list(self.portfolio.holdings.keys()):
                signal_series = signals.get(code)
                if signal_series is not None and today in signal_series.index:
                    if signal_series[today] == -1:  # 卖出信号
                        df = market_data.get(code)
                        if df is not None and today in df.index:
                            price = df.loc[today, 'close']
                            self.portfolio.sell(code, price, date=today, reason='signal',
                                              commission=self.commission,
                                              stamp_tax=self.stamp_tax)
                            # 清除持仓信息
                            if code in self.position_info:
                                del self.position_info[code]

            # 检查买入信号
            for code, df in market_data.items():
                if code in self.portfolio.holdings:
                    continue  # 已持仓，跳过

                signal_series = signals.get(code)
                if signal_series is None or today not in signal_series.index:
                    continue

                if signal_series[today] == 1:  # 买入信号
                    if today not in df.index:
                        continue

                    price = df.loc[today, 'close']

                    # 计算ATR
                    df_slice = df.loc[:today]
                    if len(df_slice) < self.atr_period:
                        continue

                    atr_series = atr(df_slice, self.atr_period)
                    current_atr = atr_series.iloc[-1]

                    if pd.isna(current_atr) or current_atr <= 0:
                        continue

                    # 计算头寸规模
                    shares = self._calculate_unit_size(price, current_atr, current_nav)
                    if shares > 0:
                        # 获取股票名称
                        stock_name = self.stock_names.get(code, f'股票{code}')
                        self.portfolio.buy(code, price, shares, date=today,
                                          name=stock_name, commission=self.commission)

                        # 初始化持仓信息
                        self.position_info[code] = {
                            'units': 1,
                            'last_entry_price': price,
                            'stop_loss': price - self.stop_loss_atr * current_atr,
                            'entry_atr': current_atr,
                        }

            # 检查加仓信号
            for code in list(self.portfolio.holdings.keys()):
                current_holding = self.portfolio.holdings.get(code)
                if not current_holding:
                    continue

                # 获取ATR值
                df = market_data.get(code)
                if df is None or today not in df.index:
                    continue

                df_slice = df.loc[:today]
                if len(df_slice) < self.atr_period:
                    continue

                atr_series = atr(df_slice, self.atr_period)
                current_atr = atr_series.iloc[-1]

                if pd.isna(current_atr) or current_atr <= 0:
                    continue

                # 更新止损价
                self._update_stop_loss(code, current_holding.current_price, current_atr)

                # 检查是否应该加仓
                if self._should_add_position(code, current_holding.current_price, current_atr):
                    # 计算加仓股数
                    shares = self._calculate_unit_size(
                        current_holding.current_price, current_atr, current_nav
                    )
                    if shares > 0:
                        stock_name = self.stock_names.get(code, f'股票{code}')
                        self.portfolio.buy(code, current_holding.current_price, shares,
                                          date=today, name=stock_name, commission=self.commission)

                        # 更新持仓信息
                        info = self.position_info.get(code, {})
                        info['units'] = info.get('units', 1) + 1
                        info['last_entry_price'] = current_holding.current_price
                        self.position_info[code] = info

            # 记录净值
            self.portfolio.nav_history.append(self.portfolio.snapshot(today))

            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{len(trading_days)}, 净值: {current_nav:,.0f}, "
                      f"持仓: {len(self.portfolio.holdings)}")

        print(f"回测完成! 最终净值: {self.portfolio.value():,.0f}, "
              f"总交易: {len(self.portfolio.trade_log)} 笔")

        # 返回净值曲线
        return pd.DataFrame(self.portfolio.nav_history).set_index('date')
