"""ATR 通道突破策略"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from strategy.portfolio import Portfolio, Holding
from strategy.signals import ATRChannelBreakoutSignal
from indicators.technical import atr, keltner_channel


class ATRChannelStrategy:
    """ATR 通道突破策略

    策略规则：
    1. 买入：价格突破通道上轨（阻力位）
    2. 卖出：价格跌破通道下轨（支撑位）
    3. 加仓：每上涨 0.5 个 ATR，加仓 0.5 倍当前头寸
    4. 最大持仓：每只股票最大持仓 10%
    5. 初始头寸：每只股票初始头寸 5%
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        ma_period: int = 20,
        atr_period: int = 20,
        atr_multiplier: float = 2.0,
        initial_position_pct: float = 0.05,  # 初始头寸比例
        max_position_pct: float = 0.10,      # 最大持仓比例
        add_position_pct: float = 0.005,     # 加仓比例（相对于初始头寸）
        atr_add_threshold: float = 0.5,      # 加仓阈值（ATR 倍数）
        commission: float = 0.0003,
        stamp_tax: float = 0.0005,
        stock_names: Optional[Dict[str, str]] = None,  # 股票名称映射
    ):
        self.initial_capital = initial_capital
        self.ma_period = ma_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.initial_position_pct = initial_position_pct
        self.max_position_pct = max_position_pct
        self.add_position_pct = add_position_pct
        self.atr_add_threshold = atr_add_threshold
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.stock_names = stock_names or {}

        # 初始化组合管理器
        self.portfolio = Portfolio(initial_cash=initial_capital)

        # 信号生成器
        self.signal_generator = ATRChannelBreakoutSignal(
            ma_period=ma_period,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier
        )

        # 记录每只股票的加仓次数
        self.add_count: Dict[str, int] = {}

    def _calculate_position_size(self, code: str, price: float, current_nav: float) -> float:
        """计算买入金额"""
        # 计算初始头寸金额
        initial_amount = current_nav * self.initial_position_pct

        # 检查是否超过最大持仓
        current_holding = self.portfolio.holdings.get(code)
        if current_holding:
            current_value = current_holding.market_value
            max_value = current_nav * self.max_position_pct
            if current_value >= max_value:
                return 0.0

        return initial_amount

    def _calculate_add_position_size(self, code: str, price: float, current_nav: float) -> float:
        """计算加仓金额"""
        current_holding = self.portfolio.holdings.get(code)
        if not current_holding:
            return 0.0

        # 计算加仓金额（相对于初始头寸）
        initial_amount = current_nav * self.initial_position_pct
        add_amount = initial_amount * self.add_position_pct

        # 检查是否超过最大持仓
        current_value = current_holding.market_value
        max_value = current_nav * self.max_position_pct
        if current_value + add_amount > max_value:
            add_amount = max_value - current_value

        return add_amount

    def _should_add_position(self, code: str, current_price: float, entry_price: float, atr_value: float) -> bool:
        """判断是否应该加仓"""
        if atr_value <= 0:
            return False

        # 计算当前价格上涨了多少个 ATR
        price_increase = current_price - entry_price
        atr_multiple = price_increase / atr_value

        # 获取已加仓次数
        add_count = self.add_count.get(code, 0)

        # 每上涨 0.5 个 ATR 加仓一次
        if atr_multiple >= (add_count + 1) * self.atr_add_threshold:
            return True

        return False

    def _generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """为所有股票生成信号"""
        return {
            code: self.signal_generator.generate(frame)
            for code, frame in market_data.items()
            if len(frame) >= max(self.ma_period, self.atr_period)
        }

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

        # 预计算每只股票的 ATR 通道
        channels = {}
        for code, df in market_data.items():
            if code in signals:
                channels[code] = keltner_channel(
                    df, self.ma_period, self.atr_period, self.atr_multiplier
                )

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

            # 检查卖出信号
            for code in list(self.portfolio.holdings.keys()):
                signal_series = signals.get(code)
                if signal_series is not None and today in signal_series.index:
                    if signal_series[today] == -1:  # 卖出信号
                        # 从 market_data 中获取价格
                        df = market_data.get(code)
                        if df is not None and today in df.index:
                            price = df.loc[today, 'close']
                        else:
                            price = None
                        
                        if price and price > 0:
                            self.portfolio.sell(code, price, date=today, reason='signal',
                                              commission=self.commission,
                                              stamp_tax=self.stamp_tax)

            # 检查买入信号
            for code, df in market_data.items():
                if code in self.portfolio.holdings:
                    continue  # 已持仓，跳过

                signal_series = signals.get(code)
                if signal_series is None or today not in signal_series.index:
                    continue

                if signal_series[today] == 1:  # 买入信号
                    # 从 market_data 中获取价格
                    if today in df.index:
                        price = df.loc[today, 'close']
                    else:
                        price = None
                    
                    if price and price > 0:
                        # 计算买入金额
                        buy_amount = self._calculate_position_size(code, price, current_nav)
                        if buy_amount > 0:
                            # 计算股数（100 股的整数倍）
                            shares = int(buy_amount / price / 100) * 100
                            if shares > 0:
                                # 获取股票名称
                                stock_name = self.stock_names.get(code, f'股票{code}')
                                self.portfolio.buy(code, price, shares, date=today,
                                                  name=stock_name, commission=self.commission)
                                # 重置加仓计数
                                self.add_count[code] = 0

            # 检查加仓信号
            for code in list(self.portfolio.holdings.keys()):
                current_holding = self.portfolio.holdings.get(code)
                if not current_holding:
                    continue

                # 获取 ATR 值
                df = market_data.get(code)
                if df is None or today not in df.index:
                    continue

                # 计算当前 ATR
                df_slice = df.loc[:today]
                if len(df_slice) < self.atr_period:
                    continue

                atr_series = atr(df_slice, self.atr_period)
                current_atr = atr_series.iloc[-1]

                if pd.isna(current_atr) or current_atr <= 0:
                    continue

                # 检查是否应该加仓
                if self._should_add_position(code, current_holding.current_price,
                                          current_holding.avg_cost, current_atr):
                    # 计算加仓金额
                    add_amount = self._calculate_add_position_size(code, current_holding.current_price,
                                                              current_nav)
                    if add_amount > 0:
                        # 计算股数
                        shares = int(add_amount / current_holding.current_price / 100) * 100
                        if shares > 0:
                            # 获取股票名称
                            stock_name = self.stock_names.get(code, f'股票{code}')
                            self.portfolio.buy(code, current_holding.current_price, shares,
                                              date=today, name=stock_name, commission=self.commission)
                            # 增加加仓计数
                            self.add_count[code] = self.add_count.get(code, 0) + 1

            # 记录净值
            self.portfolio.nav_history.append(self.portfolio.snapshot(today))

            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{len(trading_days)}, 净值: {current_nav:,.0f}, "
                      f"持仓: {len(self.portfolio.holdings)}")

        print(f"回测完成! 最终净值: {self.portfolio.value():,.0f}, "
              f"总交易: {len(self.portfolio.trade_log)} 笔")

        # 返回净值曲线
        return pd.DataFrame(self.portfolio.nav_history).set_index('date')
