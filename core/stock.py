"""个股研究框架"""
import pandas as pd
import numpy as np
import akshare as ak
from typing import Dict, List, Optional
from core.base import BaseResearcher
from visualization import charts


class StockResearcher(BaseResearcher):
    """个股多维度分析（行情 + 技术指标）"""

    def __init__(self, stock_code: str):
        super().__init__(stock_code)
        print(f"📊 初始化个股研究器: {self.stock_code}")

    @property
    def stock_code(self):
        return self.code

    def get_basic_info(self):
        try:
            stock_info = ak.stock_info_a_code_name()
            info = stock_info[stock_info['code'] == self.code]
            self.meta['name'] = info.iloc[0]['name'] if not info.empty else self.code
        except Exception as e:
            print(f"警告: 获取股票信息失败 - {e}")
            self.meta['name'] = self.code
        print(f"   [√] 股票: {self.meta['name']}")
        return self

    def load_market_data(self, days: int = 365 * 3, force_download: bool = False):
        self.data['price'] = self.data_manager.get_stock_market_data(
            self.code, days=days, force_download=force_download,
        )
        return self

    def calculate_indicators(self, period_days: int = 365):
        if 'price' not in self.data or self.data['price'].empty:
            raise ValueError("缺少价格数据")

        df = self.data['price']

        start_price = df.iloc[-period_days]['close'] if len(df) >= period_days else df.iloc[0]['close']
        end_price = df.iloc[-1]['close']
        period_return = ((end_price - start_price) / start_price) * 100

        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100

        self.metrics = {
            'current_price': df.iloc[-1]['close'],
            'period_return': period_return,
            'volatility': volatility,
            'high_52w': df['high'].tail(252).max(),
            'low_52w': df['low'].tail(252).min(),
        }
        self._print_report()
        return self

    def _print_report(self):
        print("-" * 50)
        print(f"📊 个股分析报告 ({self.meta.get('name')})")
        print(f"   当前价格 : {self.metrics['current_price']:.2f} 元")
        status = "↑ 上升" if self.metrics['period_return'] > 0 else "↓ 下跌"
        print(f"   期间涨跌 : {self.metrics['period_return']:+.2f}% [{status}]")
        print(f"   年化波动 : {self.metrics['volatility']:.2f}%")
        print(f"   52周高点: {self.metrics['high_52w']:.2f} 元")
        print(f"   52周低点: {self.metrics['low_52w']:.2f} 元")
        print("-" * 50)

    def plot_analysis(self, period_days: int = 365, show_mas: Optional[List[int]] = None):
        if show_mas is None:
            show_mas = [5, 20, 60]

        self.calculate_moving_averages(show_mas)

        if 'price' not in self.data or self.data['price'].empty:
            raise ValueError("无可用数据用于绘图")

        df = self.data['price'].copy()
        if len(df) > period_days:
            df = df.tail(period_days)

        title = f"{self.meta.get('name')} ({self.code}) - 走势分析"
        charts.plot_stock_analysis(
            df, title,
            f"{self.code}_{period_days}d.png",
            show_mas=show_mas,
        )
        return self
