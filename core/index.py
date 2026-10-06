"""指数研究框架"""
import pandas as pd
import akshare as ak
from typing import Dict, List, Optional
from core.base import BaseResearcher
from indicators.valuation import calculate_percentiles
from visualization import charts


class IndexResearcher(BaseResearcher):
    """指数多维度分析（行情 + 估值 + 技术）"""

    def __init__(self, index_code: str):
        super().__init__(index_code)
        self.meta['name'] = self.index_code = index_code
        print(f"📊 初始化研究器: {self.index_code}")

    def get_basic_info(self):
        try:
            stock_info = ak.index_stock_info()
            name = stock_info[stock_info['index_code'] == self.code]['display_name'].values
            self.meta['name'] = name[0] if len(name) > 0 else self.code
        except Exception as e:
            print(f"警告: 获取指数名称失败 - {e}")
            self.meta['name'] = self.code
        print(f"   [√] 指数: {self.meta['name']}")
        return self

    def load_market_data(self, days: int = 365 * 3, force_download: bool = False):
        self.data['price'] = self.data_manager.get_index_market_data(
            self.code, days=days, force_download=force_download,
        )
        return self

    def load_valuation_data(self, force_download: bool = False):
        self.data['valuation'] = self.data_manager.get_index_valuation_data(
            self.code, force_download=force_download,
        )
        return self

    def merge_data(self):
        if 'price' not in self.data and 'valuation' not in self.data:
            raise ValueError("缺少行情和估值数据")

        if 'price' in self.data:
            full = self.data['price'].copy()
            full.index = full.index.normalize()
            if 'valuation' in self.data:
                val = self.data['valuation'].copy()
                val.index = val.index.normalize()
                full = full.join(val, how='outer')
                val_cols = [c for c in ['pe_ttm', 'pb', 'dividend_yield'] if c in full.columns]
                if val_cols:
                    full[val_cols] = full[val_cols].ffill()
            self.data['full'] = full
        elif 'valuation' in self.data:
            self.data['full'] = self.data['valuation']
        return self

    def calculate_indicators(self, period_years: int = 5):
        self.merge_data()
        if 'full' not in self.data or self.data['full'].empty:
            raise ValueError("合并后数据为空")
        self.metrics = calculate_percentiles(self.data['full'], period_years)
        self._print_report(period_years)
        return self

    def _print_report(self, period_years: int = 5):
        print("-" * 50)
        print(f"📊 估值分析报告 ({self.meta.get('name')} - {period_years}年)")

        if 'pe_ttm' in self.metrics:
            p = self.metrics['pe_ttm']
            status = "低估" if p['percentile'] < 0.2 else ("高估" if p['percentile'] > 0.8 else "适中")
            print(f"   PE-TTM  : {p['current']:.2f} (分位: {p['percentile']:.1%}) [{status}]")

        if 'pb' in self.metrics:
            b = self.metrics['pb']
            status = "低估" if b['percentile'] < 0.2 else ("高估" if b['percentile'] > 0.8 else "适中")
            print(f"   PB      : {b['current']:.2f} (分位: {b['percentile']:.1%}) [{status}]")

        if 'dividend_yield' in self.metrics:
            d = self.metrics['dividend_yield']
            status = "高息" if d['percentile'] > 0.8 else ("低息" if d['percentile'] < 0.2 else "适中")
            print(f"   股息率  : {d['current']:.2f}% (分位: {d['percentile']:.1%}) [{status}]")

        print("-" * 50)

    def plot_analysis(self, period: str = '3Y', show_mas: Optional[List[int]] = None):
        if show_mas is None:
            show_mas = [5, 30, 120]

        self.calculate_moving_averages(show_mas)
        self.merge_data()

        if 'full' not in self.data or self.data['full'].empty:
            raise ValueError("无可用数据用于绘图")

        df = self.data['full'].copy()
        period_map = {'10Y': 10, '5Y': 5, '3Y': 3, '1Y': 1}
        years = period_map.get(period, 3)
        df_plot = df[df.index >= df.index[-1] - pd.DateOffset(years=years)]

        if 'close' not in df_plot.columns or df_plot['close'].isnull().all():
            raise ValueError("无可用的价格数据")

        title = f"{self.meta.get('name')} - 日K走势 ({period})"
        charts.plot_price_with_ma(
            df_plot, title,
            f"{self.code}_price_{period}.png",
            show_mas=show_mas,
        )

        if 'pe_ttm' in df_plot.columns and not df_plot['pe_ttm'].isnull().all():
            val_title = f"{self.meta.get('name')} - PE估值分析 ({period})"
            charts.plot_valuation(
                df_plot, val_title,
                f"{self.code}_valuation_{period}.png",
            )

        return self

    def plot_valuation(self, period: str = '3Y'):
        self.merge_data()
        if 'full' not in self.data or self.data['full'].empty:
            raise ValueError("无可用数据用于绘图")

        df = self.data['full'].copy()
        period_map = {'10Y': 10, '5Y': 5, '3Y': 3, '1Y': 1}
        years = period_map.get(period, 3)
        df_plot = df[df.index >= df.index[-1] - pd.DateOffset(years=years)]

        if 'pe_ttm' not in df_plot.columns or df_plot['pe_ttm'].isnull().all():
            print("   [!] 缺少估值数据，跳过估值图表")
            return self

        title = f"{self.meta.get('name')} - PE估值分析 ({period})"
        charts.plot_valuation(
            df_plot, title,
            f"{self.code}_valuation_{period}.png",
        )
        return self
