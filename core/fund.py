"""基金研究框架"""
import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from core.base import BaseResearcher
from indicators.attribution import (
    calc_beta, calc_volatility, calc_residual_risk,
    calc_realized_alpha, calc_realized_risk_premium,
    calc_exceptional_benchmark_return, calc_expected_benchmark_excess_return,
    analyze_rolling, decompose_risk,
)


class FundResearcher(BaseResearcher):
    """基金业绩归因与风险分析"""

    def __init__(self, fund_code: str):
        super().__init__(fund_code)
        print(f"📊 初始化基金研究器: {self.fund_code}")

    @property
    def fund_code(self):
        return self.code

    def get_basic_info(self):
        """暂做简单名称获取"""
        self.meta['name'] = self.code
        return self

    def load_market_data(self, **kwargs):
        raise NotImplementedError("基金请使用 load_fund_nav() 加载净值")

    def load_fund_nav(self, force_download: bool = False):
        self.data['nav'] = self.data_manager.get_fund_nav_data(
            self.code, force_download=force_download,
        )
        return self

    def load_benchmark_data(self, index_code: str, days: int = 365 * 3,
                            force_download: bool = False):
        """加载基准指数行情（用于业绩归因）"""
        self.data['benchmark'] = self.data_manager.get_index_market_data(
            index_code, days=days, force_download=force_download,
        )
        return self

    def load_benchmark_fund_data(self, fund_code: str, force_download: bool = False):
        """加载基准基金净值（用于业绩归因）"""
        self.data['benchmark_fund'] = self.data_manager.get_fund_nav_data(
            fund_code, force_download=force_download,
        )
        return self

    # ------------------------------------------------------------------
    # 业绩归因分析
    # ------------------------------------------------------------------

    def analyze_attribution(self, benchmark_weight: Optional[Dict[str, float]] = None,
                            rf_annual: float = 0.02) -> pd.DataFrame:
        """执行完整的收益率分解与风险分析

        参数
        ----------
        benchmark_weight : dict, optional
            基准构成，如 {'930917': 0.8, '161119': 0.2}
            键为指数/基金代码，值为权重
        rf_annual : float
            年化无风险利率

        返回
        -------
        DataFrame : 包含各个归因序列的合并数据
        """
        if benchmark_weight:
            return self._analyze_multi_benchmark(benchmark_weight, rf_annual)

        if 'benchmark' not in self.data and 'benchmark_fund' not in self.data:
            raise ValueError("需要先调用 load_benchmark_data() 或 load_benchmark_fund_data()")
        return self._analyze_single_benchmark(rf_annual)

    def _analyze_single_benchmark(self, rf_annual: float = 0.02) -> pd.DataFrame:
        """单基准归因"""
        if 'nav' not in self.data:
            raise ValueError("缺少基金净值数据，请先调用 load_fund_nav()")

        df_fund = self.data['nav'].copy()
        fund_ret = df_fund['daily_return'] / 100.0 if 'daily_return' in df_fund.columns else None
        if fund_ret is None:
            raise ValueError("基金净值数据缺少日增长率列")

        if 'benchmark' in self.data:
            df_bm = self.data['benchmark'].copy()
            bm_ret = df_bm['close'].pct_change().dropna()
        elif 'benchmark_fund' in self.data:
            df_bm = self.data['benchmark_fund'].copy()
            bm_ret = df_bm['daily_return'] / 100.0 if 'daily_return' in df_bm.columns else None
        else:
            raise ValueError("缺少基准数据")

        # 对齐日期
        common_idx = fund_ret.dropna().index.intersection(bm_ret.index)
        rP = fund_ret.loc[common_idx]
        rB = bm_ret.loc[common_idx]

        rf_daily = rf_annual / 252
        df = pd.DataFrame(index=common_idx)
        df['R_P'] = rP
        df['R_B'] = rB
        df['R_F'] = rf_daily
        df['r_P'] = df['R_P'] - df['R_F']
        df['r_B'] = df['R_B'] - df['R_F']

        self._compute_attribution(df, rf_annual)
        return df

    def _analyze_multi_benchmark(self, weights: Dict[str, float],
                                 rf_annual: float = 0.02) -> pd.DataFrame:
        """多基准归因（如 80% 指数 + 20% 债券）"""
        if 'nav' not in self.data:
            raise ValueError("缺少基金净值数据")

        df_fund = self.data['nav'].copy()
        fund_ret = df_fund['daily_return'] / 100.0

        # 合并各基准收益
        all_returns = []
        for code, weight in weights.items():
            if code in self.data.get('benchmark_cache', {}):
                bm_df = self.data['benchmark_cache'][code]
            else:
                bm_df = self.data_manager.get_index_market_data(code, days=365 * 3)
                self.data.setdefault('benchmark_cache', {})[code] = bm_df

            if 'close' in bm_df.columns:
                ret = bm_df['close'].pct_change().dropna()
                all_returns.append(ret * weight)
            elif 'daily_return' in bm_df.columns:
                ret = bm_df['daily_return'] / 100.0
                all_returns.append(ret * weight)

        if not all_returns:
            raise ValueError("无法计算基准收益")

        combined = pd.concat(all_returns, axis=1)
        bm_ret = combined.sum(axis=1).dropna()

        common_idx = fund_ret.dropna().index.intersection(bm_ret.index)
        rP = fund_ret.loc[common_idx]
        rB = bm_ret.loc[common_idx]

        rf_daily = rf_annual / 252
        df = pd.DataFrame(index=common_idx)
        df['R_P'] = rP
        df['R_B'] = rB
        df['R_F'] = rf_daily
        df['r_P'] = df['R_P'] - df['R_F']
        df['r_B'] = df['R_B'] - df['R_F']

        self._compute_attribution(df, rf_annual)
        return df

    def _compute_attribution(self, df: pd.DataFrame, rf_annual: float):
        """计算归因指标并打印报告"""
        rP, rB = df['r_P'].dropna(), df['r_B'].dropna()
        f_B = calc_expected_benchmark_excess_return(rB)
        beta = calc_beta(rP, rB)

        df['Time_Premium'] = df['R_F']
        df['Realized_Risk_Premium'] = calc_realized_risk_premium(rP, rB)
        df['Exceptional_BM_Return'] = calc_exceptional_benchmark_return(rB, f_B)
        df['Realized_Alpha'] = calc_realized_alpha(rP, rB)

        risk = decompose_risk(rP, rB)

        self.metrics = {
            'beta': beta,
            'f_B': f_B,
            'alpha_annual': rP.mean() * 252 - beta * rB.mean() * 252,
            **risk,
        }

        self._print_attribution(rf_annual, risk)
        return df

    def _print_attribution(self, rf_annual: float, risk: dict):
        print("--- 收益率分解汇总 ---")
        print(f"年化无风险利率: {rf_annual * 100:.2f}%")
        print(f"组合 Beta: {risk['beta']:.4f}")
        print(f"业绩基准预期超额收益 (年化): {self.metrics.get('f_B', 0) * 252 * 100:.4f}%")
        print(f"实现的 Alpha (年化): {self.metrics.get('alpha_annual', 0) * 100:.4f}%")
        print()
        print("--- 风险分解 ---")
        print(f"基金年化波动: {risk['sigma_fund'] * 100:.2f}%")
        print(f"基准年化波动: {risk['sigma_benchmark'] * 100:.2f}%")
        print(f"残余风险: {risk['omega'] * 100:.2f}%")
        print(f"系统性方差: {risk['variance_systematic']:.6f}")
        print(f"残余方差: {risk['variance_residual']:.6f}")
        print(f"总方差: {risk['variance_total']:.6f}")

    def analyze_rolling(self, window: int = 60):
        """滚动 Beta 和 Alpha 分析"""
        if 'r_P' not in self.data or 'r_B' not in self.data:
            raise ValueError("请先运行 analyze_attribution()")

        df = pd.DataFrame({
            'date': self.data.get('_result_index', []),
            'r_P': self.data.get('_result_rP', []),
            'r_B': self.data.get('_result_rB', []),
        })
        if df.empty:
            raise ValueError("归因结果为空")

        rolling_df = analyze_rolling(df, window)
        self._plot_rolling(rolling_df)
        return rolling_df

    def _plot_rolling(self, rolling_df: pd.DataFrame):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax1.plot(rolling_df['date'], rolling_df['beta'], color='orange', label='Rolling Beta')
        ax1.axhline(self.metrics.get('beta', 0), color='black', linestyle='--', alpha=0.5, label='Full Beta')
        ax1.set_ylabel('Beta')
        ax1.set_title('Dynamic Risk Exposure')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(rolling_df['date'], rolling_df['alpha'], color='green', label='Rolling Alpha')
        ax2.axhline(0, color='red', linestyle='-', linewidth=1)
        ax2.set_ylabel('Annualized Alpha')
        ax2.set_title('Dynamic Active Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        import os
        plot_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        filename = os.path.join(plot_dir, f"{self.code}_rolling.png")
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"   [√] 滚动分析图已保存: {filename}")
        plt.close()

    def _print_report(self):
        pass  # 归因报告通过 _print_attribution 输出
