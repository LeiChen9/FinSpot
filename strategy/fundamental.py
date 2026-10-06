"""财务因子信号 — 基于基本面选股的信号生成"""
from typing import Optional
import pandas as pd
from strategy.base import Signal


class FundamentalSignal(Signal):
    """基本面信号：当股票通过财务筛选时为买入信号

    注意：这个信号是"给定一组已筛选出的股票"，
    在组合回测中，screener_fn 的筛选结果本身即是信号。
    此处作为 Signal 接口的封装供单标的回测使用。
    """

    def __init__(self, df_financial: pd.DataFrame):
        """
        Args:
            df_financial: 包含 code, date, 财务指标的 DataFrame
        """
        self.df = df_financial

    def generate(self, df: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=df.index)
        if 'code' not in df.columns:
            return signal

        code = df['code'].iloc[0] if 'code' in df.columns else None
        if code is None:
            return signal

        fin = self.df[self.df['code'] == code].sort_values('date')
        if fin.empty:
            return signal

        signal[:] = 1
        return signal
