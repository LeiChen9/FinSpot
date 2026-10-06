"""市场面选股筛选器（市值、PE、价格等）"""
from typing import Dict, List, Optional
import pandas as pd
from screener.base import Screener, FilterResult


class MarketScreener(Screener):
    """市场因子筛选器

    支持基于实时/历史行情数据的过滤，如：
      - 市值范围
      - PE(TTM) 范围
      - 价格 > N
      - 排除 ST / 退市
    """

    def __init__(
        self,
        pe_max: Optional[float] = 30.0,
        pe_min: Optional[float] = 0.0,
        price_min: Optional[float] = None,
        market_cap_min: Optional[float] = None,
        market_cap_max: Optional[float] = None,
        exclude_st: bool = True,
    ):
        super().__init__()
        self.pe_max = pe_max
        self.pe_min = pe_min
        self.price_min = price_min
        self.market_cap_min = market_cap_min
        self.market_cap_max = market_cap_max
        self.exclude_st = exclude_st

    def run(self, as_of_date, pool: Optional[List[str]] = None) -> FilterResult:
        raise NotImplementedError(
            "子类需实现 _fetch_market_data(codes, as_of_date) "
            "返回包含 code, price, pe_ttm, market_cap, name 的 DataFrame"
        )


class TopNByMarketCap(MarketScreener):
    """取市值最大的 N 只"""

    def __init__(self, n: int = 50, pe_max: float = 30.0):
        super().__init__(pe_max=pe_max)
        self.n = n

    def run(self, as_of_date, pool: Optional[List[str]] = None) -> FilterResult:
        raise NotImplementedError(
            "子类需实现 _fetch_top_n(as_of_date, n, pe_max, pool)"
        )
