"""基于财务摘要的横截面因子。"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from data.financial import FinancialDataLoader


FINANCIAL_FACTORS = ["pe_inv", "pb_inv", "roe", "profit_margin", "rev_growth"]
FINANCIAL_FACTOR_LABELS = {
    "pe_inv": "PE 倒数", "pb_inv": "PB 倒数", "roe": "ROE",
    "profit_margin": "销售净利率", "rev_growth": "营收增长率",
}


def _find_fin_value(fin: dict, candidates: list[str]) -> Optional[float]:
    for key in candidates:
        value = fin.get(key)
        if value is not None and np.isfinite(value):
            return float(value)
    return None


def _compute_ttm_eps(fin_data: dict, loader: FinancialDataLoader,
                     stock: str, as_of: datetime) -> Optional[float]:
    eps = _find_fin_value(fin_data, ["基本每股收益", "每股收益", "每股盈余"])
    if eps is None:
        return None
    cutoff = as_of.year * 12 + as_of.month - 3
    previous = loader.get_latest_financial(
        stock, datetime(cutoff // 12, max(1, cutoff % 12), 1)
    )
    return eps if previous is None else eps


def _price(close: pd.Series, as_of) -> float | None:
    values = close[close.index <= as_of]
    if values.empty or values.iloc[-1] <= 0:
        return None
    return float(values.iloc[-1])


def calc_pe_inv(close: pd.Series, as_of: datetime, **kwargs) -> Optional[float]:
    fin, price = kwargs.get("financial_data"), _price(close, as_of)
    if fin is None or kwargs.get("fin_loader") is None or kwargs.get("stock") is None or price is None:
        return None
    eps = _find_fin_value(fin, ["基本每股收益", "每股收益", "每股盈余"])
    return eps / price if eps is not None and eps > 0 else None


def calc_pb_inv(close: pd.Series, as_of: datetime, **kwargs) -> Optional[float]:
    fin, price = kwargs.get("financial_data"), _price(close, as_of)
    if fin is None or price is None:
        return None
    bvps = _find_fin_value(fin, ["每股净资产", "每股净资产_最新股数", "摊薄每股净资产_期末股数"])
    return bvps / price if bvps is not None and bvps > 0 else None


def calc_roe(close: pd.Series, as_of: datetime, **kwargs) -> Optional[float]:
    fin = kwargs.get("financial_data")
    value = _find_fin_value(fin, ["净资产收益率(ROE)", "净资产收益率", "摊薄净资产收益率"]) if fin else None
    return value if value is not None and abs(value) <= 200 else None


def calc_profit_margin(close: pd.Series, as_of: datetime, **kwargs) -> Optional[float]:
    fin = kwargs.get("financial_data")
    value = _find_fin_value(fin, ["销售净利率", "营业利润率"]) if fin else None
    return value if value is not None and abs(value) <= 200 else None


def calc_rev_growth(close: pd.Series, as_of: datetime, **kwargs) -> Optional[float]:
    fin = kwargs.get("financial_data")
    value = _find_fin_value(fin, ["营业总收入增长率", "归属母公司净利润增长率"]) if fin else None
    return value if value is not None and abs(value) <= 1000 else None


FUNDAMENTAL_CALCS = {
    "pe_inv": calc_pe_inv, "pb_inv": calc_pb_inv, "roe": calc_roe,
    "profit_margin": calc_profit_margin, "rev_growth": calc_rev_growth,
}
