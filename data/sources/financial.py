"""财务数据源 — 同花顺/东方财富 财报数据"""
import re
import pandas as pd
import akshare as ak
from typing import Optional, List


_UNITS = {'亿': 1e8, '万': 1e4, '元': 1}
_PATTERN = re.compile(r'^([-]?[\d,.]+)([亿万]?)$')


def parse_financial_value(val) -> Optional[float]:
    """将 '615.22亿' 转为 float，None 表示缺失"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s or s in ('--', '', '-', 'False'):
        return None
    m = _PATTERN.match(s)
    if m:
        num = float(m.group(1).replace(',', ''))
        unit = m.group(2)
        return num * _UNITS.get(unit, 1)
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def fetch_cash_flow(code: str) -> pd.DataFrame:
    """获取单只股票的现金流量表（年报）

    Returns:
        DataFrame columns:
        - report_date: 报告期 (datetime)
        - operating_cashflow: 经营活动产生的现金流量净额
        - capex: 购建固定资产、无形资产和其他长期资产支付的现金
        - net_profit: 净利润
        - publish_date: 同报告期记为发布日期（保守）
    """
    df = ak.stock_financial_cash_ths(symbol=code)

    yearly = df[df['报告期'].str.endswith('12-31')].copy()
    if yearly.empty:
        return pd.DataFrame()

    result = pd.DataFrame()
    result['report_date'] = pd.to_datetime(yearly['报告期'])
    result['operating_cashflow'] = yearly['*经营活动产生的现金流量净额'].apply(parse_financial_value)
    result['capex'] = yearly['购建固定资产、无形资产和其他长期资产支付的现金'].apply(parse_financial_value)
    result['net_profit'] = yearly['净利润'].apply(parse_financial_value)

    # 估计发布日期: 年报截止日为次年 4-30
    result['publish_date'] = result['report_date'].apply(
        lambda d: d + pd.DateOffset(years=1, month=4, day=30)
    )
    result['code'] = code
    result.sort_values('report_date', ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)

    return result


def fetch_abstract(code: str) -> pd.DataFrame:
    """获取单只股票的财务摘要（含预计算比率）

    Returns:
        DataFrame columns:
        - report_date: 报告期
        - operating_cashflow: 经营现金流量净额
        - net_profit: 净利润
        - cashflow_to_profit_ratio: 经营活动净现金/归属母公司的净利润
    """
    df = ak.stock_financial_abstract(symbol=code)

    def _extract(name):
        row = df[df['指标'] == name]
        if row.empty:
            return pd.Series(dtype=float)
        s = row.iloc[:, 2:].T.squeeze()
        s.index = pd.to_datetime(s.index, format='%Y%m%d', errors='coerce')
        s.name = name
        return s

    series_list = []
    for name in ['经营现金流量净额', '净利润', '经营活动净现金/归属母公司的净利润']:
        series_list.append(_extract(name))
    if not series_list:
        return pd.DataFrame()

    result = pd.concat(series_list, axis=1)
    result.columns = ['operating_cashflow', 'net_profit', 'cashflow_to_profit_ratio']
    result = result.dropna(how='all')
    result['code'] = code
    result.reset_index(inplace=True)
    result.rename(columns={'index': 'report_date'}, inplace=True)
    result.sort_values('report_date', ascending=False, inplace=True)

    return result
