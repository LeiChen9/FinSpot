"""Cached local inputs and parsers for the Graham screener."""

import os
import re
from typing import Dict, Optional, Tuple
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIN_DIR = os.path.join(DATA_DIR, 'financial')
DIV_DIR = os.path.join(DATA_DIR, 'dividend')
UNITS = {'亿': 1e8, '万': 1e4, '元': 1}
PATTERN = re.compile(r'^([-]?[\d,.]+)([亿万]?)$')
FIN_CACHE: Dict[str, pd.DataFrame] = {}
DIV_CACHE: Dict[str, pd.DataFrame] = {}
MKT_CACHE: Dict[str, pd.DataFrame] = {}

def read_fin(code: str) -> pd.DataFrame:
    if code not in FIN_CACHE:
        path = os.path.join(FIN_DIR, f'{code}_fin.csv')
        FIN_CACHE[code] = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    return FIN_CACHE[code]

def read_dividend(code: str) -> pd.DataFrame:
    if code not in DIV_CACHE:
        path = os.path.join(DIV_DIR, f'{code}_dividend.csv')
        DIV_CACHE[code] = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    return DIV_CACHE[code]

def read_market(code: str) -> pd.DataFrame:
    if code not in MKT_CACHE:
        path = os.path.join(DATA_DIR, f'{code}_market.csv')
        MKT_CACHE[code] = (pd.read_csv(path, index_col='date', parse_dates=True,
                                        usecols=['date', 'close'])
                           if os.path.exists(path)
                           else pd.DataFrame(index=pd.DatetimeIndex([])))
    return MKT_CACHE[code]

def parse_value(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text in ('--', '-', 'False', 'nan', 'None'):
        return None
    match = PATTERN.match(text)
    if match:
        return float(match.group(1).replace(',', '')) * UNITS.get(match.group(2), 1)
    try:
        return float(text)
    except (ValueError, TypeError):
        return None

def indicator_series(df: pd.DataFrame, name: str) -> pd.Series:
    row = df[df['指标'] == name]
    if row.empty:
        return pd.Series(dtype=float)
    series = row.iloc[0, 2:].astype(object).map(parse_value)
    series.index = pd.to_datetime(series.index, format='%Y%m%d', errors='coerce')
    series = pd.to_numeric(series, errors='coerce')
    return series[series.index.notna()].sort_index()
