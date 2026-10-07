"""Pure adapters for normalising Graham source data."""

import numpy as np
import pandas as pd


def exchange_of(code: str) -> str:
    if code[0] in ('6', '9'):
        return 'sh'
    if code[0] in ('4', '8') or code.startswith('92'):
        return 'bj'
    return 'sz'


def symbol_sina(code: str) -> str:
    return f'{exchange_of(code)}{code}'


def parse_ymd(value) -> pd.Timestamp:
    try:
        return pd.to_datetime(str(int(value)), format='%Y%m%d', errors='coerce')
    except Exception:
        return pd.NaT


def slice_sina_report(df: pd.DataFrame, keep_cols: list) -> pd.DataFrame:
    out = pd.DataFrame({'报告日': df['报告日'], '公告日期': df['公告日期']})
    for column in keep_cols:
        out[column] = (pd.to_numeric(df[column], errors='coerce')
                       if column in df.columns else np.nan)
    out['报告日'] = out['报告日'].map(parse_ymd)
    out['公告日期'] = out['公告日期'].map(parse_ymd)
    return out.dropna(subset=['报告日']).sort_values('报告日')
