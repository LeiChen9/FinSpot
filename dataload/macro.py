"""Macro indicators and benchmark acquisition for the Graham dataset."""

from datetime import datetime
import os

import numpy as np
import pandas as pd


def _normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    if "date" not in frame.columns and "日期" not in frame.columns:
        frame = frame.rename(columns={frame.index.name or "index": "date"})
    frame = frame.rename(columns={
        "日期": "date", "开盘": "open", "最高": "high",
        "最低": "low", "收盘": "close", "成交量": "volume",
    })
    columns = ["date", "open", "high", "low", "close", "volume"]
    for column in columns[1:]:
        if column not in frame:
            frame[column] = np.nan
    frame["date"] = pd.to_datetime(frame["date"])
    return frame[columns].set_index("date").sort_index()


def fetch_macro(data_dir: str, meta_dir: str, log) -> None:
    import akshare as ak

    try:
        rates = ak.bond_zh_us_rate()
        rates = rates.rename(columns={"日期": "date"})
        rates["date"] = pd.to_datetime(rates["date"])
        rates = rates[[
            "date", "中国国债收益率2年", "中国国债收益率5年",
            "中国国债收益率10年", "中国国债收益率30年",
        ]].set_index("date").sort_index()
        rates.to_csv(os.path.join(data_dir, "zh_10y_treasury.csv"))
        log(f"10y treasury cached: {len(rates)} rows")
    except Exception as error:
        log(f"10y treasury ERR {error}")

    try:
        pe = ak.stock_a_ttm_lyr()
        pe["date"] = pd.to_datetime(pe["date"])
        pe = pe[["date", "middlePETTM", "averagePETTM", "middlePELYR", "averagePELYR"]]
        pe = pe.set_index("date").sort_index()
        os.makedirs(meta_dir, exist_ok=True)
        pe.to_csv(os.path.join(meta_dir, "all_a_pe.csv"))
        log(f"all-a PE cached: {len(pe)} rows")
    except Exception as error:
        log(f"all-a PE ERR {error}")

    index = None
    try:
        index = ak.stock_zh_index_daily(symbol="sh000906")
        if index is not None:
            index = index.reset_index() if "date" not in index.columns else index
    except Exception as error:
        log(f"000906 sina ERR {error}")
    if index is None or index.empty:
        try:
            index = ak.index_zh_a_hist(
                symbol="000906", period="daily", start_date="20150101",
                end_date=datetime.now().strftime("%Y%m%d"),
            )
        except Exception as error:
            log(f"000906 em ERR {error}")
    if index is not None and not index.empty:
        index = _normalize_index(index)
        index.to_csv(os.path.join(data_dir, "000906_market.csv"))
        log(f"000906 cached: {len(index)} rows")

    for symbol, tag in (("sh000300", "000300"), ("sh000905", "000905")):
        try:
            index = ak.stock_zh_index_daily(symbol=symbol)
        except Exception as error:
            log(f"{tag} sind ERR {error}")
            continue
        if index is not None and not index.empty:
            index = _normalize_index(index)
            index.to_csv(os.path.join(data_dir, f"{tag}_market.csv"))
            log(f"{tag} cached: {len(index)} rows")
