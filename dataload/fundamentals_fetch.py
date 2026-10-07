"""Financial-summary and dividend acquisition for the dividend pipeline."""

import os
import time

import pandas as pd


def fetch_financial_ths(code: str) -> pd.DataFrame | None:
    import akshare as ak
    frame = ak.stock_financial_abstract(symbol=code)
    return frame if frame is not None and not frame.empty else None


def stage_financial(codes: list[str], financial_dir: str) -> None:
    for index, code in enumerate(codes):
        path = os.path.join(financial_dir, f"{code}_fin.csv")
        if os.path.exists(path):
            header = pd.read_csv(path, nrows=1)
            if any(column.startswith("2025") for column in header.columns):
                continue
        try:
            frame = fetch_financial_ths(code)
            if frame is None:
                print(f"   [x] {code}: 无摘要")
                continue
            frame.to_csv(path, index=False)
            print(f"   [{index + 1}/{len(codes)}] {code}: {frame.shape}")
        except Exception as error:
            print(f"   [x] {code}: {error}")
        time.sleep(0.3)
    print("   [√] 财务摘要完成")


def stage_dividend(codes: list[str], dividend_dir: str) -> None:
    import akshare as ak
    for index, code in enumerate(codes):
        path = os.path.join(dividend_dir, f"{code}_dividend.csv")
        if os.path.exists(path):
            continue
        try:
            frame = ak.stock_dividend_cninfo(symbol=code)
            if frame is None or frame.empty:
                print(f"   [x] {code}: 无分红记录")
                continue
            frame.to_csv(path, index=False)
            print(f"   [{index + 1}/{len(codes)}] {code}: {len(frame)} 条")
        except Exception as error:
            print(f"   [x] {code}: {error}")
        time.sleep(0.3)
    print("   [√] 分红数据完成")
