"""Index constituent universe for the dividend pipeline."""

import os
import time

import pandas as pd


def build_universe(meta_dir: str, index_codes: list[str]) -> list[str]:
    import akshare as ak

    codes = []
    for index_code in index_codes:
        for attempt in range(3):
            try:
                frame = ak.index_stock_cons(symbol=index_code)
                codes.extend(frame["品种代码"].astype(str).str.zfill(6).tolist())
                print(f"   [√] {index_code} 成分 {len(frame)} 只")
                break
            except Exception as error:
                print(f"   [x] {index_code} 第{attempt + 1}次失败: {error}")
                time.sleep(2)
    codes = sorted(set(codes))
    os.makedirs(meta_dir, exist_ok=True)
    pd.DataFrame({"code": codes}).to_csv(os.path.join(meta_dir, "universe.csv"), index=False)
    print(f"   并集共 {len(codes)} 只")
    return codes


def load_universe(meta_dir: str) -> list[str]:
    path = os.path.join(meta_dir, "universe.csv")
    if not os.path.exists(path):
        raise RuntimeError("先运行 --stage constituents")
    return pd.read_csv(path, dtype={"code": str})["code"].astype(str).str.zfill(6).tolist()
