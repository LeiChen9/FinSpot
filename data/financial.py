"""本地财务摘要的读取与公告滞后截断。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


class FinancialDataLoader:
    """加载并缓存个股财务摘要，只暴露公告滞后后的最新记录。"""

    def __init__(self, fin_dir: str | Path = "data/financial", lag_months: int = 3):
        self.fin_dir = Path(fin_dir)
        self.lag_months = lag_months
        self._cache: dict[str, dict[str, dict[str, float]]] = {}

    def get_latest_financial(self, stock: str, as_of: datetime) -> dict | None:
        if stock not in self._cache:
            self._cache[stock] = self._read_stock(stock)
        records = self._cache[stock]
        cutoff = as_of.year * 12 + as_of.month - self.lag_months
        valid = [
            date for date in records
            if date[:4].isdigit() and int(date[:4]) * 12 + int(date[4:6]) <= cutoff
        ]
        return records[max(valid)] if valid else None

    def _read_stock(self, stock: str) -> dict[str, dict[str, float]]:
        path = self.fin_dir / f"{stock}_fin.csv"
        if not path.exists():
            return {}
        raw = pd.read_csv(path)
        if raw.empty or "指标" not in raw:
            return {}
        records: dict[str, dict[str, float]] = {}
        for _, row in raw.dropna(subset=["指标"]).iterrows():
            indicator = str(row["指标"]).strip()
            for date, value in row.iloc[2:].items():
                number = pd.to_numeric(value, errors="coerce")
                if pd.notna(number):
                    records.setdefault(str(date), {})[indicator] = float(number)
        return records
