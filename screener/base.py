"""选股筛选器基类 — 支持链式调用"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class FilterResult:
    codes: List[str] = field(default_factory=list)
    names: Dict[str, str] = field(default_factory=dict)  # 股票代码 -> 名称映射
    info: Dict[str, dict] = field(default_factory=dict)
    passed: pd.Series = None


class Screener(ABC):
    def __init__(self):
        self._steps: List[str] = []

    @abstractmethod
    def run(self, as_of_date, pool: Optional[List[str]] = None) -> FilterResult:
        ...

    def _record_step(self, name: str, before: int, after: int):
        self._steps.append(f"{name}: {before} -> {after}")

    def summary(self) -> List[str]:
        return self._steps

    def then(self, other: "Screener") -> "ChainScreener":
        return ChainScreener(self, other)


class ChainScreener(Screener):
    def __init__(self, *screeners: Screener):
        super().__init__()
        self._screeners = screeners

    def run(self, as_of_date, pool: Optional[List[str]] = None) -> FilterResult:
        result = None
        for s in self._screeners:
            result = s.run(as_of_date, pool=pool if result is None else result.codes)
            if result and result.codes:
                pool = result.codes
        return result
