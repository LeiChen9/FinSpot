"""基础研究框架 — 所有 Researcher 的抽象基类"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional
from data.manager import DataManager
from indicators.technical import moving_averages


class BaseResearcher(ABC):
    """所有研究器的抽象基类

    提供统一的接口和共享方法（如移动平均线计算）。
    """

    def __init__(self, code: str, data_manager: Optional[DataManager] = None):
        self.code = code
        self.data_manager = data_manager or DataManager()
        self.meta: Dict = {}
        self.data: Dict[str, pd.DataFrame] = {}
        self.metrics: Dict = {}

    @abstractmethod
    def get_basic_info(self):
        ...

    @abstractmethod
    def load_market_data(self, **kwargs):
        ...

    def calculate_moving_averages(self, mas: Optional[List[int]] = None):
        """计算移动平均线（所有 Researcher 共享）"""
        if mas is None:
            mas = [5, 30, 120]
        if 'price' in self.data and not self.data['price'].empty:
            self.data['price'] = moving_averages(self.data['price'], mas)
        return self

    @abstractmethod
    def _print_report(self):
        ...

    def plot_analysis(self, **kwargs):
        raise NotImplementedError("子类需实现 plot_analysis")

    def __repr__(self) -> str:
        name = self.meta.get('name', self.code)
        return f"<{type(self).__name__}: {name} ({self.code})>"
