"""策略模型之间共享的数据对象。"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class PeriodSnapshot:
    """单个调仓时点的因子和前瞻收益快照。"""

    date: datetime
    stocks: list[str]
    factor_df: pd.DataFrame
    forward_ret: pd.Series
    factor_returns: np.ndarray
    residual_var: float


@dataclass
class BacktestResult:
    """多因子回测的权重、净值和诊断结果。"""

    weights_history: list[dict] = field(default_factory=list)
    daily_nav: pd.Series = field(default_factory=pd.Series)
    performance: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)
