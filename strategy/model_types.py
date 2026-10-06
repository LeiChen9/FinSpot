"""策略模型之间共享的数据对象。"""

from dataclasses import dataclass
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
