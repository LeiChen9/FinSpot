# FinSpot 架构

notebooks 是研究入口，只保留业务编排；策略定义、回测机制、筛选和数据访问分别归属独立领域。

conda 环境名：fund

## 目录职责

| 目录 | 职责 |
|---|---|
| `notebooks/` | 加载数据、配置和调用策略、回测结果复核 |
| `strategy/` | 各个具体策略的筛选、持仓规则与策略回测编排 |
| `backtest/` | 目标权重/组合执行引擎、成本、交易分析和绩效指标 |
| `screener/` | 股票池与 point-in-time 基本面筛选 |
| `dataload/` | 本地数据读取、数据源和缓存准备管线 |
| `common/` | 共享路径和基础配置 |
| `data/` | 本地数据文件，全部 gitignored |
| `reports/` | 研究归档及 gitignored 的生成结果 |
| `tests/` | 离线回归测试 |

运行测试：`conda run -n fund python -m pytest tests/`。
