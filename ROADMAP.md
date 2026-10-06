# 生产策略路线图

目标：以可追溯的市场、财务与分红数据生产策略，执行可重复回测，并输出可解释的绩效分析。

conda 环境名：fund

## 当前主线

1. 准备数据：`python scripts/prepare_data.py --stage all`。
2. 执行红利 MA120 主回测：`python scripts/run_dividend_backtest.py`。
3. 分析输出：使用 `analysis/` 的绩效、风险、归因、持有期和优化工具；策略历史与已验证结论见 `reports/strategy_review.md`。
4. 需要保留 CSV/图表时，显式指定目录：`python scripts/run_dividend_backtest.py --output-dir reports/generated/dividend_ma120`。

## 目录职责

| 目录 | 职责 |
|---|---|
| `data/` | 可复用的行情、基金净值、财务、分红、成分股及数据源适配器 |
| `screener/` | 股票池与基本面筛选 |
| `strategy/` | 信号、组合构建、风险控制与回测引擎 |
| `analysis/` | 绩效、风险、归因、持仓和优化分析 |
| `core/`、`indicators/`、`visualization/` | 通用研究对象、指标和可视化能力 |
| `scripts/` | 数据准备、手动数据导入和可重复主回测入口 |
| `reports/` | 策略研究归档及按需生成的结果 |
| `tests/` | 公共数据与回测工具的回归测试 |

## 近期工作

- 为每个生产策略补齐独立配置、固定的数据截点和交易成本假设。
- 使用 walk-forward/样本外窗口评估多因子与均值-方差优化，避免样本内拟合。
- 统一各策略的交易成本、再平衡和基准口径，再进行横向比较。
- 在具备完整依赖的环境中运行测试，并将关键策略加入端到端回测测试。
