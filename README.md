# FinSpot

中国市场策略研究与回测工具。

## Runtime

本项目固定使用 Conda 环境 `fund`，不创建或使用 virtualenv。运行 Python
脚本时使用 `conda run -n fund python ...`；每次 AI 协助编程也默认该环境。

## Layout

- `notebooks/`: 研究入口，只负责加载数据、配置策略、运行回测和检查结果
- `strategy/`: 每个文件维护一个完整策略
- `backtest/`: 通用组合执行、交易成本、日历、回测结果与指标
- `screener/`: point-in-time 股票池与基本面筛选
- `dataload/`: 本地缓存读取、在线数据源与数据准备管线
- `common/`: 数据路径等跨领域基础配置
- `data/`: 本地行情、财务、分红和缓存，整个目录不入 Git

notebook 可从项目根目录执行，例如 `conda run -n fund jupyter lab`。抓取管线通过
`python -m dataload.pipeline_graham` 或 `python -m dataload.pipeline_local` 调用。
报告输出位于被忽略的 `reports/generated/`。
