# FinSpot

中国市场策略研究与回测工具。

## Runtime

本项目固定使用 Conda 环境 `fund`，不创建或使用 virtualenv。运行 Python
脚本时使用 `conda run -n fund python ...`；每次 AI 协助编程也默认该环境。

## Layout

- `data/`: 本地数据源、缓存与数据准备管线
- `strategy/`: 策略规则、信号、交易执行与组合净值
- `screener/`: point-in-time 股票池与基本面筛选
- `analysis/`: 回测、绩效、风险和归因分析
- `notebooks/`: 研究入口与结果复核

本地行情、财务、分红、meta、报告和凭据文件均由 `.gitignore` 排除。
