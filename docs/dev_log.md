# 开发日志

## 2026-10-08 红利低波利差定投策略立项与回测

### 策略定义（用户确认稿）
- 标的：510880 红利ETF；利差 = TTM股息率(近4次分红/现价) − 10Y国债利率
- 利差 >3% 买入持仓市值的 40%；>1.5% 买入 20%；1~1.5% 不动；<1% 清仓
- 首次建仓等信号首次 >1.5%，固定 1 万；清仓后信号再现可重新进场
- 分红当日现金全部再投入；本金 10 万，允许空仓

### 实现文件
- `strategy/dividend_bond_spread.py`
  - `DividendBondSpreadStrategy`：事件驱动回测，成本公式 `COMMISSION+SLIPPAGE`(买) / `+STAMP_TAX`(卖)
  - `fetch_data`(159307缓存) 或外部 `price_df`；`load_etf_dividends(symbol)` 取新浪分红累计差
  - 后续新增开关：`cap_mode`('legacy' 20%/40% | 'signal' 50%/100% 目标上限)、`trend_ma`（10月线趋势过滤+超额削减）、`vol_target`（波动率倒数缩放）
- `scripts/run_dividend_bond_spread.py`：159307 入口（2024 起，仅 15 个月数据）
- `scripts/run_510880.py`：510880 入口；输出 NAV、交易、买入持有对比(000300/399006/512890)、月度收益率到 `reports/dividend_yield_510880_monthly.csv`

### 数据结论
- 930955 中证红利低波动指数新浪/tx/东财均不可拉取；且中证红利低波 ETF 体系最早 2019(512890)，不满足 2010 起
- 159307 分红季度化但 2024-04 才上市；510880 2007-01 上市、2009-03 起有分红，可支持 2010 起回测
- 510880 分红年配/半年配 → "近4次分红" 实际跨 2-4 年，TTM 信号滞后（已知局限）

### 回测结果（510880，2007~2026-09，扣成本）
- legacy：终值 306,422 (+206.4%)，最大回撤 -46.5%，夏普(超10Y) 0.33（持仓起 2011-11）
- B legacy+trend：失效，3445 笔削减、+6.6%（加仓规则与削减互相冲突）
- C legacy+trend+vol：+24.4%，最大回撤 -36.8%，4765 笔削减
- **D signal+trend+vol：+52.8%，最大回撤 -21.0%，749 笔**（削减抖动需 >5% 滞回缓冲已加）

### 关键决策
- 阈值 1.5%/1%/3% 用户明确要求不动；仓位上限档位(如 50%/100%)作为新可选参数，默认关闭
- 踩空敏感度低、回撤敏感度高 → 推荐 D 档(目标上限+趋势+波动率)作为出风险优先配置
- 基准买入持有用未复权价格、未重投分红，D/对照口径不对称，尚未改为全收益指数

## 2026-10-08 复盘与修正

上一轮实现的回测结果不可信，已按领域规范重做：

- **分红重复计算**：`dataload.etf.fetch_data` 走前复权价(QQ qfq / akshare qfq)，策略又单独加现金分红再投 → 双计。现统一为不复权价 + 现金分红，`fetch_data(..., adjust='')`。
- **缓存截断**：`fetch_data` 命中旧缓存即返回，导致「2013 起」实跑 2024+。现按请求区间校验，不足则补抓。
- **成本口径**：忽略 `MIN_FEE`(5 元)，现统一用 `backtest.engine.buy_cost/sell_cost/affordable_shares`。
- **仓位基准**：legacy 市值 vs signal 初始本金混用；现统一以当前 NAV 为基。
- **数据获取归位**：`load_etf_dividends` 从 strategy 移入 `dataload.etf.fetch_etf_dividends`。
- **去重**：删除 `scripts/test_right_side_filters.py` 与两个复制回测循环的 notebook，合并为 `notebooks/dividend_bond_spread.ipynb`；买入持有复用 `backtest.engine.buy_and_hold_nav`。
- 生成物改写到 `reports/generated/`。上述规则已写入根 `AGENTS.md`。
