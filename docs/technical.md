# FinSpot Technical Reference

This is an implementation map for starting work in a fresh session. Prefer these APIs over recreating behavior in a notebook.

## Backtest

| Module | Implemented API | Use |
|---|---|---|
| `backtest.engine` | `Portfolio`, `Holding`, `market_days`, `whole_lot_shares`, `affordable_shares`, `buy_cost`, `sell_cost`, `dividends_by_day`, `buy_and_hold_nav`; `COMMISSION`, `STAMP_TAX`, `SLIPPAGE`, `MIN_FEE` | Cash/holdings ledger, trade log, daily snapshots, cost/sizing/benchmark helpers, common trading constants |
| `backtest.weights` | `BTResult`, `quarterly_rebalance_dates(dates, start, end)`, `run_weight_backtest(all_data, target_w, rebal_dates, init_cap=..., comm=..., tax=..., cap_code=None, cap=None)` | Daily-marked, periodic target-weight portfolio; optional single-asset cap with excess redistributed proportionally |
| `backtest.metrics` | `perf_metrics`, `performance_summary`, `nav_summary`, `calc_perf`, `formatted_perf_row`, `monthly_returns` | NAV and return summaries, notebook result tables |
| `backtest.trades` | `holdings_frame`, `turnover_summary`, `fifo_pnl`, `fifo_trade_outcomes`, `trade_statistics` | Trade and holding analysis |
| `backtest.rebalance` | `first_trading_day`, `scheduled_rebalance_dates`, `fixed_weight_history` | Calendar and fixed-weight history helpers |

The target-weight engine is suitable for periodic asset allocation. Event-driven or strategy-specific rules use the appropriate strategy class and share only the ledger/calendar/cost primitives they need.

## Strategies

| Module | Public entrypoints | Current use |
|---|---|---|
| `strategy.graham_dodd` | `snapshot`, `evaluate`, `screen_all`, `rebalance_dates`, `GrahamStrategy` | Graham & Dodd screening, point-in-time evaluation, dividend/corporate-action accounting, backtest |
| `strategy.buffett_quality` | `annual_signal_dates`, `rank_candidates`, `build_ranking_map`, `QualityStrategy`, `audit_borderline` | Conservative annual quality proxy; notebook `magic_formula_a_share.ipynb` |
| `strategy.buffetts_alpha` | `big_pool`, `value_rank`, `gated_pool`, `add_composite`, `select_topN`, `quarter_dates`, `run_full` | Value/safety/quality composite; notebook `buffetts_alpha_a_share.ipynb` |
| `strategy.magic_formula` | `screen_pool`, `rank_candidates`, `pick_top`, `build_ranking_map`, `MagicStrategy`, `cyclical_comparison` | Greenblatt-style ranking and cyclical-sector comparison |
| `strategy.turtle` | `TurtleSignal`, `TurtleStrategy` | ATR position sizing and Turtle breakout; notebook `atr_channel_strategy.ipynb` |
| `strategy.band_ma120` | `BandStrategy` | High-dividend pool with MA120 entry/add/sell bands |
| `strategy.dividend_ma120` | `MeanReversionStrategy`, `run_backtest`, `target_weight` | MA120 target-weight mean reversion; retained for direct strategy use |
| `strategy.donchian_value` | `DonchianValueStrategy` | Monthly value pool with Donchian entry/exit |
| `strategy.double_bottom` | `find_pivots`, `detect_double_bottom`, `DoubleBottomStrategy` | Double-bottom breakout, adds, and segment liquidation |
| `strategy.all_weather` | `TARGET_WEIGHTS`, `BENCH_6040_WEIGHTS` | A-share all-weather allocation configuration |
| `strategy.permanent_portfolio` | `TARGET_WEIGHTS`, `QDII_CODE`, `QDII_CAP`, benchmark/asset maps | China permanent-portfolio allocation configuration |
| `strategy.bank_pb` | `load_bank_data`, `screen_top10_banks`, `BankPBStrategy` | Bank-sector low-PB screen and daily rebalance/add/switch rules |
| `strategy.growth_stock` | `load_all_market_data`, `precompute_indicators`, `screen_stocks`, `GrowthStockStrategy`, `get_rebalance_dates` | Momentum/value-zone growth-stock experiment |
| `strategy.dividend_bond_spread` | `DividendBondSpreadStrategy` | Dividend-yield minus 10Y treasury spread threshold DCA; unadjusted price + cash-dividend reinvest; notebook `dividend_bond_spread.ipynb` |

Strategy classes generally return NAV DataFrames or maintain NAV/trade histories on the instance. Read each class signature before choosing the input data shape; `TurtleStrategy` and `DoubleBottomStrategy` accept `{code: DataFrame}` market maps, while `BandStrategy` receives its market map and screener at construction.

## Screening

| Module | Entry points | Inputs and result |
|---|---|---|
| `screener.liquidity` | `LiquidityScreener`, `FilterResult` | Reads local `_qfq.csv`; `run(as_of_date=None, pool=None)` returns codes, names, and diagnostic info |
| `screener.pool` | `POOL`, `signal_frame`, `screen_pool`, `build_pool_screener`, `ma120_dev`, `vol120`, `dividend_yield`, `pe_ttm` | Fixed 28-stock pool; point-in-time signal frames and daily lookup screener |
| `screener.dividend_value` | `build_value_screener`, `universe_codes`, `market_cap`, `roe_3y`, `factor_detail`, `stock_name` | Whole-market monthly top-N screen using dividend, PE, ROE, and market-cap factors |
| `screener.graham` | `GrahamScreener`, `GrahamHolding` | THS financial-summary based Graham/dividend pool; `run(as_of, codes)` returns passed codes and details |

## Data Access and Acquisition

### Local readers

`dataload.readers` is the common local read API:

- Universe/macro: `load_universe()`, `load_all_a_pe()`, `load_10y()`.
- Market: `load_market(code, qfq=False)`, `load_qfq(code)`, `load_raw(code)`, `load_index(code)`.
- Fundamentals/actions: `load_balance(code)`, `load_profit(code)`, `load_fin_summary(code)`, `load_dividend(code)`.
- Batch/as-of: `load_market_frames(codes, data_dir=DATA_DIR, min_rows=1)`, `cached_market_loader(data_dir=DATA_DIR, min_rows=1)`.

`load_qfq` falls back to the market file when a qfq file is absent. Report readers parse report and announcement dates for point-in-time filtering. The process-local cache is intentional; restart the kernel to reload changed files.

### Sources and pipelines

| Path | Functionality |
|---|---|
| `dataload.sources.qq` | Tencent QQ daily market fetcher |
| `dataload.sources.baostock` | Baostock A-share/HK market fetchers |
| `dataload.market_fetch` | Sina/Akshare/QQ/Baostock fallback for Graham market history |
| `dataload.market_local` | QQ forward-adjusted market downloader for the local pipeline |
| `dataload.pipeline_graham` | Full-A universe, statements, dividends, market, macro, and industry preparation |
| `dataload.pipeline_local` | Constituents, market, financials, dividends, and benchmarks by stage |
| `dataload.all_weather` | QQ/Sina/ChinaBond cache-backed input assembly for all-weather notebook |
| `dataload.etf` | `fetch_data(code, start, end, adjust='qfq')` (QQ/Akshare, cache `data/etf_{code}[_raw].csv`, refetches when cache misses the range) and `fetch_etf_dividends(symbol)` |

Examples from the project root:

```bash
conda run -n fund python -m dataload.pipeline_graham prepare-universe
conda run -n fund python -m dataload.pipeline_graham fetch-all --workers 8
conda run -n fund python -m dataload.pipeline_local --stage all
```

## Notebook Map

| Notebook | Strategy/API | Notes |
|---|---|---|
| `graham_dodd_a_share.ipynb` | `strategy.graham_dodd` | Main defensive Graham backtest |
| `graham_trend_compare.ipynb` | `strategy.graham_dodd`, `strategy.magic_formula.cyclical_comparison` | Graham ranking comparison and cyclical formula diagnostics |
| `magic_formula_a_share.ipynb` | `strategy.buffett_quality` | Despite the filename, this is the Buffett quality proxy |
| `buffetts_alpha_a_share.ipynb` | `strategy.buffetts_alpha` | Buffett Alpha gates and quarterly portfolio |
| `atr_channel_strategy.ipynb` | `strategy.turtle`, `screener.liquidity` | Despite the filename, this runs Turtle rules |
| `band_highdiv_ma120.ipynb` | `strategy.band_ma120`, `screener.pool` | High-dividend MA120 band strategy |
| `donchian_value.ipynb` | `strategy.donchian_value`, `screener.dividend_value` | Value pool × Donchian |
| `double_bottom_strategy.ipynb` | `strategy.double_bottom`, `screener.liquidity` | Double-bottom segments and trade analysis |
| `all_weather_a_share.ipynb` | `dataload.all_weather`, `strategy.all_weather`, `backtest.weights` | Local cache-backed all-weather portfolio |
| `permanent_portfolio_cn.ipynb` | `dataload.etf`, `strategy.permanent_portfolio`, `backtest.weights` | ETF/index data may be refreshed from network |
| `bank_pb_strategy.ipynb` | `strategy.bank_pb` | The retained, more complete bank-PB iteration |
| `growth_stock_strategy.ipynb` | `strategy.growth_stock` | Growth-stock experiment |
| `dividend_bond_spread.ipynb` | `strategy.dividend_bond_spread`, `dataload.etf` | Dividend-yield/treasury spread DCA variants and benchmark |

## Tests and Reproducibility

Tests are offline unit tests for the portfolio ledger, FIFO/trade analytics, local market loading, and point-in-time liquidity screening. Run them in the existing Conda environment:

```bash
conda run -n fund python -m pytest tests/
```

Notebook outputs can drift from older embedded results when local CSVs have changed, adjusted history is revised, or live data is fetched. In particular, treat `permanent_portfolio_cn.ipynb` as data-date dependent. Do not change strategy behavior to force a stale embedded output to match; compare using the same local input snapshot when exact parity is required.
