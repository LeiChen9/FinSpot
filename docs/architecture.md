# FinSpot Architecture

## Purpose

FinSpot is a local-first research and backtesting project for China-market strategies. Jupyter notebooks are the user-facing research entrypoints. Python modules own reusable strategy, execution, screening, and data-access behavior.

## Dependency Direction

```text
notebooks
  -> strategy
      -> backtest
      -> screener
      -> dataload
          -> common
  -> dataload
```

`backtest` is strategy-agnostic. `strategy` may use backtest primitives, but backtest must not import strategy modules. `screener` and `dataload` are shared domains; notebooks should not duplicate their implementation. Keep dependencies flowing inward toward shared mechanisms and data access, not back toward notebooks.

## Domain Ownership

| Path | Owns | Put here |
|---|---|---|
| `notebooks/` | Research workflow and result inspection | Data selection, strategy configuration, calls, tables, plots, comparisons |
| `scripts/` | Ephemeral scratch (gitignored) | Throwaway validation harnesses, CLI smoke runs; promote validated logic up the stack |
| `strategy/` | Complete strategy concepts | Screening policy specific to a strategy, portfolio rules, strategy-specific backtest orchestration |
| `backtest/` | Reusable execution and result mechanics | Portfolio ledger, generic target-weight engine, calendar, costs, trade analysis, performance metrics |
| `screener/` | Point-in-time candidate selection | Reusable liquidity, dividend-value, Graham, and fixed-pool screens |
| `dataload/` | Local data access and acquisition | Cached readers, source adapters, market/fundamental pipelines |
| `common/` | Cross-domain configuration | Project root and shared data-directory paths |
| `data/` | Local datasets and caches only | Market, financial, dividend, metadata, and fetch caches; never source code |
| `reports/` | Research records and generated artifacts | Keep source notes under version control; `reports/generated/` is local output |

Each `strategy/*.py` file should describe one strategy. Shared policy belongs in `screener/`; reusable execution mechanics belong in `backtest/`. Do not create compatibility shims for obsolete module paths.

## Notebook Contract

A notebook should follow this sequence:

1. Locate the project root and import domain APIs.
2. Load local inputs through `dataload` or a strategy-owned data loader.
3. Set dates, capital, costs, and strategy parameters.
4. Construct and run a strategy or the generic weight engine.
5. Inspect returned NAV, trades, holdings, metrics, and plots.

The notebook is the experiment record, not the implementation home for reusable trading rules, signal generation, order execution, or data-source adapters. Analysis and visualization of returned results are appropriate notebook content.

## Data and Reproducibility

`common.paths` defines `ROOT`, `DATA_DIR`, `FIN_DIR`, `DIV_DIR`, and `META_DIR`. Data files live under the project-root `data/` directory. `.gitignore` excludes all contents of `data/`, local baseline outputs under `tests/baseline/`, and generated reports. Only code and authored documentation should be committed.

Readers in `dataload.readers` cache loaded frames in-process. Clear the Python process or restart the notebook kernel after replacing a cached file. Historical notebook outputs are not guaranteed to reproduce if local market files were refreshed, especially adjusted prices, or if a notebook fetches live data.

## Runtime

Use the existing Conda environment `fund`; do not create a virtual environment. Core Python dependencies are declared in `pyproject.toml`. Network fetchers use the `data` optional dependency group. Run tests with:

```bash
conda run -n fund python -m pytest tests/
```

The two data preparation entrypoints are `python -m dataload.pipeline_graham <command>` and `python -m dataload.pipeline_local --stage <stage>`. Run them from the project root in the `fund` environment.
