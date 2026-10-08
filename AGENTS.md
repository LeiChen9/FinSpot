# Project Instructions

- The project uses the Conda environment `fund`.
- Do not create or use Python virtual environments for this repository.
- Run project scripts and tests with the existing `fund` Conda environment.
- AI coding assistance should assume `fund` is the active project environment.
- Keep credentials, tokens, passwords, and local environment files out of Git.
- Preserve strategy behavior. Obsolete module paths may be removed as part of an approved refactor.
- Before making structural or cross-module changes, read `docs/architecture.md` and `docs/technical.md`; they document the project architecture and organization.

# File Organization & Domain Standards

Follow these rules for every change. They are the enforceable summary of `docs/architecture.md`; keep the two in sync.

## Dependency direction (never reverse)

```text
notebooks / scripts
  -> strategy
      -> backtest
      -> screener
      -> dataload
          -> common
  -> dataload
```

`backtest` must never import `strategy`. `strategy` may use `backtest`, `screener`, `dataload`. Never import back from a lower layer.

## Domain ownership

| Path | Owns | Put here | Never here |
|---|---|---|---|
| `notebooks/` | Research record | Data selection, parameters, calls, tables, plots, comparisons | Reusable rules, order execution, data adapters, a copied backtest loop |
| `scripts/` | Ephemeral scratch (gitignored) | Throwaway validation harnesses, CLI smoke runs | Durable code, reusable logic, duplicated engines, anything the user will rely on |
| `strategy/` | One file = one complete strategy | Signals, portfolio rules, strategy-specific orchestration | Raw data acquisition, printing/reporting, ledger/trade mechanics that already exist in `backtest` |
| `backtest/` | Reusable execution mechanics | Ledger, calendar, costs, sizing, benchmark NAV, metrics, trade analysis | Any specific strategy logic |
| `screener/` | Point-in-time candidate selection | Reusable screens | Strategy-specific rules |
| `dataload/` | Data access and acquisition | Cached readers, source adapters, fetch/caching | Strategy logic; akshare/requests calls outside this package |
| `common/` | Cross-domain config | Root and shared paths | Domain logic |
| `data/` | Local datasets/caches only | Market/fundamental/dividend files | Source code (and never commit contents) |
| `reports/` | Research artifacts | Authored notes at `reports/` root; generated output under `reports/generated/` | Committed generated CSVs/PNGs |
| `tests/` | Offline unit tests | Deterministic, network-free tests | Live-data experiments, misnamed `test_*` scripts |

## Hard rules

1. **Data acquisition lives in `dataload`.** ETF/index price via `dataload.etf.fetch_data`, dividends via `dataload.etf.fetch_etf_dividends`, macro via `dataload.readers`. Do not call `akshare`/`requests` directly from `strategy`, `notebooks`, or `scripts`.
2. **Reuse `backtest` primitives; do not hand-roll them.** Costs: `buy_cost`/`sell_cost` (they include the 5 元 minimum), sizing: `whole_lot_shares`/`affordable_shares`, benchmark: `buy_and_hold_nav`, dividend alignment: `dividends_by_day`, metrics: `backtest.metrics`. If a mechanic is missing, add it to `backtest` and reuse it, not copy-paste.
3. **One strategy per `strategy/*.py` file**, with an `__all__` and a module docstring stating the rules and data 口径. Name the strategy class `<Name>Strategy` (not `<Name>Backtest`; reusable execution mechanics belong in `backtest`).
4. **Notebooks are records, not engines.** A notebook must call the strategy class / domain APIs and inspect returned results. Never paste the backtest loop or signal generation into a notebook.
5. **`scripts/` is ephemeral scratch, gitignored.** Use it to run an idea end to end, then discard it. Promote validated behavior to a notebook (visualization and user tuning) or to `strategy`/`backtest`/`dataload` for reuse. Never persist durable logic there.
6. **Price 口径 must be self-consistent.** Either unadjusted prices + cash dividends, or forward-adjusted prices with NO separate dividend accounting. Never both (double-counts dividends). State the choice in the docstring. `fetch_data(adjust='')` = unadjusted; `adjust='qfq'` = forward-adjusted.
7. **No dead or duplicated code.** Experimental switches belong as documented, default-off strategy parameters, exercised from a notebook — not as copied variant scripts. Delete superseded files instead of keeping parallel copies.
8. **Reproducibility.** Cache readers must not silently return truncated history for a requested range. Generated artifacts go under `reports/generated/` (gitignored).
9. **Keep `docs/technical.md` current.** When adding/removing a strategy or public dataload/backtest API, update its tables in the same change.

## Verification

Run tests in the existing environment (never create a venv):

```bash
conda run -n fund python -m pytest tests/
```
