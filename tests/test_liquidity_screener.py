import pandas as pd

from screener.liquidity import LiquidityScreener


def _write_stock(path, code, dates, volumes):
    pd.DataFrame({
        "date": dates,
        "open": [10.0] * len(dates),
        "high": [10.0] * len(dates),
        "low": [10.0] * len(dates),
        "close": [10.0] * len(dates),
        "volume": volumes,
    }).to_csv(path / f"{code}_qfq.csv", index=False)


def test_liquidity_screen_uses_only_data_before_as_of(tmp_path):
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    _write_stock(tmp_path, "000001", dates, [2_000_000] * 5 + [100_000_000])
    _write_stock(tmp_path, "000002", dates, [3_000_000] * 6)

    screener = LiquidityScreener(
        data_dir=tmp_path,
        top_n=1,
        lookback_days=5,
        min_history_days=5,
        min_daily_amount=1e7,
        min_daily_volume=1e6,
    )

    early = screener.run(as_of_date="2024-01-05")
    late = screener.run(as_of_date="2024-01-06")

    assert early.codes == ["000002"]
    assert late.codes == ["000001"]
    assert early.info["as_of_date"] == pd.Timestamp("2024-01-05")
