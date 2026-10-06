import pandas as pd

from data.local import cached_market_loader, load_market_frames


def test_local_market_loader_reads_and_caches_csv(tmp_path):
    pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "close": [10, 11]}).to_csv(
        tmp_path / "000001_market.csv", index=False
    )
    frames = load_market_frames(["000001", "missing"], tmp_path, min_rows=2)
    assert list(frames) == ["000001"]
    loader = cached_market_loader(tmp_path, min_rows=2)
    assert len(loader("000001", "2024-01-02")) == 1
    assert loader("missing", "2024-01-02") is None
