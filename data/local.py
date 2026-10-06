"""兼容入口：本地行情加载实现位于 :mod:`data.loaders`。"""

from data.loaders import cached_market_loader, load_market_frames

__all__ = ["cached_market_loader", "load_market_frames"]
