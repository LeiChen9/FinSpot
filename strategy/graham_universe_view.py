"""Cached security metadata used by Graham-derived strategies."""

from typing import Callable, Dict, Optional


class UniverseView:
    def __init__(self, load_universe: Callable):
        self._load_universe = load_universe
        self._names: Optional[Dict[str, str]] = None
        self._industries: Optional[Dict[str, str]] = None

    def name_of(self, code: str) -> str:
        if self._names is None:
            universe = self._load_universe()
            self._names = dict(zip(universe["code"], universe["name"]))
        return self._names.get(code, code)

    def industry_of(self, code: str) -> str:
        if self._industries is None:
            universe = self._load_universe()
            self._industries = {}
            if "industry" in universe.columns:
                for symbol, industry in zip(universe["code"], universe["industry"].fillna("")):
                    self._industries[str(symbol)] = str(industry)
        return self._industries.get(code, "")


def is_st_name(name: str) -> bool:
    """Reject currently labelled ST securities from defensive candidates."""
    return bool(name and "ST" in str(name).upper())
