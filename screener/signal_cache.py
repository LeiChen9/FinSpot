"""Small explicit cache for expensive point-in-time signal frames."""

from typing import Dict, Optional
import pandas as pd


class SignalFrameCache:
    def __init__(self):
        self._frames: Dict[str, Optional[pd.DataFrame]] = {}

    def get(self, code: str) -> Optional[pd.DataFrame]:
        return self._frames.get(code)

    def contains(self, code: str) -> bool:
        return code in self._frames

    def put(self, code: str, frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        self._frames[code] = frame
        return frame
