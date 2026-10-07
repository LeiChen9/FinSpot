"""Pure double-bottom pattern detection."""

import numpy as np
import pandas as pd


def find_pivots(close, left=3, right=3):
    high = np.zeros(len(close), dtype=bool)
    low = np.zeros(len(close), dtype=bool)
    for index in range(left, len(close) - right):
        window = close[index - left:index + right + 1]
        value = close[index]
        if value == window.max() and (window == value).sum() == 1:
            high[index] = True
        if value == window.min() and (window == value).sum() == 1:
            low[index] = True
    return high, low


def detect_double_bottom(close, pivot_left=3, pivot_right=3,
                         bottom_tolerance=0.03, min_rebound=0.05,
                         min_separation=5, max_separation=60,
                         breakout_buffer=0.005):
    close = np.asarray(close, dtype=float)
    highs, lows = find_pivots(close, left=pivot_left, right=pivot_right)
    high_points, low_points = np.where(highs)[0], np.where(lows)[0]
    signals = []
    for first_index, first in enumerate(low_points):
        for second in low_points[first_index + 1:]:
            distance = second - first
            if distance < min_separation:
                continue
            if distance > max_separation:
                break
            if abs(close[first] - close[second]) / close[first] > bottom_tolerance:
                continue
            peaks = high_points[(high_points > first) & (high_points < second)]
            if len(peaks) == 0:
                continue
            peak = peaks[np.argmax(close[peaks])]
            bottom = max(close[first], close[second])
            if (close[peak] - bottom) / bottom < min_rebound:
                continue
            neckline = close[peak]
            first_breakout = second + pivot_right + 1
            for breakout in range(first_breakout, len(close)):
                if close[breakout] > neckline * (1 + breakout_buffer):
                    signals.append({
                        "pattern_start": first, "bottom1": first,
                        "middle_peak": peak, "bottom2": second,
                        "neckline": neckline, "breakout": breakout,
                        "breakout_price": close[breakout],
                        "breakout_level": neckline * (1 + breakout_buffer),
                    })
                    break
    return pd.DataFrame(signals)
