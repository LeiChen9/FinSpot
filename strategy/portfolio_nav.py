"""Shared portfolio NAV construction from dated target weights."""

from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd


def build_nav_from_weights(weights_history: List[dict], all_data: Dict[str, pd.DataFrame]) -> Tuple[pd.Series, pd.DataFrame]:
    """Build daily NAV and drifted weights using rebalance snapshots."""
    if not weights_history:
        return pd.Series(), pd.DataFrame()
    all_dates = sorted(set(d for df in all_data.values() for d in df.index))
    shares: Dict[str, float] = {}
    nav_value = 1.0
    nav_records, weight_records = [], []
    rebalance_map = {e['date']: e['weights'] for e in weights_history}
    for d in all_dates:
        dt = datetime(d.year, d.month, d.day)
        if dt in rebalance_map:
            weights = rebalance_map[dt]
            if shares:
                nav_value = sum(sh * float(all_data[a].loc[d, 'close']) for a, sh in shares.items() if a in all_data and d in all_data[a].index) or nav_value
            nav_records.append({'date': dt, 'nav': nav_value})
            weight_records.append({'date': dt, **weights})
            new_shares, total_weight = {}, 0.0
            for asset, weight in weights.items():
                if weight <= 0 or asset not in all_data or d not in all_data[asset].index:
                    continue
                price = float(all_data[asset].loc[d, 'close'])
                if price > 0:
                    new_shares[asset] = weight / price
                    total_weight += weight
            shares = ({asset: quantity / total_weight * nav_value for asset, quantity in new_shares.items()}
                      if new_shares and total_weight > 0 else {})
            continue
        if not shares:
            continue
        total = sum(sh * float(all_data[a].loc[d, 'close']) for a, sh in shares.items() if a in all_data and d in all_data[a].index)
        if total <= 0:
            continue
        nav_records.append({'date': dt, 'nav': total})
        weight_records.append({'date': dt, **{asset: sh * float(all_data[asset].loc[d, 'close']) / total for asset, sh in shares.items() if asset in all_data and d in all_data[asset].index}})
    if not nav_records:
        return pd.Series(), pd.DataFrame()
    df_nav = pd.DataFrame(nav_records).set_index('date')
    df_nav['nav'] = df_nav['nav'] / df_nav['nav'].iloc[0]
    return df_nav['nav'], pd.DataFrame(weight_records).set_index('date')
