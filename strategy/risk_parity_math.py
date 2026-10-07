"""Pure risk-parity weight optimization."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calc_erc_weights(cov: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
    assets = cov.index
    matrix = cov.values
    n = len(assets)

    def risk_contribution(weights):
        sigma = np.sqrt(weights @ matrix @ weights)
        return weights * (matrix @ weights) / sigma

    def objective(weights):
        contributions = risk_contribution(weights)
        return np.sum((contributions - contributions.mean()) ** 2)

    constraints = [{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}]
    bounds = [(0.0, max_weight)] * n
    best = None
    best_objective = np.inf
    seeds = [np.ones(n) / n]
    for index in range(n):
        seed = np.zeros(n)
        seed[index] = 1.0
        seeds.append(seed)
    for seed in seeds:
        result = minimize(objective, seed, method="SLSQP", bounds=bounds,
                          constraints=constraints,
                          options={"ftol": 1e-12, "maxiter": 10000})
        if result.success and result.fun < best_objective:
            best_objective = result.fun
            best = result
    if best is None:
        return pd.Series(np.ones(n) / n, index=assets)
    return pd.Series(best.x, index=assets)
