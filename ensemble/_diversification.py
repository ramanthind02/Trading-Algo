"""Shared correlation helpers for diversification multipliers."""

from __future__ import annotations

import numpy as np
import pandas as pd

_DIVERSIFICATION_EPSILON = 0.01


def _finite_off_diagonal_correlations(corr_matrix: pd.DataFrame) -> np.ndarray:
    n_cols = len(corr_matrix.columns)
    if n_cols <= 1:
        return np.array([], dtype=float)
    mask = np.triu(np.ones((n_cols, n_cols), dtype=bool), k=1)
    correlations = corr_matrix.to_numpy(dtype=float)[mask]
    return correlations[np.isfinite(correlations)]


def positive_clipped_correlation(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    clipped = corr_matrix.clip(lower=0.0)
    np.fill_diagonal(clipped.values, 1.0)
    return clipped


def mean_off_diagonal_correlation(corr_matrix: pd.DataFrame) -> float:
    correlations = _finite_off_diagonal_correlations(corr_matrix)
    if len(correlations) == 0:
        return 1.0
    return float(correlations.mean())


def capped_diversification_multiplier(
    corr_matrix: pd.DataFrame,
    *,
    multiplier_cap: float,
    epsilon: float = _DIVERSIFICATION_EPSILON,
) -> tuple[float, float]:
    correlations = _finite_off_diagonal_correlations(corr_matrix)
    if len(correlations) == 0:
        return 1.0, 1.0
    mean_corr = float(correlations.mean())
    multiplier = float(np.sqrt(1.0 / (mean_corr + epsilon)))
    return mean_corr, min(multiplier, multiplier_cap)
