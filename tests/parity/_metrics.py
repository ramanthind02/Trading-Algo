"""Headline metric extraction and deterministic shaping for parity snapshots.

The comparison surface is the project's own metrics layer
(``feature_selection.validation.objective_metrics`` → ``quantfoundry_core.metrics``)
plus ``lib.metrics.drawdown``. We compute the *headline* scalars the spec calls
for (Sharpe, Sortino, max drawdown, Calmar, total return) from a daily returns
series, using the project functions so the snapshot reflects production math, not
a re-implementation.

Imports here are intentionally lightweight: the metrics layer imports fine even
when the pipeline *config* modules do not, so this module never blocks collection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.validation.objective_metrics import (
    metric_calmar,
    metric_sharpe,
    metric_sortino,
)
from lib.metrics.drawdown import max_drawdown

# Daily research series → annualised ratios use the trading-day convention.
ANNUALIZATION_FACTOR = 252.0


def headline_metrics(returns: pd.Series) -> dict[str, float]:
    """Compute the headline metric scalars from a daily returns series.

    Returns a dict with: sharpe, sortino, max_drawdown, calmar, total_return,
    n_obs, mean_return, std_return. Empty input yields all-zero metrics.
    """
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    clean = clean.sort_index()
    if clean.empty:
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "total_return": 0.0,
            "n_obs": 0.0,
            "mean_return": 0.0,
            "std_return": 0.0,
        }
    sharpe = float(metric_sharpe(clean, annualization_factor=ANNUALIZATION_FACTOR))
    sortino = float(metric_sortino(clean, annualization_factor=ANNUALIZATION_FACTOR))
    calmar = float(metric_calmar(clean, annualization_factor=ANNUALIZATION_FACTOR))
    mdd = float(max_drawdown(returns=clean))
    # Total return as cumulative product of (1 + r) - 1 (research returns are
    # per-bar simple/log-additive; compounding gives a stable headline number).
    total_return = float(np.expm1(np.log1p(clean).sum()))
    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "total_return": total_return,
        "n_obs": float(len(clean)),
        "mean_return": float(clean.mean()),
        "std_return": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
    }


def normalize_returns_series(returns: pd.Series, name: str = "strategy_return") -> pd.Series:
    """Deterministic returns series: datetime index, sorted, named, float."""
    s = returns.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s.astype(float)
    s.name = name
    return s


def normalize_positions_frame(positions: pd.DataFrame) -> pd.DataFrame:
    """Deterministic per-(ticker, datetime) position_fraction frame.

    Output: columns ['ticker', 'datetime', 'position_fraction'], sorted by
    (ticker, datetime), RangeIndex reset. Datetime floored to seconds to match
    the pipeline's own granularity normalization.
    """
    cols = ["ticker", "datetime", "position_fraction"]
    df = positions.loc[:, cols].copy()
    df["ticker"] = df["ticker"].astype(str)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("s")
    df["position_fraction"] = df["position_fraction"].astype(float)
    df = (
        df.sort_values(["ticker", "datetime"])
        .reset_index(drop=True)
    )
    return df
