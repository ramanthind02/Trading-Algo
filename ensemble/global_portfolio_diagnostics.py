from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd


def collect_global_strategy_health_diagnostics(
    forecast_vectors: List[pd.DataFrame],
    global_returns: Optional[pd.Series],
) -> tuple[List[pd.DataFrame], Dict[str, Set[str]], Dict[str, Any]]:
    """Collect low-information diagnostics without filtering out any strategy."""
    min_obs = 252
    min_activity_ratio = 0.02
    min_effective_obs = 21

    returns_norm: Optional[pd.Series] = None
    if global_returns is not None and not global_returns.empty:
        returns_norm = global_returns.astype(float).copy()
        returns_norm.index = pd.to_datetime(returns_norm.index).normalize()

    combined = pd.concat(forecast_vectors, ignore_index=True)
    if combined.empty:
        return forecast_vectors, {}, {
            "min_obs": min_obs,
            "min_activity_ratio": min_activity_ratio,
            "min_effective_obs": min_effective_obs,
            "tickers": {},
        }

    tickers_diag: Dict[str, Any] = {}
    eligible_by_ticker: Dict[str, Set[str]] = {}

    for ticker, ticker_df in combined.groupby("ticker", sort=False):
        ticker_key = str(ticker)
        model_stats = []
        for model_name, model_df in ticker_df.groupby("model_name", sort=False):
            signal = model_df["signal"].astype(float)
            obs = int(len(signal))
            activity_ratio = float((signal.abs() > 1e-12).mean()) if obs else 0.0
            if returns_norm is not None:
                model_dates = pd.to_datetime(model_df["datetime"]).dt.normalize()
                model_rets = returns_norm.reindex(model_dates).fillna(0.0).to_numpy(dtype=float)
                effective_obs = int(
                    np.sum(np.abs(signal.to_numpy(dtype=float) * model_rets) > 1e-12)
                )
            else:
                effective_obs = int(np.sum(np.abs(signal.to_numpy(dtype=float)) > 1e-12))
            is_eligible = (
                obs >= min_obs
                and activity_ratio >= min_activity_ratio
                and effective_obs >= min_effective_obs
            )
            model_stats.append(
                {
                    "model_name": str(model_name),
                    "obs": obs,
                    "activity_ratio": activity_ratio,
                    "effective_obs": effective_obs,
                    "eligible": is_eligible,
                }
            )

        eligible_models = {str(stats["model_name"]) for stats in model_stats}
        eligible_by_ticker[ticker_key] = eligible_models
        tickers_diag[ticker_key] = {
            "eligible_models": sorted(eligible_models),
            "n_models_before": len(model_stats),
            "n_models_after": len(eligible_models),
            "model_stats": model_stats,
        }

    diagnostics = {
        "min_obs": min_obs,
        "min_activity_ratio": min_activity_ratio,
        "min_effective_obs": min_effective_obs,
        "mode": "diagnostics_only_no_filtering",
        "tickers": tickers_diag,
    }
    return forecast_vectors, eligible_by_ticker, diagnostics
