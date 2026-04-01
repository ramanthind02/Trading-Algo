from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .portfolio_allocation import effective_instrument_weights


def aggregate_forecast_vectors_fallback(
    forecast_vectors: List[pd.DataFrame],
) -> pd.DataFrame:
    """Fallback aggregation that averages raw forecasts by ticker and datetime."""
    if not forecast_vectors:
        return pd.DataFrame(columns=["ticker", "datetime", "forecast_score"])

    all_forecasts = pd.concat(forecast_vectors, ignore_index=True)
    if "ticker" not in all_forecasts.columns or "datetime" not in all_forecasts.columns:
        return pd.DataFrame(columns=["ticker", "datetime", "forecast_score"])

    aggregated = (
        all_forecasts.groupby(["datetime", "ticker"])["forecast"]
        .mean()
        .reset_index()
    )
    aggregated.columns = ["datetime", "ticker", "forecast_score"]
    return aggregated[["ticker", "datetime", "forecast_score"]]


def apply_forecast_risk_management(
    forecasts_df: pd.DataFrame,
    candles_df: pd.DataFrame,
    instrument_weights: Optional[Dict[str, float]],
    idm: Optional[float],
    max_position_pct: Optional[float],
) -> pd.DataFrame:
    """Vectorized ticker/time alignment plus weight, IDM, and cap application."""
    forecasts_clean = forecasts_df[["ticker", "datetime", "forecast_score"]].copy()
    forecasts_clean["datetime"] = pd.to_datetime(forecasts_clean["datetime"]).dt.floor("s")

    candles_subset = candles_df[["ticker", "datetime"]].copy()
    candles_subset["datetime"] = pd.to_datetime(candles_subset["datetime"]).dt.floor("s")

    result = candles_subset.merge(
        forecasts_clean,
        on=["ticker", "datetime"],
        how="left",
    )
    result["forecast_score"] = result["forecast_score"].fillna(0.0)

    idm_value = idm if idm is not None else 1.0
    result["position_fraction"] = result["forecast_score"] * idm_value

    weights_by_ticker = effective_instrument_weights(
        result["ticker"].tolist(),
        instrument_weights,
    )
    result["position_fraction"] *= result["ticker"].map(weights_by_ticker)

    if max_position_pct is not None:
        result["position_fraction"] = result["position_fraction"].clip(
            lower=-max_position_pct,
            upper=max_position_pct,
        )

    return result[["ticker", "datetime", "forecast_score", "position_fraction"]]
