from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .portfolio_allocation import effective_instrument_weights
from lib.core.enums import TimeFrame


def collect_tf_forecast_streams(
    tf_portfolios: list[Any],
    candles_per_tf: Dict[TimeFrame, pd.DataFrame],
    daily_volatility_df: pd.DataFrame,
) -> Dict[TimeFrame, pd.DataFrame]:
    """Collect per-timeframe base-model forecast vectors with shared validation."""
    tf_forecast_streams: Dict[TimeFrame, pd.DataFrame] = {}
    for tf_portfolio in tf_portfolios:
        tf_candles = candles_per_tf.get(tf_portfolio.trading_timeframe)
        if tf_candles is None:
            raise ValueError(
                f"No candles provided for timeframe {tf_portfolio.trading_timeframe.name}"
            )
        vectors = tf_portfolio.predict_base_model_vectors_from_candles(
            tf_candles,
            daily_volatility_df=daily_volatility_df,
        )
        if vectors.empty:
            continue
        tf_forecast_streams[tf_portfolio.trading_timeframe] = vectors
    return tf_forecast_streams


def build_global_returns_proxy(
    instrument_returns: pd.DataFrame | pd.Series,
) -> Optional[pd.Series]:
    """Normalize aggregate global returns proxy for diagnostics and scaling."""
    global_returns: Optional[pd.Series] = None
    if isinstance(instrument_returns, pd.DataFrame) and not instrument_returns.empty:
        global_returns = instrument_returns.mean(axis=1).astype(float)
    elif isinstance(instrument_returns, pd.Series) and not instrument_returns.empty:
        global_returns = instrument_returns.astype(float)

    if global_returns is not None:
        global_returns.index = pd.to_datetime(global_returns.index).normalize()
    return global_returns


def build_reference_grid_from_daily_candles(
    candles_per_tf: Dict[TimeFrame, pd.DataFrame],
) -> Optional[pd.Index]:
    """Use daily candles as the preferred predict-time alignment grid when available."""
    daily_candles = candles_per_tf.get(TimeFrame.D)
    if isinstance(daily_candles, pd.DataFrame) and "datetime" in daily_candles.columns:
        return pd.to_datetime(daily_candles["datetime"]).dt.normalize().dropna()
    return None


def apply_global_position_constraints(
    combined: pd.DataFrame,
    instrument_weights: Optional[Dict[str, float]],
    global_idm: Optional[float],
    max_position_pct: float,
) -> pd.DataFrame:
    """Apply global instrument weights, IDM, and position cap to combined forecasts."""
    if combined.empty:
        return pd.DataFrame(
            columns=["ticker", "datetime", "forecast_score", "position_fraction"]
        )

    weights_by_ticker = effective_instrument_weights(
        combined["ticker"].tolist(),
        instrument_weights,
    )
    result = combined.copy()
    result["position_weighted"] = (
        result["forecast_score"] * result["ticker"].map(weights_by_ticker)
    )
    result["idm_scaled"] = result["position_weighted"] * (
        global_idm if global_idm is not None else 1.0
    )
    result["position_fraction"] = result["idm_scaled"].clip(
        lower=-max_position_pct,
        upper=max_position_pct,
    )
    return result[["ticker", "datetime", "forecast_score", "position_fraction"]].reset_index(
        drop=True
    )
