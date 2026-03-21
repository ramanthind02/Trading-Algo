from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prop_firms.base.portfolio_models import ReturnEngineConfig


DEFAULT_ES_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "raw_data" / "ES.txt"
)


def load_es_price_history(data_path: Path | None = None) -> pd.DataFrame:
    """Load ES daily OHLCV history from the repository data file."""

    resolved_path = DEFAULT_ES_DATA_PATH if data_path is None else data_path
    frame = pd.read_csv(
        resolved_path,
        header=None,
        names=["date", "open", "high", "low", "close", "volume"],
        parse_dates=["date"],
    )
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    return frame


def build_buy_hold_returns(close_prices: pd.Series) -> pd.Series:
    """Build simple daily buy-and-hold returns from a close series."""

    returns = close_prices.astype(float).pct_change().dropna()
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("close_prices must use a DatetimeIndex")
    returns.name = "es_buy_hold_returns"
    return returns


def load_es_buy_hold_returns(data_path: Path | None = None) -> pd.Series:
    """Load ES prices and compute simple buy-and-hold returns."""

    history = load_es_price_history(data_path=data_path)
    close_series = pd.Series(
        history["close"].astype(float).to_numpy(),
        index=pd.DatetimeIndex(history["date"]),
        name="ES_close",
    )
    return build_buy_hold_returns(close_series)


def apply_target_volatility(
    returns: pd.Series,
    config: ReturnEngineConfig,
) -> pd.Series:
    """Scale returns to a target annualized volatility when configured."""

    normalized = normalize_return_series(returns)
    if config.target_annual_volatility is None:
        return normalized

    realized_vol = float(normalized.std(ddof=1) * np.sqrt(config.annualization_factor))
    if realized_vol == 0.0:
        return normalized.copy()

    scale_factor = config.target_annual_volatility / realized_vol
    scaled = normalized * scale_factor
    scaled.name = normalized.name
    return scaled


def build_return_series(
    config: ReturnEngineConfig,
    external_returns: pd.Series | None = None,
    data_path: Path | None = None,
) -> pd.Series:
    """Build a replay return series from external data or a synthetic Sharpe path."""

    source_returns = (
        _build_synthetic_sharpe_returns(config=config, data_path=data_path)
        if external_returns is None
        else normalize_return_series(external_returns)
    )
    filtered_returns = _filter_return_series_by_date(
        returns=source_returns,
        start_date=config.start_date,
        end_date=config.end_date,
    )
    return (
        filtered_returns
        if external_returns is None
        else apply_target_volatility(filtered_returns, config=config)
    )


def normalize_return_series(returns: pd.Series) -> pd.Series:
    """Validate and normalize a return series."""

    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must use a DatetimeIndex")
    if returns.empty:
        raise ValueError("returns must be non-empty")
    if returns.isna().any():
        raise ValueError("returns cannot contain NaN values")

    normalized = returns.copy().astype(float)
    normalized.index = pd.to_datetime(normalized.index).tz_localize(None)
    normalized = normalized.sort_index(kind="stable")
    if normalized.name is None:
        normalized.name = "returns"
    return normalized


def _filter_return_series_by_date(
    returns: pd.Series,
    start_date: str | None,
    end_date: str | None,
) -> pd.Series:
    if start_date is None and end_date is None:
        return returns

    filtered = returns.copy()
    if start_date is not None:
        filtered = filtered.loc[filtered.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        filtered = filtered.loc[filtered.index <= pd.Timestamp(end_date)]
    if filtered.empty:
        raise ValueError("date filter removed all return observations")
    return filtered


def _build_synthetic_sharpe_returns(
    config: ReturnEngineConfig,
    data_path: Path | None,
) -> pd.Series:
    if config.target_annual_volatility is None:
        raise ValueError(
            "target_annual_volatility must be set when external_returns is not provided"
        )

    date_index = _resolve_synthetic_date_index(config=config, data_path=data_path)
    if len(date_index) < 2:
        raise ValueError("synthetic return generation requires at least 2 dates")

    annualized_volatility = config.target_annual_volatility
    daily_volatility = annualized_volatility / np.sqrt(config.annualization_factor)
    daily_mean = (
        config.target_sharpe
        * annualized_volatility
        / config.annualization_factor
    )
    shocks = _build_standardized_shocks(
        n_observations=len(date_index),
        random_seed=config.random_seed,
    )
    synthetic_returns = pd.Series(
        daily_mean + (daily_volatility * shocks),
        index=date_index,
        name=f"synthetic_sharpe_{config.target_sharpe:.2f}",
    )
    return normalize_return_series(synthetic_returns)


def _resolve_synthetic_date_index(
    config: ReturnEngineConfig,
    data_path: Path | None,
) -> pd.DatetimeIndex:
    if config.start_date is not None and config.end_date is not None:
        return pd.bdate_range(config.start_date, config.end_date)

    history = load_es_price_history(data_path=data_path)
    calendar_index = pd.DatetimeIndex(history["date"])
    if config.start_date is None and config.end_date is None:
        return calendar_index

    filtered_index = calendar_index
    if config.start_date is not None:
        filtered_index = filtered_index[filtered_index >= pd.Timestamp(config.start_date)]
    if config.end_date is not None:
        filtered_index = filtered_index[filtered_index <= pd.Timestamp(config.end_date)]
    return filtered_index


def _build_standardized_shocks(
    n_observations: int,
    random_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    raw_shocks = rng.normal(loc=0.0, scale=1.0, size=n_observations)
    centered_shocks = raw_shocks - raw_shocks.mean()
    sample_std = centered_shocks.std(ddof=1)
    if sample_std == 0.0:
        raise ValueError("synthetic shock generation produced zero sample variance")
    return centered_shocks / sample_std
