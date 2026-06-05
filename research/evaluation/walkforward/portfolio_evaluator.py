from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from research.feature._internal.core_helpers import combo_key
from features.validation.objective_metrics import (
    resolve_objective_metric_name as resolve_objective_metric,
)
from lib.core.enums import TimeFrame
from lib.core.helpers import build_feature_column_name


@dataclass(frozen=True)
class FoldPortfolioResult:
    fold_id: int
    oos_portfolio_sharpe: float
    oos_portfolio_returns: pd.Series
    n_params_selected: int
    per_signal_oos_sharpe: dict[str, float] | None = None
    per_signal_oos_returns: dict[str, pd.Series] | None = None
    per_ticker_oos_returns: dict[str, pd.Series] | None = None


def _normalize_timeframe(value: Any) -> TimeFrame:
    if isinstance(value, TimeFrame):
        return value
    if isinstance(value, str):
        return TimeFrame[value.replace("TimeFrame.", "")]
    raise ValueError(f"Unsupported timeframe value: {value}")


def ensure_portfolio_candle_columns(
    candles: pd.DataFrame,
    trading_timeframe: TimeFrame | str,
) -> pd.DataFrame:
    required_cols = {"datetime", "open", "high", "low", "close", "ticker"}
    missing_cols = required_cols - set(candles.columns)
    if missing_cols:
        raise ValueError(f"Missing required candle columns: {sorted(missing_cols)}")

    normalized = candles.copy()
    timeframe = _normalize_timeframe(trading_timeframe)
    normalized["datetime"] = pd.to_datetime(normalized["datetime"], utc=False)
    if "volume" not in normalized.columns:
        normalized["volume"] = 0.0
    if "timeframe" not in normalized.columns:
        normalized["timeframe"] = timeframe
    else:
        normalized["timeframe"] = normalized["timeframe"].apply(
            lambda value: _normalize_timeframe(value) if not isinstance(value, TimeFrame) else value
        )
    return normalized


def _calculate_oos_returns_from_positions(
    positions_df: pd.DataFrame,
    candles_df: pd.DataFrame,
    *,
    series_name: str,
) -> pd.Series:
    required_columns = {"ticker", "datetime", "position_fraction"}
    missing_columns = required_columns - set(positions_df.columns)
    if missing_columns:
        raise ValueError(
            f"positions_df is missing required columns for return calculation: {sorted(missing_columns)}"
        )

    from ensemble.portfolio_impl.portfolio_tester import calculate_strategy_returns_from_positions

    returns = calculate_strategy_returns_from_positions(
        positions_df=positions_df,
        candles_df=candles_df,
    )
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.DatetimeIndex(pd.to_datetime(returns.index, utc=False))
    if getattr(returns.index, "tz", None) is not None:
        returns.index = returns.index.tz_localize(None)
    returns = returns.sort_index()
    full_dates = pd.DatetimeIndex(pd.to_datetime(candles_df["datetime"]).unique()).sort_values()
    if getattr(full_dates, "tz", None) is not None:
        full_dates = full_dates.tz_localize(None)
    return returns.reindex(full_dates, fill_value=0.0).rename(series_name)


def _normalize_ticker_label(value: object) -> str:
    raw_name = getattr(value, "name", None)
    if isinstance(value, str):
        return value
    if isinstance(raw_name, str) and raw_name:
        return raw_name
    return str(value).replace("Ticker.", "")


def _signal_return_series(
    combo_data: pd.DataFrame,
) -> pd.Series:
    if "returns" in combo_data.columns:
        result = combo_data["returns"].copy()
    else:
        signal_column = "signal" if "signal" in combo_data.columns else "feature"
        signal = combo_data[signal_column]
        target = combo_data["target"]
        result = signal.mul(target)
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.DatetimeIndex(pd.to_datetime(result.index, utc=False))
    if result.index.duplicated().any():
        result = result.groupby(level=0).mean()
    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_localize(None)
    return result.sort_index()


def build_research_portfolio(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError("Legacy fitted portfolio construction is unsupported in frozen-signal research.")


_IDM_MAX: float = 2.5
_FORECAST_CAP: float = 2.0


def _vol_scaled_portfolio_returns(
    selected_params: list[dict[str, Any]],
    feature_data_by_combo: Mapping[tuple[tuple[str, object], ...], pd.DataFrame],
    train_candles: pd.DataFrame,
    test_candles: pd.DataFrame,
    *,
    target_volatility: float,
    timeframe: TimeFrame,
    module_name: str,
    instrument_return_kind: str = "log_intraday",
) -> pd.Series:
    """Compute production-equivalent OOS returns for the tearsheet.

    For each selected combo:
      1. EWSD is computed causally from full candles (train + test) so the EWMA
         is warm before the test window begins.
      2. ``forecast_score = min(tau / EWSD[t], 2.0) × signal``
      3. TFPortfolio is fit on train_candles to obtain IDM + instrument weights.
      4. ``position_fraction = forecast_score × instrument_weight × IDM``
      5. ``strategy_return = position_fraction × log_return(close)``

    Returns the mean return series across combos, reindexed to test dates.
    Falls back to an empty Series if real-candle data is unavailable.
    """
    from ensemble.portfolio_impl.portfolio_returns import calculate_returns_from_candles
    from ensemble.portfolio_impl.portfolio_tester import calculate_strategy_returns_from_positions
    from ensemble.portfolio_impl.tf_portfolio import TFPortfolio
    from lib.compute.daily_ewsd_volatility import DailyEWSDVolatilityService
    from lib.core.ticker_key import normalize_ticker_key

    svc = DailyEWSDVolatilityService()
    # Deduplicate before computing EWSD: when the train tearsheet is generated the
    # runner passes test_candles=train_candles, so naively concatenating would double
    # every row, inserting spurious zero-return bars that understate volatility.
    full_candles = (
        pd.concat([train_candles, test_candles], ignore_index=True)
        .drop_duplicates(subset=["ticker", "datetime"])
        .sort_values(["ticker", "datetime"])
    )
    daily_vol_df = svc.compute_daily_series(full_candles)

    instrument_returns = calculate_returns_from_candles(train_candles)
    portfolio = TFPortfolio(ensembles=[], trading_timeframe=timeframe, idm_max=_IDM_MAX)
    portfolio.fit(instrument_returns)

    test_dt_series = pd.to_datetime(test_candles["datetime"]).dt.tz_localize(None)
    test_lo, test_hi = test_dt_series.min(), test_dt_series.max()

    all_returns: list[pd.Series] = []
    for params in selected_params:
        key = combo_key(params)
        combo_data = feature_data_by_combo.get(key)
        if combo_data is None or combo_data.empty:
            continue

        signal_col = "signal" if "signal" in combo_data.columns else "feature"
        signal_full = combo_data[signal_col].copy()
        ticker_col = combo_data["ticker"] if "ticker" in combo_data.columns else None

        sig_dt = pd.to_datetime(signal_full.index).tz_localize(None)
        test_mask = (sig_dt >= test_lo) & (sig_dt <= test_hi)
        signal = signal_full.loc[test_mask]
        if signal.empty:
            continue

        if ticker_col is not None:
            ticker = ticker_col.loc[test_mask].astype(str)
        else:
            ticker = pd.Series(
                [_normalize_ticker_label(test_candles["ticker"].iloc[0])] * len(signal),
                index=signal.index,
            )

        signal_dt_clean = pd.to_datetime(signal.index).tz_localize(None)
        align_frame = pd.DataFrame(
            {"datetime": signal_dt_clean, "ticker": ticker.values}
        )
        try:
            aligned = svc.align_daily_volatility_to_candles(daily_vol_df, align_frame)
        except ValueError:
            continue

        aligned["_dt_key"] = pd.to_datetime(aligned["datetime"]).dt.tz_localize(None).dt.floor("s")
        aligned["_tk_key"] = aligned["ticker"].map(normalize_ticker_key)
        vol_lookup = (
            aligned
            .drop_duplicates(subset=["_tk_key", "_dt_key"], keep="last")
            .set_index(["_tk_key", "_dt_key"])["ewsd_annual_vol"]
        )

        ticker_norm = ticker.map(normalize_ticker_key)
        dt_keys = signal_dt_clean.floor("s")
        mi = pd.MultiIndex.from_arrays([ticker_norm.values, dt_keys])
        ewsd_arr = vol_lookup.reindex(mi).fillna(target_volatility).to_numpy(dtype=float)
        ewsd_arr = np.maximum(ewsd_arr, 1e-6)

        forecast_vals = np.minimum(target_volatility / ewsd_arr, _FORECAST_CAP) * signal.to_numpy(dtype=float)
        forecasts_df = pd.DataFrame(
            {"ticker": ticker.values, "forecast_score": forecast_vals},
            index=pd.to_datetime(signal.index),
        )
        positions_result = portfolio.predict(forecasts_df)
        positions_df = pd.DataFrame(
            {
                "ticker": positions_result["ticker"].to_numpy(),
                "datetime": pd.to_datetime(signal.index),
                "position_fraction": positions_result["position_fraction"].to_numpy(dtype=float),
            }
        )

        returns = calculate_strategy_returns_from_positions(
            positions_df, test_candles,
            instrument_return_kind=instrument_return_kind,
        )
        if returns.empty:
            continue
        if getattr(returns.index, "tz", None) is not None:
            returns.index = returns.index.tz_localize(None)
        all_returns.append(returns)

    if not all_returns:
        return pd.Series(dtype=float, name="portfolio_returns")

    combined = pd.concat(all_returns, axis=1).mean(axis=1).rename("portfolio_returns")
    test_full_index = pd.DatetimeIndex(test_dt_series.unique()).sort_values()
    return combined.reindex(test_full_index, fill_value=0.0).rename("portfolio_returns")


def evaluate_fold_portfolio(
    train_candles: pd.DataFrame,
    test_candles: pd.DataFrame,
    selected_params: list[dict[str, Any]],
    target_series: pd.Series,
    binning_config: Any,
    tickers: list[Any],
    trading_timeframe: TimeFrame | str = TimeFrame.D,
    target_volatility: float = 0.15,
    module_name: str = "rsi",
    objective_metric_name: str = "sharpe",
    weight_layer_config: Any | None = None,
    member_prediction_mode: str | None = None,
    feature_data_by_combo: Mapping[tuple[tuple[str, object], ...], pd.DataFrame] | None = None,
    feature_type: object = None,
    instrument_return_kind: str = "log_intraday",
) -> FoldPortfolioResult:
    _ = (binning_config, module_name, weight_layer_config, member_prediction_mode, feature_type)
    timeframe = _normalize_timeframe(trading_timeframe)
    train_ready = ensure_portfolio_candle_columns(train_candles, timeframe)
    test_ready = ensure_portfolio_candle_columns(test_candles, timeframe)
    if train_ready.empty or test_ready.empty:
        raise ValueError("train_candles and test_candles must contain rows")
    if not feature_data_by_combo:
        raise ValueError("Frozen-signal portfolio evaluation requires feature_data_by_combo.")

    metric = resolve_objective_metric(objective_metric_name)
    if not selected_params:
        raise ValueError("no_selected_params")

    selected_signal_returns: dict[str, pd.Series] = {}
    for params in selected_params:
        key = combo_key(params)
        combo_data = feature_data_by_combo.get(key)
        if combo_data is None or combo_data.empty:
            continue
        signal_name = build_feature_column_name(
            module=module_name,
            feature="signal",
            tf=timeframe,
            params=params,
        )
        selected_signal_returns[signal_name] = _signal_return_series(combo_data)

    if not selected_signal_returns:
        raise ValueError("No feature data found for any selected param in feature_data_by_combo")

    test_index = pd.DatetimeIndex(pd.to_datetime(test_ready["datetime"]).unique()).sort_values()
    per_signal_oos_returns = {
        name: series.reindex(test_index, fill_value=0.0).rename("returns")
        for name, series in selected_signal_returns.items()
    }
    per_signal_oos_sharpe = {
        name: float(metric(series[series != 0.0])) if not series[series != 0.0].empty else float("nan")
        for name, series in per_signal_oos_returns.items()
    }

    # Use the full production vol-targeting path for portfolio returns when real
    # OHLCV candles with a ticker column are available (i.e., portfolio_candles_df
    # was provided to the runner).  Fall back to the legacy EWSD-normalised path
    # only if vol-scaling fails or candles lack the required columns.
    combined: pd.Series
    _has_real_candles = "ticker" in test_ready.columns and "close" in test_ready.columns
    if _has_real_candles:
        try:
            combined = _vol_scaled_portfolio_returns(
                selected_params=selected_params,
                feature_data_by_combo=feature_data_by_combo,
                train_candles=train_ready,
                test_candles=test_ready,
                target_volatility=target_volatility,
                timeframe=timeframe,
                module_name=module_name,
                instrument_return_kind=instrument_return_kind,
            )
            if combined.empty:
                raise ValueError("vol-scaled path returned empty series")
        except Exception as exc:
            warnings.warn(
                f"Vol-scaled portfolio returns failed ({exc}); "
                "falling back to EWSD-normalised signal returns.",
                UserWarning,
                stacklevel=2,
            )
            combined = pd.concat(per_signal_oos_returns, axis=1).mean(axis=1).rename("portfolio_returns")
    else:
        combined = pd.concat(per_signal_oos_returns, axis=1).mean(axis=1).rename("portfolio_returns")

    combined = combined.reindex(test_index, fill_value=0.0)
    active_returns = combined[combined != 0.0]
    oos_sharpe = float(metric(active_returns)) if not active_returns.empty else float("nan")

    per_ticker_oos_returns: dict[str, pd.Series] | None = None
    if "ticker" in test_ready.columns:
        ticker_labels = test_ready["ticker"].map(_normalize_ticker_label)
        per_ticker_oos_returns = {}
        for label in sorted(set(ticker_labels.dropna().unique())):
            ticker_index = pd.DatetimeIndex(
                pd.to_datetime(test_ready.loc[ticker_labels == label, "datetime"]).unique()
            ).sort_values()
            ticker_returns = combined.reindex(ticker_index, fill_value=0.0)
            if not ticker_returns.empty:
                per_ticker_oos_returns[label] = ticker_returns.rename(f"{label}_returns")

    return FoldPortfolioResult(
        fold_id=-1,
        oos_portfolio_sharpe=oos_sharpe,
        oos_portfolio_returns=combined,
        n_params_selected=len(selected_params),
        per_signal_oos_sharpe=per_signal_oos_sharpe,
        per_signal_oos_returns=per_signal_oos_returns,
        per_ticker_oos_returns=per_ticker_oos_returns,
    )
