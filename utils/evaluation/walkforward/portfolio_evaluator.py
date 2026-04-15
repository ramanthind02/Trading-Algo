from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from feature_research.core_helpers import combo_key
from utils.core.enums import TimeFrame
from utils.core.helpers import build_feature_column_name
from utils.evaluation.walkforward.metrics import resolve_objective_metric


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
) -> FoldPortfolioResult:
    _ = (binning_config, target_volatility, module_name, weight_layer_config, member_prediction_mode, feature_type)
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
