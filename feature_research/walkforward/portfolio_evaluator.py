from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd

from ensemble.diversified_ensemble import DiversifiedEnsemble
from ensemble.portfolio import Portfolio
from ensemble.weight_layer import WeightLayer, WeightLayerConfig
from feature_research.walkforward.metrics import resolve_objective_metric
from utils.enums import TimeFrame, Ticker
from utils.helpers import build_feature_column_name


@dataclass(frozen=True)
class FoldPortfolioResult:
    fold_id: int
    oos_portfolio_sharpe: float
    oos_portfolio_returns: pd.Series
    n_params_selected: int


def _normalize_strategy(strategy: str) -> str:
    return "long_short" if strategy == "long-short" else strategy


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


def _build_control_file_payload(
    selected_params: list[dict[str, Any]],
    binning_config: Any,
    tickers: list[Ticker],
    trading_timeframe: TimeFrame,
    module_name: str,
) -> dict[str, Any]:
    strategy = _normalize_strategy(str(binning_config.strategy))
    model_configs: list[dict[str, Any]] = []
    ticker_names = [ticker.name for ticker in tickers]
    for params in selected_params:
        feature_params = {key: value for key, value in params.items() if key != "bin_count"}
        feature_column = build_feature_column_name(
            module=module_name,
            feature="signal",
            tf=trading_timeframe,
            params=feature_params,
        )
        bin_count = int(params.get("bin_count", binning_config.bin_counts[0]))
        members = [{"member_id": f"member_{i}", "bin_index": i} for i in range(bin_count)]
        model_configs.append(
            {
                "name": f"{feature_column}_{strategy}",
                "model_type": "continuous_binning",
                "feature_column": feature_column,
                "strategy": strategy,
                "tickers": ticker_names,
                "members": members,
                "constructor_params": {
                    "n_bins": bin_count,
                    "bin_counts": [bin_count],
                    "selection_metric": binning_config.selection_metric,
                    "strategy": strategy,
                    "metric_threshold": binning_config.metric_threshold,
                    "t_threshold": binning_config.t_threshold,
                    "min_region_width": binning_config.min_region_width,
                    "shrinkage_k": binning_config.shrinkage_k,
                    "long_clip_min": binning_config.long_clip_min,
                    "long_clip_max": binning_config.long_clip_max,
                    "short_clip_min": binning_config.short_clip_min,
                    "short_clip_max": binning_config.short_clip_max,
                    "use_coverage_bonus": binning_config.use_coverage_bonus,
                    "coverage_bonus_per_10pct": binning_config.coverage_bonus_per_10pct,
                    "max_coverage_bonus": binning_config.max_coverage_bonus,
                },
            }
        )

    return {
        "metadata": {"is_fit": False, "base_tf": trading_timeframe.name},
        "tickers": ticker_names,
        "base_models": model_configs,
    }


def build_research_portfolio(
    selected_params: list[dict[str, Any]],
    binning_config: Any,
    tickers: list[Ticker],
    trading_timeframe: TimeFrame | str = TimeFrame.D,
    target_volatility: float = 0.15,
    module_name: str = "rsi",
    weight_layer_config: WeightLayerConfig | None = None,
) -> Portfolio:
    timeframe = _normalize_timeframe(trading_timeframe)
    control_payload = _build_control_file_payload(
        selected_params=selected_params,
        binning_config=binning_config,
        tickers=tickers,
        trading_timeframe=timeframe,
        module_name=module_name,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        handle.write(json.dumps(control_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
        control_file = Path(handle.name)

    try:
        ensemble = DiversifiedEnsemble(
            target_volatility=target_volatility,
            control_file_path=str(control_file),
            base_tf=timeframe,
        )
    finally:
        control_file.unlink(missing_ok=True)

    weight_layer = WeightLayer(config=weight_layer_config) if weight_layer_config is not None else None
    return Portfolio(
        ensembles=[ensemble],
        trading_timeframe=timeframe,
        target_volatility=target_volatility,
        weight_layer=weight_layer,
    )


def evaluate_fold_portfolio(
    train_candles: pd.DataFrame,
    test_candles: pd.DataFrame,
    selected_params: list[dict[str, Any]],
    target_series: pd.Series,
    binning_config: Any,
    tickers: list[Ticker],
    trading_timeframe: TimeFrame | str = TimeFrame.D,
    target_volatility: float = 0.15,
    module_name: str = "rsi",
    objective_metric_name: str = "sharpe",
    weight_layer_config: WeightLayerConfig | None = None,
) -> FoldPortfolioResult:
    timeframe = _normalize_timeframe(trading_timeframe)
    train_ready = ensure_portfolio_candle_columns(train_candles, timeframe)
    test_ready = ensure_portfolio_candle_columns(test_candles, timeframe)
    if train_ready.empty or test_ready.empty:
        raise ValueError("train_candles and test_candles must contain rows")

    portfolio = build_research_portfolio(
        selected_params=selected_params,
        binning_config=binning_config,
        tickers=tickers,
        trading_timeframe=timeframe,
        target_volatility=target_volatility,
        module_name=module_name,
        weight_layer_config=weight_layer_config,
    )

    train_index = pd.DatetimeIndex(pd.to_datetime(train_ready["datetime"], utc=False))
    test_index = pd.DatetimeIndex(pd.to_datetime(test_ready["datetime"], utc=False))
    train_target = target_series.reindex(train_index)
    portfolio.fit_from_candles(train_ready, target_data=train_target)

    predictions = portfolio.predict_from_candles(test_ready)
    portfolio_predictions = (
        predictions["portfolio"] if isinstance(predictions, dict) else predictions
    )
    if portfolio_predictions.empty:
        raise ValueError("Portfolio produced no predictions for test fold")

    positions = (
        portfolio_predictions.groupby("datetime")["position_fraction"].mean()
        if "datetime" in portfolio_predictions.columns
        else pd.Series(0.0, index=test_index)
    )
    test_target = target_series.reindex(test_index)
    aligned = pd.DataFrame({"position": positions, "target": test_target}).dropna()
    oos_returns = (aligned["position"] * aligned["target"]).rename("portfolio_returns")

    active_returns = oos_returns[oos_returns != 0.0]
    metric = resolve_objective_metric(objective_metric_name)
    oos_sharpe = float(metric(active_returns)) if not active_returns.empty else float("nan")

    return FoldPortfolioResult(
        fold_id=-1,
        oos_portfolio_sharpe=oos_sharpe,
        oos_portfolio_returns=oos_returns,
        n_params_selected=len(selected_params),
    )
