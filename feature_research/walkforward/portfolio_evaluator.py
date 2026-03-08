from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import pandas as pd

from ensemble.diversified_ensemble import DiversifiedEnsemble
from ensemble.portfolio import Portfolio
from ensemble.weight_layer import WeightLayer, WeightLayerConfig
from feature_research.config import FeatureType
from feature_research.walkforward.metrics import resolve_objective_metric
from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from feature_selection.base_models.feature_base_model import BaseModel
from feature_selection.base_models.rule_based import RuleBasedModel
from utils.core.enums import TimeFrame, Ticker
from utils.core.helpers import build_feature_column_name
from utils.data.cross_ticker_store import CrossTickerDataStore, extract_cross_ticker_names

# RuleBasedModel has exactly 3 bins (-1, 0, 1 -> indices 0, 1, 2).
RULE_BASED_BIN_COUNT: int = 3


# Param keys that belong to the binning model only; never pass to the bias node (e.g. RSI).
BINNING_ONLY_PARAM_KEYS: frozenset[str] = frozenset({"bin_count", "selected_bin"})


def _ensure_cross_ticker_data(
    selected_params: list[dict[str, Any]],
    timeframes: list[TimeFrame],
) -> None:
    """Ensure cross-ticker data is loaded into the store for all selected params.

    Called before ``BaseModel`` creation to make cross-ticker loading explicit
    rather than relying on prior singleton state.  Safe to call multiple times
    — already-loaded (ticker, tf) pairs are skipped.
    """
    cross_names: set[str] = set()
    for params in selected_params:
        cross_names.update(extract_cross_ticker_names(params))
    if not cross_names:
        return

    store = CrossTickerDataStore.get_instance()
    for name in cross_names:
        try:
            ct = Ticker[name]
        except KeyError:
            continue
        for tf in timeframes:
            if not store.is_loaded(ct, tf):
                store.load(ct, tf)


def _combo_key(params: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    """Convert params dict to hashable sorted tuple for use as dict key."""
    return tuple(sorted(params.items(), key=lambda item: item[0]))


@dataclass(frozen=True)
class FoldPortfolioResult:
    fold_id: int
    oos_portfolio_sharpe: float
    oos_portfolio_returns: pd.Series
    n_params_selected: int
    per_signal_oos_sharpe: dict[str, float] | None = None
    per_signal_oos_returns: dict[str, pd.Series] | None = None


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


def _calculate_oos_returns_from_positions(
    positions_df: pd.DataFrame,
    candles_df: pd.DataFrame,
    *,
    series_name: str,
) -> pd.Series:
    """Convert ticker-level position predictions into realized returns using candles."""
    required_columns = {"ticker", "datetime", "position_fraction"}
    missing_columns = required_columns - set(positions_df.columns)
    if missing_columns:
        raise ValueError(
            f"positions_df is missing required columns for return calculation: {sorted(missing_columns)}"
        )

    from ensemble.portfolio_tester import calculate_strategy_returns_from_positions

    returns = calculate_strategy_returns_from_positions(
        positions_df=positions_df,
        candles_df=candles_df,
    )
    if not isinstance(returns.index, pd.DatetimeIndex):
        normalized_index = pd.to_datetime(returns.index, utc=False)
        returns.index = pd.DatetimeIndex(normalized_index)
    if getattr(returns.index, "tz", None) is not None:
        returns.index = returns.index.tz_localize(None)
    returns = returns.sort_index()
    # Reindex to full candle calendar so flat days are 0.0; avoids tearsheet exposure showing 100%.
    full_dates = pd.DatetimeIndex(pd.to_datetime(candles_df["datetime"]).unique()).sort_values()
    if getattr(full_dates, "tz", None) is not None:
        full_dates = full_dates.tz_localize(None)
    returns = returns.reindex(full_dates, fill_value=0.0).rename(series_name)
    return returns


def _build_control_file_payload(
    selected_params: list[dict[str, Any]],
    binning_config: Any,
    tickers: list[Ticker],
    trading_timeframe: TimeFrame,
    module_name: str,
    feature_type: FeatureType,
) -> dict[str, Any]:
    strategy = _normalize_strategy(str(binning_config.strategy))
    model_configs: list[dict[str, Any]] = []
    ticker_names = [ticker.name for ticker in tickers]
    is_rule_based = feature_type == FeatureType.RULE_BASED
    for params in selected_params:
        feature_params = {k: v for k, v in params.items() if k not in BINNING_ONLY_PARAM_KEYS}
        feature_column = build_feature_column_name(
            module=module_name,
            feature="signal",
            tf=trading_timeframe,
            params=feature_params,
        )
        if is_rule_based:
            bin_count = RULE_BASED_BIN_COUNT
            member_indices = range(0, RULE_BASED_BIN_COUNT)
        else:
            bin_count = int(params.get("bin_count", binning_config.bin_counts[0]))
            bin_index_max_val = getattr(binning_config, "bin_index_max", None)
            start = max(0, getattr(binning_config, "bin_index_min", 0))
            end = (bin_index_max_val + 1) if bin_index_max_val is not None else bin_count
            end = min(bin_count, end)
            member_indices = range(start, end)
        model_name = f"{feature_column}_{strategy}"
        members = [
            {
                "member_name": f"{model_name}__member_{i}",
                "params": {"bin_index": i},
                "member_id": f"member_{i}",
                "bin_index": i,
            }
            for i in member_indices
        ]
        if is_rule_based:
            constructor_params = {
                "selection_metric": binning_config.selection_metric,
                "strategy": strategy,
                "metric_threshold": binning_config.metric_threshold,
                "shrinkage_k": binning_config.shrinkage_k,
                "long_clip_min": binning_config.long_clip_min,
                "long_clip_max": binning_config.long_clip_max,
                "short_clip_min": binning_config.short_clip_min,
                "short_clip_max": binning_config.short_clip_max,
            }
        else:
            constructor_params = {
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
                "bin_index_min": getattr(binning_config, "bin_index_min", 0),
                "bin_index_max": getattr(binning_config, "bin_index_max", None),
            }
        model_configs.append(
            {
                "name": model_name,
                "model_type": "rule_based" if is_rule_based else "continuous_binning",
                "feature_column": feature_column,
                "strategy": strategy,
                "tickers": ticker_names,
                "members": members,
                "constructor_params": constructor_params,
                "bias_node_spec": {
                    "module_name": module_name,
                    "timeframes": [trading_timeframe.name],
                    "params": feature_params,
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
    member_prediction_mode: str | None = None,
    feature_type: FeatureType = FeatureType.CONTINUOUS,
) -> Portfolio:
    timeframe = _normalize_timeframe(trading_timeframe)
    control_payload = _build_control_file_payload(
        selected_params=selected_params,
        binning_config=binning_config,
        tickers=tickers,
        trading_timeframe=timeframe,
        module_name=module_name,
        feature_type=feature_type,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        handle.write(json.dumps(control_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
        control_file = Path(handle.name)

    try:
        ensemble = DiversifiedEnsemble(
            target_volatility=target_volatility,
            control_file_path=str(control_file),
            base_tf=timeframe,
            member_forecast_scaling_mode=member_prediction_mode or "sharpe_weighted",
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


def _build_one_base_model_with_members(
    selected_params: list[dict[str, Any]],
    binning_config: Any,
    tickers: list[Ticker],
    trading_timeframe: TimeFrame,
    module_name: str,
    feature_data_by_combo: Mapping[tuple[tuple[str, object], ...], pd.DataFrame],
    train_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    train_target: pd.Series,
    feature_type: FeatureType,
) -> tuple[BaseModel, pd.DataFrame, pd.DataFrame]:
    """Build one BaseModel with one member per selected param and train/test feature DataFrames."""
    strategy = _normalize_strategy(str(binning_config.strategy))
    is_rule_based = feature_type == FeatureType.RULE_BASED
    train_parts: dict[str, pd.Series] = {}
    test_parts: dict[str, pd.Series] = {}
    feature_columns: list[str] = []
    for params in selected_params:
        # Feature data is keyed by (lookback, bin_count) only; selected_bin is not in the key.
        key = _combo_key({k: v for k, v in params.items() if k != "selected_bin"})
        if key not in feature_data_by_combo:
            continue
        df = feature_data_by_combo[key]
        if "feature" not in df.columns:
            continue
        feature_params = {k: v for k, v in params.items() if k not in BINNING_ONLY_PARAM_KEYS}
        feature_column = build_feature_column_name(
            module=module_name,
            feature="signal",
            tf=trading_timeframe,
            params=feature_params,
        )
        ser = df["feature"].copy()
        ser.index = pd.DatetimeIndex(pd.to_datetime(ser.index)).floor("s")
        ser_unique = (
            ser.groupby(level=0).first()
            if ser.index.duplicated().any()
            else ser
        )
        train_parts[feature_column] = ser_unique.reindex(train_index).dropna()
        test_parts[feature_column] = ser_unique.reindex(test_index).dropna()
        feature_columns.append(feature_column)
    if not feature_columns:
        raise ValueError("No feature data found for any selected param in feature_data_by_combo")
    train_features_df = pd.DataFrame(train_parts)
    test_features_df = pd.DataFrame(test_parts)

    first_params = selected_params[0]
    first_feature_params = {k: v for k, v in first_params.items() if k not in BINNING_ONLY_PARAM_KEYS}
    first_feature_column = build_feature_column_name(
        module=module_name,
        feature="signal",
        tf=trading_timeframe,
        params=first_feature_params,
    )
    bin_count = (
        RULE_BASED_BIN_COUNT
        if is_rule_based
        else int(first_params.get("bin_count", getattr(binning_config, "bin_counts", [10])[0]))
    )
    bin_index_max_val = None if is_rule_based else getattr(binning_config, "bin_index_max", None)
    if is_rule_based:
        first_binning = RuleBasedModel(
            selection_metric=getattr(binning_config, "selection_metric", "sharpe"),
            strategy=strategy,
            metric_threshold=getattr(binning_config, "metric_threshold", 0.0),
            shrinkage_k=getattr(binning_config, "shrinkage_k", 20.0),
            long_clip_min=getattr(binning_config, "long_clip_min", 0.5),
            long_clip_max=getattr(binning_config, "long_clip_max", 2.0),
            short_clip_min=getattr(binning_config, "short_clip_min", 0.5),
            short_clip_max=getattr(binning_config, "short_clip_max", 2.0),
        )
    else:
        first_binning = ContinuousBinningModel(
            n_bins=bin_count,
            bin_counts=[bin_count],
            selection_metric=getattr(binning_config, "selection_metric", "sharpe"),
            strategy=strategy,
            metric_threshold=getattr(binning_config, "metric_threshold", 0.0),
            t_threshold=getattr(binning_config, "t_threshold", 2.0),
            min_region_width=getattr(binning_config, "min_region_width", 2),
            shrinkage_k=getattr(binning_config, "shrinkage_k", 20.0),
            long_clip_min=getattr(binning_config, "long_clip_min", 0.5),
            long_clip_max=getattr(binning_config, "long_clip_max", 2.0),
            short_clip_min=getattr(binning_config, "short_clip_min", 0.5),
            short_clip_max=getattr(binning_config, "short_clip_max", 2.0),
            use_coverage_bonus=getattr(binning_config, "use_coverage_bonus", False),
            coverage_bonus_per_10pct=getattr(binning_config, "coverage_bonus_per_10pct", 0.02),
            max_coverage_bonus=getattr(binning_config, "max_coverage_bonus", 0.2),
            bin_index_min=getattr(binning_config, "bin_index_min", 0),
            bin_index_max=bin_index_max_val,
        )
    bias_node_spec = {
        "module_name": module_name,
        "timeframes": [trading_timeframe],
        "params": first_feature_params,
    }
    feature_config = {
        "bias_node_spec": bias_node_spec,
        "model_type": "rule_based" if is_rule_based else "continuous_binning",
        "constructor_params": {},
        "strategy": strategy,
    }
    base_model = BaseModel(
        feature_config=feature_config,
        tickers=list(tickers),
        binning_model=first_binning,
        use_cache=False,
    )
    base_model.feature_column = first_feature_column
    first_ser = train_features_df[first_feature_column].dropna()
    first_target_aligned = train_target.reindex(first_ser.index).dropna()
    first_ser = first_ser.reindex(first_target_aligned.index).dropna()
    first_binning.fit(first_ser, first_target_aligned)
    # Include the primary selected param in member emission so top_k selections
    # are all represented in the fast-path ensemble forecasts.
    base_model.add_member(first_feature_column, first_binning, feature_column=first_feature_column)

    for i, params in enumerate(selected_params[1:], start=1):
        feature_params = {k: v for k, v in params.items() if k not in BINNING_ONLY_PARAM_KEYS}
        feature_column = build_feature_column_name(
            module=module_name,
            feature="signal",
            tf=trading_timeframe,
            params=feature_params,
        )
        bin_count = (
            RULE_BASED_BIN_COUNT
            if is_rule_based
            else int(params.get("bin_count", getattr(binning_config, "bin_counts", [10])[0]))
        )
        bin_index_max_val = None if is_rule_based else getattr(binning_config, "bin_index_max", None)
        if is_rule_based:
            member_binning = RuleBasedModel(
                selection_metric=getattr(binning_config, "selection_metric", "sharpe"),
                strategy=strategy,
                metric_threshold=getattr(binning_config, "metric_threshold", 0.0),
                shrinkage_k=getattr(binning_config, "shrinkage_k", 20.0),
                long_clip_min=getattr(binning_config, "long_clip_min", 0.5),
                long_clip_max=getattr(binning_config, "long_clip_max", 2.0),
                short_clip_min=getattr(binning_config, "short_clip_min", 0.5),
                short_clip_max=getattr(binning_config, "short_clip_max", 2.0),
            )
        else:
            member_binning = ContinuousBinningModel(
                n_bins=bin_count,
                bin_counts=[bin_count],
                selection_metric=getattr(binning_config, "selection_metric", "sharpe"),
                strategy=strategy,
                metric_threshold=getattr(binning_config, "metric_threshold", 0.0),
                t_threshold=getattr(binning_config, "t_threshold", 2.0),
                min_region_width=getattr(binning_config, "min_region_width", 2),
                shrinkage_k=getattr(binning_config, "shrinkage_k", 20.0),
                long_clip_min=getattr(binning_config, "long_clip_min", 0.5),
                long_clip_max=getattr(binning_config, "long_clip_max", 2.0),
                short_clip_min=getattr(binning_config, "short_clip_min", 0.5),
                short_clip_max=getattr(binning_config, "short_clip_max", 2.0),
                use_coverage_bonus=getattr(binning_config, "use_coverage_bonus", False),
                coverage_bonus_per_10pct=getattr(binning_config, "coverage_bonus_per_10pct", 0.02),
                max_coverage_bonus=getattr(binning_config, "max_coverage_bonus", 0.2),
                bin_index_min=getattr(binning_config, "bin_index_min", 0),
                bin_index_max=bin_index_max_val,
            )
        ser = train_features_df[feature_column].dropna()
        target_aligned = train_target.reindex(ser.index).dropna()
        ser = ser.reindex(target_aligned.index).dropna()
        member_binning.fit(ser, target_aligned)
        base_model.add_member(feature_column, member_binning, feature_column=feature_column)

    return base_model, train_features_df, test_features_df


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
    member_prediction_mode: str | None = None,
    feature_data_by_combo: Mapping[tuple[tuple[str, object], ...], pd.DataFrame] | None = None,
    feature_type: FeatureType = FeatureType.CONTINUOUS,
) -> FoldPortfolioResult:
    timeframe = _normalize_timeframe(trading_timeframe)
    train_ready = ensure_portfolio_candle_columns(train_candles, timeframe)
    test_ready = ensure_portfolio_candle_columns(test_candles, timeframe)
    if train_ready.empty or test_ready.empty:
        raise ValueError("train_candles and test_candles must contain rows")

    train_dt = pd.to_datetime(train_ready["datetime"], utc=False)
    test_dt = pd.to_datetime(test_ready["datetime"], utc=False)
    train_index_unique = pd.DatetimeIndex(train_dt.unique()).sort_values()
    test_index_unique = pd.DatetimeIndex(test_dt.unique()).sort_values()
    # target_series may have duplicate index (multi-ticker); reindex requires unique index on the caller
    target_unique = (
        target_series.groupby(level=0).first()
        if target_series.index.duplicated().any()
        else target_series
    )
    train_target = target_unique.reindex(train_index_unique)
    metric = resolve_objective_metric(objective_metric_name)
    per_signal_oos_sharpe: dict[str, float] | None = None
    per_signal_oos_returns: dict[str, pd.Series] | None = None

    _ensure_cross_ticker_data(selected_params, [timeframe])

    if feature_data_by_combo and len(selected_params) > 0:
        base_model, train_features_df, test_features_df = _build_one_base_model_with_members(
            selected_params=selected_params,
            binning_config=binning_config,
            tickers=tickers,
            trading_timeframe=timeframe,
            module_name=module_name,
            feature_data_by_combo=feature_data_by_combo,
            train_index=train_index_unique,
            test_index=test_index_unique,
            train_target=train_target,
            feature_type=feature_type,
        )
        model_name = base_model.feature_column or "ensemble"
        required_columns = base_model.get_member_feature_columns()
        ensemble = DiversifiedEnsemble(
            target_volatility=target_volatility,
            base_models={model_name: base_model},
            required_columns=required_columns,
            base_tf=timeframe,
            member_forecast_scaling_mode=member_prediction_mode or "sharpe_weighted",
        )
        ensemble._member_feature_data = {model_name: train_features_df}
        weight_layer = (
            WeightLayer(config=weight_layer_config) if weight_layer_config is not None else None
        )
        portfolio = Portfolio(
            ensembles=[ensemble],
            trading_timeframe=timeframe,
            target_volatility=target_volatility,
            weight_layer=weight_layer,
        )
        portfolio.fit_from_candles(train_ready, target_data=train_target)
        ensemble._member_feature_data = {model_name: test_features_df}
        predictions = portfolio.predict_from_candles(
            test_ready, return_base_model_predictions=True
        )
        portfolio_predictions = (
            predictions["portfolio"] if isinstance(predictions, dict) else predictions
        )
        if portfolio_predictions.empty:
            raise ValueError("Portfolio produced no predictions for test fold")
        oos_returns = _calculate_oos_returns_from_positions(
            portfolio_predictions,
            test_ready,
            series_name="portfolio_returns",
        )
        active_returns = oos_returns[oos_returns != 0.0]
        oos_sharpe = float(metric(active_returns)) if not active_returns.empty else float("nan")
        base_models = (predictions or {}).get("base_models", {})
        if base_models:
            per_signal_oos_sharpe = {}
            per_signal_oos_returns = {}
            for sig_name, pos_df in base_models.items():
                if pos_df is None or pos_df.empty or "position_fraction" not in pos_df.columns:
                    continue
                ret = _calculate_oos_returns_from_positions(
                    pos_df,
                    test_ready,
                    series_name="returns",
                )
                ar = ret[ret != 0.0]
                per_signal_oos_sharpe[sig_name] = (
                    float(metric(ar)) if not ar.empty else float("nan")
                )
                per_signal_oos_returns[sig_name] = ret
        return FoldPortfolioResult(
            fold_id=-1,
            oos_portfolio_sharpe=oos_sharpe,
            oos_portfolio_returns=oos_returns,
            n_params_selected=len(selected_params),
            per_signal_oos_sharpe=per_signal_oos_sharpe,
            per_signal_oos_returns=per_signal_oos_returns if base_models else None,
        )

    portfolio = build_research_portfolio(
        selected_params=selected_params,
        binning_config=binning_config,
        tickers=tickers,
        trading_timeframe=timeframe,
        target_volatility=target_volatility,
        module_name=module_name,
        weight_layer_config=weight_layer_config,
        member_prediction_mode=member_prediction_mode,
        feature_type=feature_type,
    )
    portfolio.fit_from_candles(train_ready, target_data=train_target)
    predictions = portfolio.predict_from_candles(test_ready)
    portfolio_predictions = (
        predictions["portfolio"] if isinstance(predictions, dict) else predictions
    )
    if portfolio_predictions.empty:
        raise ValueError("Portfolio produced no predictions for test fold")
    oos_returns = _calculate_oos_returns_from_positions(
        portfolio_predictions,
        test_ready,
        series_name="portfolio_returns",
    )
    active_returns = oos_returns[oos_returns != 0.0]
    oos_sharpe = float(metric(active_returns)) if not active_returns.empty else float("nan")
    return FoldPortfolioResult(
        fold_id=-1,
        oos_portfolio_sharpe=oos_sharpe,
        oos_portfolio_returns=oos_returns,
        n_params_selected=len(selected_params),
    )
