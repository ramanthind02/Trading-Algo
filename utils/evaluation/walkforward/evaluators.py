from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

import pandas as pd

from feature_research.core_helpers import (
    combo_key,
    normalize_datetime_index,
    normalize_series_datetime_index,
)
from feature_selection.base_models.continuous_binning import ContinuousBinningModel

if TYPE_CHECKING:
    from feature_research.in_sample.config import ResearchConfig


def build_continuous_walkforward_evaluator(
    combo_feature_target: dict[tuple[tuple[str, object], ...], pd.DataFrame],
    config: "ResearchConfig",
) -> Callable[..., pd.Series | tuple[pd.Series, dict[str, object]]]:
    """Build evaluator for continuous binning walkforward evaluation."""
    strategy = config.binning_params.strategy

    def evaluate_param_combo(
        fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        params: dict[str, object],
        *,
        train_end: pd.Timestamp | None = None,
    ) -> tuple[pd.Series, dict[str, object]]:
        combo = combo_key({k: v for k, v in params.items() if k != "selected_bin"})
        combo_data = combo_feature_target[combo]
        fold_index_norm = normalize_datetime_index(fold_candles.index)
        combo_index_norm = normalize_datetime_index(combo_data.index)
        in_fold = combo_index_norm.isin(fold_index_norm)
        fold_data = combo_data.loc[in_fold].dropna()
        if fold_data.empty:
            return (pd.Series(dtype=float), {"selected_long_bin": None})

        train_cutoff = pd.Timestamp(train_end) if train_end is not None else pd.Timestamp(fold_data.index.max())
        train_data = fold_data.loc[fold_data.index <= train_cutoff]
        if train_data.empty:
            return (pd.Series(dtype=float), {"selected_long_bin": None})

        bin_count = int(cast(int, params.get("bin_count", config.binning_params.bin_counts[0])))
        model = ContinuousBinningModel(
            n_bins=bin_count,
            bin_counts=[bin_count],
            strategy=config.binning_params.strategy,
            bin_index_min=config.binning_params.bin_index_min,
            bin_index_max=config.binning_params.bin_index_max,
        )
        try:
            model.fit(train_data["feature"], train_data["target"])
        except ValueError:
            return (pd.Series(dtype=float), {"selected_long_bin": None})

        if "selected_bin" in params:
            requested_bin = int(cast(int, params["selected_bin"]))
            bin_assignments = model.assign_bins(fold_data["feature"])
            signal_forced = (bin_assignments == requested_bin).astype(float)
            series_forced = normalize_series_datetime_index(signal_forced.mul(fold_data["target"]))
            train_cutoff_ts = pd.Timestamp(train_cutoff)
            train_series = series_forced.loc[series_forced.index <= train_cutoff_ts]
            forced_has_no_train_trades = train_series.empty or int((train_series.fillna(0) != 0).sum()) == 0
            forced_has_no_trades_anywhere = int((series_forced.fillna(0) != 0).sum()) == 0
            if forced_has_no_train_trades or forced_has_no_trades_anywhere:
                signal = model.predict(fold_data["feature"], strategy=strategy)
                series = normalize_series_datetime_index(signal.mul(fold_data["target"]))
                return (series, {"selected_long_bin": requested_bin})
            return (series_forced, {"selected_long_bin": requested_bin})

        selected_long_bin = model.selected_bins_.get("long")
        signal = model.predict(fold_data["feature"], strategy=strategy)
        series = normalize_series_datetime_index(signal.mul(fold_data["target"]))
        return (series, {"selected_long_bin": selected_long_bin})

    return evaluate_param_combo


def build_rule_based_walkforward_evaluator(
    combo_returns: dict[tuple[tuple[str, object], ...], pd.Series],
) -> Callable[[pd.DataFrame, pd.Series, dict[str, object]], pd.Series]:
    """Build evaluator for rule-based walkforward evaluation."""

    def evaluate_param_combo(
        fold_candles: pd.DataFrame,
        _fold_target: pd.Series,
        params: dict[str, object],
    ) -> pd.Series:
        returns = combo_returns[combo_key(params)]
        fold_mask = returns.index.isin(fold_candles.index)
        fold_returns = returns.loc[fold_mask].dropna()
        if fold_returns.empty:
            return pd.Series(dtype=float)
        if fold_returns.index.duplicated().any():
            fold_returns = fold_returns[~fold_returns.index.duplicated(keep="first")]
        return fold_returns

    return evaluate_param_combo
