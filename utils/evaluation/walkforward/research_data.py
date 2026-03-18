from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from feature_research.core_helpers import (
    combo_key,
    expand_params_with_bin_count,
    normalize_datetime_index,
    normalize_series_datetime_index,
    unique_sorted_datetime_index,
)
from feature_research.in_sample.data_loader import (
    load_candles_for_config,
    load_features_for_combo,
    param_combo_label,
)

if TYPE_CHECKING:
    from feature_research.in_sample.config import ResearchConfig


ComboMap = dict[tuple[tuple[str, object], ...], pd.DataFrame]
ReturnsMap = dict[tuple[tuple[str, object], ...], pd.Series]


@dataclass(frozen=True)
class ContinuousResearchData:
    combo_feature_target: ComboMap
    successful_param_grid: list[dict[str, object]]
    reference_index: pd.DatetimeIndex | None
    reference_target_series: pd.Series | None


@dataclass(frozen=True)
class RuleBasedResearchData:
    combo_returns: ReturnsMap
    combo_feature_target: ComboMap
    successful_param_grid: list[dict[str, object]]
    reference_index: pd.DatetimeIndex | None
    reference_target_series: pd.Series | None


def load_continuous_research_data(
    config: "ResearchConfig",
    expanded_specs: list[dict[str, object]],
    *,
    candles_override: pd.DataFrame | None = None,
    print_loaded: bool = False,
) -> ContinuousResearchData:
    combo_feature_target: ComboMap = {}
    successful_param_grid: list[dict[str, object]] = []
    reference_index: pd.DatetimeIndex | None = None
    reference_target_series: pd.Series | None = None

    for single_spec in expanded_specs:
        combo = single_spec["params"]
        label = param_combo_label(combo)
        data = load_features_for_combo(single_spec, config, candles_override=candles_override)
        if data is None:
            if print_loaded:
                print(f"  [{label}] SKIP -- no data")
            continue

        feature, target, _ = data
        paired = pd.DataFrame({"feature": feature, "target": target}).dropna()
        if paired.empty:
            if print_loaded:
                print(f"  [{label}] SKIP -- aligned feature/target empty")
            continue

        feature_series = paired["feature"]
        target_series = paired["target"]
        expanded_combo_params = expand_params_with_bin_count(
            params=dict(combo),
            bin_counts=config.binning_params.bin_counts,
        )
        normalized_feature = normalize_series_datetime_index(feature_series)
        normalized_target = normalize_series_datetime_index(target_series)
        for combo_params in expanded_combo_params:
            combo_feature_target[combo_key(combo_params)] = pd.DataFrame(
                {"feature": normalized_feature, "target": normalized_target}
            )
            successful_param_grid.append(combo_params)
        if reference_index is None:
            reference_index = normalize_datetime_index(target_series.index)
            reference_target_series = normalized_target.reindex(reference_index)
        if print_loaded:
            print(f"  [{label}] loaded n={len(feature_series):,}")

    return ContinuousResearchData(
        combo_feature_target=combo_feature_target,
        successful_param_grid=successful_param_grid,
        reference_index=reference_index,
        reference_target_series=reference_target_series,
    )


def load_rule_based_research_data(
    config: "ResearchConfig",
    expanded_specs: list[dict[str, object]],
    *,
    candles_override: pd.DataFrame | None = None,
    dedupe_before_multiply: bool = False,
    capture_target_as_reference: bool = True,
    print_loaded: bool = False,
) -> RuleBasedResearchData:
    combo_returns: ReturnsMap = {}
    combo_feature_target: ComboMap = {}
    successful_param_grid: list[dict[str, object]] = []
    reference_index: pd.DatetimeIndex | None = None
    reference_target_series: pd.Series | None = None

    for single_spec in expanded_specs:
        combo = single_spec["params"]
        label = param_combo_label(combo)
        data = load_features_for_combo(single_spec, config, candles_override=candles_override)
        if data is None:
            if print_loaded:
                print(f"  [{label}] SKIP -- no data")
            continue

        feature, target, _ = data
        paired = pd.DataFrame({"feature": feature, "target": target}).dropna()
        if paired.empty:
            if print_loaded:
                print(f"  [{label}] SKIP -- aligned feature/target empty")
            continue

        feature_series = paired["feature"]
        target_series = paired["target"]
        feature_norm = normalize_series_datetime_index(feature_series)
        target_norm = normalize_series_datetime_index(target_series)

        if dedupe_before_multiply:
            if feature_norm.index.duplicated().any():
                feature_norm = feature_norm.groupby(level=0).first()
            if target_norm.index.duplicated().any():
                target_norm = target_norm.groupby(level=0).first()

        key = combo_key(combo)
        combo_returns[key] = normalize_series_datetime_index(feature_norm.mul(target_norm))
        combo_feature_target[key] = pd.DataFrame({"feature": feature_norm, "target": target_norm})
        successful_param_grid.append(dict(combo))

        target_reference = (
            target_norm.groupby(level=0).mean()
            if target_norm.index.duplicated().any()
            else target_norm
        )
        if reference_index is None:
            reference_index = unique_sorted_datetime_index(target_norm.index)
            if capture_target_as_reference:
                reference_target_series = target_reference.reindex(reference_index).fillna(0.0)
                reference_target_series.name = "walkforward_target"
        if reference_target_series is None and capture_target_as_reference:
            reference_target_series = target_reference

        if print_loaded:
            print(f"  [{label}] loaded n={len(feature_series):,}")

    return RuleBasedResearchData(
        combo_returns=combo_returns,
        combo_feature_target=combo_feature_target,
        successful_param_grid=successful_param_grid,
        reference_index=reference_index,
        reference_target_series=reference_target_series,
    )


def build_reference_target(
    reference_index: pd.DatetimeIndex,
    reference_target_series: pd.Series | None,
) -> pd.Series:
    if reference_target_series is None:
        return pd.Series(0.0, index=reference_index, name="walkforward_target")
    return reference_target_series.reindex(reference_index).fillna(0.0).rename("walkforward_target")


def load_portfolio_candles(config: "ResearchConfig") -> pd.DataFrame:
    return load_candles_for_config(config)
