from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from feature_research.core_helpers import (
    combo_key,
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


@dataclass(frozen=True)
class FrozenSignalResearchData:
    combo_signal_target: ComboMap
    successful_param_grid: list[dict[str, object]]
    reference_index: pd.DatetimeIndex | None
    reference_target_series: pd.Series | None


def load_signed_signal_research_data(
    config: "ResearchConfig",
    expanded_specs: list[dict[str, object]],
    *,
    candles_override: pd.DataFrame | None = None,
    print_loaded: bool = False,
) -> FrozenSignalResearchData:
    combo_signal_target: ComboMap = {}
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
        paired = pd.DataFrame({"signal": feature, "target": target}).dropna()
        if paired.empty:
            if print_loaded:
                print(f"  [{label}] SKIP -- aligned feature/target empty")
            continue

        signal = normalize_series_datetime_index(paired["signal"])
        target_series = normalize_series_datetime_index(paired["target"])
        key = combo_key(combo)
        combo_signal_target[key] = pd.DataFrame({"signal": signal, "target": target_series})
        successful_param_grid.append(dict(combo))

        if reference_index is None:
            reference_index = normalize_datetime_index(target_series.index)
            reference_target_series = target_series.reindex(reference_index)

        if print_loaded:
            print(f"  [{label}] loaded n={len(signal):,}")

    return FrozenSignalResearchData(
        combo_signal_target=combo_signal_target,
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
