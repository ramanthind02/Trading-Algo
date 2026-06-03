from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os import cpu_count
from typing import TYPE_CHECKING

import pandas as pd

from feature_research._internal.core_helpers import (
    combo_key,
    normalize_datetime_index,
    normalize_series_datetime_index,
    unique_sorted_datetime_index,
)
from feature_research.in_sample.data_loader import (
    enrich_param_combo_with_module,
    load_candles_for_config,
    load_features_for_combo,
    param_combo_label,
)

if TYPE_CHECKING:
    from feature_research.config import ResearchConfig


ComboMap = dict[tuple[tuple[str, object], ...], pd.DataFrame]


@dataclass(frozen=True)
class FrozenSignalResearchData:
    combo_signal_target: ComboMap
    successful_param_grid: list[dict[str, object]]
    reference_index: pd.Index | None
    reference_target_series: pd.Series | None


def _resolve_combo_workers(config: "ResearchConfig", combo_count: int) -> int:
    raw_n_jobs = int(getattr(config, "n_jobs", 1) or 1)
    if raw_n_jobs == -1:
        desired = cpu_count() or 1
    elif raw_n_jobs < 1:
        desired = 1
    else:
        desired = raw_n_jobs
    return max(1, min(combo_count, desired))


def _load_combo_signal_target(
    single_spec: dict[str, object],
    config: "ResearchConfig",
    *,
    candles_override: pd.DataFrame | None = None,
) -> tuple[dict[str, object], pd.DataFrame | None]:
    combo = enrich_param_combo_with_module(
        single_spec["params"],
        single_spec.get("module_name"),
    )
    data = load_features_for_combo(
        single_spec,
        config,
        candles_override=candles_override,
        populate_on_miss=True,
    )
    if data is None:
        return combo, None

    feature, target, _, ticker_s = data
    paired = pd.concat(
        [
            feature.rename("signal"),
            target.rename("target"),
            ticker_s.rename("ticker"),
        ],
        axis=1,
    ).dropna(how="any")
    if paired.empty:
        return combo, None

    signal = normalize_series_datetime_index(paired["signal"])
    target_series = normalize_series_datetime_index(paired["target"])
    returns = signal.mul(target_series).rename("returns")
    # Do not reindex with duplicate labels (multi-ticker panels); row order matches paired.
    ticker_aligned = pd.Series(
        paired["ticker"].to_numpy(),
        index=signal.index,
        dtype=str,
        name="ticker",
    )
    return combo, pd.concat(
        {
            "signal": signal,
            "target": target_series,
            "returns": returns,
            "ticker": ticker_aligned,
        },
        axis=1,
    )


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
    worker_count = _resolve_combo_workers(config, len(expanded_specs))
    loader = lambda spec: _load_combo_signal_target(
        spec,
        config,
        candles_override=candles_override,
    )
    combo_frames: list[tuple[dict[str, object], pd.DataFrame | None]]
    if worker_count == 1 or len(expanded_specs) <= 1:
        combo_frames = [loader(single_spec) for single_spec in expanded_specs]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            combo_frames = list(executor.map(loader, expanded_specs))

    for single_spec, (combo, paired) in zip(expanded_specs, combo_frames, strict=False):
        label = param_combo_label(combo)
        if paired is None:
            if print_loaded:
                print(f"  [{label}] SKIP -- no data")
            continue

        signal = paired["signal"]
        target_series = paired["target"]
        key = combo_key(combo)
        combo_signal_target[key] = paired
        successful_param_grid.append(dict(combo))

        if reference_index is None:
            reference_index = (
                target_series.index
                if isinstance(target_series.index, pd.MultiIndex)
                else normalize_datetime_index(target_series.index)
            )
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
    reference_index: pd.Index,
    reference_target_series: pd.Series | None,
) -> pd.Series:
    if reference_target_series is None:
        return pd.Series(0.0, index=reference_index, name="walkforward_target")
    return reference_target_series.reindex(reference_index).fillna(0.0).rename("walkforward_target")


def load_portfolio_candles(config: "ResearchConfig") -> pd.DataFrame:
    return load_candles_for_config(config)
