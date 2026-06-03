"""Quantile binning for permutation when the EDA grid is continuous.

Each param combo: load continuous features → ``pd.qcut`` per ticker (Phase 0 style) →
map selected bins to ±1/0 via :class:`~utils.core.enums.Direction` (lowest bin long for
``Direction.LONG``, highest short for ``Direction.SHORT``, both tails for ``LONG_SHORT``).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pandas as pd

from feature_extraction.feature_extractor import extract_features_for_bias_node
from feature_research.binning.transforms import (
    assign_quantile_bins,
    map_bin_selection_to_position,
    merge_feature_target_panel,
    quantile_bin_sets_for_strategy,
)
from feature_research._internal.core_helpers import combo_key, normalize_series_datetime_index
from feature_research.in_sample.data_loader import param_combo_label
from utils.evaluation.walkforward.research_data import (
    FrozenSignalResearchData,
    _resolve_combo_workers,
)

if TYPE_CHECKING:
    from feature_research.config import ResearchConfig


def permutation_quantile_n_bins(config: ResearchConfig) -> int:
    bc = config.binning_params.bin_counts
    return int(bc[0]) if bc else 10


def _continuous_to_quantile_signed_frame(
    merged: pd.DataFrame,
    *,
    n_bins: int,
    long_bins: frozenset[int],
    short_bins: frozenset[int],
) -> pd.DataFrame | None:
    if merged.empty or "ticker" not in merged.columns:
        return None
    bin_panel = merged[["ticker", "feature"]].copy()
    binned = assign_quantile_bins(bin_panel, n_bins=n_bins)
    if binned.empty or "bin_index" not in binned.columns:
        return None

    m = merged.reset_index()
    datetime_col = m.columns[0]
    b = binned.reset_index()
    keys = [datetime_col, "ticker"]
    if not all(k in b.columns for k in keys):
        return None
    joined = pd.merge(m, b[keys + ["bin_index"]], on=keys, how="inner")
    if joined.empty:
        return None
    joined = joined.sort_values(keys, kind="mergesort")
    signs = [
        float(map_bin_selection_to_position(bx, long_bins=long_bins, short_bins=short_bins))
        for bx in joined["bin_index"]
    ]
    mi = pd.MultiIndex.from_arrays([joined[datetime_col], joined["ticker"]])
    signal = pd.Series(signs, index=mi, name="signal", dtype=float)
    target_series = pd.Series(joined["target"].to_numpy(dtype=float), index=mi, name="target")
    paired = pd.DataFrame({"signal": signal, "target": target_series}).dropna()
    if paired.empty:
        return None
    sig = normalize_series_datetime_index(paired["signal"])
    tgt = normalize_series_datetime_index(paired["target"])
    returns = sig.mul(tgt).rename("returns")
    return pd.DataFrame({"signal": sig, "target": tgt, "returns": returns})


def _load_combo_quantile_binned_signed(
    single_spec: dict[str, object],
    config: ResearchConfig,
    *,
    candles_override: pd.DataFrame | None = None,
) -> tuple[dict[str, object], pd.DataFrame | None]:
    params = single_spec.get("params", {})
    if not isinstance(params, dict):
        return {}, None
    features_df, targets_df = extract_features_for_bias_node(
        bias_spec=single_spec,
        ticker=config.tickers,
        start=config.start,
        end=config.end,
        target_col=config.target_col,
        use_cache=True,
        candles_override=candles_override,
    )
    if features_df is None or features_df.empty or targets_df is None or targets_df.empty:
        return params, None

    feature_cols = [c for c in features_df.columns if c != "ticker"]
    if not feature_cols:
        return params, None
    feature_col = feature_cols[0]
    if config.target_col not in targets_df.columns:
        return params, None

    merged = merge_feature_target_panel(
        features_df,
        targets_df,
        feature_col=feature_col,
        target_col=config.target_col,
    )
    if merged.empty:
        return params, None

    n_bins = permutation_quantile_n_bins(config)
    lb, sb = quantile_bin_sets_for_strategy(config.strategy, n_bins)
    paired = _continuous_to_quantile_signed_frame(merged, n_bins=n_bins, long_bins=lb, short_bins=sb)
    if paired is None:
        return params, None
    return params, paired


def load_quantile_binned_permutation_research_data(
    config: ResearchConfig,
    expanded_specs: list[dict[str, object]],
    *,
    candles_override: pd.DataFrame | None = None,
    print_loaded: bool = False,
) -> FrozenSignalResearchData:
    """Like ``load_signed_signal_research_data`` but signals are quantile-binned ±1/0 per combo."""
    combo_signal_target: dict[tuple[tuple[str, object], ...], pd.DataFrame] = {}
    successful_param_grid: list[dict[str, object]] = []
    reference_index: pd.Index | None = None
    reference_target_series: pd.Series | None = None

    worker_count = _resolve_combo_workers(config, len(expanded_specs))
    loader = lambda spec: _load_combo_quantile_binned_signed(
        spec,
        config,
        candles_override=candles_override,
    )
    if worker_count == 1 or len(expanded_specs) <= 1:
        combo_frames = [loader(single_spec) for single_spec in expanded_specs]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            combo_frames = list(executor.map(loader, expanded_specs))

    for single_spec, (combo, paired) in zip(expanded_specs, combo_frames, strict=False):
        label = param_combo_label(combo)
        if paired is None:
            if print_loaded:
                print(f"  [{label}] SKIP -- no binned data")
            continue

        signal = paired["signal"]
        target_series = paired["target"]
        key = combo_key(combo)
        combo_signal_target[key] = paired
        successful_param_grid.append(dict(combo))

        if reference_index is None:
            reference_index = target_series.index
            reference_target_series = target_series.reindex(reference_index)

        if print_loaded:
            print(f"  [{label}] binned n={len(signal):,}")

    return FrozenSignalResearchData(
        combo_signal_target=combo_signal_target,
        successful_param_grid=successful_param_grid,
        reference_index=reference_index,
        reference_target_series=reference_target_series,
    )
