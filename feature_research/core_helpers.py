from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import pandas as pd

from utils.core.enums import TimeFrame

if TYPE_CHECKING:
    from feature_research.in_sample.config import ResearchConfig
    from utils.evaluation.walkforward.config import WalkforwardResearchConfig


def normalize_timeframe_from_bias_spec(
    bias_spec: Mapping[str, Any],
    fallback: TimeFrame = TimeFrame.D,
) -> TimeFrame:
    """Extract and normalize timeframe from bias spec."""
    raw = bias_spec.get("timeframes", [fallback])
    first = raw[0] if isinstance(raw, list) else raw
    return TimeFrame[first] if isinstance(first, str) else first


def combo_key(params: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    """Convert params dict to hashable sorted tuple for use as dict key."""
    return tuple(sorted(params.items(), key=lambda item: item[0]))


def normalize_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    """Normalize datetime index to timezone-naive UTC."""
    datetime_index = pd.DatetimeIndex(index)
    return datetime_index.tz_localize(None) if datetime_index.tz is not None else datetime_index


def unique_sorted_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    """Deduplicate and sort datetime index."""
    normalized = normalize_datetime_index(index)
    unique_vals = normalized.unique()
    return pd.DatetimeIndex(unique_vals).sort_values()


def normalize_series_datetime_index(series: pd.Series) -> pd.Series:
    """Normalize series datetime index to timezone-naive UTC."""
    if not isinstance(series.index, pd.DatetimeIndex):
        return series
    normalized_series = series.copy()
    normalized_series.index = normalize_datetime_index(series.index)
    return normalized_series


def expand_params_with_bin_count(
    params: Mapping[str, object],
    bin_counts: list[int],
) -> list[dict[str, object]]:
    """Expand params by bin_count for continuous grids."""
    if "bin_count" in params:
        return [dict(params)]
    if not bin_counts:
        return [dict(params)]
    return [{**params, "bin_count": int(bin_count)} for bin_count in bin_counts]


def expand_params_with_selected_bin(
    params_list: list[Mapping[str, object]],
    *,
    bin_index_min: int = 0,
    bin_index_max: int | None = None,
) -> list[dict[str, object]]:
    """Expand each param dict to include selected_bin."""
    expanded: list[dict[str, object]] = []
    for params in params_list:
        bin_count = params.get("bin_count")
        if bin_count is None:
            expanded.append(dict(params))
            continue
        n_bins = int(bin_count)
        if bin_index_max is not None:
            start = max(0, bin_index_min)
            end = min(n_bins, bin_index_max + 1)
            bin_range = range(start, end)
        else:
            bin_range = range(n_bins)
        for selected_bin in bin_range:
            expanded.append({**params, "selected_bin": selected_bin})
    return expanded


def build_runtime_walkforward_config(
    config: "ResearchConfig",
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> "WalkforwardResearchConfig":
    """Build runtime walkforward configuration from date bounds.

    Uses fixed defaults for top_k, objective metric, and smoothing so that
    OOS/validation single-fold runs do not depend on removed ResearchConfig
    fields. Permutation objective is specified via PermutationResearchConfig.

    Parameters
    ----------
    config : ResearchConfig
        Research configuration with output_root and n_jobs.
    train_start : pd.Timestamp
        Training period start date.
    train_end : pd.Timestamp
        Training period end date.
    test_start : pd.Timestamp
        Test period start date.
    test_end : pd.Timestamp
        Test period end date.

    Returns
    -------
    WalkforwardResearchConfig
        Configuration with test_step computed from date range.
    """
    from utils.evaluation.walkforward.config import (
        WalkforwardResearchConfig,
        WalkforwardSelectionMethod,
    )

    test_step = max(1, int((test_end - test_start).days))
    return WalkforwardResearchConfig(
        train_start=train_start.to_pydatetime(),
        train_end=train_end.to_pydatetime(),
        enabled=True,
        test_step=test_step,
        num_steps=1,
        top_k=1,
        objective_metric_name="t_stat",
        min_fold_samples=10,
        output_root=config.output_root,
        selection_method=WalkforwardSelectionMethod.TOP_K,
        n_jobs=config.n_jobs,
        smoothing_self_weight=3.0,
    )
