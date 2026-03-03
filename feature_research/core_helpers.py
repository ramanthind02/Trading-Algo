from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from utils.core.enums import TimeFrame


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
