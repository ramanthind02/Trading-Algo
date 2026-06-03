from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping

import pandas as pd

from utils.core.enums import TimeFrame

if TYPE_CHECKING:
    from feature_research.config import ResearchConfig
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
    """Convert params dict to a recursively hashable sorted tuple."""

    def _hashable(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return tuple(
                sorted(
                    ((str(key), _hashable(inner_value)) for key, inner_value in value.items()),
                    key=lambda item: item[0],
                )
            )
        if isinstance(value, list):
            return tuple(_hashable(item) for item in value)
        if isinstance(value, tuple):
            return tuple(_hashable(item) for item in value)
        return value

    return tuple(
        sorted(((key, _hashable(value)) for key, value in params.items()), key=lambda item: item[0])
    )


def normalize_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    """Normalize datetime index to timezone-naive UTC."""

    datetime_index = pd.DatetimeIndex(index)
    return (
        datetime_index.tz_localize(None)
        if datetime_index.tz is not None
        else datetime_index
    )


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


def build_runtime_walkforward_config(
    config: "ResearchConfig",
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> "WalkforwardResearchConfig":
    """Build runtime walkforward configuration from date bounds."""

    from utils.evaluation.walkforward.config import WalkforwardResearchConfig

    test_step = max(1, int((test_end - test_start).days))
    return WalkforwardResearchConfig(
        train_start=train_start.to_pydatetime(),
        train_end=train_end.to_pydatetime(),
        enabled=True,
        test_step=test_step,
        num_steps=1,
        objective_metric_name="t_stat",
        min_fold_samples=10,
        output_root=config.output_root,
        n_jobs=config.n_jobs,
    )
