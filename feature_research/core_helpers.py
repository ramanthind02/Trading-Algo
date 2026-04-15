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
    """Convert params dict to hashable sorted tuple for use as dict key.

    Recursively normalizes nested ``dict`` / ``list`` values in research payloads
    frozen params with ``source_bias_node_spec``) so the result is hashable.
    """

    def _hashable(v: object) -> object:
        if isinstance(v, Enum):
            return v.value
        if isinstance(v, dict):
            return tuple(
                sorted(
                    ((str(k), _hashable(val)) for k, val in v.items()),
                    key=lambda item: item[0],
                )
            )
        if isinstance(v, list):
            return tuple(_hashable(x) for x in v)
        if isinstance(v, tuple):
            return tuple(_hashable(x) for x in v)
        return v

    return tuple(sorted(((k, _hashable(v)) for k, v in params.items()), key=lambda item: item[0]))


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


def build_runtime_walkforward_config(
    config: "ResearchConfig",
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> "WalkforwardResearchConfig":
    """Build runtime walkforward configuration from date bounds.

    Uses fixed defaults for top_k and objective metric so OOS/validation
    single-fold runs do not depend on removed ResearchConfig fields.
    Permutation objective is specified via PermutationResearchConfig.

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
