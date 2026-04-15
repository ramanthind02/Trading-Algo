"""Centralized cache helpers for feature extraction workflows."""

from __future__ import annotations

from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .central_cache import CentralCacheStore
from .central_cache_errors import CacheCoverageError
from .central_cache_models import (
    ArtifactDescriptor,
    ArtifactScope,
    CacheRequest,
    LookupMode,
)
from .cross_ticker_store import (
    CrossTickerDataStore,
    SCALAR_LIST_PARAM_KEYS,
    extract_cross_ticker_names,
)
from utils.core.enums import Ticker, TimeFrame


def is_grid_axis(key: str, value: object) -> bool:
    """Return True if *value* should be expanded as a grid dimension."""
    return isinstance(value, list) and key not in SCALAR_LIST_PARAM_KEYS


# Inner param blobs for composite nodes (dual_signal, filter_gate, filter_and_signal).
NESTED_GRID_PARAM_KEYS: frozenset[str] = frozenset(
    {"paramsA", "paramsB", "filter_params", "signal_params"}
)


def _composite_nested_side_has_grid(params: Dict[str, Any]) -> bool:
    return any(
        k in NESTED_GRID_PARAM_KEYS
        and isinstance(params[k], dict)
        and any(is_grid_axis(ik, iv) for ik, iv in params[k].items())
        for k in params
    )


def expand_param_grid(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand parameter grid to a list of parameter dicts.

    Supports nested grids under ``paramsA`` / ``paramsB`` (``dual_signal``) and
    ``filter_params`` / ``signal_params`` (``filter_gate``, ``filter_and_signal``):
    each nested dict is expanded independently, then combined by Cartesian product.
    """
    if not isinstance(params, dict):
        return [{}]

    nested_keys = sorted(
        k for k in params if k in NESTED_GRID_PARAM_KEYS and isinstance(params[k], dict)
    )
    if nested_keys and _composite_nested_side_has_grid(params):
        rest = {k: v for k, v in params.items() if k not in nested_keys}
        inner_seqs = [expand_param_grid(params[k]) for k in nested_keys]
        merged_nested = [
            dict(zip(nested_keys, combo)) for combo in product(*inner_seqs)
        ]
        return [
            expanded
            for piece in merged_nested
            for expanded in expand_param_grid({**rest, **piece})
        ]

    has_grid = any(is_grid_axis(k, v) for k, v in params.items())
    if not has_grid:
        return [params]

    keys = list(params.keys())
    values = [
        value if is_grid_axis(key, value) else [value]
        for key, value in params.items()
    ]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def normalize_ticker_str(ticker: object) -> str:
    """Normalize ticker-like objects to stable strings."""
    if hasattr(ticker, "name"):
        return str(getattr(ticker, "name"))
    if isinstance(ticker, str):
        return ticker
    return str(ticker)


def normalize_ticker_series(series: pd.Series) -> pd.Series:
    """Normalize a ticker column without per-row apply()."""
    if series.empty:
        return series
    first = series.iloc[0]
    if hasattr(first, "name"):
        return pd.Series([value.name for value in series], index=series.index, dtype=str)
    if isinstance(first, str):
        return series
    return series.astype(str)


def ensure_utc_datetime_index(values: object) -> pd.DatetimeIndex:
    """Convert arbitrary datetime-like values to a UTC-normalized index."""
    datetime_values = pd.to_datetime(values)
    if isinstance(datetime_values, pd.Series):
        datetime_index = pd.DatetimeIndex(datetime_values.to_numpy())
    else:
        datetime_index = pd.DatetimeIndex(datetime_values)
    if datetime_index.tz is None:
        return datetime_index.tz_localize("UTC")
    return datetime_index.tz_convert("UTC")


def prepare_candles_override(
    candles_override: pd.DataFrame,
    tickers: List[Ticker],
    use_millisecond_offset: bool,
) -> pd.DataFrame:
    """Validate and normalize override candles for forward-return computation."""
    _ = use_millisecond_offset
    required_columns = {"datetime", "open", "high", "low", "close", "ticker"}
    missing_columns = sorted(required_columns.difference(candles_override.columns))
    if missing_columns:
        raise ValueError(
            "candles_override missing required columns: "
            f"{missing_columns}. Expected columns include {sorted(required_columns)}"
        )

    normalized = candles_override.copy()
    normalized["datetime"] = ensure_utc_datetime_index(normalized["datetime"])
    normalized["ticker"] = normalize_ticker_series(normalized["ticker"])

    requested_tickers = [normalize_ticker_str(ticker) for ticker in tickers]
    requested_set = set(requested_tickers)
    override_ticker_set = set(normalized["ticker"].unique())
    missing_tickers = sorted(requested_set.difference(override_ticker_set))
    if missing_tickers:
        raise ValueError(
            "candles_override missing ticker data for requested tickers: "
            f"{missing_tickers}. Available tickers: {sorted(override_ticker_set)}"
        )

    return normalized


def preload_cross_ticker_data(
    param_combos: List[Dict[str, Any]],
    timeframes: List[TimeFrame],
    start: datetime,
    end: datetime,
) -> None:
    """Scan param combos for ``cross_tickers`` and pre-load referenced tickers."""
    cross_names: set[str] = set()
    for combo in param_combos:
        cross_names.update(extract_cross_ticker_names(combo))

    if not cross_names:
        return

    store = CrossTickerDataStore.get_instance()
    for name in cross_names:
        try:
            cross_ticker = Ticker[name]
        except KeyError:
            continue
        for timeframe in timeframes:
            if not store.is_loaded(cross_ticker, timeframe):
                store.load(cross_ticker, timeframe, start=start, end=end)


def preload_cross_ticker_override_data(
    candles_override: pd.DataFrame,
    timeframes: List[TimeFrame],
) -> None:
    """Load override candles into the cross-ticker store via ``set_data()``."""
    if candles_override.empty or "ticker" not in candles_override.columns:
        return

    normalized = candles_override.copy()
    normalized["ticker"] = normalize_ticker_series(normalized["ticker"])

    store = CrossTickerDataStore.get_instance()
    for ticker_name, ticker_df in normalized.groupby("ticker"):
        try:
            ticker_enum = Ticker[ticker_name]
        except KeyError:
            continue

        payload = ticker_df.drop(
            columns=["ticker", "timeframe", "timestamp"],
            errors="ignore",
        )
        for timeframe in timeframes:
            store.set_data(ticker_enum, timeframe, payload)


def build_bias_node_descriptor(
    module_name: str,
    params: Dict[str, Any],
    ticker: Ticker,
    tf: TimeFrame,
    scope: ArtifactScope,
) -> ArtifactDescriptor:
    """Build the canonical cache descriptor for a bias-node artifact."""
    return ArtifactDescriptor(
        family="bias",
        module_name=module_name,
        params=params,
        ticker=ticker,
        timeframe=tf,
        scope=scope,
    )


def read_aligned_feature_artifact(
    cache_store: CentralCacheStore,
    descriptor: ArtifactDescriptor,
    request: CacheRequest,
    price_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Read an artifact from cache and align it to the current price index."""
    clamped_request = request
    if request.start is not None or request.end is not None:
        record = cache_store.describe_artifact(descriptor)
        if record is not None:
            effective_start = request.start
            effective_end = request.end
            if request.start is not None and record.coverage.start is not None:
                effective_start = max(
                    pd.Timestamp(request.start),
                    pd.Timestamp(record.coverage.start),
                ).to_pydatetime()
            if request.end is not None and record.coverage.end is not None:
                effective_end = min(
                    pd.Timestamp(request.end),
                    pd.Timestamp(record.coverage.end),
                ).to_pydatetime()
            if (
                effective_start is not None
                and effective_end is not None
                and pd.Timestamp(effective_start) > pd.Timestamp(effective_end)
            ):
                raise CacheCoverageError(
                    module_name=descriptor.module_name or descriptor.family,
                    ticker=descriptor.ticker,
                    timeframe=descriptor.timeframe,
                    start=request.start,
                    end=request.end,
                    coverage_start=record.coverage.start,
                    coverage_end=record.coverage.end,
                )
            clamped_request = CacheRequest(
                start=effective_start,
                end=effective_end,
                exact_dt=request.exact_dt,
                as_of_dt=request.as_of_dt,
            )

    cached_df = cache_store.read_artifact(
        descriptor,
        request=clamped_request,
        lookup_mode=LookupMode.EXACT,
    )
    cached_aligned = cached_df
    if cached_aligned.index.tz is not None:
        cached_aligned = cached_aligned.reindex(price_index.tz_convert(cached_aligned.index.tz))
    else:
        cached_aligned = cached_aligned.reindex(price_index.tz_localize(None))

    if cached_aligned.isna().all().all():
        price_index_naive = price_index.tz_localize(None) if price_index.tz else price_index
        cached_index_naive = cached_df.index.tz_localize(None) if cached_df.index.tz else cached_df.index
        cached_df_naive = cached_df.copy()
        cached_df_naive.index = cached_index_naive
        cached_aligned = cached_df_naive.reindex(price_index_naive)
    return cached_aligned


def assign_cached_feature_values(
    feature_data: np.ndarray,
    cached_aligned: pd.DataFrame,
    start_col: int,
    end_col: int,
) -> None:
    """Copy cached artifact columns into the target feature matrix."""
    if "value" in cached_aligned.columns:
        feature_data[:, start_col] = cached_aligned["value"].values
        return

    for idx, column in enumerate(cached_aligned.columns):
        if start_col + idx >= end_col:
            break
        feature_data[:, start_col + idx] = cached_aligned[column].values


def write_feature_artifact(
    cache_store: CentralCacheStore,
    descriptor: ArtifactDescriptor,
    cached_slice: pd.DataFrame,
    depends_on: Sequence[tuple[Ticker, TimeFrame]],
) -> None:
    """Write a feature slice back through the central cache facade."""
    if cached_slice.shape[1] == 1:
        cache_store.write_artifact(
            descriptor,
            cached_slice.rename(columns={cached_slice.columns[0]: "value"}),
            depends_on=depends_on,
        )
        return
    cache_store.write_artifact(descriptor, cached_slice, depends_on=depends_on)


__all__ = [
    "NESTED_GRID_PARAM_KEYS",
    "SCALAR_LIST_PARAM_KEYS",
    "assign_cached_feature_values",
    "build_bias_node_descriptor",
    "ensure_utc_datetime_index",
    "expand_param_grid",
    "extract_cross_ticker_names",
    "is_grid_axis",
    "normalize_ticker_series",
    "normalize_ticker_str",
    "prepare_candles_override",
    "preload_cross_ticker_data",
    "preload_cross_ticker_override_data",
    "read_aligned_feature_artifact",
    "write_feature_artifact",
]
