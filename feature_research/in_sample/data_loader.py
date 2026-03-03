"""Unified data loading and cache management for both continuous and rule-based research.

Dispatches on feature_type for validation-specific logic:
  - CONTINUOUS: no additional validation beyond standard checks
  - RULE_BASED: skips multi-ticker raw return validation (handled at config level)
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from feature_research.in_sample.config import ResearchConfig

from feature_extraction.feature_extractor import extract_features_for_bias_node
from feature_research.bootstrap import find_repo_root
from feature_research.config import FeatureType
from utils.cache.cache_manager import CacheManager
from utils.core.enums import Ticker, TimeFrame
from utils.core.helpers import load_data_multi_ticker

_UNNORMALIZED_RETURN_COLS: frozenset[str] = frozenset({"log_return", "raw_return"})


def _resolve_project_root() -> Path | None:
    this_file = Path(__file__).resolve()
    return find_repo_root(this_file)


def expand_bias_specs(bias_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a bias_spec with list-valued params into one spec per param combo.

    Parameters
    ----------
    bias_spec : dict[str, Any]
        Bias specification with optional list-valued params.

    Returns
    -------
    list[dict[str, Any]]
        List of fully expanded specs, one per parameter combination.
    """
    params = bias_spec.get("params", {})
    keys = list(params.keys())
    values = [v if isinstance(v, list) else [v] for v in params.values()]
    combos = [dict(zip(keys, combo)) for combo in product(*values)] if keys else [{}]
    return [
        {
            "module_name": bias_spec["module_name"],
            "params": combo,
            "timeframes": bias_spec.get("timeframes", [TimeFrame.D]),
        }
        for combo in combos
    ]


def load_candles_for_config(config: "ResearchConfig") -> pd.DataFrame:
    """Load candles for the config universe and normalize to DatetimeIndex.

    Parameters
    ----------
    config : ResearchConfig
        Research configuration with tickers, dates, and bias_spec.

    Returns
    -------
    pd.DataFrame
        OHLCV data with DatetimeIndex, sorted by datetime.
    """
    timeframes = config.bias_spec.get("timeframes", [TimeFrame.D])
    raw_timeframe = timeframes[0] if isinstance(timeframes, list) else timeframes
    timeframe = TimeFrame[raw_timeframe] if isinstance(raw_timeframe, str) else raw_timeframe

    # Primary key is (datetime, ticker); no millisecond offsets
    candles_df = load_data_multi_ticker(
        tickers=config.tickers,
        timeframe=timeframe,
        start=config.start,
        end=config.end,
    )

    normalized = candles_df.copy()
    if "datetime" not in normalized.columns:
        raise ValueError("Candles data must include a 'datetime' column.")

    datetimes = pd.to_datetime(normalized["datetime"], utc=False)
    if getattr(datetimes.dt, "tz", None) is not None:
        datetimes = datetimes.dt.tz_localize(None)

    normalized["datetime"] = datetimes
    normalized = normalized.sort_values("datetime")
    normalized = normalized.set_index("datetime", drop=False)
    normalized.index = pd.DatetimeIndex(normalized.index, name="datetime_index")
    return normalized


def param_combo_label(combo: dict[str, Any]) -> str:
    """Return a human-readable folder name for a param combo dict.

    Parameters
    ----------
    combo : dict[str, Any]
        Parameter combination mapping.

    Returns
    -------
    str
        Folder-safe label, e.g., "lookback_10__other_5" (sorted by key).

    Examples
    --------
    >>> param_combo_label({"lookback": 5})
    'lookback_5'
    >>> param_combo_label({"lookback": 20, "atr_length": 14})
    'atr_length_14__lookback_20'
    """
    parts = [f"{k}_{v}" for k, v in sorted(combo.items())]
    return "__".join(parts)


def populate_cache_if_needed(config: "ResearchConfig") -> None:
    """Populate the feature cache if config.populate_cache is True.

    Invariant: cache population always uses all possible data (full date range
    discovered from OHLC parquet files), not config.start/end. This avoids
    partial cache when config is later extended.

    Safe to call even if cache already exists — ``overwrite_existing=False``
    means only missing entries are computed.

    Parameters
    ----------
    config : ResearchConfig
        Research configuration with populate_cache flag and bias_spec.
    """
    if not config.populate_cache:
        return

    project_root = _resolve_project_root()
    if project_root is None:
        project_root = Path(__file__).resolve().parents[3]

    candle_dir = project_root / "data" / "ohlc_data"
    if not candle_dir.exists():
        print(
            f"[data_loader] WARNING: candle directory not found at {candle_dir}. "
            "Skipping cache population."
        )
        return

    manager = CacheManager(candle_dir=str(candle_dir))
    expanded = expand_bias_specs(config.bias_spec)

    # Invariant: use full available data range from OHLC, not config dates
    timeframes_raw = config.bias_spec.get("timeframes", [TimeFrame.D])
    timeframes = [
        TimeFrame[t] if isinstance(t, str) else t
        for t in (timeframes_raw if isinstance(timeframes_raw, list) else [timeframes_raw])
    ]
    date_range = manager.get_available_date_range(tickers=config.tickers, timeframes=timeframes)
    if date_range is not None:
        start_date, end_date = date_range
        print(
            f"[data_loader] Populating cache with full OHLC range: {start_date.date()} -> {end_date.date()}"
        )
    else:
        start_date = config.start
        end_date = config.end
        print(
            f"[data_loader] Could not discover OHLC date range; using config: {start_date.date()} -> {end_date.date()}"
        )

    summary = manager.populate_cache(
        bias_node_specs=expanded,
        tickers=config.tickers,
        start_date=start_date,
        end_date=end_date,
        show_progress=True,
        overwrite_existing=False,
    )
    print(f"[data_loader] Cache populated: {summary}")


def get_tickers_with_coverage_for_config(config: "ResearchConfig") -> list[Ticker]:
    """Return tickers that have OHLC data covering [config.start, config.end].

    Tickers whose data starts after config.start or ends before config.end are
    excluded so pipelines (walkforward, permutation, OOS) do not hit cache
    misses for partial ranges.

    Returns
    -------
    list
        Subset of config.tickers with full coverage. May be empty if no ticker
        has data for the config range.
    """
    project_root = _resolve_project_root()
    if project_root is None:
        return list(config.tickers)
    candle_dir = project_root / "data" / "ohlc_data"
    if not candle_dir.exists():
        return list(config.tickers)
    manager = CacheManager(candle_dir=str(candle_dir))
    timeframes_raw = config.bias_spec.get("timeframes", [TimeFrame.D])
    timeframes = [
        TimeFrame[t] if isinstance(t, str) else t
        for t in (
            timeframes_raw
            if isinstance(timeframes_raw, list)
            else [timeframes_raw]
        )
    ]
    ranges = manager.get_available_date_range_per_ticker(
        tickers=config.tickers,
        timeframes=timeframes,
    )
    start_ts = pd.Timestamp(config.start)
    end_ts = pd.Timestamp(config.end)
    covered = [
        t
        for t in config.tickers
        if t in ranges
        and ranges[t][0] <= start_ts.to_pydatetime()
        and ranges[t][1] >= end_ts.to_pydatetime()
    ]
    return covered


def get_available_date_ranges_for_tickers(
    config: "ResearchConfig", tickers: list[Ticker]
) -> dict[Ticker, tuple[pd.Timestamp, pd.Timestamp]]:
    """Return per-ticker (min_date, max_date) from OHLC data for error messages.

    Used when no tickers have full coverage so we can report what range each
    ticker actually has (e.g. suggest narrowing config.start/end or oos_window).
    """
    project_root = _resolve_project_root()
    if project_root is None or not (project_root / "data" / "ohlc_data").exists():
        return {}
    manager = CacheManager(candle_dir=str(project_root / "data" / "ohlc_data"))
    timeframes_raw = config.bias_spec.get("timeframes", [TimeFrame.D])
    timeframes = [
        TimeFrame[t] if isinstance(t, str) else t
        for t in (
            timeframes_raw
            if isinstance(timeframes_raw, list)
            else [timeframes_raw]
        )
    ]
    ranges = manager.get_available_date_range_per_ticker(
        tickers=tickers,
        timeframes=timeframes,
    )
    return {
        t: (pd.Timestamp(ranges[t][0]), pd.Timestamp(ranges[t][1]))
        for t in tickers
        if t in ranges
    }


def load_features_for_combo(
    single_combo_spec: dict[str, Any],
    config: "ResearchConfig",
    candles_override: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series, str] | None:
    """Extract feature + target Series for a single param combo across all config tickers.

    Dispatches on feature_type for validation logic.

    Parameters
    ----------
    single_combo_spec : dict[str, Any]
        Single (non-expanded) bias spec with module_name, params, timeframes.
    config : ResearchConfig
        Research configuration with target_col, tickers, feature_type, etc.
    candles_override : pd.DataFrame | None, default=None
        Optional override candles passed through to ``extract_features_for_bias_node``
        for forward-return computation. Expected schema: one row per ticker/timestamp
        with columns ``datetime``, ``open``, ``high``, ``low``, ``close``, ``ticker``.

    Returns
    -------
    (feature, target, feature_col) or None
        Aligned (feature, target) Series and feature column name.
        Returns None if extraction fails or returns empty data.

    Raises
    ------
    ValueError
        If target data is missing or misaligned.
    """
    features_df, targets_df = extract_features_for_bias_node(
        bias_spec=single_combo_spec,
        ticker=config.tickers,
        start=config.start,
        end=config.end,
        target_col=config.target_col,
        use_cache=config.use_cache,
        candles_override=candles_override,
    )

    if features_df is None or features_df.empty:
        print(f"[data_loader] Empty features for {single_combo_spec['params']}. Skipping.")
        return None

    feature_cols = [c for c in features_df.columns if c != "ticker"]
    if not feature_cols:
        return None
    feature_col = feature_cols[0]

    params = single_combo_spec.get("params", {})
    if targets_df is None:
        raise ValueError(f"Target extraction returned no dataframe for params={params}.")
    if targets_df.empty:
        raise ValueError(f"Target dataframe is empty for params={params}.")
    if config.target_col not in targets_df.columns:
        available_target_cols = [c for c in targets_df.columns if c != "ticker"]
        raise ValueError(
            f"Configured target_col '{config.target_col}' is missing for params={params}. "
            f"Available target columns: {available_target_cols}."
        )
    target_col_name = config.target_col

    aligned = pd.DataFrame(
        {"feature": features_df[feature_col], "target": targets_df[target_col_name]}
    ).dropna()

    if aligned.empty:
        return None

    # DISPATCH: Feature-type-specific validation
    # For CONTINUOUS: all validations are ok
    # For RULE_BASED: skip multi-ticker raw return check (already checked at config level)
    if config.feature_type == FeatureType.CONTINUOUS:
        if target_col_name in _UNNORMALIZED_RETURN_COLS:
            unique_tickers = (
                int(features_df["ticker"].nunique())
                if "ticker" in features_df.columns
                else len(config.tickers)
            )
            if unique_tickers > 1:
                raise ValueError(
                    f"load_features_for_combo received target_col='{target_col_name}' with "
                    f"{unique_tickers} tickers. Raw return targets must not be mixed "
                    "across tickers. Use 'log_return_ewsd' or 'log_return_atr'."
                )

    feature_series = aligned["feature"].copy()
    feature_series.name = feature_col
    target_series = aligned["target"].copy()
    target_series.name = target_col_name

    return feature_series, target_series, feature_col
