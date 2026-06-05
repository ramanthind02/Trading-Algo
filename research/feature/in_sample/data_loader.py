"""Unified data loading and cache management for both continuous and rule-based research.

Dispatches on feature_type:

  - **CONTINUOUS:** raw bias-node columns are **quantile-binned per ticker** (same as permutation:
    ``binning_params.bin_counts[0]``, ``strategy`` → ±1/0). Returned ``feature`` series is that
    discrete signal, not the raw continuous values.
  - **SIGNED_SIGNAL:** native discrete output from the node (no binning).

Validation: CONTINUOUS still rejects multi-ticker **raw** return targets (``log_return`` / ``raw_return``).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd

if TYPE_CHECKING:
    from research.feature.config import ResearchConfig

from research.feature._internal.bias_spec_catalog import first_bias_spec


class SupportsBiasCachePopulation(Protocol):
    """Subset of config used by cache bootstrap / bias coverage helpers."""

    tickers: list[Ticker]
    bias_spec: dict[str, Any] | list[dict[str, Any]]
    start: datetime
    end: datetime

from features.extraction.feature_extractor import extract_features_for_bias_node
from research.feature._internal.bootstrap import find_repo_root
from research.feature.config import CachePopulationMode, FeatureType, RAW_TARGET_COLS
from lib.cache import ArtifactScope
from lib.cache.runtime.cache_manager import CacheManager
from lib.core.enums import Ticker, TimeFrame
from lib.core.helpers import load_data_multi_ticker
from lib.cache.runtime.feature_pipeline_support import (
    NESTED_GRID_PARAM_KEYS,
    expand_param_grid as _expand_bias_param_grid,
)
from utils.data.cross_ticker_store import extract_cross_ticker_names


def _resolve_project_root() -> Path | None:
    this_file = Path(__file__).resolve()
    return find_repo_root(this_file)


def _cache_coverage_console_message(summary: dict[str, Any]) -> str:
    """One-line cache summary for the console; omits per-task ``details`` (can be 100+ rows)."""
    parts = [
        f"total_tasks={summary.get('total_tasks')}",
        f"rebuilt={summary.get('rebuilt')}",
        f"validated={summary.get('validated')}",
        f"failed={summary.get('failed')}",
        f"tickers={summary.get('requested_tickers')}",
        f"timeframes={summary.get('timeframes')}",
        f"refresh_mode={summary.get('refresh_mode')}",
    ]
    msg = "[data_loader] Cache coverage ensured: " + ", ".join(parts)
    if int(summary.get("failed") or 0) <= 0:
        return msg
    details = summary.get("details")
    if not isinstance(details, list):
        return msg
    fails = [d for d in details if isinstance(d, dict) and d.get("status") == "failed"]
    if not fails:
        return msg
    cap = 12
    tail = f" ({len(fails)} total)" if len(fails) > cap else ""
    return f"{msg}; failed_tasks={fails[:cap]!r}{tail}"


def _normalize_bias_module_key(module_name: object) -> str:
    return str(module_name or "").replace("_", "").lower()


BIAS_MODULE_COMBO_KEY = "_bias_module"


def enrich_param_combo_with_module(
    params: Mapping[str, Any],
    module_name: object,
) -> dict[str, Any]:
    """Attach taxonomy ``module_name`` so identical inner params stay distinct (e.g. gate variants)."""
    enriched = dict(params)
    if module_name is not None:
        enriched[BIAS_MODULE_COMBO_KEY] = str(module_name)
    return enriched


def params_for_bias_node(params: Mapping[str, Any]) -> dict[str, Any]:
    """Strip combo-identity keys before instantiating a bias node from ``params``."""
    return {
        key: value
        for key, value in params.items()
        if key != BIAS_MODULE_COMBO_KEY
    }


def bias_spec_for_node_instantiation(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a bias spec safe to pass into ``extract_features_for_bias_node``."""
    params = spec.get("params", {})
    if not isinstance(params, dict):
        return dict(spec)
    return {**dict(spec), "params": params_for_bias_node(params)}


def expand_bias_specs(
    bias_spec: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand bias_spec(s) with list-valued params into one spec per parameter combination.

    Supports:

    - **Catalog:** ``bias_spec`` is a non-empty ``list[dict]`` — expand each entry and
      concatenate (e.g. raw breakout + gated breakout branches).
    - **Multi-module:** ``module_name`` is a list of taxonomy keys sharing the same
      ``params`` grid — Cartesian product (e.g. ``filter_gate`` vs ``filter_gate_entry_only``).

    Parameters
    ----------
    bias_spec : dict[str, Any] | list[dict[str, Any]]
        Single spec or ordered catalog of specs.

    Returns
    -------
    list[dict[str, Any]]
        Fully expanded specs.
    """
    if isinstance(bias_spec, list):
        return [spec for branch in bias_spec for spec in expand_bias_specs(branch)]

    params = bias_spec.get("params", {})
    if not isinstance(params, dict):
        params = {}
    combos = _expand_bias_param_grid(params)
    module_names = bias_spec.get("module_name")
    timeframes = bias_spec.get("timeframes", [TimeFrame.D])
    if isinstance(module_names, list):
        return [
            {
                "module_name": mn,
                "params": combo,
                "timeframes": timeframes,
            }
            for mn in module_names
            for combo in combos
        ]
    return [
        {
            "module_name": bias_spec["module_name"],
            "params": combo,
            "timeframes": timeframes,
        }
        for combo in combos
    ]


def _normalize_timeframes(
    bias_spec: dict[str, Any],
    *,
    include_daily_ewsd: bool = True,
) -> list[TimeFrame]:
    """Return normalized cache timeframes for a bias spec."""
    raw_timeframes = bias_spec.get("timeframes", [TimeFrame.D])
    requested_timeframes = (
        raw_timeframes if isinstance(raw_timeframes, list) else [raw_timeframes]
    )
    normalized: list[TimeFrame] = []
    seen: set[TimeFrame] = set()

    def add_timeframe(timeframe_like: Any) -> None:
        timeframe = (
            TimeFrame[timeframe_like]
            if isinstance(timeframe_like, str)
            else timeframe_like
        )
        if timeframe in seen:
            return
        seen.add(timeframe)
        normalized.append(timeframe)

    for timeframe_like in requested_timeframes:
        add_timeframe(timeframe_like)
    if include_daily_ewsd:
        add_timeframe(TimeFrame.D)
    return normalized


def _cache_requirements_for_bias_spec(
    config: SupportsBiasCachePopulation,
    bias_spec: dict[str, Any] | None = None,
    *,
    include_daily_ewsd: bool = True,
) -> tuple[list[Ticker], list[Ticker], list[Ticker], list[TimeFrame]]:
    """Return primary tickers, dependency tickers, bootstrap tickers, and timeframes."""
    selected_bias_spec = config.bias_spec if bias_spec is None else bias_spec
    primary_tickers = list(config.tickers)
    dependency_tickers: list[Ticker] = []
    seen_dependency_tickers: set[Ticker] = set(primary_tickers)

    for expanded_spec in expand_bias_specs(selected_bias_spec):
        params = expanded_spec.get("params", {})
        if not isinstance(params, dict):
            continue
        for ticker_name in sorted(extract_cross_ticker_names(params)):
            if ticker_name not in Ticker.__members__:
                continue
            ticker = Ticker[ticker_name]
            if ticker in seen_dependency_tickers:
                continue
            seen_dependency_tickers.add(ticker)
            dependency_tickers.append(ticker)

    bootstrap_tickers = [*primary_tickers, *dependency_tickers]
    timeframes = _normalize_timeframes(
        first_bias_spec(selected_bias_spec),
        include_daily_ewsd=include_daily_ewsd,
    )
    return primary_tickers, dependency_tickers, bootstrap_tickers, timeframes


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
    timeframes = first_bias_spec(config.bias_spec).get("timeframes", [TimeFrame.D])
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


_INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Keep final path segments short (Windows MAX_PATH); prefer abbreviated labels before hashing.
_MAX_PARAM_COMBO_LABEL_LEN = 80

# Omitted from compact labels when constant across a typical signed-signal grid.
_LABEL_SKIP_KEYS: frozenset[str] = frozenset(
    {
        "strategy_mode",
        "strategyMode",
        "exit_policy",
        "exitPolicy",
    }
)

_PARAM_KEY_ABBREV: dict[str, str] = {
    "lookback": "lb",
    "avg_period": "ap",
    "avgPeriod": "ap",
    "short_period": "sp",
    "shortPeriod": "sp",
    "ma_period": "ma",
    "maPeriod": "ma",
    "oversold": "os",
    "overbought": "ob",
    "momentum_lookback": "ml",
    "momentumLookback": "ml",
    "spanFast": "sf",
    "spanSlow": "ss",
    "rsi_period": "rsi",
    "rsiPeriod": "rsi",
    "exit_bars": "eb",
    "exitBars": "eb",
    "max_hold_bars": "mh",
    "maxHoldBars": "mh",
    "macd_fast": "mf",
    "macdFast": "mf",
    "macd_slow": "ms",
    "macdSlow": "ms",
    "macd_signal": "mc",
    "macdSignal": "mc",
    "trend_ema_period": "te",
    "trendEmaPeriod": "te",
    "pullback_ema_period": "pe",
    "pullbackEmaPeriod": "pe",
    "atr_len": "al",
    "atrLen": "al",
    "atr_mult": "am",
    "atrMult": "am",
    "atr_pullback_mult": "ap",
    "atrPullbackMult": "ap",
    "rsi_max": "rsiMax",
    "rsiMax": "rsiMax",
    "channel_lookback": "ch",
    "channelLookback": "ch",
    "sma_period": "sma",
    "smaPeriod": "sma",
}


def _sanitize_path_label(raw: str) -> str:
    """Strip characters illegal in Windows path segments (and collapse to safe token)."""
    cleaned = _INVALID_PATH_CHARS.sub("_", raw)
    cleaned = cleaned.strip(" .")
    return cleaned if cleaned else "combo"


def _format_label_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        as_int = int(value)
        return str(as_int) if value == as_int else str(value)
    return str(value)


def _combo_label_token(key: str, value: object, *, abbreviate: bool) -> str:
    if isinstance(value, dict):
        nested = _build_combo_label_tokens(value, abbreviate=abbreviate, skip_fixed=False)
        nested_body = "_".join(nested) if nested else "empty"
        prefix = _PARAM_KEY_ABBREV.get(key, key) if abbreviate else key
        return f"{prefix}_{nested_body}"
    if isinstance(value, (list, tuple)):
        flat = "_".join(_format_label_scalar(v) for v in value)
        prefix = _PARAM_KEY_ABBREV.get(key, key) if abbreviate else key
        return f"{prefix}_{flat}"
    prefix = _PARAM_KEY_ABBREV.get(key, key) if abbreviate else key
    return f"{prefix}{_format_label_scalar(value)}"


def _build_combo_label_tokens(
    combo: Mapping[str, Any],
    *,
    abbreviate: bool,
    skip_fixed: bool,
) -> list[str]:
    tokens: list[str] = []
    for key, val in sorted(combo.items()):
        if skip_fixed and key in _LABEL_SKIP_KEYS:
            continue
        tokens.append(_combo_label_token(key, val, abbreviate=abbreviate))
    return tokens


def param_combo_display_label(combo: Mapping[str, Any]) -> str:
    """Human-readable combo label for tables and markdown (never hashed).

  Compact ``lb2_ap3_os25_ob70`` style; skips grid-fixed keys like ``exit_policy``.
    """
    if not combo:
        return "empty_params"
    tokens = _build_combo_label_tokens(combo, abbreviate=True, skip_fixed=True)
    if not tokens:
        tokens = _build_combo_label_tokens(combo, abbreviate=True, skip_fixed=False)
    return "_".join(tokens)


def expanded_combo_param_value(combo: dict[str, Any], key: str) -> Any | None:
    """Resolve *key* from an expanded param dict (top-level or composite nested blobs)."""
    if key in combo:
        return combo[key]
    for nk in NESTED_GRID_PARAM_KEYS:
        blob = combo.get(nk)
        if isinstance(blob, dict) and key in blob:
            return blob[key]
    return None


def param_combo_label(combo: dict[str, Any]) -> str:
    """Return a filesystem-safe folder name for a param combo dict.

    Nested dicts (e.g. ``filter_params`` / ``signal_params`` on ``filter_gate``) are
    flattened recursively. Scalar values are never passed through ``repr()`` of a dict
    (which would inject ``:`` and quotes and break Windows ``mkdir``).

    If the flattened label would exceed ``_MAX_PARAM_COMBO_LABEL_LEN`` characters, it is
    replaced by ``combo_<sha256>`` so paths stay within Windows ``MAX_PATH`` limits.

    Parameters
    ----------
    combo : dict[str, Any]
        Parameter combination mapping.

    Returns
    -------
    str
        Folder-safe label, e.g. ``lookback_10__other_5`` (sorted by key).

    Examples
    --------
    >>> param_combo_label({"lookback": 5})
    'lookback_5'
    >>> param_combo_label({"lookback": 20, "atr_length": 14})
    'atr_length_14__lookback_20'
    """
    if not combo:
        return "empty_params"
    for abbreviate, skip_fixed in ((True, True), (True, False), (False, False)):
        tokens = _build_combo_label_tokens(
            combo, abbreviate=abbreviate, skip_fixed=skip_fixed
        )
        if not tokens:
            continue
        joined = "__".join(tokens)
        sanitized = _sanitize_path_label(joined)
        if len(sanitized) <= _MAX_PARAM_COMBO_LABEL_LEN:
            return sanitized
    joined = json.dumps(dict(combo), sort_keys=True, default=str)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]
    return _sanitize_path_label(f"combo_{digest}")


def expanded_spec_combo_label(single_spec: Mapping[str, Any]) -> str:
    """Unique label for one expanded bias row (composite ``module_name`` + params).

    Params alone can match across gate variants (e.g. ``filter_gate`` vs
    ``filter_gate_entry_only``); prefixing ``module_name`` keeps paths and visualization keys distinct.
    """
    mod = single_spec.get("module_name")
    mod_token = str(mod) if mod is not None else "unknown"
    params = single_spec.get("params", {})
    pdict = params if isinstance(params, dict) else {}
    body = param_combo_label(pdict)
    return f"{mod_token}__{body}" if body else mod_token


def permutation_combo_display_name(params: dict[str, Any] | None) -> str:
    """Human-readable label for permutation / validation tables."""
    from research.feature.filter_research_labels import permutation_combo_display_name as _display

    return _display(params)


@dataclass(frozen=True)
class CachePopulationWindow:
    """Resolved date bounds for central-cache bootstrap and bias artifact coverage."""

    bootstrap_start: datetime
    bootstrap_end: datetime
    bias_coverage_start: datetime | None
    bias_coverage_end: datetime
    used_config_fallback: bool


def _iter_numeric_param_values(params: object) -> list[int]:
    """Collect positive integer-like values from nested bias param dicts."""

    if isinstance(params, dict):
        return [
            value
            for nested in params.values()
            for value in _iter_numeric_param_values(nested)
        ]
    if isinstance(params, list):
        return [
            value
            for item in params
            for value in _iter_numeric_param_values(item)
        ]
    if isinstance(params, int) and params > 0:
        return [params]
    if isinstance(params, float) and params > 0 and float(params).is_integer():
        return [int(params)]
    return []


def estimate_bias_lookback_buffer_days(
    bias_spec: dict[str, Any] | list[dict[str, Any]],
    *,
    minimum_buffer_days: int = 504,
) -> int:
    """Conservative calendar-day buffer before ``config.start`` for indicator warmup."""

    expanded = expand_bias_specs(bias_spec)
    max_period = max(
        (
            value
            for spec in expanded
            for value in _iter_numeric_param_values(spec.get("params", {}))
        ),
        default=0,
    )
    return max(minimum_buffer_days, max_period * 2 + 1)


def resolve_cache_population_window(
    config: SupportsBiasCachePopulation,
    *,
    ranges: Mapping[Ticker, tuple[datetime, datetime]],
    bootstrap_tickers: list[Ticker],
    cache_population_mode: CachePopulationMode = CachePopulationMode.FULL_HISTORY,
    lookback_buffer_days: int | None = None,
) -> CachePopulationWindow:
    """Resolve OHLC bootstrap and bias-artifact bounds for cache population.

    Bootstrap candles use the intersection of all bootstrap tickers' OHLC spans.
    In ``FULL_HISTORY`` mode, bias/EWSD coverage uses ``bias_coverage_start=None``
    (earliest candle coverage) through ``bias_coverage_end``.

    In ``ANALYSIS_PLUS_LOOKBACK`` mode, bias coverage starts at
    ``config.start - lookback_buffer`` (clamped to available OHLC).

    When OHLC discovery is incomplete, falls back to ``config.start`` / ``config.end``.
    """
    req_start = pd.Timestamp(config.start).to_pydatetime()
    req_end = pd.Timestamp(config.end).to_pydatetime()
    buffer_days = (
        lookback_buffer_days
        if lookback_buffer_days is not None
        else estimate_bias_lookback_buffer_days(config.bias_spec)
    )
    analysis_bias_start = (pd.Timestamp(req_start) - pd.Timedelta(days=buffer_days)).to_pydatetime()

    if all(ticker in ranges for ticker in bootstrap_tickers):
        common_start = max(ranges[ticker][0] for ticker in bootstrap_tickers)
        common_end = min(ranges[ticker][1] for ticker in bootstrap_tickers)
        overlap_start = max(common_start, req_start)
        overlap_end = min(common_end, req_end)
        if overlap_start > overlap_end:
            raise ValueError(
                "Config analysis window does not overlap available OHLC common range: "
                f"analysis [{req_start.date()} .. {req_end.date()}], "
                f"common [{common_start.date()} .. {common_end.date()}]."
            )
        bias_start = (
            max(common_start, analysis_bias_start)
            if cache_population_mode is CachePopulationMode.ANALYSIS_PLUS_LOOKBACK
            else None
        )
        return CachePopulationWindow(
            bootstrap_start=common_start,
            bootstrap_end=common_end,
            bias_coverage_start=bias_start,
            bias_coverage_end=common_end,
            used_config_fallback=False,
        )

    bias_start = (
        analysis_bias_start
        if cache_population_mode is CachePopulationMode.ANALYSIS_PLUS_LOOKBACK
        else req_start
    )
    return CachePopulationWindow(
        bootstrap_start=req_start,
        bootstrap_end=req_end,
        bias_coverage_start=bias_start,
        bias_coverage_end=req_end,
        used_config_fallback=True,
    )


def _cache_population_log_message(
    window: CachePopulationWindow,
    *,
    analysis_start: datetime,
    analysis_end: datetime,
    cache_population_mode: CachePopulationMode,
) -> str:
    if window.used_config_fallback:
        return (
            "[data_loader] Could not discover common OHLC date range; "
            f"populating cache from config: {window.bootstrap_start.date()} -> "
            f"{window.bootstrap_end.date()}"
        )
    mode_label = (
        "analysis+lookback"
        if cache_population_mode is CachePopulationMode.ANALYSIS_PLUS_LOOKBACK
        else "full common OHLC history"
    )
    bias_start = (
        window.bias_coverage_start.date()
        if window.bias_coverage_start is not None
        else "inception"
    )
    return (
        f"[data_loader] Populating cache ({mode_label}): "
        f"bootstrap {window.bootstrap_start.date()} -> {window.bootstrap_end.date()}, "
        f"bias {bias_start} -> {window.bias_coverage_end.date()} "
        f"(analysis window: {analysis_start.date()} -> {analysis_end.date()})"
    )


def populate_cache_if_needed(
    config: SupportsBiasCachePopulation,
    *,
    bias_spec: dict[str, Any] | None = None,
    artifact_scope: ArtifactScope = ArtifactScope.LIVE,
    bias_cache_max_workers: int = 6,
    cache_population_mode: CachePopulationMode = CachePopulationMode.FULL_HISTORY,
    lookback_buffer_days: int | None = None,
) -> None:
    """Bootstrap OHLC into the central cache, then ensure bias (+EWSD) artifact coverage.

    Feature-research pipelines call this before cache-backed extraction so reads
    hit ``CentralCacheStore`` with missing/stale artifacts refreshed first.

    The helper bootstraps the required source candles into the runtime cache,
    then refreshes bias artifacts from that cache. Population uses the **full
    common OHLC history** across required tickers and timeframes (not
    ``config.start`` / ``config.end``), so indicator warmup happens once at data
    inception. Analysis pipelines should narrow ``config.start`` / ``config.end``
    (e.g. to ``training_window_bounds``) only *after* calling this helper.

    Safe to call even if cache already exists. Missing, stale, and out-of-range
    artifacts are refreshed; fresh artifacts are left untouched.

    Parameters
    ----------
    config : SupportsBiasCachePopulation
        Any config with ``tickers``, ``bias_spec``, ``start``, ``end``
        (e.g. ``ResearchConfig`` or ``BinningResearchConfig``).
    artifact_scope :
        Namespace for bias-node parquet (default live). Use ``RESEARCH`` for
        binning-phase runs that should not write under ``artifacts/live``.
    bias_cache_max_workers :
        Thread count for parallel bias-node **rebuilds** in
        ``CacheManager.ensure_bias_cache_coverage`` (default ``6``). Set to ``1`` for
        fully sequential rebuilds.
    """
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
    selected_bias_spec = config.bias_spec if bias_spec is None else bias_spec
    expanded = expand_bias_specs(selected_bias_spec)
    (
        primary_tickers,
        _dependency_tickers,
        bootstrap_tickers,
        timeframes,
    ) = _cache_requirements_for_bias_spec(config, selected_bias_spec)

    ranges = manager.get_available_date_range_per_ticker(
        tickers=bootstrap_tickers,
        timeframes=timeframes,
    )
    analysis_start = pd.Timestamp(config.start).to_pydatetime()
    analysis_end = pd.Timestamp(config.end).to_pydatetime()
    effective_mode = cache_population_mode
    if hasattr(config, "cache_population_mode"):
        effective_mode = getattr(config, "cache_population_mode", cache_population_mode)
    window = resolve_cache_population_window(
        config,
        ranges=ranges,
        bootstrap_tickers=bootstrap_tickers,
        cache_population_mode=effective_mode,
        lookback_buffer_days=lookback_buffer_days,
    )
    print(
        _cache_population_log_message(
            window,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            cache_population_mode=effective_mode,
        )
    )

    bootstrap_summary = manager.bootstrap_source_candles(
        tickers=bootstrap_tickers,
        timeframes=timeframes,
        start_date=window.bootstrap_start,
        end_date=window.bootstrap_end,
    )
    if bootstrap_summary["failed"] > 0:
        raise ValueError(f"Failed to bootstrap candle cache coverage: {bootstrap_summary}")

    summary = manager.ensure_bias_cache_coverage(
        bias_node_specs=expanded,
        tickers=primary_tickers,
        start_date=window.bias_coverage_start,
        end_date=window.bias_coverage_end,
        refresh_mode="missing_stale_only",
        include_daily_ewsd=True,
        artifact_scope=artifact_scope,
        max_workers=bias_cache_max_workers,
    )
    if summary["failed"] > 0:
        raise ValueError(f"Failed to ensure feature cache coverage: {summary}")
    print(_cache_coverage_console_message(summary))


def get_tickers_with_coverage_for_config(
    config: "ResearchConfig",
    *,
    bias_spec: dict[str, Any] | None = None,
) -> list[Ticker]:
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
    (
        primary_tickers,
        dependency_tickers,
        bootstrap_tickers,
        timeframes,
    ) = _cache_requirements_for_bias_spec(config, bias_spec)
    ranges = manager.get_available_date_range_per_ticker(
        tickers=bootstrap_tickers,
        timeframes=timeframes,
    )
    start_ts = pd.Timestamp(config.start)
    end_ts = pd.Timestamp(config.end)
    dependencies_have_full_coverage = all(
        dep_ticker in ranges
        and ranges[dep_ticker][0] <= start_ts.to_pydatetime()
        and ranges[dep_ticker][1] >= end_ts.to_pydatetime()
        for dep_ticker in dependency_tickers
    )
    if not dependencies_have_full_coverage:
        return []
    covered = [
        t
        for t in primary_tickers
        if t in ranges
        and ranges[t][0] <= start_ts.to_pydatetime()
        and ranges[t][1] >= end_ts.to_pydatetime()
    ]
    return covered


def get_available_date_ranges_for_tickers(
    config: "ResearchConfig",
    tickers: list[Ticker],
    *,
    bias_spec: dict[str, Any] | None = None,
) -> dict[Ticker, tuple[pd.Timestamp, pd.Timestamp]]:
    """Return per-ticker (min_date, max_date) from OHLC data for error messages.

    Used when no tickers have full coverage so we can report what range each
    ticker actually has (e.g. suggest narrowing config.start/end or oos_window).
    """
    project_root = _resolve_project_root()
    if project_root is None or not (project_root / "data" / "ohlc_data").exists():
        return {}
    manager = CacheManager(candle_dir=str(project_root / "data" / "ohlc_data"))
    (
        _primary_tickers,
        dependency_tickers,
        _bootstrap_tickers,
        timeframes,
    ) = _cache_requirements_for_bias_spec(config, bias_spec)
    requested_tickers = list(
        dict.fromkeys([*tickers, *dependency_tickers])
    )
    ranges = manager.get_available_date_range_per_ticker(
        tickers=requested_tickers,
        timeframes=timeframes,
    )
    return {
        t: (pd.Timestamp(ranges[t][0]), pd.Timestamp(ranges[t][1]))
        for t in requested_tickers
        if t in ranges
    }


def get_effective_range_and_tickers(
    config: "ResearchConfig",
    *,
    bias_spec: dict[str, Any] | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp, list[Ticker]] | None:
    """Return effective date range and tickers when no ticker has full coverage.

    Used as fallback so pipelines can run with whatever OHLC data is available
    instead of raising. Intersection of all tickers' ranges (clipped to config)
    is used when non-empty; otherwise the single ticker with largest overlap.

    Returns
    -------
    tuple of (effective_start, effective_end, tickers) or None
        None if no OHLC data exists (e.g. no data/ohlc_data or empty ranges).
    """
    primary_tickers, dependency_tickers, _bootstrap_tickers, _timeframes = (
        _cache_requirements_for_bias_spec(config, bias_spec)
    )
    ranges = get_available_date_ranges_for_tickers(
        config,
        primary_tickers,
        bias_spec=bias_spec,
    )
    if not ranges:
        return None
    start_ts = pd.Timestamp(config.start)
    end_ts = pd.Timestamp(config.end)
    # Intersection clipped to requested range: all tickers have data in [s, e]
    effective_start = max(start_ts, max(r[0] for r in ranges.values()))
    effective_end = min(end_ts, min(r[1] for r in ranges.values()))
    if effective_start < effective_end:
        covering = [
            t
            for t in config.tickers
            if t in ranges
            and ranges[t][0] <= effective_start
            and ranges[t][1] >= effective_end
        ]
        if covering:
            return (effective_start, effective_end, covering)
    # Fallback: single ticker with largest overlap with [config.start, config.end]
    def overlap_seconds(ticker: Ticker) -> float:
        if ticker not in ranges:
            return 0.0
        interval_starts = [ranges[ticker][0]]
        interval_ends = [ranges[ticker][1]]
        for dependency_ticker in dependency_tickers:
            if dependency_ticker not in ranges:
                return 0.0
            interval_starts.append(ranges[dependency_ticker][0])
            interval_ends.append(ranges[dependency_ticker][1])
        low = max(start_ts, *interval_starts)
        high = min(end_ts, *interval_ends)
        return (pd.Timestamp(high) - pd.Timestamp(low)).total_seconds()

    best = max(
        (t for t in primary_tickers),
        key=overlap_seconds,
        default=None,
    )
    if best is None or overlap_seconds(best) <= 0:
        return None
    interval_starts = [ranges[best][0]]
    interval_ends = [ranges[best][1]]
    for dependency_ticker in dependency_tickers:
        if dependency_ticker not in ranges:
            return None
        interval_starts.append(ranges[dependency_ticker][0])
        interval_ends.append(ranges[dependency_ticker][1])
    eff_start = max(start_ts, *interval_starts)
    eff_end = min(end_ts, *interval_ends)
    return (eff_start, eff_end, [best])


def load_features_for_combo(
    single_combo_spec: dict[str, Any],
    config: "ResearchConfig",
    candles_override: pd.DataFrame | None = None,
    *,
    populate_on_miss: bool = False,
) -> tuple[pd.Series, pd.Series, str, pd.Series] | None:
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
    populate_on_miss : bool, default=False
        When ``True`` and ``use_cache=True``, stream and persist bias artifacts for
        parameter combos not already in the central cache (e.g. min-step perturbation
        neighbours off the exploration grid).

    Returns
    -------
    (feature, target, feature_col, ticker) or None
        Aligned (feature, target) Series, bias-node column name, and per-row ``ticker``.
        For ``feature_type=CONTINUOUS``, ``feature`` values are **quantile-binned** ±1/0
        (permutation-equivalent), not raw continuous levels.

    Raises
    ------
    ValueError
        If target data is missing or misaligned.
    """
    features_df, targets_df = extract_features_for_bias_node(
        bias_spec=bias_spec_for_node_instantiation(single_combo_spec),
        ticker=config.tickers,
        start=config.start,
        end=config.end,
        target_col=config.target_col,
        use_cache=True,
        populate_on_miss=populate_on_miss,
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

    # concat (not DataFrame(dict)) so duplicate (datetime,) rows for multi-ticker do not
    # trigger pandas homogenize/reindex, which fails on duplicate index labels.
    # Include ticker in the same concat — never ``.reindex(aligned.index)`` when the index
    # has duplicates (pandas raises ValueError).
    feat_s = features_df[feature_col].rename("feature")
    targ_s = targets_df[target_col_name].rename("target")
    if "ticker" in features_df.columns:
        tick_part = features_df["ticker"].astype(str).rename("ticker")
    else:
        first_ticker = config.tickers[0]
        ticker_label = first_ticker.name if hasattr(first_ticker, "name") else str(first_ticker)
        tick_part = pd.Series(ticker_label, index=feat_s.index, dtype=str).rename("ticker")
    aligned = pd.concat([feat_s, targ_s, tick_part], axis=1).dropna(how="any")

    if aligned.empty:
        return None

    if config.feature_type == FeatureType.CONTINUOUS:
        if target_col_name in RAW_TARGET_COLS:
            unique_tickers = (
                int(features_df["ticker"].nunique())
                if "ticker" in features_df.columns
                else len(config.tickers)
            )
            if unique_tickers > 1:
                raise ValueError(
                    f"load_features_for_combo received target_col='{target_col_name}' with "
                    f"{unique_tickers} tickers. Raw return targets must not be mixed "
                    "across tickers. Use 'log_return_ewsd'."
                )

    target_series = aligned["target"].copy()
    target_series.name = target_col_name
    ticker_series = aligned["ticker"].copy()
    ticker_series.name = "ticker"
    feature_series = aligned["feature"].copy()
    feature_series.name = feature_col

    return feature_series, target_series, feature_col, ticker_series
