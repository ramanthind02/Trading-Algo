"""Data loading and cache management helpers for the continuous binning research pipeline."""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from feature_research.in_sample.continuous_binning.config import ResearchConfig

from feature_extraction.feature_extractor import extract_features_for_bias_node
from utils.cache.cache_manager import CacheManager
from utils.core.enums import TimeFrame
from utils.core.helpers import load_data_multi_ticker

_UNNORMALIZED_RETURN_COLS: frozenset[str] = frozenset({"log_return", "raw_return"})


def expand_bias_specs(bias_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a bias_spec with list-valued params into one spec per param combo."""
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
    """Load candles for the config universe and normalize to DatetimeIndex."""
    timeframes = config.bias_spec.get("timeframes", [TimeFrame.D])
    raw_timeframe = timeframes[0] if isinstance(timeframes, list) else timeframes
    timeframe = TimeFrame[raw_timeframe] if isinstance(raw_timeframe, str) else raw_timeframe

    candles_df = load_data_multi_ticker(
        tickers=config.tickers,
        timeframe=timeframe,
        start=config.start,
        end=config.end,
        use_millisecond_offset=True,
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

    Safe to call even if cache already exists — ``overwrite_existing=False``
    means only missing entries are computed.
    """
    if not config.populate_cache:
        return

    project_root = next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "pyproject.toml").exists()
        ),
        Path(__file__).resolve().parents[3],
    )
    candle_dir = project_root / "data" / "ohlc_data"
    if not candle_dir.exists():
        print(
            f"[data_loader] WARNING: candle directory not found at {candle_dir}. "
            "Skipping cache population."
        )
        return

    manager = CacheManager(candle_dir=str(candle_dir))
    expanded = expand_bias_specs(config.bias_spec)
    summary = manager.populate_cache(
        bias_node_specs=expanded,
        tickers=config.tickers,
        start_date=config.start,
        end_date=config.end,
        show_progress=True,
        overwrite_existing=False,
    )
    print(f"[data_loader] Cache populated: {summary}")


def load_features_for_combo(
    single_combo_spec: dict[str, Any],
    config: "ResearchConfig",
    candles_override: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series, str] | None:
    """Extract feature + target Series for a single param combo across all config tickers.

    Returns
    -------
    (feature, target, feature_col) or None if extraction fails / returns empty data.

    Parameters
    ----------
    candles_override : pd.DataFrame | None, default=None
        Optional override candles passed through to
        ``extract_features_for_bias_node`` for forward-return computation.
        Expected schema: one row per ticker/timestamp with columns
        ``datetime``, ``open``, ``high``, ``low``, ``close``, ``ticker``.
        ``datetime`` values may be tz-naive or tz-aware; they are normalized
        to UTC by the downstream extractor.

    The returned Series are aligned (same index, NaNs dropped) and concatenated
    across all tickers in ``config.tickers``.
    """
    features_df, targets_df = extract_features_for_bias_node(
        bias_spec=single_combo_spec,
        ticker=config.tickers,
        start=config.start,
        end=config.end,
        use_millisecond_offset=True,
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
