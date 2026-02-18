"""Data loading and cache management helpers for the rule-based EDA research pipeline."""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from feature_research.rule_based.config import RuleBasedResearchConfig

from feature_extraction.feature_extractor import extract_features_for_bias_node
from utils.cache_manager import CacheManager
from utils.enums import TimeFrame


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


def param_combo_label(combo: dict[str, Any]) -> str:
    """Return a human-readable folder name for a param combo dict.

    Examples
    --------
    >>> param_combo_label({"rsi_period": 2})
    'rsi_period_2'
    >>> param_combo_label({"rsi_period": 2, "oversold": 25.0})
    'oversold_25.0__rsi_period_2'
    """
    parts = [f"{k}_{v}" for k, v in sorted(combo.items())]
    return "__".join(parts)


def populate_cache_if_needed(config: "RuleBasedResearchConfig") -> None:
    """Populate the feature cache if config.populate_cache is True.

    Safe to call even if cache already exists — ``overwrite_existing=False``
    means only missing entries are computed.
    """
    if not config.populate_cache:
        return

    project_root = Path(__file__).resolve().parents[2]
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
    config: "RuleBasedResearchConfig",
) -> tuple[pd.Series, pd.Series, str] | None:
    """Extract feature + target Series for a single param combo across all config tickers.

    Returns
    -------
    (feature, target, feature_col) or None if extraction fails / returns empty data.

    The returned Series are aligned (same index, NaNs dropped) and concatenated
    across all tickers in ``config.tickers``.
    """
    try:
        features_df, targets_df = extract_features_for_bias_node(
            bias_spec=single_combo_spec,
            ticker=config.tickers,
            start=config.start,
            end=config.end,
            use_millisecond_offset=True,
            target_col=config.target_col,
            use_cache=config.use_cache,
        )
    except Exception as exc:
        print(f"[data_loader] Feature extraction failed for {single_combo_spec['params']}: {exc}")
        return None

    if features_df is None or features_df.empty:
        print(f"[data_loader] Empty features for {single_combo_spec['params']}. Skipping.")
        return None

    feature_cols = [c for c in features_df.columns if c != "ticker"]
    if not feature_cols:
        return None
    feature_col = feature_cols[0]

    target_col_name = (
        config.target_col
        if config.target_col in targets_df.columns
        else [c for c in targets_df.columns if c != "ticker"][0]
    )

    aligned = pd.DataFrame(
        {"feature": features_df[feature_col], "target": targets_df[target_col_name]}
    ).dropna()

    if aligned.empty:
        return None

    return aligned["feature"], aligned["target"], feature_col
