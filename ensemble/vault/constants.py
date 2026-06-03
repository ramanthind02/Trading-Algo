"""Shared vault layout constants."""

from __future__ import annotations

from typing import Final

# Top-level folders under vault/<D|W|M>/ that contain ensemble leaf directories
# (manual weight-hierarchy grouping). Must match ``weight_hierarchy_group`` JSON values.
VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES: frozenset[str] = frozenset(
    {
        "mean_reversion_indices",
        "buy_hold",
        "es_tlt",
        "seasonal",
        "momentum",
        "trend_following",
        "momentum_gc",
        "crude_oil_mr",
        "gc_breakout",
        "cl_breakout",
        "breakout",
        "silver_mr",
        "silver_trend",
    }
)

# Asset-class labels for asset-first ``hierarchy_equal`` specs (see docs/SaaS/weight_layer_spec.md).
TICKER_ASSET_CLASS: Final[dict[str, str]] = {
    "ES": "equity_indices",
    "NQ": "equity_indices",
    "RTY": "equity_indices",
    "YM": "equity_indices",
    "DAX": "equity_indices",
    "GC": "commodities",
    "SI": "commodities",
    "CL": "commodities",
    "NG": "commodities",
    "ZB": "fixed_income",
    "ZN": "fixed_income",
    "TLT": "fixed_income",
    "EUR": "fx",
    "JPY": "fx",
    "GBP": "fx",
    "EU": "fx",
    "JY": "fx",
    "BP": "fx",
    "CD": "fx",
    "SF": "fx",
}

# Strategy groups that always land in a fixed top-level asset bucket (not ticker-derived).
# ``buy_hold`` and ``es_tlt`` are omitted: each stream uses ``TICKER_ASSET_CLASS``
# (ES/NQ → equity_indices, GC → commodities, TLT → fixed_income). ES/TLT rebalancing
# flow trades ES only → ``equity_indices/es_tlt`` (TLT is a cross-ticker peer, not a stream).
STRATEGY_GROUP_ASSET_OVERRIDE: Final[dict[str, str]] = {}

ASSET_CLASS_ORDER: Final[tuple[str, ...]] = (
    "equity_indices",
    "commodities",
    "fixed_income",
    "fx",
    "diversified",
)

__all__ = [
    "ASSET_CLASS_ORDER",
    "STRATEGY_GROUP_ASSET_OVERRIDE",
    "TICKER_ASSET_CLASS",
    "VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES",
]
