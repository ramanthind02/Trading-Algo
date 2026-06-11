"""Shared vault layout constants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

# Built-in top-level folders under vault/<D|W|M>/ that contain ensemble leaf directories
# (manual weight-hierarchy grouping). Must match ``weight_hierarchy_group`` JSON values.
_BASE_WEIGHT_HIERARCHY_GROUP_DIR_NAMES: Final[frozenset[str]] = frozenset(
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

# User-defined sleeves live in a persisted JSON list (additive to the built-ins) so new sleeves
# can be created from the UI without a code change. The built-ins are never removed.
_CUSTOM_SLEEVES_PATH: Final[Path] = Path(__file__).resolve().parent / "custom_sleeves.json"


def _load_custom_sleeves() -> frozenset[str]:
    if not _CUSTOM_SLEEVES_PATH.exists():
        return frozenset()
    try:
        data = json.loads(_CUSTOM_SLEEVES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    return frozenset(str(name).strip() for name in data if str(name).strip())


def valid_weight_hierarchy_groups() -> frozenset[str]:
    """Built-in sleeves unioned with any user-defined ones (read fresh — runtime-additive)."""

    return _BASE_WEIGHT_HIERARCHY_GROUP_DIR_NAMES | _load_custom_sleeves()


def add_custom_sleeve(name: str) -> frozenset[str]:
    """Persist a new user-defined sleeve name (idempotent). Returns the full valid set."""

    clean = str(name).strip()
    if not clean:
        raise ValueError("Sleeve name must be a non-empty string.")
    if not all(ch.isalnum() or ch in "_-" for ch in clean):
        raise ValueError("Sleeve name may only contain letters, digits, '_' and '-'.")
    if clean in _BASE_WEIGHT_HIERARCHY_GROUP_DIR_NAMES:
        return valid_weight_hierarchy_groups()
    existing = set(_load_custom_sleeves())
    existing.add(clean)
    _CUSTOM_SLEEVES_PATH.write_text(json.dumps(sorted(existing), indent=2) + "\n", encoding="utf-8")
    return valid_weight_hierarchy_groups()


# Snapshot at import (base ∪ custom). Prefer ``valid_weight_hierarchy_groups()`` where runtime
# additions must be seen without a restart (e.g. spec validation, the form schema).
VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES: frozenset[str] = valid_weight_hierarchy_groups()

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
    "AUDNZD": "fx",
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
    "valid_weight_hierarchy_groups",
    "add_custom_sleeve",
]
