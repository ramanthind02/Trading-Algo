"""Shared vault layout constants."""

from __future__ import annotations

# Top-level folders under vault/<D|W|M>/ that contain ensemble leaf directories
# (manual weight-hierarchy grouping). Must match ``weight_hierarchy_group`` JSON values.
VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES: frozenset[str] = frozenset(
    {
        "mean_reversion_indices",
        "buy_hold",
        "es_tlt",
        "seasonal",
        "momentum",
        "momentum_gc",
    }
)
