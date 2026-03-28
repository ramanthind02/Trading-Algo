"""Compatibility shim for portfolio materialization helpers."""

from .runtime.portfolio_materialization import (
    BaseModelMaterializationIdentity,
    CleanupSummary,
    MaterializationSummary,
    materialize_global_portfolio_predictions,
    prune_inactive_base_model_materializations,
)

__all__ = [
    "BaseModelMaterializationIdentity",
    "CleanupSummary",
    "MaterializationSummary",
    "materialize_global_portfolio_predictions",
    "prune_inactive_base_model_materializations",
]
