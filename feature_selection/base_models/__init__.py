"""Base model shims for the frozen signed-signal contract."""

from __future__ import annotations

import importlib

from feature_selection.base_models.feature_base_model import BaseModel

__all__ = ["BaseModel"]


def __getattr__(name: str) -> object:
    legacy = frozenset({"ContinuousBinningModel", "RuleBasedModel", "BinningModelBase"})
    if name in legacy:
        module = importlib.import_module(f"{__name__}.base_model")
        return getattr(module, name)
    raise AttributeError(name)
