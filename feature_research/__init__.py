"""Canonical public surface for `feature_research`.

Prefer the phase packages:

- `feature_research.exploration`
- `feature_research.validation`
- `feature_research.portfolio_addition`

Use `feature_research.pipeline` only as a legacy compatibility facade.
"""
from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from .shared import FeatureResearchPhase

if TYPE_CHECKING:
    from . import exploration, portfolio_addition, validation

_PHASE_MODULES = frozenset({"exploration", "validation", "portfolio_addition"})
_LEGACY_MODULE_ALIASES = {
    "bias_spec_catalog": "feature_research._internal.bias_spec_catalog",
    "bootstrap": "feature_research._internal.bootstrap",
    "core_helpers": "feature_research._internal.core_helpers",
    "permutation_script_support": "feature_research._internal.permutation_script_support",
}


class _LazyAliasModule(ModuleType):
    """Lazy compatibility alias for moved internal modules."""

    def __init__(self, alias_name: str, target_name: str) -> None:
        super().__init__(f"{__name__}.{alias_name}")
        self._alias_name = alias_name
        self._target_name = target_name

    def _load(self) -> ModuleType:
        name = self.__name__
        cached = sys.modules.get(name)
        if cached is not self:
            return cached  # type: ignore[return-value]

        target_name = self._target_name
        alias_name = self._alias_name
        module = import_module(target_name)
        sys.modules[name] = module
        globals()[alias_name] = module
        return module

    def __getattr__(self, item: str) -> object:
        return getattr(self._load(), item)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(dir(self._load())))


def _register_legacy_module_aliases() -> None:
    for alias_name, target_name in _LEGACY_MODULE_ALIASES.items():
        fullname = f"{__name__}.{alias_name}"
        if fullname not in sys.modules:
            sys.modules[fullname] = _LazyAliasModule(alias_name, target_name)


_register_legacy_module_aliases()


def __getattr__(name: str) -> ModuleType:
    """Lazily expose the canonical phase packages at the root."""

    if name in _PHASE_MODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FeatureResearchPhase",
    "exploration",
    "portfolio_addition",
    "validation",
]
