"""
Composable signal filter framework for bias nodes.

Provides a decorator pattern to wrap any BiasNode with one or more stateful
filters.  When a filter blocks, the wrapped node's output is replaced by a
configurable neutral value (default ``0.0``).

Key types
---------
SignalFilter      – ABC for stateful per-candle filter logic.
FilterSpec        – Immutable configuration describing a filter and its params.
FilteredBiasNode  – BiasNode decorator (lives in ``nodes.filtered``; re-exported here).
create_filter     – Factory that turns a FilterSpec into a SignalFilter instance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Type

from utils.core.models import Candle
import utils.core.helpers as _helpers


# ---------------------------------------------------------------------------
# FilterSpec – immutable config (mirrors BiasNodeSpec)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterSpec:
    """Immutable specification for a signal filter.

    Parameters
    ----------
    filter_name : str
        Registry key used by :func:`create_filter` (e.g. ``'vol'``).
    params : dict[str, object]
        Keyword arguments forwarded to the filter constructor.
    """
    filter_name: str
    params: dict[str, object]


# ---------------------------------------------------------------------------
# SignalFilter ABC
# ---------------------------------------------------------------------------

class SignalFilter(ABC):
    """Stateful per-candle filter that gates bias-node signals.

    Subclasses must implement :meth:`update`, :meth:`should_pass`, and
    :meth:`reset`.  The ``filter_name``, ``params``, and ``warmup``
    attributes are used for column naming, hashing, and warmup tracking.
    """

    filter_name: str
    """Short identifier for column naming (e.g. ``'vol'``, ``'regime'``)."""

    params: Dict[str, Any]
    """Filter parameters (used in column suffix and singleton key)."""

    warmup: int
    """Number of candles required before :meth:`should_pass` is meaningful."""

    # ---- abstract interface ------------------------------------------------

    @abstractmethod
    def update(self, candle: Candle) -> None:
        """Ingest *candle* to maintain internal state.

        Called once per candle **before** :meth:`should_pass`.
        """

    @abstractmethod
    def should_pass(self) -> bool:
        """Return ``True`` if the current candle's signal should pass."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for reuse across back-tests."""

    # ---- column-naming helper ---------------------------------------------

    def get_filter_suffix(self) -> str:
        """Build the ``__f_<name>_<params>`` column-name suffix.

        Parameter keys are converted to camelCase to stay consistent with
        :func:`utils.core.helpers.build_feature_column_name`.
        """
        parts: list[str] = [self.filter_name]
        for key in sorted(self.params.keys()):
            parts.append(_helpers._to_camel_case(str(key)))
            parts.append(str(self.params[key]))
        return "__f_" + "_".join(parts)


# ---------------------------------------------------------------------------
# Filter registry + factory
# ---------------------------------------------------------------------------

_FILTER_REGISTRY: Dict[str, Type[SignalFilter]] = {}


def register_filter(cls: Type[SignalFilter]) -> Type[SignalFilter]:
    """Class decorator that registers a SignalFilter subclass."""
    _FILTER_REGISTRY[cls.filter_name] = cls
    return cls


def create_filter(spec: FilterSpec) -> SignalFilter:
    """Instantiate a SignalFilter from a FilterSpec.

    Raises
    ------
    KeyError
        If *spec.filter_name* is not registered.
    """
    if spec.filter_name not in _FILTER_REGISTRY:
        raise KeyError(
            f"Unknown filter '{spec.filter_name}'. "
            f"Registered: {sorted(_FILTER_REGISTRY)}"
        )
    return _FILTER_REGISTRY[spec.filter_name](**spec.params)


# ---------------------------------------------------------------------------
# Re-export FilteredBiasNode for convenience (lazy to avoid circular import)
# ---------------------------------------------------------------------------

def __getattr__(name: str):  # noqa: N807
    if name == "FilteredBiasNode":
        from nodes.filtered import FilteredBiasNode
        return FilteredBiasNode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FilterSpec",
    "SignalFilter",
    "FilteredBiasNode",
    "create_filter",
    "register_filter",
]
