"""FilteredBiasNode – decorator that wraps a BiasNode with signal filters.

Lives alongside other node types in the ``nodes/`` package so that it is
discoverable as a first-class node variant.

Usage
-----
>>> from nodes.filtered import FilteredBiasNode
>>> from filters import FilterSpec, create_filter
>>> from filters.volatility import VolatilityFilter
>>>
>>> vol_filter = VolatilityFilter(atr_period=14, regime='high')
>>> wrapped = FilteredBiasNode(base_node, [vol_filter])
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Tuple, ClassVar

from nodes import BiasNode, LookbackContribution
from filters import SignalFilter
from utils.core.models import Candle


class FilteredBiasNode(BiasNode):
    """Decorator that wraps a :class:`BiasNode` with a chain of filters.

    When **all** filters pass, the wrapped node's signal propagates unchanged.
    When **any** filter blocks, every output element is replaced by
    *neutral_value* (default ``0.0``).

    Parameters
    ----------
    wrapped : BiasNode
        The underlying signal-producing node.
    filters : Sequence[SignalFilter]
        One or more filters to apply (AND logic).
    neutral_value : float, optional
        Value emitted when a filter blocks.  ``0.0`` is safe for rule-based
        features (means "flat") and continuous features (centre of most
        symmetric distributions).  Set to ``float('nan')`` for explicit
        exclusion semantics if desired.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        wrapped: BiasNode,
        filters: Sequence[SignalFilter],
        *,
        neutral_value: float = 0.0,
    ) -> None:
        super().__init__(wrapped.ticker, wrapped.tf)
        self.wrapped = wrapped
        self.filters: Tuple[SignalFilter, ...] = tuple(filters)
        self.neutral_value = neutral_value

        # Inherit metadata from wrapped node
        self.module_name = wrapped.module_name
        self.output_features = list(wrapped.output_features)
        self.params = self._build_combined_params()

        # Warmup = max of wrapped node warmup and all filter warmups
        wrapped_front_bad = getattr(wrapped, 'front_bad', 0)
        self.front_bad = max(
            wrapped_front_bad,
            *(f.warmup for f in self.filters),
        )

        self._n_candles: int = 0

        # Column naming
        self.ensure_standardized_columns()
        self._init_cache_after_params()

    # ---- internal helpers --------------------------------------------------

    def _build_combined_params(self) -> Dict[str, Any]:
        """Merge wrapped-node params with filter-chain params for hashing/caching."""
        combined: Dict[str, Any] = dict(self.wrapped.params) if self.wrapped.params else {}
        filter_descriptors: list[str] = []
        for f in self.filters:
            filter_descriptors.append(f.get_filter_suffix())
        combined["__filters__"] = tuple(filter_descriptors)
        return combined

    def _neutral_for(self, result: List) -> List:
        """Return a list of *neutral_value* matching the length of *result*."""
        return [self.neutral_value] * len(result)

    def _extra_lookback_contributions(self) -> tuple[LookbackContribution, ...]:
        """Include wrapped-node and filter warmups in the outer node warmup."""
        filter_windows = tuple(
            LookbackContribution(label=f"filter_{index}", bars=int(f.warmup))
            for index, f in enumerate(self.filters)
            if getattr(f, "warmup", 0) > 0
        )
        return (*self.wrapped.lookback_contributions(), *filter_windows)

    # ---- BiasNode interface ------------------------------------------------

    def _compute_candle(self, candle: Candle) -> List:
        self._n_candles += 1

        # Always update filters so they track continuous state
        for f in self.filters:
            f.update(candle)

        # Delegate to wrapped node
        result = self.wrapped.add_candle(candle)

        # During warmup the wrapped node produces its own placeholder — pass through
        if self._n_candles <= self.front_bad:
            return result

        # Gate: if any filter blocks, replace with neutral
        if not all(f.should_pass() for f in self.filters):
            return self._neutral_for(result)

        return result

    # ---- column naming (append filter suffixes) ----------------------------

    def get_column_names(self) -> List[str]:
        """Wrapped node columns + ``__f_…`` suffixes for each filter."""
        base_names = self.wrapped.get_column_names()
        if not base_names:
            return base_names
        suffix = "".join(f.get_filter_suffix() for f in self.filters)
        return [name + suffix for name in base_names]
