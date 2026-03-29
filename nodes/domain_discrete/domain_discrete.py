from __future__ import annotations

from typing import ClassVar, List

import numpy as np

from feature_selection.domain_discrete import load_domain_discrete_spec
from filters import FilterSpec, create_filter
from nodes import BiasNode, LookbackContribution
from nodes.filtered import FilteredBiasNode
from utils.core import helpers
from utils.core.enums import Direction, Ticker, TimeFrame
from utils.core.models import Candle


class DomainDiscreteNode(BiasNode):
    """Generic frozen discrete-signal wrapper around a source bias node."""

    lookback_param_names: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, ticker: Ticker, tf: TimeFrame, **params: object) -> None:
        super().__init__(ticker, tf)

        self.domain_spec = load_domain_discrete_spec(params)
        if not self.domain_spec.ticker_scope.allows(ticker):
            raise ValueError(
                f"Ticker {ticker.name} is outside ticker_scope "
                f"{[item.name for item in self.domain_spec.ticker_scope.tickers]}"
            )

        source_spec = self.domain_spec.source_bias_node_spec
        if tf not in source_spec.timeframes:
            raise ValueError(
                f"Timeframe {tf.name} is not declared by source_bias_node_spec "
                f"{[item.name for item in source_spec.timeframes]}"
            )

        wrapped = helpers.create_fresh_bias_node(
            source_spec.module_name,
            ticker,
            tf,
            dict(source_spec.params),
        )
        if source_spec.filters:
            filters = tuple(
                create_filter(spec if isinstance(spec, FilterSpec) else FilterSpec(**spec))
                for spec in source_spec.filters
            )
            wrapped = FilteredBiasNode(wrapped, filters)
        self._wrapped = wrapped

        self.module_name = "domain_discrete"
        self.output_features = ["signal"]
        self.params = {
            "sourceModule": source_spec.module_name,
            "direction": self.domain_spec.direction.value,
            "specVersion": self.domain_spec.spec_version,
        }
        self.front_bad = getattr(self._wrapped, "front_bad", 0)

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _extra_lookback_contributions(self) -> tuple[LookbackContribution, ...]:
        return self._wrapped.lookback_contributions()

    def _assign_bin(self, raw_value: float) -> int:
        return int(np.digitize(raw_value, np.asarray(self.domain_spec.edges, dtype=float)))

    def _map_bin_to_signal(self, bin_idx: int) -> float:
        direction = self.domain_spec.direction
        if bin_idx in self.domain_spec.long_bins and direction in (Direction.LONG, Direction.LONG_SHORT):
            return 1.0
        if bin_idx in self.domain_spec.short_bins and direction in (Direction.SHORT, Direction.LONG_SHORT):
            return -1.0
        return 0.0

    def _compute_candle(self, candle: Candle) -> List[float]:
        wrapped_result = self._wrapped.add_candle(candle)
        raw_value = float(wrapped_result[0]) if wrapped_result else 0.0

        if getattr(self._wrapped, "output", None):
            # Warmup follows the wrapped node contract; preserve neutral output until ready.
            candle_count = len(self._wrapped.output)
        else:
            candle_count = 0

        if candle_count <= self.front_bad:
            self.output.append(0.0)
            return [0.0]

        bin_idx = self._assign_bin(raw_value)
        signal = self._map_bin_to_signal(bin_idx)
        self.output.append(signal)
        return [signal]
