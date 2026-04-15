"""FilterGateNode — pass-through signal when the filter is active.

The filter's first output is treated as a gate: **non-zero means open**. When open,
the node's output is the signal child's first output (unchanged). When closed
(filter first output is 0), the output is 0.0.

Use discrete filters that emit ``0`` when off and any non-zero value when on; for a
strictly boolean filter, use ``0`` / ``1``.

Usage
-----
>>> bias_spec = {
...     "module_name": "filter_gate",
...     "timeframes": [TimeFrame.D],
...     "params": {
...         "filter_module": "rsi_regime",
...         "filter_params": {"lookback": 14},
...         "signal_module": "ewmac",
...         "signal_params": {"span_fast": 16, "span_slow": 64},
...     },
... }
"""

from __future__ import annotations

from typing import ClassVar, List

from nodes import BiasNode, LookbackContribution
from utils.core import helpers
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


class FilterGateNode(BiasNode):
    """Boolean-style gate on the signal using the filter's first output."""

    lookback_param_names: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, ticker: Ticker, tf: TimeFrame, **params: object) -> None:
        super().__init__(ticker, tf)

        filter_mod = str(params["filter_module"])
        signal_mod = str(params["signal_module"])
        filter_params: dict[str, object] = dict(params.get("filter_params") or {})  # type: ignore[arg-type]
        signal_params: dict[str, object] = dict(params.get("signal_params") or {})  # type: ignore[arg-type]

        self._node_filter = helpers.create_fresh_bias_node(filter_mod, ticker, tf, filter_params)
        self._node_signal = helpers.create_fresh_bias_node(signal_mod, ticker, tf, signal_params)

        self.module_name = "filter_gate"
        self.output_features = ["signal"]
        self.params = {
            "filter_module": filter_mod,
            "signal_module": signal_mod,
            **{f"f_{k}": v for k, v in filter_params.items()},
            **{f"s_{k}": v for k, v in signal_params.items()},
        }
        self.front_bad = max(
            getattr(self._node_filter, "front_bad", 0),
            getattr(self._node_signal, "front_bad", 0),
        )

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _extra_lookback_contributions(self) -> tuple[LookbackContribution, ...]:
        return (
            *self._node_filter.lookback_contributions(),
            *self._node_signal.lookback_contributions(),
        )

    def _compute_candle(self, candle: Candle) -> List[float]:
        filter_result = self._node_filter.add_candle(candle)
        signal_result = self._node_signal.add_candle(candle)

        filt = float(filter_result[0]) if filter_result else 0.0
        sig = float(signal_result[0]) if signal_result else 0.0

        gate_open = filt != 0.0
        out = sig if gate_open else 0.0
        self.output.append(out)
        return [out]
