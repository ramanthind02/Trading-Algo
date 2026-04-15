"""FilterAndSignalNode — signed AND between filter and signal.

Emits ``+1`` only when both first outputs are ``+1``, ``-1`` only when both are
``-1``, and ``0.0`` otherwise (same agreement rule as :class:`DualSignalNode`,
with filter/signal naming).

Usage
-----
>>> bias_spec = {
...     "module_name": "filter_and_signal",
...     "timeframes": [TimeFrame.D],
...     "params": {
...         "filter_module": "rsi_signal",
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


class FilterAndSignalNode(BiasNode):
    """Discrete confirmation: filter and signal must match in sign and be non-zero."""

    lookback_param_names: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, ticker: Ticker, tf: TimeFrame, **params: object) -> None:
        super().__init__(ticker, tf)

        filter_mod = str(params["filter_module"])
        signal_mod = str(params["signal_module"])
        filter_params: dict[str, object] = dict(params.get("filter_params") or {})  # type: ignore[arg-type]
        signal_params: dict[str, object] = dict(params.get("signal_params") or {})  # type: ignore[arg-type]

        self._node_filter = helpers.create_fresh_bias_node(filter_mod, ticker, tf, filter_params)
        self._node_signal = helpers.create_fresh_bias_node(signal_mod, ticker, tf, signal_params)

        self.module_name = "filter_and_signal"
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

        sig_f = float(filter_result[0]) if filter_result else 0.0
        sig_s = float(signal_result[0]) if signal_result else 0.0

        out = sig_f if (sig_f == sig_s and sig_f != 0.0) else 0.0
        self.output.append(out)
        return [out]
