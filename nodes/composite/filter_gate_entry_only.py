"""FilterGateEntryOnlyNode — apply the filter only when entering a position.

The signal child's first output is the **underlying** position (e.g. Donchian 0/1).
The filter's first output is the gate: **non-zero = open**. Unlike :class:`FilterGateNode`,
once the gated output is **non-zero**, subsequent bars keep passing the signal through
**even if the filter closes**, until the signal returns to zero (exit). New entries
(signal transitions from 0 to non-zero) require the filter to be open on that bar
(or a later bar while the signal stays on — see below).

Rules per bar (after child warmups):

- If signal ``sig == 0``: output ``0`` (flat).
- If ``sig != 0`` and previous **gated** output was non-zero: output ``sig`` (hold).
- If ``sig != 0`` and previous gated output was ``0``: output ``sig`` if filter is open,
  else ``0`` (entry blocked).

This matches “ATR regime only for entry; do not force exit when ATR gate turns off.”
"""

from __future__ import annotations

from typing import ClassVar, List

from nodes import BiasNode, LookbackContribution
from lib.core import helpers
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle


class FilterGateEntryOnlyNode(BiasNode):
    """Gate **entries** with the filter; allow **holds** without re-checking the filter."""

    lookback_param_names: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, ticker: Ticker, tf: TimeFrame, **params: object) -> None:
        super().__init__(ticker, tf)

        filter_mod = str(params["filter_module"])
        signal_mod = str(params["signal_module"])
        filter_params: dict[str, object] = dict(params.get("filter_params") or {})  # type: ignore[arg-type]
        signal_params: dict[str, object] = dict(params.get("signal_params") or {})  # type: ignore[arg-type]

        self._node_filter = helpers.create_fresh_bias_node(filter_mod, ticker, tf, filter_params)
        self._node_signal = helpers.create_fresh_bias_node(signal_mod, ticker, tf, signal_params)

        self.module_name = "filter_gate_entry_only"
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

        self._prev_gated_out: float = 0.0

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
        prev = self._prev_gated_out

        if sig == 0.0:
            out = 0.0
        elif prev != 0.0:
            out = sig
        else:
            out = sig if gate_open else 0.0

        self._prev_gated_out = out
        self.output.append(out)
        return [out]
