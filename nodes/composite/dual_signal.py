"""DualSignalNode – confirmation wrapper for two independent bias nodes.

Emits the agreed direction when both inner nodes produce the same non-zero
discrete signal (+1 or -1). Outputs 0.0 whenever the nodes disagree or either
node is flat.

Usage
-----
>>> bias_spec = {
...     "module_name": "dual_signal",
...     "timeframes": [TimeFrame.D],
...     "params": {
...         "moduleA": "rsi_signal",
...         "paramsA": {"lookback": 14},
...         "moduleB": "ewmac",
...         "paramsB": {"span_fast": 16, "span_slow": 64},
...     },
... }
"""

from __future__ import annotations

from typing import ClassVar, List

from nodes import BiasNode, LookbackContribution
from lib.core import helpers
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle


class DualSignalNode(BiasNode):
    """Confirmation node: emits a direction only when two child nodes agree.

    Both child nodes are instantiated fresh (no singleton reuse) to ensure
    independent state, following the same pattern as other wrapped bias nodes.

    Parameters
    ----------
    ticker, tf
        Forwarded to both inner nodes and the base ``BiasNode``.
    moduleA
        Taxonomy key for inner node A (e.g. ``"rsi_signal"``).
    paramsA
        Parameter dict for inner node A.
    moduleB
        Taxonomy key for inner node B.
    paramsB
        Parameter dict for inner node B.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, ticker: Ticker, tf: TimeFrame, **params: object) -> None:
        super().__init__(ticker, tf)

        module_a: str = str(params["moduleA"])
        module_b: str = str(params["moduleB"])
        params_a: dict[str, object] = dict(params.get("paramsA") or {})  # type: ignore[arg-type]
        params_b: dict[str, object] = dict(params.get("paramsB") or {})  # type: ignore[arg-type]

        self._node_a = helpers.create_fresh_bias_node(module_a, ticker, tf, params_a)
        self._node_b = helpers.create_fresh_bias_node(module_b, ticker, tf, params_b)

        self.module_name = "dual_signal"
        self.output_features = ["signal"]
        self.params = {
            "moduleA": module_a,
            "moduleB": module_b,
            **{f"a_{k}": v for k, v in params_a.items()},
            **{f"b_{k}": v for k, v in params_b.items()},
        }
        self.front_bad = max(
            getattr(self._node_a, "front_bad", 0),
            getattr(self._node_b, "front_bad", 0),
        )

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    # ---- BiasNode interface ------------------------------------------------

    def _extra_lookback_contributions(self) -> tuple[LookbackContribution, ...]:
        return (*self._node_a.lookback_contributions(), *self._node_b.lookback_contributions())

    def _compute_candle(self, candle: Candle) -> List[float]:
        result_a = self._node_a.add_candle(candle)
        result_b = self._node_b.add_candle(candle)

        sig_a = float(result_a[0]) if result_a else 0.0
        sig_b = float(result_b[0]) if result_b else 0.0

        signal = sig_a if (sig_a == sig_b and sig_a != 0.0) else 0.0
        self.output.append(signal)
        return [signal]
