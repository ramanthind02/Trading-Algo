"""
Mean-reversion bias node: inverse of BasicBreakout (fade prior-bar breakouts).

Wraps BasicBreakout and negates its stacked output. Same hold/stack semantics;
output in [-max_positions, max_positions] (or mode-clamped). No SL/TP/orders
inside the node.
"""

from typing import ClassVar, List

from nodes import BiasNode
from nodes.basic_breakout import BasicBreakout
from lib.core.enums import PositionMode, Ticker, TimeFrame
from lib.core.models import Candle


class BasicMR(BiasNode):
    """
    Mean-reversion bias node: inverse of BasicBreakout (fade prior-bar breakouts).
    Wraps BasicBreakout and negates stacked output. Same hold/stack semantics;
    output in [-max_positions, max_positions] (or mode-clamped). Supports same
    position mode and max_positions.
    """

    hardcoded_lookbacks: ClassVar[tuple[tuple[str, int], ...]] = (("warmup_bars", 2),)

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        mode: PositionMode = PositionMode.LONG_SHORT,
        max_positions: int = 3,
    ) -> None:
        super().__init__(ticker, tf)
        self.mode = mode
        self.max_positions = max_positions
        # Inner breakout must use LONG_SHORT so we get full signed output;
        # we then negate (MR view) and apply our own position mode.
        self._breakout = BasicBreakout(
            ticker, tf, mode=PositionMode.LONG_SHORT, max_positions=max_positions
        )

        self.module_name = "basic_mr"
        self.output_features = ["signal"]
        self.params = {"mode": mode, "max_positions": max_positions}
        self.front_bad = 2

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[int]:
        raw = self._breakout._compute_candle(candle)[0]
        negated = -int(raw)

        if self.mode == PositionMode.LONG_ONLY and negated < 0:
            signal = 0
        elif self.mode == PositionMode.SHORT_ONLY and negated > 0:
            signal = 0
        else:
            signal = negated

        self.output.append(signal)
        return [signal]
