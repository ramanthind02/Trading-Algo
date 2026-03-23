"""
Prior-bar high/low breakout bias node (Market Edge / StatOasis EA–style).

Rule-based, stateful: holds last non-zero output on inside bars and stacks
same-direction breakouts up to max_positions (default 5). Output is an integer
in [-max_positions, max_positions]; 0 for warmup or when position mode clamps.
No SL/TP/limit/stop inside the node; bias output is the position directive for
downstream systems.
"""

from typing import ClassVar, List

from nodes import BiasNode
from utils.core.enums import PositionMode, Ticker, TimeFrame
from utils.core.models import Candle


class BasicBreakout(BiasNode):
    """
    Prior-bar high/low breakout bias node (Market Edge). Rule-based, stateful.
    Holds on inside bar; stacks same-direction breakouts up to max_positions (default 5).
    Output: integer in [-max_positions, max_positions]; 0 for warmup or mode clamp.
    Position mode: LONG_SHORT (default), LONG_ONLY, or SHORT_ONLY.
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
        self.module_name = "basicbreakout"
        self.output_features = ["signal"]
        self.params = {"mode": mode, "max_positions": max_positions}
        self.front_bad = 2

        self.prev_high: float = 0.0
        self.prev_low: float = 0.0
        self.n_prices: int = 0
        self.last_signal: int = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[int]:
        self.n_prices += 1
        if self.n_prices < self.front_bad:
            self.output.append(0)
            self.prev_high = candle.high
            self.prev_low = candle.low
            return [0]

        if candle.close > self.prev_high:
            raw = 1
        elif candle.close < self.prev_low:
            raw = -1
        else:
            raw = 0

        if raw == 0:
            stacked = self.last_signal
        elif raw == 1:
            stacked = (
                1
                if self.last_signal <= 0
                else min(self.max_positions, self.last_signal + 1)
            )
        else:
            stacked = (
                -1
                if self.last_signal >= 0
                else -min(self.max_positions, -self.last_signal + 1)
            )

        if self.mode == PositionMode.LONG_ONLY and stacked < 0:
            signal = 0
        elif self.mode == PositionMode.SHORT_ONLY and stacked > 0:
            signal = 0
        else:
            signal = stacked

        self.last_signal = signal
        self.prev_high = candle.high
        self.prev_low = candle.low
        self.output.append(signal)
        return [signal]
