import collections
from typing import List

from nodes import BiasNode
from utils.core.enums import TimeFrame, Ticker
from utils.core.models import Candle


_VALID_DIRECTIONS = ("long", "short", "long_short")


class TurtleTrading(BiasNode):
    """
    Turtle Trading bias node — RULE-BASED, direction configurable.

    Outputs signed position: 1 (long), -1 (short), 0 (flat) depending on direction:
    - direction='long': 1 or 0
    - direction='short': -1 or 0
    - direction='long_short': 1, 0, or -1

    Classic rules: entry on breakout, exit on stop channel. Re-entry allowed on any new breakout.
    """
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        entry_lookback: int,
        stop_lookback: int,
        direction: str = "long",
    ) -> None:
        """
        Initializes the TurtleTrading bias node.

        Parameters:
        - ticker: The ticker symbol.
        - tf: The timeframe of the candles.
        - entry_lookback: Periods to look back for entry (classic: 4).
        - stop_lookback: Periods to look back for stop (classic: 2).
        - direction: 'long', 'short', or 'long_short'. Controls which positions are output.
        """
        super().__init__(ticker, tf)

        if not (entry_lookback > 0 and stop_lookback > 0):
            raise ValueError("Lookback periods must be positive integers.")
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(f"direction must be one of {_VALID_DIRECTIONS}, got {direction!r}")

        self.entry_lookback = entry_lookback
        self.stop_lookback = stop_lookback
        self.direction = direction

        self.module_name = "turtle"
        self.output_features = ["signal"]
        self.params = {
            "entry_lookback": entry_lookback,
            "stop_lookback": stop_lookback,
            "direction": direction,
        }
        self.front_bad = max(entry_lookback, stop_lookback) + 1

        required_history_len = max(entry_lookback, stop_lookback) + 10
        self.candles_history: collections.deque = collections.deque(maxlen=required_history_len)

        self.current_position = 0  # 0=flat, 1=long, -1=short (internal)
        self.entry_price: float | None = None

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _calculate_donchian_channel(self, lookback: int) -> tuple[float, float]:
        """
        Donchian channel (highest high, lowest low) over the last lookback bars, excluding the current candle.
        Computed from candles_history so behavior is correct regardless of DonchianChannel's Cython path.
        """
        if len(self.candles_history) < lookback + 1:
            return 0.0, 0.0
        recent = list(self.candles_history)[-(lookback + 1) : -1]
        if not recent:
            return 0.0, 0.0
        highest_high = max(c.high for c in recent)
        lowest_low = min(c.low for c in recent)
        return highest_high, lowest_low

    def _compute_candle(self, candle: Candle) -> List[float]:
        """
        Compute Turtle Trading bias for the given candle.

        Returns signed position: 1 (long), -1 (short), or 0 (flat) according to direction.
        """
        self.candles_history.append(candle)

        if len(self.candles_history) < self.front_bad:
            signal = 0.0
            self.output.append(signal)
            return [signal]

        entry_high, entry_low = self._calculate_donchian_channel(self.entry_lookback)
        stop_high, stop_low = self._calculate_donchian_channel(self.stop_lookback)
        current_price = candle.close

        if self.current_position == 0:
            if current_price > entry_high and self.direction in ("long", "long_short"):
                self.current_position = 1
                self.entry_price = current_price
            elif current_price < entry_low and self.direction in ("short", "long_short"):
                self.current_position = -1
                self.entry_price = current_price
        elif self.current_position == 1:
            if current_price < stop_low:
                self.current_position = 0
                self.entry_price = None
        elif self.current_position == -1:
            if current_price > stop_high:
                self.current_position = 0
                self.entry_price = None

        if self.direction == "long":
            signal = 1.0 if self.current_position == 1 else 0.0
        elif self.direction == "short":
            signal = -1.0 if self.current_position == -1 else 0.0
        else:
            signal = float(self.current_position)

        self.output.append(signal)
        return [signal]