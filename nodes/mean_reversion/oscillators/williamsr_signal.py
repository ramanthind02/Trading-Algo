"""
Williams %R Signal Bias Node

Discrete signals from Williams %R threshold crosses. Generates -1, 0, or 1
signals based on oversold/overbought cross events and optional fixed-exit rules.

Classic %R scale: -100 (at range low) through 0 (at range high).

Output: Rule-based -1, 0, or 1
"""

from collections import deque
from typing import ClassVar, List, Optional

from nodes import BiasNode
from lib.core.enums import DirectionInput, Ticker, TimeFrame, coerce_direction
from lib.core.models import Candle


class WilliamsRSignal(BiasNode):
    """
    Williams %R Signal — discrete signals from %R threshold crosses.

    Uses the classic formula over ``lookback`` bars (including the current bar):
    ``%R = (highest_high - close) / (highest_high - lowest_low) × -100``.

    - Signal = 1 (long) when %R crosses at or below ``oversold`` from above.
    - Signal = -1 (short) when %R crosses at or above ``overbought`` from below.
    - Signal = 0 when flat or fixed-exit fires.

    Default thresholds match common textbook levels: oversold -80, overbought -20.

    Output Range: -1, 0, or 1
    Neutral Value: 0 (during warmup)
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"lookback"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        lookback: int = 14,
        oversold: float = -80.0,
        overbought: float = -20.0,
        strategy_mode: DirectionInput = "long",
        exit_policy: str = "threshold_or_bars",
        exit_bars: int = 5,
    ) -> None:
        super().__init__(ticker, tf)

        if oversold >= overbought:
            raise ValueError(
                "oversold must be strictly less than overbought on the -100..0 %R scale"
            )

        self.lookback = lookback
        self.oversold = oversold
        self.overbought = overbought
        self.strategy_mode = self._normalize_strategy_mode(strategy_mode)
        self.exit_policy = exit_policy.lower().strip()
        if self.exit_policy not in {"threshold", "threshold_or_bars"}:
            raise ValueError(
                "exit_policy must be 'threshold' or 'threshold_or_bars'"
            )
        if exit_bars < 1:
            raise ValueError("exit_bars must be >= 1")
        self.exit_bars = exit_bars

        self.module_name = "williamsrsignal"
        self.output_features = ["signal"]
        self.params = {
            "lookback": lookback,
            "oversold": oversold,
            "overbought": overbought,
            "strategyMode": self.strategy_mode,
            "exitPolicy": self.exit_policy,
            "exitBars": exit_bars,
        }

        self.front_bad = lookback

        self.high_prices: deque[float] = deque(maxlen=lookback)
        self.low_prices: deque[float] = deque(maxlen=lookback)

        self.position = 0
        self.bars_in_position = 0
        self.prev_wr: Optional[float] = None

        self.n_candles = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[float]:
        self.n_candles += 1

        self.high_prices.append(candle.high)
        self.low_prices.append(candle.low)

        if len(self.high_prices) < self.lookback:
            self.position = 0
            self.bars_in_position = 0
            self.prev_wr = None
            self.output.append(0.0)
            return [0.0]

        highest_high = max(self.high_prices)
        lowest_low = min(self.low_prices)
        price_range = highest_high - lowest_low

        if price_range > 0.0:
            wr = ((highest_high - candle.close) / price_range) * -100.0
        else:
            wr = 0.0

        signal = self._apply_position_rules(wr)
        self.prev_wr = wr
        self.output.append(float(signal))
        return [float(signal)]

    def _normalize_strategy_mode(self, strategy_mode: DirectionInput) -> str:
        return coerce_direction(strategy_mode, field_name="strategy_mode").value

    def _crossed_below(self, prev_wr: float, wr: float) -> bool:
        return prev_wr > self.oversold and wr <= self.oversold

    def _crossed_above(self, prev_wr: float, wr: float) -> bool:
        return prev_wr < self.overbought and wr >= self.overbought

    def _apply_position_rules(self, wr: float) -> int:
        if self.prev_wr is None:
            return 0

        crossed_below = self._crossed_below(self.prev_wr, wr)
        crossed_above = self._crossed_above(self.prev_wr, wr)

        next_position = self.position

        if self.strategy_mode == "long":
            if self.position == 0 and crossed_below:
                next_position = 1
            elif self.position == 1 and crossed_above:
                next_position = 0
        elif self.strategy_mode == "short":
            if self.position == 0 and crossed_above:
                next_position = -1
            elif self.position == -1 and crossed_below:
                next_position = 0
        else:
            if crossed_below:
                next_position = 1
            elif crossed_above:
                next_position = -1

        if next_position == self.position and next_position != 0:
            self.bars_in_position += 1
        elif next_position != 0:
            self.bars_in_position = 1
        else:
            self.bars_in_position = 0

        if (
            self.exit_policy == "threshold_or_bars"
            and next_position != 0
            and self.bars_in_position >= self.exit_bars
        ):
            next_position = 0
            self.bars_in_position = 0

        self.position = next_position
        return next_position
