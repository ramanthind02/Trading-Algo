"""
Five-day washout mean-reversion bias node (index-style MR).

Inspired by short-horizon index mean-reversion: enter long after a cascade of
consecutive lower lows (default four steps over five daily bars), exit on a
simple recovery trigger (close above the prior bar's high) or a hard bar-count
stop. Optional SMA filter (Connors-style) when ma_period > 0.

Output: 0.0 flat, 1.0 long (rule-based discrete signal).
"""

from collections import deque
from enum import Enum
from typing import ClassVar, List

from nodes import BiasNode
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


class WashoutRecoveryExit(str, Enum):
    """How to define the recovery / take-profit style exit while long."""

    PRIOR_HIGH_BREAK = "prior_high_break"


class FiveDayWashoutMR(BiasNode):
    """
    Stateful long-only mean-reversion bias from a lower-low washout cascade.

    Entry (flat): after ``cascade_length`` consecutive strict lower lows
    (each low < previous low), optionally require close > SMA(ma_period) when
    ``ma_period`` > 0.

    Exit (long): close > prior bar high (recovery), or ``max_hold_bars``
    closes in the position (time stop), whichever comes first.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset(
        {"cascadeLength", "maPeriod"}
    )

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        cascade_length: int = 4,
        ma_period: int = 0,
        max_hold_bars: int = 5,
        recovery_exit: WashoutRecoveryExit = WashoutRecoveryExit.PRIOR_HIGH_BREAK,
    ) -> None:
        super().__init__(ticker, tf)

        if cascade_length < 1:
            raise ValueError("cascade_length must be >= 1")
        if ma_period < 0:
            raise ValueError("ma_period must be >= 0 (0 disables the filter)")
        if max_hold_bars < 1:
            raise ValueError("max_hold_bars must be >= 1")

        self.cascade_length = cascade_length
        self.ma_period = ma_period
        self.max_hold_bars = max_hold_bars
        self.recovery_exit = recovery_exit

        self.module_name = "five_day_washout_mr"
        self.output_features = ["signal"]
        self.params = {
            "cascadeLength": cascade_length,
            "maPeriod": ma_period,
            "maxHoldBars": max_hold_bars,
            "recoveryExit": recovery_exit,
        }

        self.front_bad = max(cascade_length + 1, ma_period if ma_period > 0 else 0, 2)

        self._lows: deque[float] = deque(maxlen=cascade_length + 1)
        self._ma_buffer: deque[float] = (
            deque(maxlen=ma_period) if ma_period > 0 else deque()
        )
        self._ma_sum = 0.0

        self._prior_bar_high: float | None = None
        self._n_candles = 0
        self._current_position = 0
        self._bars_in_position = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _strict_lower_low_cascade(self, lows: List[float]) -> bool:
        return all(
            lows[-1 - i] < lows[-2 - i] for i in range(self.cascade_length)
        )

    def _trend_ok(self, close: float) -> bool:
        if self.ma_period <= 0:
            return True
        if len(self._ma_buffer) < self.ma_period:
            return False
        return close > (self._ma_sum / self.ma_period)

    def _maybe_exit_long(self, candle: Candle) -> None:
        if self._current_position != 1:
            return
        self._bars_in_position += 1
        recovery = False
        if self.recovery_exit is WashoutRecoveryExit.PRIOR_HIGH_BREAK:
            prior = self._prior_bar_high
            recovery = prior is not None and candle.close > prior
        timed_out = self._bars_in_position >= self.max_hold_bars
        if recovery or timed_out:
            self._current_position = 0
            self._bars_in_position = 0

    def _maybe_enter_long(self, candle: Candle) -> None:
        if self._current_position != 0:
            return
        if len(self._lows) < self.cascade_length + 1:
            return
        lows_list = list(self._lows)
        if not self._strict_lower_low_cascade(lows_list):
            return
        if not self._trend_ok(candle.close):
            return
        self._current_position = 1
        self._bars_in_position = 1

    def _compute_candle(self, candle: Candle) -> List[float]:
        self._n_candles += 1
        curr_close = float(candle.close)
        curr_low = float(candle.low)

        if self.ma_period > 0:
            if len(self._ma_buffer) >= self.ma_period:
                self._ma_sum -= self._ma_buffer[0]
            self._ma_sum += curr_close
            self._ma_buffer.append(curr_close)

        self._lows.append(curr_low)

        if self._n_candles < self.front_bad:
            self._prior_bar_high = float(candle.high)
            self.output.append(0.0)
            return [0.0]

        self._maybe_exit_long(candle)
        self._maybe_enter_long(candle)

        self._prior_bar_high = float(candle.high)
        signal = float(self._current_position)
        self.output.append(signal)
        return [signal]
