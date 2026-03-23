"""
Turnaround Tuesday seasonal bias node (rule-based).

Long-only. Signal fires the day BEFORE entry so the system enters at the correct open.

- TUESDAY: Signal fires on Monday when today(Mon) < yesterday(Fri) < 2d-ago(Thu); enter at Tuesday open.
- TUE_WED: Signal fires on Monday (enter Tue open) or Tuesday (enter Wed open) when same declining-close pattern holds.

Exit: when today's close > yesterday's high.

Designed for daily bars (e.g. NQ).
"""

from enum import Enum
from typing import ClassVar, List, Tuple

from nodes import BiasNode
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


# datetime.weekday(): Monday=0, Tuesday=1, Wednesday=2
# Signal fires the day BEFORE entry so execution happens at the correct open.
_MONDAY = 0          # pre-Tuesday entry day
_MON_TUE = (0, 1)   # pre-TUE_WED entry days


class TurnaroundTuesdayMode(Enum):
    """Mode for turnaround Tuesday entry day filter."""

    TUESDAY = "tuesday"
    """Enter only on Tuesday when 3-day declining close pattern holds."""

    TUE_WED = "tue_wed"
    """Enter on Tuesday or Wednesday when 3-day declining close pattern holds."""


class TurnaroundTuesday(BiasNode):
    """
    Turnaround Tuesday Bias Node (rule-based).

    Single output signal (0 or 1). Mode selects entry rule:
    - TUESDAY: Signal fires Monday when Mon < Fri < Thu closes; enters long at Tuesday open.
    - TUE_WED: Signal fires Monday (enters Tue open) or Tuesday (enters Wed open) on same declining-close pattern.

    Exit: when close > yesterday's high.

    **Type**: Rule-based (outputs 0 or 1).
    """
    hardcoded_lookbacks: ClassVar[tuple[tuple[str, int], ...]] = (("pattern_window", 3),)

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        mode: TurnaroundTuesdayMode | str = TurnaroundTuesdayMode.TUE_WED,
    ) -> None:
        super().__init__(ticker, tf)

        self._mode = (
            TurnaroundTuesdayMode(mode) if isinstance(mode, str) else mode
        )

        self.module_name = "turnaroundtuesday"
        self.output_features = ["signal"]
        self.params = {"mode": self._mode.value}

        self.front_bad = 3

        # Last 3 bars (close, high): index 0 = 3d ago, 1 = 2d ago, 2 = yesterday
        self._buffer: List[Tuple[float, float]] = []
        self._position_tuesday = 0
        self._position_tue_wed = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[int]:
        close = candle.close
        high = candle.high
        wd = candle.datetime.weekday()

        if len(self._buffer) < 3:
            self._buffer.append((close, high))
            self.output = [0]
            return [0]

        close_1d, high_1d = self._buffer[2]
        close_2d, _ = self._buffer[1]

        # Exit: today's close > yesterday's high
        if close > high_1d:
            self._position_tuesday = 0
            self._position_tue_wed = 0

        # Entry condition: today < yesterday < 2d-ago (signal fires day before entry open)
        # On Monday: close=Mon, close_1d=Fri, close_2d=Thu → enters at Tuesday open
        # On Tuesday (TUE_WED only): close=Tue, close_1d=Mon, close_2d=Fri → enters at Wednesday open
        close_pattern = close < close_1d and close_1d < close_2d

        if close_pattern:
            if wd == _MONDAY:
                self._position_tuesday = 1
            if wd in _MON_TUE:
                self._position_tue_wed = 1

        # Rotate buffer: drop oldest, append current
        self._buffer.pop(0)
        self._buffer.append((close, high))

        signal = (
            self._position_tuesday
            if self._mode == TurnaroundTuesdayMode.TUESDAY
            else self._position_tue_wed
        )
        self.output = [signal]
        return [signal]
