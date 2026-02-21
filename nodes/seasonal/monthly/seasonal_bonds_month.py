"""
Seasonal Bonds Month (window dressing) bias node.

Rule-based: configurable short/f flat/long windows by calendar day.
Designed for TLT / bond seasonal (window dressing) strategy.

Modes:
- SHORT_THEN_LONG: Short first N days, then long for the rest of the month.
- SHORT_FLAT_LONG: Short first N days, flat in the middle, long last M days.
"""

import calendar
from enum import Enum
from typing import List

from nodes import BiasNode
from utils.enums import Ticker, TimeFrame
from utils.models import Candle


class SeasonalBondsMonthMode(Enum):
    """Mode for seasonal bonds month logic."""

    SHORT_THEN_LONG = "short_then_long"
    """Short first short_days, then long for the rest of the month."""

    SHORT_FLAT_LONG = "short_flat_long"
    """Short first short_days, flat in the middle, long the last long_days."""


class SeasonalBondsMonth(BiasNode):
    """
    Seasonal Bonds Month Bias Node (rule-based).

    Window-dressing style calendar rules. No price or indicator logic.
    Mode is selected via `mode` param; column naming includes mode and days params.

    **Type**: Rule-based (outputs -1, 0, or 1).

    Modes:
    - SHORT_THEN_LONG: -1 for days 1..short_days, 1 for the rest.
    - SHORT_FLAT_LONG: -1 for days 1..short_days, 0 in the middle, 1 for the last long_days.

    Parameters:
    - short_days: Number of days at start of month to short (default: 7).
    - long_days: Number of days at end of month to long; used only when mode is SHORT_FLAT_LONG (default: 7).
    - mode: SeasonalBondsMonthMode or string 'short_then_long' | 'short_flat_long' (default: SHORT_THEN_LONG).
    """

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        short_days: int = 7,
        long_days: int = 7,
        mode: SeasonalBondsMonthMode | str = SeasonalBondsMonthMode.SHORT_THEN_LONG,
    ) -> None:
        super().__init__(ticker, tf)

        self.short_days = short_days
        self.long_days = long_days
        self._mode = (
            SeasonalBondsMonthMode(mode) if isinstance(mode, str) else mode
        )

        self.module_name = "seasonalbondsmonth"
        self.output_features = ["signal"]
        self.params = {
            "short_days": short_days,
            "long_days": long_days,
            "mode": self._mode.value,
        }

        self.front_bad = 1

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[int]:
        day = candle.datetime.day
        if self._mode == SeasonalBondsMonthMode.SHORT_THEN_LONG:
            signal = -1 if day <= self.short_days else 1
        else:
            _, last_day = calendar.monthrange(
                candle.datetime.year, candle.datetime.month
            )
            if day <= self.short_days:
                signal = -1
            elif day > last_day - self.long_days:
                signal = 1
            else:
                signal = 0
        self.output.append(signal)
        return [signal]
