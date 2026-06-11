"""
%B (Percent B) Bollinger signal — long below lower band, exit above upper band.

``%B = (close - lower) / (upper - lower)`` on standard Bollinger bands (SMA ± std_dev × σ).

Long mean reversion:
- Enter on cross below ``lower_threshold`` (default 0 = lower band).
- Exit on cross above ``upper_threshold`` (default 1 = upper band).

Long/short mode — independent exit thresholds (optional):
- ``long_exit_threshold``: if set, long exits when %B crosses above this level (e.g. 0.5 = mean)
  before the short entry at ``upper_threshold``.  Omit → original flip-at-upper behavior.
- ``short_exit_threshold``: if set, short exits when %B crosses below this level (e.g. 0.5 = mean)
  before the long entry at ``lower_threshold``.  Omit → original flip-at-lower behavior.

Example — symmetric mean-reversion with flat zone at middle:
  lower_threshold=0, upper_threshold=1, long_exit_threshold=0.5, short_exit_threshold=0.5
"""

from __future__ import annotations

from collections import deque
from typing import ClassVar, List, Optional

import numpy as np

from nodes import BiasNode
from lib.core.enums import DirectionInput, Ticker, TimeFrame, coerce_direction
from lib.core.models import Candle


class PercentBSignal(BiasNode):
    """Discrete -1/0/1 from Bollinger %B threshold crosses (RSI-signal semantics)."""

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"period"})
    param_choices: ClassVar[dict[str, list[str]]] = {
        "exit_policy": ["threshold", "threshold_or_bars"],
    }

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        period: int = 20,
        std_dev: float = 2.0,
        lower_threshold: float = 0.0,
        upper_threshold: float = 1.0,
        band_extension: float = 0.0,
        strategy_mode: DirectionInput = "long",
        exit_policy: str = "threshold",
        exit_bars: int = 5,
        long_exit_threshold: float = 1.0,
        short_exit_threshold: float = 0.0,
    ) -> None:
        super().__init__(ticker, tf)

        if period < 2:
            raise ValueError("period must be >= 2")
        if std_dev <= 0:
            raise ValueError("std_dev must be > 0")
        if band_extension < 0:
            raise ValueError("band_extension must be >= 0")

        # Apply band extension: widen the entry zone beyond the standard band.
        # band_extension=0.1 → enter long at %B < -0.1, short at %B > 1.1.
        effective_lower = lower_threshold - band_extension
        effective_upper = upper_threshold + band_extension
        if effective_lower >= effective_upper:
            raise ValueError("lower_threshold must be < upper_threshold after applying band_extension")

        if long_exit_threshold > effective_upper:
            raise ValueError("long_exit_threshold must be <= upper_threshold")
        if short_exit_threshold < effective_lower:
            raise ValueError("short_exit_threshold must be >= lower_threshold")

        self.period = period
        self.std_dev = std_dev
        self.lower_threshold = effective_lower
        self.upper_threshold = effective_upper
        self.band_extension = band_extension
        self._long_exit_threshold = long_exit_threshold
        self._short_exit_threshold = short_exit_threshold
        self.strategy_mode = self._normalize_strategy_mode(strategy_mode)
        self.exit_policy = exit_policy.lower().strip()
        if self.exit_policy not in {"threshold", "threshold_or_bars"}:
            raise ValueError("exit_policy must be 'threshold' or 'threshold_or_bars'")
        if exit_bars < 1:
            raise ValueError("exit_bars must be >= 1")
        self.exit_bars = exit_bars

        self.module_name = "percent_b_signal"
        self.output_features = ["signal"]
        self.params = {
            "period": period,
            "stdDev": std_dev,
            "lowerThreshold": lower_threshold,
            "upperThreshold": upper_threshold,
            "bandExtension": band_extension,
            "strategyMode": self.strategy_mode,
            "exitPolicy": self.exit_policy,
            "exitBars": exit_bars,
            "longExitThreshold": long_exit_threshold,
            "shortExitThreshold": short_exit_threshold,
        }

        self.front_bad = period
        self.close_buffer: deque[float] = deque(maxlen=period)
        self.n_candles = 0

        self.position = 0
        self.bars_in_position = 0
        self.prev_percent_b: Optional[float] = None

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _normalize_strategy_mode(self, strategy_mode: DirectionInput) -> str:
        return coerce_direction(strategy_mode, field_name="strategy_mode").value

    def _percent_b(self, close: float) -> float:
        closes = np.array(self.close_buffer, dtype=float)
        sma = float(np.mean(closes))
        std = float(np.std(closes, ddof=1))
        upper_band = sma + self.std_dev * std
        lower_band = sma - self.std_dev * std
        band_width = upper_band - lower_band
        if band_width <= 0:
            return 0.5
        return (close - lower_band) / band_width

    def _crossed_below(self, prev_b: float, percent_b: float) -> bool:
        return prev_b > self.lower_threshold and percent_b <= self.lower_threshold

    def _crossed_above(self, prev_b: float, percent_b: float) -> bool:
        return prev_b < self.upper_threshold and percent_b >= self.upper_threshold

    def _crossed_long_exit(self, prev_b: float, percent_b: float) -> bool:
        return prev_b < self._long_exit_threshold and percent_b >= self._long_exit_threshold

    def _crossed_short_exit(self, prev_b: float, percent_b: float) -> bool:
        return prev_b > self._short_exit_threshold and percent_b <= self._short_exit_threshold

    def _apply_position_rules(self, percent_b: float) -> int:
        if self.prev_percent_b is None:
            return 0

        crossed_below = self._crossed_below(self.prev_percent_b, percent_b)
        crossed_above = self._crossed_above(self.prev_percent_b, percent_b)
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
        else:  # long_short
            if self.position == 1:
                if self._crossed_long_exit(self.prev_percent_b, percent_b):
                    # Flip to short if short entry also fires on this bar (always-in), else go flat.
                    next_position = -1 if crossed_above else 0
            elif self.position == -1:
                if self._crossed_short_exit(self.prev_percent_b, percent_b):
                    # Flip to long if long entry also fires on this bar (always-in), else go flat.
                    next_position = 1 if crossed_below else 0
            else:  # flat — standard entry
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

    def _compute_candle(self, candle: Candle) -> List:
        self.n_candles += 1
        close_v = float(candle.close)
        self.close_buffer.append(close_v)

        if self.n_candles < self.front_bad:
            self.output.append(0.0)
            return [0.0]

        percent_b = self._percent_b(close_v)
        signal = float(self._apply_position_rules(percent_b))
        self.prev_percent_b = percent_b
        self.output.append(signal)
        return [signal]
