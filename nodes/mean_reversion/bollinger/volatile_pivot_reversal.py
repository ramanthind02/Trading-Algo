"""
Volatile pivot reversal (Bollinger ratio entry) — Algomatic / StatOasis-style.

Long mean reversion after a confirmed local low pivot in a volatile down-move:
- Pivot: price low vs ``pivot_bars_before`` older and ``pivot_bars_after`` newer bars.
- Formation (at pivot bar): wide Bollinger bandwidth and wide-band %B below ``bbr_wide_threshold``.
- Entry (current bar): narrow-band %B reclaimed into ``[min_percent_b, max_percent_b]`` within
  ``[min_bars_since_bottom, max_bars_since_bottom]`` bars after the pivot.
- Exit: fixed ``exit_bars`` hold (time exit), matching MQL ``EXIT_TIME_BARS``.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import ClassVar, List, Sequence, Tuple

import numpy as np

from nodes import BiasNode
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle

_OhlcBar = Tuple[float, float, float]


class PivotCompareMode(Enum):
    """Price series used for pivot centre and neighbours."""

    CLOSE = "close"
    LOW_HIGH = "low_high"


def _coerce_pivot_compare_mode(raw: PivotCompareMode | str) -> PivotCompareMode:
    if isinstance(raw, PivotCompareMode):
        return raw
    key = str(raw).strip().lower()
    for member in PivotCompareMode:
        if member.value == key or member.name.lower() == key:
            return member
    raise ValueError(
        "pivot_compare_mode must be PivotCompareMode or one of: "
        f"{', '.join(m.value for m in PivotCompareMode)}"
    )


def _bollinger_bands(
    closes: Sequence[float], std_mult: float
) -> tuple[float, float, float]:
    arr = np.asarray(closes, dtype=np.float64)
    middle = float(arr.mean())
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return lower, middle, upper


def _percent_b(close: float, lower: float, upper: float) -> float:
    width = upper - lower
    if width <= 0.0:
        return 0.5
    return (close - lower) / width


def _bb_width_pct(lower: float, middle: float, upper: float) -> float:
    if middle == 0.0:
        return 0.0
    return (upper - lower) / middle * 100.0


class VolatilePivotReversal(BiasNode):
    """Long-only discrete signal from volatile pivot + Bollinger ratio reclaim; time-bar exit."""

    lookback_param_names: ClassVar[frozenset[str]] = frozenset(
        {
            "bb_period",
            "max_bars_since_bottom",
            "pivot_bars_before",
            "pivot_bars_after",
        }
    )

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        pivot_bars_before: int = 3,
        pivot_bars_after: int = 3,
        min_bars_since_bottom: int = 3,
        max_bars_since_bottom: int = 25,
        pivot_compare_mode: PivotCompareMode | str = PivotCompareMode.CLOSE,
        bb_period: int = 20,
        bb_deviation_narrow: float = 2.0,
        bb_deviation_wide: float = 20.0,
        min_percent_b: float = 0.41,
        max_percent_b: float = 0.53,
        bbw_threshold: float = 1.0,
        bbr_wide_threshold: float = 0.8,
        exit_bars: int = 20,
    ) -> None:
        super().__init__(ticker, tf)

        if pivot_bars_before < 1 or pivot_bars_after < 1:
            raise ValueError("pivot_bars_before and pivot_bars_after must be >= 1")
        if min_bars_since_bottom < 1:
            raise ValueError("min_bars_since_bottom must be >= 1")
        if max_bars_since_bottom < min_bars_since_bottom:
            raise ValueError("max_bars_since_bottom must be >= min_bars_since_bottom")
        if bb_period < 2:
            raise ValueError("bb_period must be >= 2")
        if bb_deviation_narrow <= 0.0 or bb_deviation_wide <= 0.0:
            raise ValueError("bb deviations must be > 0")
        if min_percent_b >= max_percent_b:
            raise ValueError("min_percent_b must be < max_percent_b")
        if exit_bars < 1:
            raise ValueError("exit_bars must be >= 1")

        self.pivot_bars_before = pivot_bars_before
        self.pivot_bars_after = pivot_bars_after
        self.min_bars_since_bottom = min_bars_since_bottom
        self.max_bars_since_bottom = max_bars_since_bottom
        self.pivot_compare_mode = _coerce_pivot_compare_mode(pivot_compare_mode)
        self.bb_period = bb_period
        self.bb_deviation_narrow = bb_deviation_narrow
        self.bb_deviation_wide = bb_deviation_wide
        self.min_percent_b = min_percent_b
        self.max_percent_b = max_percent_b
        self.bbw_threshold = bbw_threshold
        self.bbr_wide_threshold = bbr_wide_threshold
        self.exit_bars = exit_bars

        self.module_name = "volatile_pivot_reversal"
        self.output_features = ["signal"]
        self.params = {
            "pivotBarsBefore": pivot_bars_before,
            "pivotBarsAfter": pivot_bars_after,
            "minBarsSinceBottom": min_bars_since_bottom,
            "maxBarsSinceBottom": max_bars_since_bottom,
            "pivotCompareMode": self.pivot_compare_mode.value,
            "bbPeriod": bb_period,
            "bbDeviationNarrow": bb_deviation_narrow,
            "bbDeviationWide": bb_deviation_wide,
            "minPercentB": min_percent_b,
            "maxPercentB": max_percent_b,
            "bbwThreshold": bbw_threshold,
            "bbrWideThreshold": bbr_wide_threshold,
            "exitBars": exit_bars,
        }

        history_len = (
            bb_period
            + max_bars_since_bottom
            + pivot_bars_before
            + pivot_bars_after
            + 2
        )
        self.front_bad = history_len
        self._history: deque[_OhlcBar] = deque(maxlen=history_len)
        self._position = 0
        self._bars_in_position = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _pivot_price(self, shift: int, *, for_low: bool) -> float:
        close_v, _high_v, low_v = self._history[-1 - shift]
        if self.pivot_compare_mode == PivotCompareMode.CLOSE:
            return close_v
        return low_v if for_low else _high_v

    def _closes_window(self, end_shift: int) -> list[float]:
        end_index = len(self._history) - 1 - end_shift
        start_index = end_index - self.bb_period + 1
        return [self._history[i][0] for i in range(start_index, end_index + 1)]

    def _band_metrics(
        self, end_shift: int, std_mult: float
    ) -> tuple[float, float, float, float, float] | None:
        window = self._closes_window(end_shift)
        if len(window) < self.bb_period:
            return None
        close_v = self._history[-1 - end_shift][0]
        lower, middle, upper = _bollinger_bands(window, std_mult)
        percent_b = _percent_b(close_v, lower, upper)
        bbw = _bb_width_pct(lower, middle, upper)
        return lower, middle, upper, percent_b, bbw

    def _is_local_low(self, shift: int) -> bool:
        if shift < self.pivot_bars_after:
            return False
        if shift + self.pivot_bars_before >= len(self._history):
            return False
        center = self._pivot_price(shift, for_low=True)
        older_violation = any(
            self._pivot_price(shift + offset, for_low=True) <= center
            for offset in range(1, self.pivot_bars_before + 1)
        )
        if older_violation:
            return False
        newer_violation = any(
            self._pivot_price(shift - offset, for_low=True) <= center
            for offset in range(1, self.pivot_bars_after + 1)
        )
        return not newer_violation

    def _is_bottom_formation_bb(self, shift: int) -> bool:
        if not self._is_local_low(shift):
            return False
        wide = self._band_metrics(shift, self.bb_deviation_wide)
        if wide is None:
            return False
        _lower, _middle, _upper, percent_b_wide, bbw_wide = wide
        return bbw_wide > self.bbw_threshold and percent_b_wide < self.bbr_wide_threshold

    def _find_bars_since_bottom(self) -> int:
        return next(
            (
                bars_ago
                for bars_ago in range(
                    self.min_bars_since_bottom, self.max_bars_since_bottom + 1
                )
                if self._is_bottom_formation_bb(bars_ago)
            ),
            -1,
        )

    def _entry_reclaim_passes(self) -> bool:
        narrow = self._band_metrics(0, self.bb_deviation_narrow)
        if narrow is None:
            return False
        _lower, _middle, _upper, percent_b, _bbw = narrow
        return self.min_percent_b <= percent_b <= self.max_percent_b

    def _long_entry_signal(self) -> bool:
        bars_since = self._find_bars_since_bottom()
        return bars_since >= 0 and self._entry_reclaim_passes()

    def _apply_position_rules(self) -> int:
        if self._position == 1:
            self._bars_in_position += 1
            if self._bars_in_position >= self.exit_bars:
                self._position = 0
                self._bars_in_position = 0

        if self._position == 0 and self._long_entry_signal():
            self._position = 1
            self._bars_in_position = 1

        return self._position

    def _compute_candle(self, candle: Candle) -> List[float]:
        self._history.append(
            (float(candle.close), float(candle.high), float(candle.low))
        )

        min_ready = (
            self.bb_period
            + self.max_bars_since_bottom
            + self.pivot_bars_before
            + self.pivot_bars_after
        )
        if len(self._history) < min_ready:
            self._position = 0
            self._bars_in_position = 0
            self.output.append(0.0)
            return [0.0]

        signal = float(self._apply_position_rules())
        self.output.append(signal)
        return [signal]
