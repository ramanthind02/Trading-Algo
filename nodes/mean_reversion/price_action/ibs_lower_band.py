"""
IBS lower-band long bias node

Rolling mean of (High - Low) over hl_mean_lookback; IBS = (Close - Low) / (High - Low);
lower band = rolling max(High) over band_high_lookback minus band_width_mult times that
mean range. Long-only: enter when Close is below the lower band and IBS is below
ibs_entry_max; exit when Close is above the prior bar's High.

Output: 1.0 (long) or 0.0 (flat). Warmup: 0.0.
"""

from __future__ import annotations

from collections import deque
from typing import ClassVar, List

from nodes import BiasNode
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


class IBSLowerBand(BiasNode):
    """
    Discrete long/flat signals from a Donchian-style lower band and weak IBS filter.

    Parameters
    ----------
    hl_mean_lookback
        Bars for rolling mean of (High - Low). Default 25.
    band_high_lookback
        Bars for rolling max of High in the lower-band formula. Default 10.
    band_width_mult
        Multiplier on the mean range subtracted from rolling max High. Default 2.5.
    ibs_entry_max
        Enter long only if IBS is strictly below this (default 0.3).
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset(
        {"hlMeanLookback", "bandHighLookback"}
    )

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        hl_mean_lookback: int = 25,
        band_high_lookback: int = 10,
        band_width_mult: float = 2.5,
        ibs_entry_max: float = 0.3,
    ) -> None:
        super().__init__(ticker, tf)

        if hl_mean_lookback < 1:
            raise ValueError("hl_mean_lookback must be >= 1")
        if band_high_lookback < 1:
            raise ValueError("band_high_lookback must be >= 1")
        if band_width_mult <= 0.0:
            raise ValueError("band_width_mult must be > 0")
        if not (0.0 < ibs_entry_max < 1.0):
            raise ValueError("ibs_entry_max must be in (0, 1)")

        self.hl_mean_lookback = hl_mean_lookback
        self.band_high_lookback = band_high_lookback
        self.band_width_mult = band_width_mult
        self.ibs_entry_max = ibs_entry_max

        self.module_name = "ibslowerband"
        self.output_features = ["signal"]
        self.params = {
            "hlMeanLookback": hl_mean_lookback,
            "bandHighLookback": band_high_lookback,
            "bandWidthMult": band_width_mult,
            "ibsEntryMax": ibs_entry_max,
        }

        self.front_bad = max(hl_mean_lookback, band_high_lookback)

        self._hl_ranges: deque[float] = deque(maxlen=hl_mean_lookback)
        self._highs: deque[float] = deque(maxlen=band_high_lookback)
        self._hl_sum = 0.0
        self._position = 0
        self._prev_high = 0.0
        self._have_prev_high = False

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _mean_hl(self) -> float:
        n = len(self._hl_ranges)
        return self._hl_sum / float(n) if n else 0.0

    def _append_bar(self, high: float, low: float) -> None:
        hl = high - low
        if hl < 0.0:
            hl = 0.0
        if len(self._hl_ranges) == self._hl_ranges.maxlen:
            dropped = self._hl_ranges[0]
            self._hl_sum -= dropped
        self._hl_ranges.append(hl)
        self._hl_sum += hl
        self._highs.append(high)

    def _compute_candle(self, candle: Candle) -> List[float]:
        h = float(candle.high)
        lo = float(candle.low)
        c = float(candle.close)

        self._append_bar(h, lo)

        if len(self._hl_ranges) < self.hl_mean_lookback or len(
            self._highs
        ) < self.band_high_lookback:
            self._position = 0
            self._prev_high = h
            self._have_prev_high = True
            self.output.append(0.0)
            return [0.0]

        mean_hl = self._mean_hl()
        max_high = max(self._highs)
        lower_band = max_high - self.band_width_mult * mean_hl
        span = h - lo
        ibs = (c - lo) / span if span > 0.0 else 0.5

        if self._position == 1 and self._have_prev_high and c > self._prev_high:
            self._position = 0

        if (
            self._position == 0
            and c < lower_band
            and ibs < self.ibs_entry_max
        ):
            self._position = 1

        self._prev_high = h
        self._have_prev_high = True

        sig = float(self._position)
        self.output.append(sig)
        return [sig]
