"""
Linear regression slope "hook" mean-reversion signal (Algomatic-style).

Daily-style rule set: 3-period OLS slope of typical price (H+L+C)/3;
hook when today's slope > yesterday's and yesterday's < two days ago;
long entry when hook holds and the bar is red (close < open). Exit when
close > SMA(close, ma_period) or close > current typical price.

Reference: Algomatic Trading — mean-reversion with linear regression slope hook.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import ClassVar, Deque, List

import numpy as np

from nodes import BiasNode
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


class LrsHookExitVariant(Enum):
    """Which published exit rule to apply while long."""

    MA5 = "ma5"
    """Exit when close > simple moving average of close (default period 5)."""

    TYPICAL_PRICE = "typical_price"
    """Exit when close > this bar's typical price (H+L+C)/3."""


class FridayEntryPolicy(Enum):
    """Optional filter to skip new longs on Friday (weekend gap exposure)."""

    ALLOW = "allow"
    SKIP = "skip"


def _ols_slope(values: List[float]) -> float:
    """Univariate OLS slope over x = 0..n-1; matches functime-style indexing."""
    n = len(values)
    if n < 2:
        return 0.0
    y = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(y)):
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x_mean = (n - 1) / 2.0
    y_mean = float(y.mean())
    xc = x - x_mean
    denom = float(np.dot(xc, xc))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(xc, y - y_mean) / denom)


def _coerce_exit_variant(raw: LrsHookExitVariant | str) -> LrsHookExitVariant:
    if isinstance(raw, LrsHookExitVariant):
        return raw
    key = str(raw).strip().lower()
    for member in LrsHookExitVariant:
        if member.value == key or member.name.lower() == key:
            return member
    raise ValueError(
        "exit_variant must be LrsHookExitVariant or one of: "
        f"{', '.join(m.value for m in LrsHookExitVariant)}"
    )


def _coerce_friday_policy(raw: FridayEntryPolicy | str) -> FridayEntryPolicy:
    if isinstance(raw, FridayEntryPolicy):
        return raw
    key = str(raw).strip().lower()
    for member in FridayEntryPolicy:
        if member.value == key or member.name.lower() == key:
            return member
    raise ValueError(
        "friday_entry_policy must be FridayEntryPolicy or one of: "
        f"{', '.join(m.value for m in FridayEntryPolicy)}"
    )


class LrsHookSignal(BiasNode):
    """
    Long-only discrete signal from typical-price LRS hook + red candle,
    with MA or typical-price exit (Algomatic Strategy #4 style).

    Stateful: 1 when long, 0 when flat. Evaluates exit before entry on each bar.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset(
        {"lrsPeriod", "maPeriod"}
    )

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        lrs_period: int = 3,
        ma_period: int = 5,
        exit_variant: LrsHookExitVariant | str = LrsHookExitVariant.MA5,
        friday_entry_policy: FridayEntryPolicy | str = FridayEntryPolicy.ALLOW,
    ) -> None:
        super().__init__(ticker, tf)

        if lrs_period < 2:
            raise ValueError("lrs_period must be >= 2")
        if ma_period < 1:
            raise ValueError("ma_period must be >= 1")

        self.lrs_period = lrs_period
        self.ma_period = ma_period
        self.exit_variant = _coerce_exit_variant(exit_variant)
        self.friday_entry_policy = _coerce_friday_policy(friday_entry_policy)

        self.module_name = "lrshooksignal"
        self.output_features = ["signal"]
        self.params = {
            "lrsPeriod": lrs_period,
            "maPeriod": ma_period,
            "exitVariant": self.exit_variant.value,
            "fridayEntryPolicy": self.friday_entry_policy.value,
        }

        self._min_bars = max(self.lrs_period + 2, self.ma_period)
        self.front_bad = self._min_bars

        self._tp_hist: Deque[float] = deque(maxlen=self._min_bars)
        self._close_hist: Deque[float] = deque(maxlen=self.ma_period)
        self._position = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _three_slopes(self) -> tuple[float, float, float] | None:
        """Return (lrs_curr, lrs_prev, lrs_prev2) or None if history insufficient."""
        if len(self._tp_hist) < self.lrs_period + 2:
            return None
        seq = list(self._tp_hist)
        w = self.lrs_period
        lrs_curr = _ols_slope(seq[-w:])
        lrs_prev = _ols_slope(seq[-w - 1 : -1])
        lrs_prev2 = _ols_slope(seq[-w - 2 : -2])
        return (lrs_curr, lrs_prev, lrs_prev2)

    def _ma_close(self) -> float | None:
        if len(self._close_hist) < self.ma_period:
            return None
        return float(sum(self._close_hist) / self.ma_period)

    def _compute_candle(self, candle: Candle) -> List[float]:
        typical = (candle.high + candle.low + candle.close) / 3.0
        self._tp_hist.append(float(typical))
        self._close_hist.append(float(candle.close))

        if len(self._tp_hist) < self._min_bars:
            self.output.append(0.0)
            return [0.0]

        slopes = self._three_slopes()
        if slopes is None:
            self.output.append(0.0)
            return [0.0]

        lrs_curr, lrs_prev, lrs_prev2 = slopes
        hook = lrs_curr > lrs_prev and lrs_prev < lrs_prev2
        red = candle.close < candle.open

        ma_val = self._ma_close()

        if self._position == 1:
            exit_fire = False
            if self.exit_variant == LrsHookExitVariant.MA5:
                if ma_val is not None and candle.close > ma_val:
                    exit_fire = True
            elif candle.close > typical:
                exit_fire = True
            if exit_fire:
                self._position = 0

        if self._position == 0 and hook and red:
            friday = candle.datetime.weekday() == 4
            if not (self.friday_entry_policy == FridayEntryPolicy.SKIP and friday):
                self._position = 1

        sig = float(self._position)
        self.output.append(sig)
        return [sig]
