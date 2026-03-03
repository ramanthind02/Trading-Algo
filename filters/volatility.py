"""ATR-based volatility regime filter.

Gates bias-node signals based on whether the current ATR percentile rank
is above or below a configurable threshold — useful for separating
trending/volatile environments from quiet/ranging ones.

Example
-------
>>> from filters import FilterSpec, create_filter
>>> spec = FilterSpec(filter_name='vol', params={'atr_period': 14, 'regime': 'high'})
>>> filt = create_filter(spec)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict

from filters import SignalFilter, register_filter
from utils.core.models import Candle


@register_filter
class VolatilityFilter(SignalFilter):
    """ATR percentile-rank volatility filter.

    Computes ATR incrementally and tracks a rolling window of ATR values to
    derive a percentile rank.  The filter passes when the rank satisfies
    the *regime* condition:

    * ``'high'`` — pass when ``percentile_rank >= threshold``
    * ``'low'``  — pass when ``percentile_rank < threshold``

    Parameters
    ----------
    atr_period : int
        Smoothing period for the ATR (Wilder-style EMA).  Default ``14``.
    rank_period : int
        Rolling window length for the percentile-rank calculation.
        Default ``252`` (≈ 1 year of daily bars).
    regime : str
        ``'high'`` or ``'low'``.  Default ``'high'``.
    threshold : float
        Percentile boundary in ``[0, 1]``.  Default ``0.5`` (median).
    """

    filter_name: str = "vol"

    def __init__(
        self,
        atr_period: int = 14,
        rank_period: int = 252,
        regime: str = "high",
        threshold: float = 0.5,
    ) -> None:
        if atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {atr_period}")
        if rank_period < 1:
            raise ValueError(f"rank_period must be >= 1, got {rank_period}")
        regime = regime.lower()
        if regime not in ("high", "low"):
            raise ValueError(f"regime must be 'high' or 'low', got '{regime}'")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        self.atr_period = atr_period
        self.rank_period = rank_period
        self.regime = regime
        self.threshold = threshold

        self.params: Dict[str, Any] = {
            "atr_period": atr_period,
            "rank_period": rank_period,
            "regime": regime,
            "threshold": threshold,
        }
        self.warmup: int = max(atr_period, rank_period)

        # ---- internal state ------------------------------------------------
        self._prev_close: float = 0.0
        self._n_candles: int = 0

        # Wilder-smoothed ATR state
        self._atr: float = 0.0
        self._tr_sum: float = 0.0  # accumulator for initial ATR seed

        # Rolling ATR history for percentile rank
        self._atr_history: deque[float] = deque(maxlen=rank_period)

        # Latest decision
        self._passes: bool = True

    # ---- SignalFilter interface --------------------------------------------

    def update(self, candle: Candle) -> None:  # noqa: D401
        self._n_candles += 1

        # True Range
        if self._n_candles == 1:
            tr = candle.high - candle.low
        else:
            tr = max(
                candle.high - candle.low,
                abs(candle.high - self._prev_close),
                abs(candle.low - self._prev_close),
            )
        self._prev_close = candle.close

        # Wilder-smoothed ATR
        if self._n_candles <= self.atr_period:
            self._tr_sum += tr
            if self._n_candles == self.atr_period:
                self._atr = self._tr_sum / self.atr_period
        else:
            self._atr = (self._atr * (self.atr_period - 1) + tr) / self.atr_period

        # Only record ATR once the initial seed is ready
        if self._n_candles >= self.atr_period:
            self._atr_history.append(self._atr)

        # Percentile rank requires a full window
        if len(self._atr_history) < self.rank_period:
            self._passes = True  # pass-through during warmup
            return

        # Compute percentile rank: fraction of history values <= current ATR
        current_atr = self._atr
        count_leq = sum(1 for v in self._atr_history if v <= current_atr)
        pct_rank = count_leq / len(self._atr_history)

        if self.regime == "high":
            self._passes = pct_rank >= self.threshold
        else:
            self._passes = pct_rank < self.threshold

    def should_pass(self) -> bool:  # noqa: D401
        return self._passes

    def reset(self) -> None:
        self._prev_close = 0.0
        self._n_candles = 0
        self._atr = 0.0
        self._tr_sum = 0.0
        self._atr_history.clear()
        self._passes = True
