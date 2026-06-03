"""
Trend pullback re-entry with ATR trailing exit (robust MACD-hook decomposition).

Captures the structural idea behind Algomatic's MACD hook on gold without MACD or
Hull MA layers:

  Regime  : close > EMA(trend_ema_period)           — macro uptrend filter
  Setup   : while flat in regime, arm on dip below EMA(pullback_ema_period)
            or close <= EMA(pullback) − atr_pullback_mult × ATR
  Entry   : armed + close reclaims above EMA(pullback) while still in regime
  Exit    : chandelier ATR trailing stop; optional max_hold_bars cap (0 = off)

Signal: 1.0 long, 0.0 flat.
"""

from __future__ import annotations

from typing import ClassVar, List

from nodes import BiasNode
from utils.compute.fast_nodes import compute_ema_fast
from utils.core.enums import TimeFrame, Ticker
from utils.core.models import Candle


def _ema_alpha(period: int) -> float:
    return 2.0 / (period + 1.0)


class PullbackContinuationSignal(BiasNode):
    """
    Long-only pullback continuation inside a macro EMA uptrend.

    Replaces MACD > 0 + histogram hook + Hull MA with EMA regime / dip / reclaim
    and ATR-based risk exit.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset(
        {
            "trendEmaPeriod",
            "pullbackEmaPeriod",
            "atrLen",
            "maxHoldBars",
        }
    )

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        trend_ema_period: int = 200,
        pullback_ema_period: int = 20,
        atr_len: int = 14,
        atr_mult: float = 3.0,
        atr_pullback_mult: float = 1.0,
        max_hold_bars: int = 0,
    ) -> None:
        super().__init__(ticker, tf)

        if trend_ema_period < 2:
            raise ValueError("trend_ema_period must be >= 2")
        if pullback_ema_period < 2:
            raise ValueError("pullback_ema_period must be >= 2")
        if trend_ema_period <= pullback_ema_period:
            raise ValueError("trend_ema_period must be > pullback_ema_period")
        if atr_len < 1:
            raise ValueError("atr_len must be >= 1")
        if not (0.0 < atr_mult <= 20.0):
            raise ValueError("atr_mult must be in (0, 20]")
        if atr_pullback_mult < 0.0:
            raise ValueError("atr_pullback_mult must be >= 0")
        if max_hold_bars < 0:
            raise ValueError("max_hold_bars must be >= 0 (0 disables time cap)")

        self.trend_ema_period = trend_ema_period
        self.pullback_ema_period = pullback_ema_period
        self.atr_len = atr_len
        self.atr_mult = atr_mult
        self.atr_pullback_mult = atr_pullback_mult
        self.max_hold_bars = max_hold_bars

        self.module_name = "pullback_continuation_signal"
        self.output_features = ["signal"]
        self.params = {
            "trendEmaPeriod": trend_ema_period,
            "pullbackEmaPeriod": pullback_ema_period,
            "atrLen": atr_len,
            "atrMult": atr_mult,
            "atrPullbackMult": atr_pullback_mult,
            **({"maxHoldBars": max_hold_bars} if max_hold_bars > 0 else {}),
        }

        self.front_bad = trend_ema_period * 2 + atr_len + 5

        self._alpha_trend = _ema_alpha(trend_ema_period)
        self._alpha_pullback = _ema_alpha(pullback_ema_period)

        self._ema_trend: float | None = None
        self._ema_pullback: float | None = None
        self._atr_seed_trs: list[float] = []
        self._atr_val: float | None = None
        self._prev_close: float | None = None

        self._position = 0
        self._chandelier_stop: float | None = None
        self._pullback_armed = False
        self._bars_in_trade = 0
        self._n_bars = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _update_emas(self, close: float) -> None:
        is_first = self._n_bars == 1
        self._ema_trend = compute_ema_fast(
            close,
            self._ema_trend or close,
            self._alpha_trend,
            is_first,
        )
        self._ema_pullback = compute_ema_fast(
            close,
            self._ema_pullback or close,
            self._alpha_pullback,
            is_first,
        )

    def _update_atr(self, high: float, low: float, close: float) -> None:
        tr = (
            max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
            if self._prev_close is not None
            else high - low
        )
        self._prev_close = close

        if self._atr_val is None:
            self._atr_seed_trs.append(tr)
            if len(self._atr_seed_trs) >= self.atr_len:
                self._atr_val = sum(self._atr_seed_trs) / len(self._atr_seed_trs)
                self._atr_seed_trs.clear()
            return

        self._atr_val = (self._atr_val * (self.atr_len - 1) + tr) / self.atr_len

    def _regime_ok(self, close: float) -> bool:
        return self._ema_trend is not None and close > self._ema_trend

    def _dip_arms_pullback(self, close: float, atr: float) -> bool:
        if self._ema_pullback is None:
            return False
        below_fast = close < self._ema_pullback
        atr_extension = (
            self.atr_pullback_mult > 0.0
            and close <= self._ema_pullback - self.atr_pullback_mult * atr
        )
        return below_fast or atr_extension

    def _apply_exits(self, low: float) -> None:
        if self._position != 1:
            return

        if self._chandelier_stop is not None and low < self._chandelier_stop:
            self._position = 0
            self._chandelier_stop = None
            self._bars_in_trade = 0
            self._pullback_armed = False
            return

        if self.max_hold_bars > 0:
            self._bars_in_trade += 1
            if self._bars_in_trade >= self.max_hold_bars:
                self._position = 0
                self._chandelier_stop = None
                self._bars_in_trade = 0
                self._pullback_armed = False

    def _apply_flat_logic(self, close: float, atr: float) -> None:
        if self._position != 0:
            return

        if not self._regime_ok(close):
            self._pullback_armed = False
            return

        if self._dip_arms_pullback(close, atr):
            self._pullback_armed = True
            return

        if (
            self._pullback_armed
            and self._ema_pullback is not None
            and close > self._ema_pullback
        ):
            self._position = 1
            self._pullback_armed = False
            self._chandelier_stop = close - self.atr_mult * atr
            self._bars_in_trade = 1

    def _trail_stop(self, close: float, atr: float) -> None:
        if self._position == 1 and self._chandelier_stop is not None:
            self._chandelier_stop = max(
                self._chandelier_stop,
                close - self.atr_mult * atr,
            )

    def _compute_candle(self, candle: Candle) -> List[float]:
        self._n_bars += 1
        close = float(candle.close)
        high = float(candle.high)
        low = float(candle.low)

        self._update_emas(close)
        self._update_atr(high, low, close)

        if self._n_bars < self.front_bad or self._atr_val is None:
            self.output.append(0.0)
            return [0.0]

        atr = self._atr_val
        self._apply_exits(low)
        self._apply_flat_logic(close, atr)
        self._trail_stop(close, atr)

        signal = float(self._position)
        self.output.append(signal)
        return [signal]
