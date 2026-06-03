"""
ADX-RSI Mean Reversion — simplified, robust alternative to RegimeLrsiSignal.

Three-layer entry filter (all must pass at entry; exits are unconditional):
  1. Trend direction (SMA): long entries only above SMA(trend_period),
     short entries only below.  Set trend_period=0 to disable.
  2. Regime quality (ADX): entries blocked when ADX >= adx_threshold (trending).
  3. Signal timing (RSI): enter on RSI cross through oversold / overbought.

Filter exploration showed SMA(252) entry-only gate raises IS Sharpe from 0.12 → 0.30.
This bakes that insight in as a first-class parameter rather than a FilterGateNode wrapper.
"""

from __future__ import annotations

from collections import deque
from typing import ClassVar, List

import numpy as np

from nodes import BiasNode
from utils.compute.rsi_helpers import compute_rsi_initial, update_rsi
from utils.core.enums import DirectionInput, Ticker, TimeFrame, coerce_direction
from utils.core.models import Candle

_RSI_SEED_SIZE = 256


class AdxRsiMR(BiasNode):
    """
    Trend-directional ADX-Regime RSI Mean-Reversion.

    Entry conditions (all three must pass; no existing position):
      Long  — close > SMA(trend_period) [if enabled]
               AND  ADX < adx_threshold
               AND  RSI crosses DOWN through oversold
      Short — close < SMA(trend_period) [if enabled]
               AND  ADX < adx_threshold
               AND  RSI crosses UP through overbought

    Exit conditions (unconditional — trend/ADX gate does not force exits):
      1. bars_in_position >= exit_bars   (time exit)
      2. RSI crosses back through 50     (mean recovery)

    Parameters
    ----------
    adx_period    : Wilder ADX period (default 14).
    adx_threshold : ADX must be below this for entries (default 25.0).
    rsi_period    : RSI lookback (default 5).
    oversold      : RSI cross-down threshold for longs (default 30.0).
    overbought    : RSI cross-up threshold for shorts (default 70.0).
    exit_bars     : Hard time exit after N bars (default 2).
    trend_period  : SMA period for directional gate; 0 = disabled (default 252).
    strategy_mode : ``"long"``, ``"short"``, or ``"long_short"`` (default ``"long_short"``).
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset(
        {"adxPeriod", "rsiPeriod", "trendPeriod"}
    )

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        rsi_period: int = 3,
        oversold: float = 30.0,
        overbought: float = 70.0,
        exit_bars: int = 2,
        trend_period: int = 252,
        strategy_mode: DirectionInput = "long_short",
    ) -> None:
        super().__init__(ticker, tf)

        if adx_period < 2:
            raise ValueError(f"adx_period must be >= 2, got {adx_period}")
        if rsi_period < 2:
            raise ValueError(f"rsi_period must be >= 2, got {rsi_period}")
        if not (0.0 < oversold < 50.0):
            raise ValueError(f"oversold must be in (0, 50), got {oversold}")
        if not (50.0 < overbought < 100.0):
            raise ValueError(f"overbought must be in (50, 100), got {overbought}")
        if exit_bars < 1:
            raise ValueError(f"exit_bars must be >= 1, got {exit_bars}")
        if trend_period < 0:
            raise ValueError(f"trend_period must be >= 0, got {trend_period}")

        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_bars = exit_bars
        self.trend_period = trend_period
        self.strategy_mode = coerce_direction(strategy_mode).value

        self.module_name = "adx_rsi_mr"
        self.output_features = ["signal"]
        self.params = {
            "adxPeriod": adx_period,
            "adxThreshold": adx_threshold,
            "rsiPeriod": rsi_period,
            "oversold": oversold,
            "overbought": overbought,
            "exitBars": exit_bars,
            **({"trendPeriod": trend_period} if trend_period > 0 else {}),
            "strategyMode": self.strategy_mode,
        }

        self.front_bad = max(2 * adx_period, rsi_period + 1, trend_period)

        # ── ADX state ────────────────────────────────────────────────────────
        self._prev_h: float | None = None
        self._prev_l: float | None = None
        self._prev_c: float | None = None
        self._tr_seeded: bool = False
        self._sum_tr = self._sum_pdm = self._sum_mdm = 0.0
        self._acc = 0
        self._s_tr = self._s_pdm = self._s_mdm = 0.0
        self._adx_val: float | None = None
        self._sum_dx = 0.0
        self._dx_bars = 0

        # ── RSI state ────────────────────────────────────────────────────────
        self._close_buf: deque[float] = deque(maxlen=max(_RSI_SEED_SIZE, trend_period + 1))
        self._rsi_upsum = 0.0
        self._rsi_dnsum = 0.0
        self._rsi_ready = False
        self._prev_close_rsi: float | None = None
        self._curr_rsi: float = 50.0
        self._prev_rsi: float = 50.0

        # ── SMA trend state ──────────────────────────────────────────────────
        self._sma_sum: float = 0.0
        self._sma_count: int = 0

        # ── Position state ───────────────────────────────────────────────────
        self.position: int = 0
        self.bars_in_position: int = 0
        self._n: int = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    # ── ADX ──────────────────────────────────────────────────────────────────

    def _step_adx(self, h: float, l_: float, c: float) -> float | None:
        if self._prev_c is None:
            self._prev_h, self._prev_l, self._prev_c = h, l_, c
            return None

        tr = max(h - l_, abs(h - self._prev_c), abs(l_ - self._prev_c))
        up = h - self._prev_h
        dn = self._prev_l - l_
        pdm = up if up > dn and up > 0.0 else 0.0
        mdm = dn if dn > up and dn > 0.0 else 0.0
        self._prev_h, self._prev_l, self._prev_c = h, l_, c

        if not self._tr_seeded:
            self._sum_tr += tr
            self._sum_pdm += pdm
            self._sum_mdm += mdm
            self._acc += 1
            if self._acc < self.adx_period:
                return None
            self._s_tr, self._s_pdm, self._s_mdm = self._sum_tr, self._sum_pdm, self._sum_mdm
            self._tr_seeded = True
        else:
            n = self.adx_period
            self._s_tr = self._s_tr - self._s_tr / n + tr
            self._s_pdm = self._s_pdm - self._s_pdm / n + pdm
            self._s_mdm = self._s_mdm - self._s_mdm / n + mdm

        dip = 100.0 * self._s_pdm / self._s_tr if self._s_tr > 0.0 else 0.0
        dim = 100.0 * self._s_mdm / self._s_tr if self._s_tr > 0.0 else 0.0
        di_sum = dip + dim
        dx = 100.0 * abs(dip - dim) / di_sum if di_sum > 0.0 else 0.0

        if self._adx_val is None:
            self._sum_dx += dx
            self._dx_bars += 1
            if self._dx_bars < self.adx_period:
                return None
            self._adx_val = self._sum_dx / self.adx_period
        else:
            n = self.adx_period
            self._adx_val = (self._adx_val * (n - 1) + dx) / n

        return self._adx_val

    # ── RSI ──────────────────────────────────────────────────────────────────

    def _step_rsi(self, close: float) -> float | None:
        self._close_buf.append(close)
        if not self._rsi_ready:
            if len(self._close_buf) <= self.rsi_period:
                return None
            prices = np.array(list(self._close_buf)[: self.rsi_period + 1], dtype=np.float64)
            self._rsi_upsum, self._rsi_dnsum = compute_rsi_initial(prices, self.rsi_period)
            self._prev_close_rsi = float(self._close_buf[-2])
            self._rsi_ready = True

        assert self._prev_close_rsi is not None
        self._rsi_upsum, self._rsi_dnsum, rsi = update_rsi(
            self._prev_close_rsi, close, self._rsi_upsum, self._rsi_dnsum, self.rsi_period
        )
        self._prev_close_rsi = close
        return float(rsi)

    # ── SMA trend direction ───────────────────────────────────────────────────

    def _trend_allows_long(self, close: float) -> bool:
        if self.trend_period == 0:
            return True
        buf = self._close_buf
        n = self.trend_period
        if len(buf) < n:
            return False
        sma = sum(list(buf)[-n:]) / n
        return close > sma

    def _trend_allows_short(self, close: float) -> bool:
        if self.trend_period == 0:
            return True
        buf = self._close_buf
        n = self.trend_period
        if len(buf) < n:
            return False
        sma = sum(list(buf)[-n:]) / n
        return close < sma

    # ── Position management ───────────────────────────────────────────────────

    def _apply_exits(self) -> None:
        if self.position == 0:
            return
        if self.bars_in_position >= self.exit_bars or (
            (self.position == 1 and self._curr_rsi > 50.0)
            or (self.position == -1 and self._curr_rsi < 50.0)
        ):
            self.position = 0
            self.bars_in_position = 0

    def _apply_entries(self, adx: float, close: float, long_sig: bool, short_sig: bool) -> None:
        if self.position != 0 or adx >= self.adx_threshold:
            return
        allow_long = self.strategy_mode in {"long", "long_short"}
        allow_short = self.strategy_mode in {"short", "long_short"}
        if long_sig and allow_long and self._trend_allows_long(close):
            self.position = 1
            self.bars_in_position = 0
        elif short_sig and not long_sig and allow_short and self._trend_allows_short(close):
            self.position = -1
            self.bars_in_position = 0

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _compute_candle(self, candle: Candle) -> List[float]:
        self._n += 1
        h, l_, c = float(candle.high), float(candle.low), float(candle.close)

        adx = self._step_adx(h, l_, c)
        rsi = self._step_rsi(c)

        if adx is None or rsi is None or self._n < self.front_bad:
            self.output.append(float(self.position))
            return [float(self.position)]

        self._prev_rsi, self._curr_rsi = self._curr_rsi, rsi

        long_cross = self._prev_rsi > self.oversold and self._curr_rsi <= self.oversold
        short_cross = self._prev_rsi < self.overbought and self._curr_rsi >= self.overbought

        if self.position != 0:
            self.bars_in_position += 1

        self._apply_exits()
        self._apply_entries(adx, c, long_cross, short_cross)

        out = float(self.position)
        self.output.append(out)
        return [out]
