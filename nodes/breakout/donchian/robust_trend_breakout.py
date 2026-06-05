"""
Robust Trend Breakout bias node — simplified Gold ATH-style system.

Architecture (3 core parameters + 2 optional):
  Regime  : close > EMA(ema_period)
  Entry   : close > highest(high, lookback)[prev bar]  [Donchian breakout]
  Exit    : chandelier ATR trailing stop ONLY
             trailing_stop = max(prev_stop, close − atr_mult × ATR(atr_len))

Optional robustness upgrades:
  atr_vol_filter : N > 0 → only trade when ATR(atr_len) > SMA(ATR, N)
                   Filters out low-volatility chop regimes where breakouts fail.
  cooldown_bars  : N > 0 → after a stop-out, wait N bars before next entry.
                   Reduces repeated false breakouts in sideways ranges.

Design rationale vs GoldAthMomentumBreakout:
  - EMA fast/slow stack replaced with single macro EMA (ema_period 100–200).
  - RSI overbought exit removed — trend systems profit from holding big runs.
  - EMA-break exit removed — ATR trailing stop alone handles all exits.
  → Fewer parameters, less correlation between exit signals, more right-tail skew.

Signal: ``1.0`` (long), ``-1.0`` (short), ``0.0`` (flat); mode via ``strategy_mode``.
"""
from __future__ import annotations

import collections
import math
from typing import ClassVar, List

from nodes import BiasNode
from lib.core.enums import DirectionInput, TimeFrame, Ticker, coerce_direction
from lib.core.models import Candle


class RobustTrendBreakout(BiasNode):
    """
    Trend breakout: EMA regime + Donchian entry + ATR chandelier exit.

    Long  — close > EMA, breakout above prior ``lookback`` highs; trail stop below.
    Short — close < EMA, breakdown below prior ``lookback`` lows; trail stop above.

    Parameters
    ----------
    lookback        : Donchian entry channel lookback in bars (default 55).
    ema_period      : Single macro EMA period for regime filter (default 100).
    atr_len         : ATR period for chandelier trailing stop (default 14).
    atr_mult        : ATR multiplier for the chandelier stop (default 3.5).
    atr_vol_filter  : 0 = disabled.  N > 0 = only enter when
                      ATR(atr_len) > SMA(ATR(atr_len), N).  Default 0.
    cooldown_bars   : 0 = disabled.  N > 0 = bars to wait after a
                      stop-out before the next entry is allowed.  Default 0.
    strategy_mode   : ``"long"``, ``"short"``, or ``"long_short"`` (default ``"long"``).
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset(
        {"lookback", "ema_period", "atr_len", "atr_vol_filter"}
    )

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        lookback: int = 55,
        ema_period: int = 100,
        atr_len: int = 14,
        atr_mult: float = 3.5,
        atr_vol_filter: int = 0,
        cooldown_bars: int = 0,
        strategy_mode: DirectionInput = "long",
    ) -> None:
        super().__init__(ticker, tf)

        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if ema_period < 2:
            raise ValueError("ema_period must be >= 2")
        if atr_len < 1:
            raise ValueError("atr_len must be >= 1")
        if not (0.0 < atr_mult <= 20.0):
            raise ValueError("atr_mult must be in (0, 20]")
        if atr_vol_filter < 0:
            raise ValueError("atr_vol_filter must be 0 (off) or a positive integer")
        if cooldown_bars < 0:
            raise ValueError("cooldown_bars must be >= 0")

        self.lookback = lookback
        self.ema_period = ema_period
        self.atr_len = atr_len
        self.atr_mult = atr_mult
        self.atr_vol_filter = atr_vol_filter
        self.cooldown_bars = cooldown_bars
        self.strategy_mode = coerce_direction(strategy_mode, field_name="strategy_mode").value

        self.module_name = "robust_trend_breakout"
        self.output_features = ["signal"]
        self.params = {
            "lookback": lookback,
            "ema_period": ema_period,
            "atr_len": atr_len,
            "atr_mult": atr_mult,
            **({"atr_vol_filter": atr_vol_filter} if atr_vol_filter > 0 else {}),
            **({"cooldown_bars": cooldown_bars} if cooldown_bars > 0 else {}),
            **({"strategy_mode": self.strategy_mode} if self.strategy_mode != "long" else {}),
        }

        # Warmup: 2× EMA period gives good convergence + ATR/Donchian headroom
        self.front_bad = ema_period * 2 + max(atr_len, lookback,
                                               atr_vol_filter if atr_vol_filter > 0 else 0) + 5

        # Candle history for Donchian lookback (previous `lookback` bars before current)
        self._candles_history: collections.deque[Candle] = collections.deque(
            maxlen=lookback + 10
        )

        # ── EMA state ──
        self._alpha_ema: float = 2.0 / (ema_period + 1)
        self._ema_val: float | None = None

        # ── ATR state (Wilder's smoothing after SMA seed) ──
        self._atr_seed_trs: list[float] = []
        self._atr_val: float | None = None
        self._prev_close: float | None = None

        # ── ATR volatility filter (SMA of ATR values) ──
        self._atr_history: collections.deque[float] = (
            collections.deque(maxlen=atr_vol_filter + 5)
            if atr_vol_filter > 0
            else collections.deque(maxlen=1)   # unused but avoids None checks
        )

        # ── Position state ──
        self._position: int = 0
        self._chandelier_stop: float | None = None

        # ── Cooldown counter (counts down from cooldown_bars after a stop-out) ──
        self._cooldown_remaining: int = 0

        # ── Bar counter ──
        self._n_bars: int = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    # ------------------------------------------------------------------
    # Incremental indicator helpers
    # ------------------------------------------------------------------

    def _update_ema(self, close: float) -> None:
        if self._ema_val is None:
            self._ema_val = close
        else:
            self._ema_val = self._alpha_ema * close + (1.0 - self._alpha_ema) * self._ema_val

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
        else:
            self._atr_val = (self._atr_val * (self.atr_len - 1) + tr) / self.atr_len

        if self._atr_val is not None and self.atr_vol_filter > 0:
            self._atr_history.append(self._atr_val)

    def _donchian_entry_level(self) -> float:
        """Highest high of the previous ``lookback`` bars (current bar excluded)."""
        history = list(self._candles_history)
        prior = history[-(self.lookback + 1) : -1]
        if len(prior) < self.lookback:
            return math.inf
        return max(float(c.high) for c in prior)

    def _donchian_short_entry_level(self) -> float:
        """Lowest low of the previous ``lookback`` bars (current bar excluded)."""
        history = list(self._candles_history)
        prior = history[-(self.lookback + 1) : -1]
        if len(prior) < self.lookback:
            return -math.inf
        return min(float(c.low) for c in prior)

    def _vol_filter_ok(self) -> bool:
        """True if ATR is above its recent SMA, or if the filter is disabled."""
        if self.atr_vol_filter == 0 or self._atr_val is None:
            return True
        if len(self._atr_history) < self.atr_vol_filter:
            return True  # not enough data yet — don't block entry
        atr_ma = sum(list(self._atr_history)[-self.atr_vol_filter :]) / self.atr_vol_filter
        return self._atr_val >= atr_ma

    def _stop_out(self) -> None:
        self._position = 0
        self._chandelier_stop = None
        if self.cooldown_bars > 0:
            self._cooldown_remaining = self.cooldown_bars

    def _allows_long(self) -> bool:
        return self.strategy_mode in {"long", "long_short"}

    def _allows_short(self) -> bool:
        return self.strategy_mode in {"short", "long_short"}

    # ------------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------------

    def _compute_candle(self, candle: Candle) -> List[float]:
        self._n_bars += 1
        self._candles_history.append(candle)

        close = float(candle.close)
        high = float(candle.high)
        low = float(candle.low)

        # Update indicators (causal — current bar data only)
        self._update_ema(close)
        self._update_atr(high, low, close)

        # Suppress signal during warmup
        if self._n_bars < self.front_bad:
            self.output.append(0.0)
            return [0.0]

        ema = self._ema_val
        atr = self._atr_val or 0.0

        # Decrement cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # ── ATR chandelier exits ──
        if self._position == 1 and self._chandelier_stop is not None and low < self._chandelier_stop:
            self._stop_out()
        elif self._position == -1 and self._chandelier_stop is not None and high > self._chandelier_stop:
            self._stop_out()

        # ── Entries (flat only) ──
        dc_long = self._donchian_entry_level()
        dc_short = self._donchian_short_entry_level()
        vol_ok = self._vol_filter_ok()
        cooldown_ok = self._cooldown_remaining == 0

        if self._position == 0 and ema is not None and vol_ok and cooldown_ok:
            if (
                self._allows_long()
                and close > ema
                and dc_long < math.inf
                and close > dc_long
            ):
                self._position = 1
                self._chandelier_stop = close - self.atr_mult * atr
            elif (
                self._allows_short()
                and close < ema
                and dc_short > -math.inf
                and close < dc_short
            ):
                self._position = -1
                self._chandelier_stop = close + self.atr_mult * atr

        # ── Trail chandelier stops (long up, short down) ──
        if self._position == 1 and self._chandelier_stop is not None:
            self._chandelier_stop = max(self._chandelier_stop, close - self.atr_mult * atr)
        elif self._position == -1 and self._chandelier_stop is not None:
            self._chandelier_stop = min(self._chandelier_stop, close + self.atr_mult * atr)

        signal = float(self._position)
        self.output.append(signal)
        return [signal]
