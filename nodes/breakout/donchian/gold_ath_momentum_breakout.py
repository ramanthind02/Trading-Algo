"""
Gold ATH Momentum Breakout bias node.

Implements the three-indicator long-only system:
  - Entry : close breaks above N-bar Donchian high (previous bar)
             AND close > EMA(fast) AND EMA(fast) > EMA(slow)  [bullish EMA stack]
             AND currently flat
  - Exits (any one clears the position):
      1. Chandelier trailing stop  : low < trailing_stop
         trailing_stop = max(prev_stop, close − atr_mult × ATR(atr_len))
      2. RSI overbought crossunder : RSI(rsi_len) crosses below rsi_ob
      3. EMA break                 : close < EMA(fast)

Signal values: 1.0 (long), 0.0 (flat).

All incremental indicators use Wilder's smoothing for ATR/RSI after the initial
SMA seed, and standard EWM for EMAs.  No signals are emitted during the warmup
period (front_bad bars) so all indicators have converged before trading begins.
"""
from __future__ import annotations

import collections
import math
from typing import ClassVar, List

from nodes import BiasNode
from utils.core.enums import TimeFrame, Ticker
from utils.core.models import Candle


class GoldAthMomentumBreakout(BiasNode):
    """
    Long-only Donchian breakout with EMA-stack filter and three exit mechanisms.

    Parameters
    ----------
    lookback  : Donchian entry channel lookback in bars (default 20).
    ema_fast  : Fast EMA period (default 21).
    ema_slow  : Slow EMA period, must be > ema_fast (default 50).
    atr_len   : ATR period for the chandelier trailing stop (default 14).
    atr_mult  : ATR multiplier for the chandelier stop (default 3.0).
    rsi_len   : RSI period for the overbought exit (default 14).
    rsi_ob    : RSI level at which an upward crossunder triggers an exit (default 70).
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset(
        {"lookback", "ema_fast", "ema_slow", "atr_len", "rsi_len"}
    )

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        lookback: int = 20,
        ema_fast: int = 21,
        ema_slow: int = 50,
        atr_len: int = 14,
        atr_mult: float = 3.0,
        rsi_len: int = 14,
        rsi_ob: int = 70,
    ) -> None:
        super().__init__(ticker, tf)

        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        if ema_fast < 2:
            raise ValueError("ema_fast must be >= 2")
        if ema_slow <= ema_fast:
            raise ValueError("ema_slow must be strictly greater than ema_fast")
        if atr_len < 1:
            raise ValueError("atr_len must be >= 1")
        if rsi_len < 2:
            raise ValueError("rsi_len must be >= 2")
        if not (0.0 < atr_mult <= 20.0):
            raise ValueError("atr_mult must be in (0, 20]")
        if not (50 < rsi_ob <= 100):
            raise ValueError("rsi_ob must be in (50, 100]")

        self.lookback = lookback
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_len = atr_len
        self.atr_mult = atr_mult
        self.rsi_len = rsi_len
        self.rsi_ob = rsi_ob

        self.module_name = "gold_ath_momentum"
        self.output_features = ["signal"]
        self.params = {
            "lookback": lookback,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "atr_len": atr_len,
            "atr_mult": atr_mult,
            "rsi_len": rsi_len,
            "rsi_ob": rsi_ob,
        }

        # Warmup: 2× slow EMA period gives good convergence; then add room for
        # RSI / ATR initialisation windows.
        self.front_bad = ema_slow * 2 + max(atr_len, rsi_len, lookback) + 5

        # Candle history for Donchian channel (need lookback bars before current)
        self._candles_history: collections.deque[Candle] = collections.deque(
            maxlen=lookback + 10
        )

        # ── EMA state (seed with first close, converges within warmup window) ──
        self._alpha_fast: float = 2.0 / (ema_fast + 1)
        self._alpha_slow: float = 2.0 / (ema_slow + 1)
        self._ema_fast_val: float | None = None
        self._ema_slow_val: float | None = None

        # ── ATR state (Wilder's smoothing) ──
        # SMA seed: accumulate `atr_len` TRs, then switch to Wilder's rolling smooth.
        self._atr_seed_trs: list[float] = []
        self._atr_val: float | None = None
        self._prev_close: float | None = None

        # ── RSI state (Wilder's smoothing) ──
        # SMA seed: accumulate `rsi_len` price changes, then switch to Wilder's smooth.
        self._rsi_seed_gains: list[float] = []
        self._rsi_seed_losses: list[float] = []
        self._rsi_avg_gain: float | None = None
        self._rsi_avg_loss: float | None = None
        self._rsi_prev_close: float | None = None
        self._rsi_current: float | None = None   # RSI of current bar (for crossunder)
        self._rsi_previous: float | None = None  # RSI of previous bar (for crossunder)

        # ── Position state ──
        self._position: int = 0          # 0 = flat, 1 = long
        self._chandelier_stop: float | None = None

        # ── Bar counter ──
        self._n_bars: int = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    # ------------------------------------------------------------------
    # Incremental indicator helpers
    # ------------------------------------------------------------------

    def _update_ema(self, close: float) -> None:
        if self._ema_fast_val is None:
            self._ema_fast_val = close
            self._ema_slow_val = close
        else:
            self._ema_fast_val = (
                self._alpha_fast * close + (1.0 - self._alpha_fast) * self._ema_fast_val
            )
            self._ema_slow_val = (
                self._alpha_slow * close + (1.0 - self._alpha_slow) * self._ema_slow_val
            )

    def _update_atr(self, high: float, low: float, close: float) -> None:
        if self._prev_close is not None:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        else:
            tr = high - low
        self._prev_close = close

        if self._atr_val is None:
            self._atr_seed_trs.append(tr)
            if len(self._atr_seed_trs) >= self.atr_len:
                self._atr_val = sum(self._atr_seed_trs) / len(self._atr_seed_trs)
                self._atr_seed_trs.clear()
        else:
            # Wilder's smoothing: EMA with alpha = 1 / atr_len
            self._atr_val = (self._atr_val * (self.atr_len - 1) + tr) / self.atr_len

    def _update_rsi(self, close: float) -> None:
        if self._rsi_prev_close is None:
            self._rsi_prev_close = close
            return
        change = close - self._rsi_prev_close
        gain = change if change > 0.0 else 0.0
        loss = -change if change < 0.0 else 0.0
        self._rsi_prev_close = close

        if self._rsi_avg_gain is None:
            self._rsi_seed_gains.append(gain)
            self._rsi_seed_losses.append(loss)
            if len(self._rsi_seed_gains) >= self.rsi_len:
                self._rsi_avg_gain = sum(self._rsi_seed_gains) / self.rsi_len
                self._rsi_avg_loss = sum(self._rsi_seed_losses) / self.rsi_len
                self._rsi_seed_gains.clear()
                self._rsi_seed_losses.clear()
        else:
            assert self._rsi_avg_loss is not None
            self._rsi_avg_gain = (
                self._rsi_avg_gain * (self.rsi_len - 1) + gain
            ) / self.rsi_len
            self._rsi_avg_loss = (
                self._rsi_avg_loss * (self.rsi_len - 1) + loss
            ) / self.rsi_len

        if self._rsi_avg_gain is not None and self._rsi_avg_loss is not None:
            avg_loss = self._rsi_avg_loss
            avg_gain = self._rsi_avg_gain
            if avg_loss == 0.0:
                rsi_val = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_val = 100.0 - 100.0 / (1.0 + rs)
            self._rsi_previous = self._rsi_current
            self._rsi_current = rsi_val

    def _donchian_entry_level(self) -> float:
        """Highest high of the previous ``lookback`` bars (excludes current bar)."""
        history = list(self._candles_history)
        # history already includes the current bar (appended before this call)
        prior = history[-(self.lookback + 1) : -1]
        if len(prior) < self.lookback:
            return math.inf  # not enough data: block entry
        return max(float(c.high) for c in prior)

    # ------------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------------

    def _compute_candle(self, candle: Candle) -> List[float]:
        self._n_bars += 1
        self._candles_history.append(candle)

        close = float(candle.close)
        high = float(candle.high)
        low = float(candle.low)

        # Update all indicators first (causal: only current and prior bars used)
        self._update_ema(close)
        self._update_atr(high, low, close)
        self._update_rsi(close)

        # Emit flat during warmup (indicators haven't converged yet)
        if self._n_bars < self.front_bad:
            self.output.append(0.0)
            return [0.0]

        ema_f = self._ema_fast_val
        ema_s = self._ema_slow_val
        atr = self._atr_val or 0.0

        trend_up = (ema_f is not None and ema_s is not None
                    and close > ema_f and ema_f > ema_s)

        # ── Exits (evaluated before entry on the same bar) ──
        if self._position == 1 and self._chandelier_stop is not None:
            exit_trail = low < self._chandelier_stop
            exit_ema = ema_f is not None and close < ema_f
            exit_rsi = (
                self._rsi_current is not None
                and self._rsi_previous is not None
                and self._rsi_previous >= self.rsi_ob
                and self._rsi_current < self.rsi_ob
            )
            if exit_trail or exit_ema or exit_rsi:
                self._position = 0
                self._chandelier_stop = None

        # ── Entry ──
        dc_level = self._donchian_entry_level()
        if (
            self._position == 0
            and trend_up
            and dc_level < math.inf
            and close > dc_level
        ):
            self._position = 1
            self._chandelier_stop = close - self.atr_mult * atr

        # ── Update chandelier stop (trails price up, never moves down) ──
        if self._position == 1 and self._chandelier_stop is not None:
            new_stop = close - self.atr_mult * atr
            self._chandelier_stop = max(self._chandelier_stop, new_stop)

        signal = float(self._position)
        self.output.append(signal)
        return [signal]
