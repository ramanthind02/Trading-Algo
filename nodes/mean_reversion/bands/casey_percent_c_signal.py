"""
Casey Bands PercentC Signal Bias Node

Discrete signals from threshold crosses on **PercentC** (Casey Bands), matching the
StatOasis / TradingView subgraph logic: EMA channel offsets by ATR × multiplier,
then SMA-smoothed; price position in the channel scaled to 0–100.

Pine reference (equivalent structure)::

    atrValue  = ta.atr(atrLookBack)
    upperBand = ta.sma(ta.ema(high, lookBack) + atrValue * multiplier, smoothing)
    lowerBand = ta.sma(ta.ema(low, lookBack) - atrValue * multiplier, smoothing)
    percentC  = upperBand - lowerBand != 0
        ? (close - lowerBand) / (upperBand - lowerBand) * 100
        : 50

ATR uses Wilder (RMA) smoothing of true range, consistent with ``ta.atr`` on TradingView.

Output: Rule-based -1, 0, or 1 (same entry/exit semantics as :class:`~nodes.mean_reversion.rsi.rsi_signal.RSISignal`).
"""

from __future__ import annotations

from collections import deque
from typing import ClassVar, List, Optional

from nodes import BiasNode
from utils.core.enums import DirectionInput, Ticker, TimeFrame, coerce_direction
from utils.core.models import Candle
from utils.compute.fast_nodes import compute_ema_fast


class CaseyPercentCSignal(BiasNode):
    """
    PercentC (0–100) with RSISignal-style crosses on ``oversold`` / ``overbought``.

    Parameters
    ----------
    ema_lookback
        ``lookBack`` in Pine — period for EMA of high and EMA of low.
    atr_lookback
        ``atrLookBack`` — Wilder ATR period.
    multiplier
        Channel half-width in ATR units (Pine ``multiplier``).
    smoothing
        SMA length applied to raw upper/lower bands; ``1`` = no smoothing.
    oversold, overbought
        PercentC scale (same spirit as RSI 30 / 70).
    strategy_mode, exit_policy, exit_bars
        Same as :class:`~nodes.mean_reversion.rsi.rsi_signal.RSISignal`.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset(
        {"emaLookback", "atrLookback", "smoothing"}
    )

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        ema_lookback: int = 20,
        atr_lookback: int = 20,
        multiplier: float = 1.25,
        smoothing: int = 3,
        oversold: float = 30.0,
        overbought: float = 70.0,
        strategy_mode: DirectionInput = "long",
        exit_policy: str = "threshold_or_bars",
        exit_bars: int = 5,
    ) -> None:
        super().__init__(ticker, tf)

        if ema_lookback < 2:
            raise ValueError("ema_lookback must be >= 2")
        if atr_lookback < 2:
            raise ValueError("atr_lookback must be >= 2")
        if smoothing < 1:
            raise ValueError("smoothing must be >= 1")
        if multiplier <= 0:
            raise ValueError("multiplier must be > 0")
        self.ema_lookback = ema_lookback
        self.atr_lookback = atr_lookback
        self.multiplier = multiplier
        self.smoothing = smoothing
        self.oversold = oversold
        self.overbought = overbought
        self.strategy_mode = self._normalize_strategy_mode(strategy_mode)
        self.exit_policy = exit_policy.lower().strip()
        if self.exit_policy not in {"threshold", "threshold_or_bars"}:
            raise ValueError(
                "exit_policy must be 'threshold' or 'threshold_or_bars'"
            )
        if exit_bars < 1:
            raise ValueError("exit_bars must be >= 1")
        self.exit_bars = exit_bars

        self.alpha = 2.0 / (float(ema_lookback) + 1.0)
        self.module_name = "casey_percent_c_signal"
        self.output_features = ["signal"]
        self.params = {
            "emaLookback": ema_lookback,
            "atrLookback": atr_lookback,
            "multiplier": multiplier,
            "smoothing": smoothing,
            "oversold": oversold,
            "overbought": overbought,
            "strategyMode": self.strategy_mode,
            "exitPolicy": self.exit_policy,
            "exitBars": exit_bars,
        }

        self.front_bad = atr_lookback + smoothing + ema_lookback

        self.prev_close: Optional[float] = None
        self._ema_high: Optional[float] = None
        self._ema_low: Optional[float] = None
        self._ema_first = True

        self._tr_init: list[float] = []
        self._atr: float = 0.0
        self._atr_ready = False

        self._upper_raw_hist: deque[float] = deque(maxlen=smoothing)
        self._lower_raw_hist: deque[float] = deque(maxlen=smoothing)

        self.position = 0
        self.bars_in_position = 0
        self.prev_percent_c: Optional[float] = None

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _normalize_strategy_mode(self, strategy_mode: DirectionInput) -> str:
        return coerce_direction(strategy_mode, field_name="strategy_mode").value

    def _true_range(self, candle: Candle) -> float:
        hl = candle.high - candle.low
        if self.prev_close is None:
            return max(hl, 0.0)
        hc = abs(candle.high - self.prev_close)
        lc = abs(candle.low - self.prev_close)
        return max(hl, hc, lc)

    def _update_wilder_atr(self, tr: float) -> None:
        if not self._atr_ready:
            self._tr_init.append(tr)
            if len(self._tr_init) < self.atr_lookback:
                return
            self._atr = sum(self._tr_init) / float(self.atr_lookback)
            self._atr_ready = True
            self._tr_init = []
            return
        self._atr = (
            self._atr * float(self.atr_lookback - 1) + tr
        ) / float(self.atr_lookback)

    def _crossed_below(self, prev_pc: float, pc: float) -> bool:
        return prev_pc > self.oversold and pc <= self.oversold

    def _crossed_above(self, prev_pc: float, pc: float) -> bool:
        return prev_pc < self.overbought and pc >= self.overbought

    def _apply_position_rules(self, percent_c: float) -> int:
        if self.prev_percent_c is None:
            return 0

        crossed_below = self._crossed_below(self.prev_percent_c, percent_c)
        crossed_above = self._crossed_above(self.prev_percent_c, percent_c)

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
        else:
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
        tr = self._true_range(candle)
        self._update_wilder_atr(tr)

        high_v = float(candle.high)
        low_v = float(candle.low)
        close_v = float(candle.close)

        if self._ema_first:
            self._ema_high = compute_ema_fast(high_v, 0.0, self.alpha, is_first=True)
            self._ema_low = compute_ema_fast(low_v, 0.0, self.alpha, is_first=True)
            self._ema_first = False
        else:
            assert self._ema_high is not None and self._ema_low is not None
            self._ema_high = compute_ema_fast(
                high_v, self._ema_high, self.alpha, is_first=False
            )
            self._ema_low = compute_ema_fast(
                low_v, self._ema_low, self.alpha, is_first=False
            )

        self.prev_close = close_v

        if not self._atr_ready or self._ema_high is None or self._ema_low is None:
            self.position = 0
            self.bars_in_position = 0
            self.prev_percent_c = None
            self._upper_raw_hist.clear()
            self._lower_raw_hist.clear()
            self.output.append(0.0)
            return [0.0]

        upper_raw = self._ema_high + self._atr * self.multiplier
        lower_raw = self._ema_low - self._atr * self.multiplier
        self._upper_raw_hist.append(upper_raw)
        self._lower_raw_hist.append(lower_raw)

        if len(self._upper_raw_hist) < self.smoothing:
            self.position = 0
            self.bars_in_position = 0
            self.prev_percent_c = None
            self.output.append(0.0)
            return [0.0]

        upper_band = sum(self._upper_raw_hist) / float(self.smoothing)
        lower_band = sum(self._lower_raw_hist) / float(self.smoothing)
        width = upper_band - lower_band

        if abs(width) < 1e-12:
            percent_c = 50.0
        else:
            percent_c = (close_v - lower_band) / width * 100.0

        signal = self._apply_position_rules(percent_c)
        self.prev_percent_c = percent_c

        self.output.append(float(signal))
        return [float(signal)]
