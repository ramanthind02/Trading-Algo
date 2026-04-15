"""
Envelope reversion bias node (Algomatic-style equity dip system).

Reverse-engineered from public strategy copy: long-only pullback entries when price
crosses below a short moving-average envelope lower band while the close remains
above a long-term (e.g. 200) SMA; exit when price crosses back above the envelope
upper band.

Band width can be a **percent** of the envelope center (classic MA envelope) or
**ATR multiplier** bands around the center SMA.

Output: discrete -1, 0, or 1 (same semantics as :class:`~nodes.mean_reversion.rsi.rsi_signal.RSISignal`).
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import ClassVar, List, Optional, Union

from nodes import BiasNode
from utils.core.enums import DirectionInput, Ticker, TimeFrame, coerce_direction
from utils.core.models import Candle


class EnvelopeWidthMode(Enum):
    """How envelope half-width is measured around the center SMA."""

    PERCENT = "percent"
    ATR_MULTIPLIER = "atr_multiplier"


def _coerce_envelope_width_mode(
    mode: Union[EnvelopeWidthMode, str],
) -> EnvelopeWidthMode:
    if isinstance(mode, EnvelopeWidthMode):
        return mode
    key = str(mode).strip().lower().replace("-", "_")
    mapping = {
        "percent": EnvelopeWidthMode.PERCENT,
        "pct": EnvelopeWidthMode.PERCENT,
        "atr_multiplier": EnvelopeWidthMode.ATR_MULTIPLIER,
        "atr": EnvelopeWidthMode.ATR_MULTIPLIER,
    }
    if key not in mapping:
        allowed = ", ".join(sorted({e.value for e in EnvelopeWidthMode}))
        raise ValueError(
            f"width_mode must be one of {allowed!r} (got {mode!r})"
        )
    return mapping[key]


class EnvelopeReversionSignal(BiasNode):
    """
    SMA envelope mean-reversion with long-horizon trend filter.

    Parameters
    ----------
    trend_period
        SMA period for the trend filter (strategy copy uses 200).
    envelope_period
        SMA period for the envelope center line.
    width_mode
        ``PERCENT`` — upper/lower = center × (1 ± percent_width/100).
        ``ATR_MULTIPLIER`` — upper/lower = center ± atr_multiplier × Wilder ATR.
    percent_width
        Half-envelope as a percent of center when ``width_mode`` is ``PERCENT``.
    atr_lookback, atr_multiplier
        Used when ``width_mode`` is ``ATR_MULTIPLIER``.
    strategy_mode, exit_policy, exit_bars
        Same pattern as :class:`~nodes.mean_reversion.rsi.rsi_signal.RSISignal`.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset(
        {"trendPeriod", "envelopePeriod", "atrLookback"}
    )

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        trend_period: int = 200,
        envelope_period: int = 20,
        width_mode: Union[EnvelopeWidthMode, str] = EnvelopeWidthMode.PERCENT,
        percent_width: float = 2.0,
        atr_lookback: int = 14,
        atr_multiplier: float = 1.5,
        strategy_mode: DirectionInput = "long",
        exit_policy: str = "threshold_or_bars",
        exit_bars: int = 5,
    ) -> None:
        super().__init__(ticker, tf)

        if trend_period < 2:
            raise ValueError("trend_period must be >= 2")
        if envelope_period < 2:
            raise ValueError("envelope_period must be >= 2")
        if percent_width <= 0:
            raise ValueError("percent_width must be > 0")
        if atr_lookback < 2:
            raise ValueError("atr_lookback must be >= 2")
        if atr_multiplier <= 0:
            raise ValueError("atr_multiplier must be > 0")

        self.trend_period = trend_period
        self.envelope_period = envelope_period
        self.width_mode = _coerce_envelope_width_mode(width_mode)
        self.percent_width = percent_width
        self.atr_lookback = atr_lookback
        self.atr_multiplier = atr_multiplier
        self.strategy_mode = self._normalize_strategy_mode(strategy_mode)
        self.exit_policy = exit_policy.lower().strip()
        if self.exit_policy not in {"threshold", "threshold_or_bars"}:
            raise ValueError(
                "exit_policy must be 'threshold' or 'threshold_or_bars'"
            )
        if exit_bars < 1:
            raise ValueError("exit_bars must be >= 1")
        self.exit_bars = exit_bars

        self.module_name = "envelope_reversion_signal"
        self.output_features = ["signal"]
        self.params = {
            "trendPeriod": trend_period,
            "envelopePeriod": envelope_period,
            "widthMode": self.width_mode.value,
            "percentWidth": percent_width,
            "atrLookback": atr_lookback,
            "atrMultiplier": atr_multiplier,
            "strategyMode": self.strategy_mode,
            "exitPolicy": self.exit_policy,
            "exitBars": exit_bars,
        }

        atr_warm = atr_lookback if self.width_mode == EnvelopeWidthMode.ATR_MULTIPLIER else 0
        self.front_bad = max(trend_period, envelope_period, atr_warm) + 1

        self._trend_closes: deque[float] = deque(maxlen=trend_period)
        self._env_closes: deque[float] = deque(maxlen=envelope_period)

        self.prev_close: Optional[float] = None
        self.prev_lower: Optional[float] = None
        self.prev_upper: Optional[float] = None

        self._tr_init: list[float] = []
        self._atr: float = 0.0
        self._atr_ready = False
        self._prev_close_for_tr: Optional[float] = None

        self.position = 0
        self.bars_in_position = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _normalize_strategy_mode(self, strategy_mode: DirectionInput) -> str:
        return coerce_direction(strategy_mode, field_name="strategy_mode").value

    def _true_range(self, candle: Candle) -> float:
        hl = float(candle.high) - float(candle.low)
        if self._prev_close_for_tr is None:
            return max(hl, 0.0)
        hc = abs(float(candle.high) - self._prev_close_for_tr)
        lc = abs(float(candle.low) - self._prev_close_for_tr)
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

    def _bands(
        self,
        center: float,
        atr: float,
    ) -> tuple[float, float]:
        if self.width_mode == EnvelopeWidthMode.PERCENT:
            f = self.percent_width / 100.0
            return center * (1.0 + f), center * (1.0 - f)
        upper = center + self.atr_multiplier * atr
        lower = center - self.atr_multiplier * atr
        return upper, lower

    def _trend_bullish(self, close_v: float, trend_sma: float) -> bool:
        return close_v > trend_sma

    def _trend_bearish(self, close_v: float, trend_sma: float) -> bool:
        return close_v < trend_sma

    def _crossed_below_lower(
        self,
        prev_c: float,
        close_v: float,
        prev_lo: float,
        lower: float,
    ) -> bool:
        return prev_c >= prev_lo and close_v < lower

    def _crossed_above_upper(
        self,
        prev_c: float,
        close_v: float,
        prev_hi: float,
        upper: float,
    ) -> bool:
        return prev_c <= prev_hi and close_v > upper

    def _apply_position_rules(
        self,
        close_v: float,
        trend_sma: float,
        upper: float,
        lower: float,
    ) -> int:
        if (
            self.prev_close is None
            or self.prev_lower is None
            or self.prev_upper is None
        ):
            return 0

        prev_c = self.prev_close
        prev_lo = self.prev_lower
        prev_hi = self.prev_upper

        below_lower = self._crossed_below_lower(prev_c, close_v, prev_lo, lower)
        above_upper = self._crossed_above_upper(prev_c, close_v, prev_hi, upper)
        bull = self._trend_bullish(close_v, trend_sma)
        bear = self._trend_bearish(close_v, trend_sma)

        next_position = self.position

        if self.strategy_mode == "long":
            if self.position == 0 and below_lower and bull:
                next_position = 1
            elif self.position == 1 and above_upper:
                next_position = 0
        elif self.strategy_mode == "short":
            if self.position == 0 and above_upper and bear:
                next_position = -1
            elif self.position == -1 and below_lower:
                next_position = 0
        else:
            if self.position == 0 and below_lower and bull:
                next_position = 1
            elif self.position == 0 and above_upper and bear:
                next_position = -1
            elif self.position == 1 and above_upper:
                next_position = 0
            elif self.position == -1 and below_lower:
                next_position = 0

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

    def _compute_candle(self, candle: Candle) -> List[float]:
        close_v = float(candle.close)
        self._trend_closes.append(close_v)
        self._env_closes.append(close_v)

        tr = self._true_range(candle)
        if self.width_mode == EnvelopeWidthMode.ATR_MULTIPLIER:
            self._update_wilder_atr(tr)
        self._prev_close_for_tr = close_v

        if (
            len(self._trend_closes) < self.trend_period
            or len(self._env_closes) < self.envelope_period
        ):
            self.position = 0
            self.bars_in_position = 0
            self.prev_close = close_v
            self.prev_lower = None
            self.prev_upper = None
            self.output.append(0.0)
            return [0.0]

        if (
            self.width_mode == EnvelopeWidthMode.ATR_MULTIPLIER
            and not self._atr_ready
        ):
            self.position = 0
            self.bars_in_position = 0
            self.prev_close = close_v
            self.prev_lower = None
            self.prev_upper = None
            self.output.append(0.0)
            return [0.0]

        trend_sma = sum(self._trend_closes) / float(self.trend_period)
        center = sum(self._env_closes) / float(self.envelope_period)
        atr_v = self._atr if self.width_mode == EnvelopeWidthMode.ATR_MULTIPLIER else 0.0
        upper, lower = self._bands(center, atr_v)

        signal = float(
            self._apply_position_rules(close_v, trend_sma, upper, lower)
        )

        self.prev_close = close_v
        self.prev_lower = lower
        self.prev_upper = upper

        self.output.append(signal)
        return [signal]
