"""
MACD histogram hook momentum signal (Algomatic-style).

Long-only daily-style rule from Algomatic Trading's MACD Hook on Gold article.
Hull MA confirmation is replaced by a rising EMA on close. Exit uses a time stop
only (the article's three consecutive bullish days exit is omitted).

Entry (flat -> long): MACD line > 0, histogram hook (h > h[-1] and h[-1] < h[-2]),
and trend EMA rising.

Exit (long -> flat): bars in trade >= max_hold_bars.
"""

from __future__ import annotations

from collections import deque
from typing import ClassVar, Deque, List

from nodes import BiasNode
from lib.compute.fast_nodes import compute_ema_fast
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle


def _histogram_hook(h_curr: float, h_prev: float, h_prev2: float) -> bool:
    """True when histogram turns up after a pullback (Algomatic hook)."""
    return h_curr > h_prev and h_prev < h_prev2


def _ema_alpha(period: int) -> float:
    return 2.0 / (period + 1.0)


class MacdHookSignal(BiasNode):
    """
    Discrete long / flat MACD hook signal.

    Article defaults: MACD(12, 26, 9), trend EMA(15), max hold 10 bars.
    Output: 1.0 long, 0.0 flat.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset(
        {"macdFast", "macdSlow", "macdSignal", "trendEmaPeriod", "maxHoldBars"}
    )
    hardcoded_lookbacks: ClassVar[tuple[tuple[str, int], ...]] = (("histLag", 2),)

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        trend_ema_period: int = 15,
        max_hold_bars: int = 10,
    ) -> None:
        super().__init__(ticker, tf)

        if macd_fast < 2:
            raise ValueError("macd_fast must be >= 2")
        if macd_slow <= macd_fast:
            raise ValueError("macd_slow must be > macd_fast")
        if macd_signal < 2:
            raise ValueError("macd_signal must be >= 2")
        if trend_ema_period < 2:
            raise ValueError("trend_ema_period must be >= 2")
        if max_hold_bars < 1:
            raise ValueError("max_hold_bars must be >= 1")

        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.trend_ema_period = trend_ema_period
        self.max_hold_bars = max_hold_bars

        self.module_name = "macd_hook_signal"
        self.output_features = ["signal"]
        self.params = {
            "macdFast": macd_fast,
            "macdSlow": macd_slow,
            "macdSignal": macd_signal,
            "trendEmaPeriod": trend_ema_period,
            "maxHoldBars": max_hold_bars,
        }

        self.front_bad = macd_slow + macd_signal + 2

        self._alpha_fast = _ema_alpha(macd_fast)
        self._alpha_slow = _ema_alpha(macd_slow)
        self._alpha_signal = _ema_alpha(macd_signal)
        self._alpha_trend = _ema_alpha(trend_ema_period)

        self._ema_fast: float | None = None
        self._ema_slow: float | None = None
        self._ema_signal: float | None = None
        self._trend_ema: float | None = None
        self._prev_trend_ema: float | None = None
        self._macd_line: float | None = None

        self._hist: Deque[float] = deque(maxlen=3)
        self._position = 0
        self._bars_in_trade = 0
        self._n_prices = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _update_emas(self, close: float) -> None:
        self._n_prices += 1
        is_first = self._n_prices == 1

        self._ema_fast = compute_ema_fast(
            close, self._ema_fast or close, self._alpha_fast, is_first
        )
        self._ema_slow = compute_ema_fast(
            close, self._ema_slow or close, self._alpha_slow, is_first
        )

        self._macd_line = self._ema_fast - self._ema_slow
        self._ema_signal = compute_ema_fast(
            self._macd_line,
            self._ema_signal or self._macd_line,
            self._alpha_signal,
            is_first,
        )

        histogram = self._macd_line - self._ema_signal
        self._hist.append(float(histogram))

        self._prev_trend_ema = self._trend_ema
        self._trend_ema = compute_ema_fast(
            close,
            self._trend_ema or close,
            self._alpha_trend,
            is_first,
        )

    def _entry_ready(self) -> bool:
        if (
            self._macd_line is None
            or self._trend_ema is None
            or self._prev_trend_ema is None
            or len(self._hist) < 3
        ):
            return False

        h_curr, h_prev, h_prev2 = self._hist[-1], self._hist[-2], self._hist[-3]
        hook = _histogram_hook(h_curr, h_prev, h_prev2)
        bullish_macd = self._macd_line > 0.0
        rising_trend = self._trend_ema > self._prev_trend_ema
        return hook and bullish_macd and rising_trend

    def _apply_position_rules(self) -> float:
        if self._position == 1:
            self._bars_in_trade += 1
            if self._bars_in_trade >= self.max_hold_bars:
                self._position = 0
                self._bars_in_trade = 0

        if self._position == 0 and self._entry_ready():
            self._position = 1
            self._bars_in_trade = 1

        return float(self._position)

    def _compute_candle(self, candle: Candle) -> List[float]:
        close = float(candle.close)
        self._update_emas(close)

        if self._n_prices <= self.front_bad:
            self.output.append(0.0)
            return [0.0]

        signal = self._apply_position_rules()
        self.output.append(signal)
        return [signal]
