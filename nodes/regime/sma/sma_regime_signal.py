"""SMA regime — long above SMA, short below SMA (signed -1 / 0 / +1)."""

from __future__ import annotations

from collections import deque
from typing import ClassVar, List

from nodes import BiasNode
from utils.core.enums import PositionMode, Ticker, TimeFrame
from utils.core.models import Candle


def _coerce_position_mode(raw: PositionMode | str) -> PositionMode:
    if isinstance(raw, PositionMode):
        return raw
    return PositionMode(raw)


class SmaRegimeSignalNode(BiasNode):
    """
    Discrete regime from price vs a simple moving average:

    - **+1** when ``close > SMA(period)``
    - **-1** when ``close < SMA(period)`` and :attr:`mode` is ``LONG_SHORT``
    - **0** when ``close < SMA(period)`` and :attr:`mode` is ``LONG_ONLY``; when
      ``close == SMA(period)`` (rare); or during warmup

    :attr:`mode` follows :class:`~utils.core.enums.PositionMode` (same as breakout/MR nodes):
    ``LONG_ONLY`` clamps shorts to flat; ``SHORT_ONLY`` clamps longs to flat.

    No hysteresis: position flips whenever price crosses the SMA (subject to mode clamp).
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"period"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        period: int = 200,
        mode: PositionMode | str = PositionMode.LONG_SHORT,
    ) -> None:
        super().__init__(ticker, tf)
        if period < 2:
            raise ValueError("period must be >= 2")

        self.period = period
        self.mode = _coerce_position_mode(mode)
        self.module_name = "sma_regime_signal"
        self.output_features = ["signal"]
        self.params = {"period": period, "mode": self.mode}
        self.front_bad = period
        self._closes: deque[float] = deque(maxlen=period)

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[float]:
        self._closes.append(float(candle.close))
        if len(self._closes) < self.period:
            raw = 0.0
        else:
            sma = sum(self._closes) / float(self.period)
            c = float(candle.close)
            if c > sma:
                raw = 1.0
            elif c < sma:
                raw = -1.0
            else:
                raw = 0.0

        if self.mode == PositionMode.LONG_ONLY and raw < 0:
            out = 0.0
        elif self.mode == PositionMode.SHORT_ONLY and raw > 0:
            out = 0.0
        else:
            out = raw

        self.output.append(out)
        return [out]
