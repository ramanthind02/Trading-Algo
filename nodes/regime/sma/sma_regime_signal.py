"""SMA regime — long above SMA, short below SMA (signed -1 / 0 / +1)."""

from __future__ import annotations

from collections import deque
from typing import ClassVar, List

from nodes import BiasNode
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


class SmaRegimeSignalNode(BiasNode):
    """
    Discrete regime from price vs a simple moving average:

    - **+1** when ``close > SMA(period)``
    - **-1** when ``close < SMA(period)``
    - **0** when ``close == SMA(period)`` (rare) or during warmup

    No hysteresis: position flips whenever price crosses the SMA.
    """

    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"period"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        period: int = 200,
    ) -> None:
        super().__init__(ticker, tf)
        if period < 2:
            raise ValueError("period must be >= 2")

        self.period = period
        self.module_name = "sma_regime_signal"
        self.output_features = ["signal"]
        self.params = {"period": period}
        self.front_bad = period
        self._closes: deque[float] = deque(maxlen=period)

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[float]:
        self._closes.append(float(candle.close))
        if len(self._closes) < self.period:
            out = 0.0
        else:
            sma = sum(self._closes) / float(self.period)
            c = float(candle.close)
            if c > sma:
                out = 1.0
            elif c < sma:
                out = -1.0
            else:
                out = 0.0
        self.output.append(out)
        return [out]
