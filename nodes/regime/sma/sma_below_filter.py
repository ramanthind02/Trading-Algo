"""SMA bearish gate for ``filter_gate`` — binary 1.0 when close < SMA(period), else 0.0."""

from __future__ import annotations

from collections import deque
from typing import ClassVar, List

from nodes import BiasNode
from lib.core.models import Candle
from lib.core.enums import Ticker, TimeFrame


class SmaBelowFilterNode(BiasNode):
    """
    Binary regime gate: **1.0** when ``close < SMA(period)``, else **0.0**.

    Intended as the **filter** leg of :class:`~nodes.composite.filter_gate.FilterGateNode`
    (non-zero = gate open). Warmup bars emit **0.0** (gate closed).
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
        self.module_name = "sma_below_filter"
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
            out = 1.0 if float(candle.close) < sma else 0.0
        self.output.append(out)
        return [out]
