"""
Long-only close breakout bias node.

Entry (Option 1): close > previous bar close → long (+1).
Flat when close <= previous bar close (0).

Intentionally parameter-free: one-bar momentum breakout for baseline research.
"""

from __future__ import annotations

from typing import ClassVar, List

from nodes import BiasNode
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle


class CloseBreakout(BiasNode):
    """
    Long-only breakout on consecutive closes.

    - **+1** when ``close > previous close``
    - **0** when ``close <= previous close`` or during warmup
    """

    hardcoded_lookbacks: ClassVar[tuple[tuple[str, int], ...]] = (("warmup_bars", 2),)

    def __init__(self, ticker: Ticker, tf: TimeFrame) -> None:
        super().__init__(ticker, tf)
        self.module_name = "close_breakout"
        self.output_features = ["signal"]
        self.params: dict[str, object] = {}
        self.front_bad = 2

        self._n_bars = 0
        self._prev_close: float | None = None

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[float]:
        self._n_bars += 1
        close = float(candle.close)

        if self._n_bars < self.front_bad or self._prev_close is None:
            self._prev_close = close
            self.output.append(0.0)
            return [0.0]

        signal = 1.0 if close > self._prev_close else 0.0
        self._prev_close = close
        self.output.append(signal)
        return [signal]
