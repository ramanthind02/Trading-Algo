"""
Seasonal gold Friday bias node (rule-based).

Fires on Thursday so that after the pipeline's 1-day forward shift the position
is held during Friday's open→close session.  Flat on all other days and on
non-gold tickers.
"""

from __future__ import annotations

from typing import ClassVar, List

from nodes import BiasNode
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle

# datetime.weekday(): Monday=0 … Friday=4
# Signal fires the day BEFORE entry (same convention as TurnaroundTuesday):
# Feature[Thursday] → shift(-1) → Return[Friday open→close].
_THURSDAY = 3
_GOLD_TICKERS = frozenset({Ticker.GC})


class SeasonalGoldFriday(BiasNode):
    """
    Seasonal Gold Friday Bias Node (rule-based).

    Outputs 1 (long) on Thursday when ticker is GC; otherwise 0.  The pipeline's
    1-day forward shift maps this to Friday's open→close return, so the strategy
    enters at Friday open and exits at Friday close.

    **Type**: Rule-based (outputs 1 or 0).
    """

    hardcoded_lookbacks: ClassVar[tuple[tuple[str, int], ...]] = (("weekday_gate", 1),)

    def __init__(self, ticker: Ticker, tf: TimeFrame) -> None:
        super().__init__(ticker, tf)

        self.module_name = "seasonalgoldfriday"
        self.output_features = ["signal"]
        self.params = {}
        self.front_bad = 1
        self._applies = ticker in _GOLD_TICKERS

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[int]:
        is_thursday = candle.datetime.weekday() == _THURSDAY
        signal = 1 if self._applies and is_thursday else 0
        self.output.append(signal)
        return [signal]
