"""
EOY SP500 seasonal bias node (rule-based).

Long from October 7 through January 7 inclusive. If Oct 7 or Jan 7 fall on
non-trading days, enter/exit on the first bar whose date is on or after that
calendar date. Outputs 1 (long), 0 (flat).
"""

from typing import List

from nodes import BiasNode
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle


class EoySp500(BiasNode):
    """
    EOY SP500 Bias Node (rule-based).

    Goes long on October 7 and holds until January 7 inclusive. Uses each
    candle's calendar date; when entry/exit dates fall on non-trading days,
    the first bar on or after that date triggers the position change.

    **Type**: Rule-based (outputs 0 or 1).
    """

    def __init__(self, ticker: Ticker, tf: TimeFrame) -> None:
        super().__init__(ticker, tf)

        self.module_name = "eoysp500"
        self.output_features = ["signal"]
        self.params: dict = {}
        self.front_bad = 0

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[int]:
        month = candle.datetime.month
        day = candle.datetime.day
        in_window = (
            (month == 10 and day >= 7)
            or month in (11, 12)
            or (month == 1 and day <= 7)
        )
        signal = 1 if in_window else 0
        self.output.append(signal)
        return [signal]
