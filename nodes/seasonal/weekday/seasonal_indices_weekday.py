from typing import ClassVar, List
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame
from nodes import BiasNode


# Monday=0, Tuesday=1, Wednesday=2 in datetime.weekday()
_WEEKDAYS_LONG = (0, 1, 2)


class SeasonalIndicesWeekday(BiasNode):
    """
    Seasonal Indices Weekday Bias Node (rule-based).

    Outputs 1 (long) when the candle's weekday is Monday, Tuesday, or Wednesday,
    and 0 (neutral) otherwise. Used as a simple seasonal/weekday filter for
    baseline or feature pipelines.

    **Type**: Rule-based (outputs 1 or 0).

    Parameters:
    - None (no parameters needed)
    """
    hardcoded_lookbacks: ClassVar[tuple[tuple[str, int], ...]] = (("weekday_gate", 1),)

    def __init__(self, ticker: Ticker, tf: TimeFrame) -> None:
        """
        Initialize Seasonal Indices Weekday node.

        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        """
        super().__init__(ticker, tf)

        self.module_name = "seasonalindicesweekday"
        self.output_features = ["signal"]
        self.params = {}

        self.front_bad = 1

        self.ensure_standardized_columns()
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List[int]:
        """
        Compute weekday signal for the given candle.

        Returns 1 if candle datetime is Mon/Tue/Wed, else 0.

        Parameters:
        - candle: The candle to process

        Returns:
        - List containing [1] or [0]
        """
        wd = candle.datetime.weekday()
        signal = 1 if wd in _WEEKDAYS_LONG else 0
        self.output.append(signal)
        return [signal]
