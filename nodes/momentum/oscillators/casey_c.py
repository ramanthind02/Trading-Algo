"""
Casey C% Bias Node

Percent change ranking oscillator. Ranks the current percent change
against historical percent changes over a lookback period.

Output: Continuous 0-100 (percentile rank)
"""

from typing import ClassVar, List
import numpy as np
from collections import deque
from lib.core.models import Candle
from lib.core.enums import Ticker, TimeFrame
from nodes import BiasNode


class CaseyC(BiasNode):
    """
    Casey C% - Percent change ranking oscillator.

    Calculates the percentile rank of the current N-period percent change
    compared to all historical N-period percent changes within the lookback window.

    Formula:
        pct_change = (close - close_N_periods_ago) / close_N_periods_ago * 100
        casey_c = percentile_rank(pct_change, historical_pct_changes) * 100

    This creates a normalized measure of momentum where:
    - 100 = current momentum is at the highest level in the lookback period
    - 0 = current momentum is at the lowest level in the lookback period
    - 50 = current momentum is at the median

    Output Range: 0.0 to 100.0 (continuous)
    Neutral Value: 50.0 (during warmup)

    Parameters:
    - change_period: Period for calculating percent change (default: 10)
    - ranking_period: Lookback period for percentile ranking (default: 100)
    """
    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"changePeriod", "rankingPeriod"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        change_period: int = 10,
        ranking_period: int = 100
    ):
        super().__init__(ticker, tf)

        self.change_period = change_period
        self.ranking_period = ranking_period

        # Standardized naming metadata
        self.module_name = 'caseyc'
        self.output_features = ['signal']
        self.params = {
            'changePeriod': change_period,
            'rankingPeriod': ranking_period
        }

        # Warmup needs enough data for percent change + ranking
        self.front_bad = change_period + ranking_period

        # Buffer for close prices (need change_period + 1 for percent change)
        self.close_buffer = deque(maxlen=change_period + 1)

        # Buffer for historical percent changes
        self.pct_change_buffer = deque(maxlen=ranking_period)

        self.n_candles = 0

        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute Casey C% for the given candle.

        Parameters:
        - candle: The candle to process

        Returns:
        - List containing the Casey C% value (0.0-100.0)
        """
        self.n_candles += 1
        curr_close = candle.close
        self.close_buffer.append(curr_close)

        # Calculate percent change if we have enough data
        if len(self.close_buffer) > self.change_period:
            past_close = self.close_buffer[0]
            if past_close != 0:
                pct_change = (curr_close - past_close) / past_close * 100
            else:
                pct_change = 0.0
            self.pct_change_buffer.append(pct_change)
        else:
            pct_change = 0.0

        # Return neutral during warmup
        if self.n_candles < self.front_bad:
            self.output.append(50.0)
            return [50.0]

        # Calculate percentile rank
        if len(self.pct_change_buffer) > 0:
            current_pct_change = self.pct_change_buffer[-1]
            historical = np.array(self.pct_change_buffer)

            # Percentile rank: what percentage of values are less than current
            count_below = np.sum(historical < current_pct_change)
            count_equal = np.sum(historical == current_pct_change)

            # Use midpoint method for ties
            percentile = (count_below + 0.5 * count_equal) / len(historical) * 100
        else:
            percentile = 50.0

        # Ensure within bounds
        percentile = max(0.0, min(100.0, percentile))

        self.output.append(percentile)
        return [percentile]
