"""
TSI (True Strength Index) Bias Node

Price-time correlation measuring trend strength. Uses double exponential
smoothing of price momentum.

Output: Continuous -100 to +100
"""

from typing import ClassVar, List
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame
from nodes import BiasNode


class TSI(BiasNode):
    """
    TSI (True Strength Index) - Price-time correlation / trend strength.

    The True Strength Index uses double exponential smoothing of price
    momentum to measure trend strength while filtering noise.

    Formula:
        price_change = close - prev_close
        double_smoothed_pc = EMA(EMA(price_change, long_period), short_period)
        double_smoothed_abs_pc = EMA(EMA(abs(price_change), long_period), short_period)
        TSI = 100 * (double_smoothed_pc / double_smoothed_abs_pc)

    Interpretation:
    - TSI > 0: Bullish momentum
    - TSI < 0: Bearish momentum
    - TSI near +100 or -100: Strong trend
    - TSI near 0: Weak or no trend

    Output Range: -100.0 to +100.0 (continuous)
    Neutral Value: 0.0 (during warmup)

    Parameters:
    - long_period: Long EMA period (default: 25)
    - short_period: Short EMA period (default: 13)
    """
    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"longPeriod", "shortPeriod"})

    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        long_period: int = 25,
        short_period: int = 13
    ):
        super().__init__(ticker, tf)

        self.long_period = long_period
        self.short_period = short_period

        # Standardized naming metadata
        self.module_name = 'tsi'
        self.output_features = ['signal']
        self.params = {
            'longPeriod': long_period,
            'shortPeriod': short_period
        }

        # Warmup needs long period + short period
        self.front_bad = long_period + short_period

        # EMA multipliers
        self.long_mult = 2.0 / (long_period + 1)
        self.short_mult = 2.0 / (short_period + 1)

        # State for double exponential smoothing of price change
        self.ema_pc_long = None  # First EMA of price change
        self.ema_pc_short = None  # Second EMA (of the first EMA)

        # State for double exponential smoothing of abs(price change)
        self.ema_abs_pc_long = None
        self.ema_abs_pc_short = None

        self.prev_close = None
        self.n_candles = 0

        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute TSI for the given candle.

        Parameters:
        - candle: The candle to process

        Returns:
        - List containing the TSI value (-100.0 to +100.0)
        """
        self.n_candles += 1
        curr_close = candle.close

        # Need at least 2 candles for price change
        if self.prev_close is None:
            self.prev_close = curr_close
            self.output.append(0.0)
            return [0.0]

        # Calculate price change
        price_change = curr_close - self.prev_close
        abs_price_change = abs(price_change)

        # Update first EMA (long period)
        if self.ema_pc_long is None:
            self.ema_pc_long = price_change
            self.ema_abs_pc_long = abs_price_change
        else:
            self.ema_pc_long = price_change * self.long_mult + self.ema_pc_long * (1 - self.long_mult)
            self.ema_abs_pc_long = abs_price_change * self.long_mult + self.ema_abs_pc_long * (1 - self.long_mult)

        # Update second EMA (short period)
        if self.ema_pc_short is None:
            self.ema_pc_short = self.ema_pc_long
            self.ema_abs_pc_short = self.ema_abs_pc_long
        else:
            self.ema_pc_short = self.ema_pc_long * self.short_mult + self.ema_pc_short * (1 - self.short_mult)
            self.ema_abs_pc_short = self.ema_abs_pc_long * self.short_mult + self.ema_abs_pc_short * (1 - self.short_mult)

        self.prev_close = curr_close

        # Return neutral during warmup
        if self.n_candles < self.front_bad:
            self.output.append(0.0)
            return [0.0]

        # Calculate TSI
        if self.ema_abs_pc_short != 0:
            tsi = 100.0 * (self.ema_pc_short / self.ema_abs_pc_short)
        else:
            tsi = 0.0

        # Ensure within bounds
        tsi = max(-100.0, min(100.0, tsi))

        self.output.append(tsi)
        return [tsi]
