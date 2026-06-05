import math
from typing import ClassVar, List
import numpy as np
from lib.core.models import Candle
from lib.core.enums import Ticker, TimeFrame
from nodes import BiasNode

try:
    from lib.compute.fast_nodes import (
        CYTHON_NODES_AVAILABLE,
        compute_sma_fast,
        compute_atr_from_slice_fast,
        normal_cdf_fast,
    )
except ImportError:
    CYTHON_NODES_AVAILABLE = False
    compute_sma_fast = None  # type: ignore[assignment]
    compute_atr_from_slice_fast = None  # type: ignore[assignment]
    normal_cdf_fast = None  # type: ignore[assignment]


class CloseMaMinusMA(BiasNode):
    """
    Close Minus MA (Moving Average) Bias Node
    
    Computes the difference between the current close price and a moving average
    of log prices, normalized by ATR (Average True Range) and transformed through
    a normal CDF to produce a bounded output in the range [-50, 50].
    
    The calculation:
    1. Compute MA of log(close) over lookback period
    2. Compute ATR over atr_length period
    3. Normalize: (log(close) - MA) / (ATR * sqrt(lookback + 1))
    4. Transform: 100 * normal_cdf(normalized_value) - 50
    
    Parameters:
    - lookback: Period for moving average calculation (default: 20)
    - atr_length: Period for ATR calculation (default: 14)
    """
    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"lookback", "atr_length"})
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, lookback: int = 20, atr_length: int = 252):
        """
        Initialize CloseMaMinusMA node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - lookback: Period for moving average (default: 20)
        - atr_length: Period for ATR calculation (default: 14)
        """
        super().__init__(ticker, tf)
        
        self.lookback = lookback
        self.atr_length = atr_length
        
        # Number of candles needed before we can compute valid output
        self.front_bad = max(lookback, atr_length)
        self.params = {"lookback": lookback, "atr_length": atr_length}
        
        # Price history for computation
        self.open_prices = []
        self.high_prices = []
        self.low_prices = []
        self.close_prices = []
        
        # Column name for output
        self.columns = [f'cmma_{lookback}_{atr_length}']
    
    def _compute_atr(self, index: int) -> float:
        """
        Compute Average True Range at the given index
        
        Parameters:
        - index: The index to compute ATR for
        
        Returns:
        - ATR value
        """
        if index < self.atr_length:
            return 0.0
        
        tr_sum = 0.0
        for i in range(index - self.atr_length + 1, index + 1):
            if i == 0:
                # First candle: TR = high - low
                tr = self.high_prices[i] - self.low_prices[i]
            else:
                # TR = max(high - low, |high - prev_close|, |low - prev_close|)
                hl = self.high_prices[i] - self.low_prices[i]
                hc = abs(self.high_prices[i] - self.close_prices[i - 1])
                lc = abs(self.low_prices[i] - self.close_prices[i - 1])
                tr = max(hl, hc, lc)
            tr_sum += tr
        
        return tr_sum / self.atr_length
    
    def _normal_cdf(self, x: float) -> float:
        """
        Compute the cumulative distribution function of the standard normal distribution
        
        Parameters:
        - x: Input value
        
        Returns:
        - CDF value between 0 and 1
        """
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute CMMA for the given candle
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the CMMA value
        """
        self.open_prices.append(candle.open)
        self.high_prices.append(candle.high)
        self.low_prices.append(candle.low)
        self.close_prices.append(candle.close)
        
        n = len(self.close_prices)
        
        # Return neutral value (0.0) until we have enough data
        if n < self.front_bad:
            self.output.append(0.0)
            return [0.0]

        # Moving average of log(close) over lookback (Cython when available)
        if CYTHON_NODES_AVAILABLE and compute_sma_fast is not None:
            log_arr = np.array(
                [math.log(p) for p in self.close_prices[-self.lookback :]],
                dtype=np.float64,
            )
            ma = compute_sma_fast(log_arr, self.lookback)
        else:
            log_sum = 0.0
            for k in range(n - self.lookback, n):
                log_sum += math.log(self.close_prices[k])
            ma = log_sum / self.lookback

        # ATR (Cython when available)
        if CYTHON_NODES_AVAILABLE and compute_atr_from_slice_fast is not None:
            highs_arr = np.array(self.high_prices, dtype=np.float64)
            lows_arr = np.array(self.low_prices, dtype=np.float64)
            closes_arr = np.array(self.close_prices, dtype=np.float64)
            atr = compute_atr_from_slice_fast(
                highs_arr, lows_arr, closes_arr, n - 1, self.atr_length
            )
        else:
            atr = self._compute_atr(n - 1)

        if atr > 0.0:
            denom = atr * math.sqrt(self.lookback + 1.0)
            normalized = (math.log(self.close_prices[-1]) - ma) / denom
            if CYTHON_NODES_AVAILABLE and normal_cdf_fast is not None:
                result = 100.0 * normal_cdf_fast(1.0 * normalized) - 50.0
            else:
                result = 100.0 * self._normal_cdf(1.0 * normalized) - 50.0
        else:
            result = 0.0

        self.output.append(result)
        return [result]
