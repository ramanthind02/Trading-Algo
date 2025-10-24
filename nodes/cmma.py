import math
from typing import List
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from nodes import BiasNode


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
        
        # Compute moving average of log(close) over lookback period
        log_sum = 0.0
        for k in range(n - self.lookback, n):
            log_sum += math.log(self.close_prices[k])
        ma = log_sum / self.lookback
        
        # Compute ATR
        atr = self._compute_atr(n - 1)
        
        if atr > 0.0:
            # Normalize by ATR * sqrt(lookback + 1)
            denom = atr * math.sqrt(self.lookback + 1.0)
            normalized = (math.log(self.close_prices[-1]) - ma) / denom
            
            # Transform through normal CDF and scale to [-50, 50]
            # The factor 1.0 controls compression (increase for more compression, decrease for less)
            result = 100.0 * self._normal_cdf(1.0 * normalized) - 50.0
        else:
            # If ATR is zero, return neutral value
            result = 0.0
        
        self.output.append(result)
        return [result]