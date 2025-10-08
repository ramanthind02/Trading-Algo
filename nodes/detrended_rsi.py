import numpy as np
from typing import List
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from nodes import BiasNode


class DetrendedRSI(BiasNode):
    """
    Detrended RSI Bias Node
    
    Computes the difference between a short-period RSI and its linear regression
    prediction based on a longer-period RSI. This removes the trend component from
    the RSI, leaving only the deviations from the trend.
    
    Algorithm:
    1. Compute short-period RSI (the one being detrended)
    2. Compute long-period RSI (the detrender)
    3. For each point, fit a linear regression of short RSI vs long RSI over a lookback window
    4. Output = actual short RSI - predicted short RSI
    
    Parameters:
    - short_length: Period for the RSI being detrended
    - long_length: Period for the detrender RSI (should be > short_length)
    - lookback: Window size for linear regression fit
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, short_length: int = 2, 
                 long_length: int = 20, lookback: int = 252):
        """
        Initialize Detrended RSI node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - short_length: RSI period being detrended (default: 2)
        - long_length: Detrender RSI period (default: 50, must be > short_length)
        - lookback: Linear regression lookback period (default: 100)
        """
        super().__init__(ticker, tf)
        
        if long_length <= short_length:
            raise ValueError(f"long_length ({long_length}) must be greater than short_length ({short_length})")
        
        self.short_length = short_length
        self.long_length = long_length
        self.lookback = lookback
        
        # Number of candles needed before we can compute valid output
        self.front_bad = long_length + lookback - 1
        
        # Storage for RSI values
        self.short_rsi = []  # Short-period RSI (detrended)
        self.long_rsi = []   # Long-period RSI (detrender)
        
        # RSI computation state
        self.short_upsum = 1e-60
        self.short_dnsum = 1e-60
        self.long_upsum = 1e-60
        self.long_dnsum = 1e-60
        
        # Price history for RSI computation
        self.close_prices = []

        self.columns = [f'detrended_rsi_{short_length}_{long_length}_{lookback}']
    
    def _compute_rsi_update(self, new_close: float, prev_close: float, 
                           upsum: float, dnsum: float, length: int) -> tuple:
        """
        Update RSI using exponential moving average
        
        Returns: (new_upsum, new_dnsum, rsi_value)
        """
        diff = new_close - prev_close
        
        if diff > 0.0:
            upsum = ((length - 1.0) * upsum + diff) / length
            dnsum *= (length - 1.0) / length
        else:
            dnsum = ((length - 1.0) * dnsum - diff) / length
            upsum *= (length - 1.0) / length
        
        rsi = 100.0 * upsum / (upsum + dnsum)
        
        # Special transformation for length=2 (from C++ code)
        if length == 2:
            rsi = -10.0 * np.log(2.0 / (1 + 0.00999 * (2 * rsi - 100)) - 1)
        
        return upsum, dnsum, rsi
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute detrended RSI for the given candle
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the detrended RSI value
        """
        self.close_prices.append(candle.close)
        n = len(self.close_prices)
        
        # Initialize short RSI if we have enough data
        if n == self.short_length:
            # Initialize with simple average
            for i in range(1, self.short_length):
                diff = self.close_prices[i] - self.close_prices[i-1]
                if diff > 0.0:
                    self.short_upsum += diff
                else:
                    self.short_dnsum -= diff
            self.short_upsum /= (self.short_length - 1)
            self.short_dnsum /= (self.short_length - 1)
        
        # Update short RSI if initialized
        if n >= self.short_length:
            self.short_upsum, self.short_dnsum, short_rsi_val = self._compute_rsi_update(
                self.close_prices[-1], self.close_prices[-2],
                self.short_upsum, self.short_dnsum, self.short_length
            )
            self.short_rsi.append(short_rsi_val)
        
        # Initialize long RSI if we have enough data
        if n == self.long_length:
            # Initialize with simple average
            for i in range(1, self.long_length):
                diff = self.close_prices[i] - self.close_prices[i-1]
                if diff > 0.0:
                    self.long_upsum += diff
                else:
                    self.long_dnsum -= diff
            self.long_upsum /= (self.long_length - 1)
            self.long_dnsum /= (self.long_length - 1)
        
        # Update long RSI if initialized
        if n >= self.long_length:
            self.long_upsum, self.long_dnsum, long_rsi_val = self._compute_rsi_update(
                self.close_prices[-1], self.close_prices[-2],
                self.long_upsum, self.long_dnsum, self.long_length
            )
            self.long_rsi.append(long_rsi_val)
        
        # Compute detrended RSI if we have enough data
        if n < self.front_bad:
            # Not enough data yet - return neutral value
            self.output.append(0.0)
            return [0.0]
        
        # We have enough data - compute linear regression and detrend
        # Get the last 'lookback' values
        short_window = self.short_rsi[-self.lookback:]
        long_window = self.long_rsi[-self.lookback:]
        
        # Compute means
        x_mean = np.mean(long_window)
        y_mean = np.mean(short_window)
        
        # Compute regression coefficient
        x_diff = np.array(long_window) - x_mean
        y_diff = np.array(short_window) - y_mean
        
        xss = np.sum(x_diff ** 2)
        xy = np.sum(x_diff * y_diff)
        coef = xy / (xss + 1e-60)
        
        # Compute detrended value: actual - predicted
        current_x_diff = self.long_rsi[-1] - x_mean
        current_y_diff = self.short_rsi[-1] - y_mean
        detrended = current_y_diff - coef * current_x_diff
        
        self.output.append(detrended)
        return [detrended]