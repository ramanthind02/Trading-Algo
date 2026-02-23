from nodes import BiasNode
from utils.core.models import Candle
from utils.core.enums import Bias, Ticker, TimeFrame
from typing import List
import numpy as np


class HistoricalVolatilityNode(BiasNode):
    """
    HistoricalVolatilityNode computes multiple historical volatility estimators.
    
    Supports:
    - Garman-Klass volatility (uses OHLC)
    - Parkinson volatility (uses high-low range)
    - Close-to-close volatility (traditional)
    
    All volatilities are annualized (multiplied by sqrt(252)).
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, lookback: int = 30, method: str = 'garman_klass'):
        """
        Initialize Historical Volatility node.
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - lookback: Period for volatility calculation (default: 30)
        - method: Volatility estimation method ('garman_klass', 'parkinson', 'close_to_close')
        """
        super().__init__(ticker, tf)
        
        self.lookback = lookback
        self.method = method
        self.front_bad = lookback
        
        # Store candle history
        self.candles = []
        
        # Column names
        self.columns = [f'{method}_vol_{lookback}d']
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute historical volatility for the current candle.
        
        Parameters:
        - candle: The current candle
        
        Returns:
        - List: Annualized volatility estimate
        """
        # Add current candle to history
        self.candles.append(candle)
        
        # Keep only the required lookback period
        if len(self.candles) > self.lookback:
            self.candles.pop(0)
        
        # Need at least lookback candles to compute
        if len(self.candles) < self.lookback:
            self.output = [np.nan]
            return self.output
        
        # Compute volatility based on method
        if self.method == 'garman_klass':
            vol = self._garman_klass_volatility()
        elif self.method == 'parkinson':
            vol = self._parkinson_volatility()
        elif self.method == 'close_to_close':
            vol = self._close_to_close_volatility()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.output = [vol]
        return self.output
    
    def _garman_klass_volatility(self) -> float:
        """
        Garman-Klass volatility estimator using OHLC data.
        More efficient than close-to-close.
        """
        hl_sum = 0.0
        co_sum = 0.0
        
        for candle in self.candles:
            hl = np.log(candle.high / candle.low)
            co = np.log(candle.close / candle.open)
            
            hl_sum += hl ** 2
            co_sum += co ** 2
        
        # Garman-Klass formula
        gk = 0.5 * (hl_sum / self.lookback) - (2 * np.log(2) - 1) * (co_sum / self.lookback)
        
        # Annualize
        vol = np.sqrt(gk) * np.sqrt(252)
        
        return vol
    
    def _parkinson_volatility(self) -> float:
        """
        Parkinson volatility estimator using high-low range.
        """
        hl_sum = 0.0
        
        for candle in self.candles:
            hl_ratio = np.log(candle.high / candle.low)
            hl_sum += hl_ratio ** 2
        
        # Parkinson formula
        vol = np.sqrt((1 / (4 * np.log(2))) * (hl_sum / self.lookback)) * np.sqrt(252)
        
        return vol
    
    def _close_to_close_volatility(self) -> float:
        """
        Traditional close-to-close volatility (standard deviation of log returns).
        """
        log_returns = []
        
        for i in range(1, len(self.candles)):
            log_ret = np.log(self.candles[i].close / self.candles[i-1].close)
            log_returns.append(log_ret)
        
        # Standard deviation of returns
        vol = np.std(log_returns) * np.sqrt(252)
        
        return vol
