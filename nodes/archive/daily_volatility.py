from nodes import BiasNode
from utils.models import Candle
from utils.enums import Bias, Ticker, TimeFrame
from typing import List
import numpy as np


class DailyVolatilityNode(BiasNode):
    """
    DailyVolatilityNode converts annualized volatility to daily volatility.
    
    Daily volatility = Annualized volatility / sqrt(252)
    
    This node depends on a HistoricalVolatilityNode to provide the annualized volatility.
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, lookback: int = 30, method: str = 'garman_klass'):
        """
        Initialize Daily Volatility node.
        
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
        method_abbrev = {'garman_klass': 'gk', 'parkinson': 'park', 'close_to_close': 'c2c'}
        abbrev = method_abbrev.get(method, method[:3])
        self.columns = [f'daily_vol_{abbrev}_{lookback}d']
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute daily volatility for the current candle.
        
        Parameters:
        - candle: The current candle
        
        Returns:
        - List: Daily volatility estimate
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
        
        # Compute annualized volatility based on method
        if self.method == 'garman_klass':
            annual_vol = self._garman_klass_volatility()
        elif self.method == 'parkinson':
            annual_vol = self._parkinson_volatility()
        elif self.method == 'close_to_close':
            annual_vol = self._close_to_close_volatility()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Convert to daily volatility
        daily_vol = annual_vol / np.sqrt(252)
        
        self.output = [daily_vol]
        return self.output
    
    def _garman_klass_volatility(self) -> float:
        """Garman-Klass volatility estimator using OHLC data."""
        hl_sum = 0.0
        co_sum = 0.0
        
        for candle in self.candles:
            hl = np.log(candle.high / candle.low)
            co = np.log(candle.close / candle.open)
            
            hl_sum += hl ** 2
            co_sum += co ** 2
        
        gk = 0.5 * (hl_sum / self.lookback) - (2 * np.log(2) - 1) * (co_sum / self.lookback)
        vol = np.sqrt(gk) * np.sqrt(252)
        
        return vol
    
    def _parkinson_volatility(self) -> float:
        """Parkinson volatility estimator using high-low range."""
        hl_sum = 0.0
        
        for candle in self.candles:
            hl_ratio = np.log(candle.high / candle.low)
            hl_sum += hl_ratio ** 2
        
        vol = np.sqrt((1 / (4 * np.log(2))) * (hl_sum / self.lookback)) * np.sqrt(252)
        
        return vol
    
    def _close_to_close_volatility(self) -> float:
        """Traditional close-to-close volatility."""
        log_returns = []
        
        for i in range(1, len(self.candles)):
            log_ret = np.log(self.candles[i].close / self.candles[i-1].close)
            log_returns.append(log_ret)
        
        vol = np.std(log_returns) * np.sqrt(252)
        
        return vol
