from nodes import BiasNode
from utils.core.models import Candle
from utils.core.enums import Bias, Ticker, TimeFrame
from typing import List
import numpy as np
from scipy import stats


class VolatilityRegimeNode(BiasNode):
    """
    VolatilityRegimeNode computes volatility regime features.
    
    Features:
    - vol_percentile: Current vol rank vs. historical period (0-100)
    - vol_zscore: Z-score of current volatility
    - vol_expansion: Ratio of short-term to long-term volatility
    - vol_acceleration: Rate of change in volatility
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, 
                 short_window: int = 20, long_window: int = 60, 
                 percentile_window: int = 252, method: str = 'garman_klass'):
        """
        Initialize Volatility Regime node.
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - short_window: Short-term volatility window (default: 20)
        - long_window: Long-term volatility window (default: 60)
        - percentile_window: Window for percentile calculation (default: 252)
        - method: Volatility estimation method (default: 'garman_klass')
        """
        super().__init__(ticker, tf)
        
        self.short_window = short_window
        self.long_window = long_window
        self.percentile_window = percentile_window
        self.method = method
        
        # Need enough data for percentile calculation
        self.front_bad = percentile_window
        
        # Store candle history
        self.candles = []
        
        # Store historical volatilities for percentile/zscore
        self.vol_history = []
        
        # Column names
        self.columns = [
            f'vol_percentile_{percentile_window}d',
            f'vol_zscore_{percentile_window}d',
            f'vol_expansion_{short_window}_{long_window}',
            f'vol_acceleration_{short_window}_{long_window}'
        ]
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute volatility regime features for the current candle.
        
        Parameters:
        - candle: The current candle
        
        Returns:
        - List: [vol_percentile, vol_zscore, vol_expansion, vol_acceleration]
        """
        # Add current candle to history
        self.candles.append(candle)
        
        # Keep only what we need
        if len(self.candles) > self.percentile_window:
            self.candles.pop(0)
        
        # Need at least long_window candles to compute
        if len(self.candles) < self.long_window:
            self.output = [np.nan, np.nan, np.nan, np.nan]
            return self.output
        
        # Compute current short-term volatility
        current_vol = self._compute_volatility(self.candles[-self.short_window:])
        
        # Compute long-term volatility
        long_vol = self._compute_volatility(self.candles[-self.long_window:])
        
        # Store current volatility in history
        self.vol_history.append(current_vol)
        if len(self.vol_history) > self.percentile_window:
            self.vol_history.pop(0)
        
        # Compute percentile (need full window)
        if len(self.vol_history) >= self.percentile_window:
            vol_percentile = stats.percentileofscore(self.vol_history, current_vol)
        else:
            vol_percentile = np.nan
        
        # Compute z-score (need full window)
        if len(self.vol_history) >= self.percentile_window:
            vol_mean = np.mean(self.vol_history)
            vol_std = np.std(self.vol_history)
            vol_zscore = (current_vol - vol_mean) / (vol_std + 1e-10)
        else:
            vol_zscore = np.nan
        
        # Compute expansion ratio
        vol_expansion = current_vol / (long_vol + 1e-10)
        
        # Compute acceleration (rate of change)
        vol_acceleration = (current_vol - long_vol) / (long_vol + 1e-10)
        
        self.output = [vol_percentile, vol_zscore, vol_expansion, vol_acceleration]
        return self.output
    
    def _compute_volatility(self, candles: List[Candle]) -> float:
        """
        Compute volatility for a given window of candles.
        
        Parameters:
        - candles: List of candles
        
        Returns:
        - float: Annualized volatility
        """
        if self.method == 'garman_klass':
            return self._garman_klass_volatility(candles)
        elif self.method == 'parkinson':
            return self._parkinson_volatility(candles)
        elif self.method == 'close_to_close':
            return self._close_to_close_volatility(candles)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _garman_klass_volatility(self, candles: List[Candle]) -> float:
        """Garman-Klass volatility estimator."""
        hl_sum = 0.0
        co_sum = 0.0
        n = len(candles)
        
        for candle in candles:
            hl = np.log(candle.high / candle.low)
            co = np.log(candle.close / candle.open)
            
            hl_sum += hl ** 2
            co_sum += co ** 2
        
        gk = 0.5 * (hl_sum / n) - (2 * np.log(2) - 1) * (co_sum / n)
        vol = np.sqrt(gk) * np.sqrt(252)
        
        return vol
    
    def _parkinson_volatility(self, candles: List[Candle]) -> float:
        """Parkinson volatility estimator."""
        hl_sum = 0.0
        n = len(candles)
        
        for candle in candles:
            hl_ratio = np.log(candle.high / candle.low)
            hl_sum += hl_ratio ** 2
        
        vol = np.sqrt((1 / (4 * np.log(2))) * (hl_sum / n)) * np.sqrt(252)
        
        return vol
    
    def _close_to_close_volatility(self, candles: List[Candle]) -> float:
        """Traditional close-to-close volatility."""
        log_returns = []
        
        for i in range(1, len(candles)):
            log_ret = np.log(candles[i].close / candles[i-1].close)
            log_returns.append(log_ret)
        
        vol = np.std(log_returns) * np.sqrt(252)
        
        return vol
