from nodes import BiasNode
from utils.models import Candle
from utils.enums import Bias, Ticker, TimeFrame
from typing import List
import numpy as np
from scipy import stats


class MovementMagnitudeNode(BiasNode):
    """
    MovementMagnitudeNode computes movement magnitude features.
    
    Features:
    - move_to_vol_ratio: abs(daily_return) / daily_vol (KEY FEATURE)
    - move_zscore: Z-score of daily return
    - move_percentile: Percentile rank of today's move
    
    These features capture how "significant" or "unusual" a price move is
    relative to recent volatility and historical movements.
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, 
                 vol_window: int = 30, zscore_window: int = 30,
                 percentile_window: int = 252, method: str = 'garman_klass'):
        """
        Initialize Movement Magnitude node.
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - vol_window: Window for volatility calculation (default: 30)
        - zscore_window: Window for z-score calculation (default: 30)
        - percentile_window: Window for percentile calculation (default: 252)
        - method: Volatility estimation method (default: 'garman_klass')
        """
        super().__init__(ticker, tf)
        
        self.vol_window = vol_window
        self.zscore_window = zscore_window
        self.percentile_window = percentile_window
        self.method = method
        
        # Need enough data for percentile calculation
        self.front_bad = percentile_window
        
        # Store candle history
        self.candles = []
        
        # Store historical returns for percentile/zscore
        self.return_history = []
        
        # Column names
        self.columns = [
            f'move_to_vol_ratio_{vol_window}d',
            f'move_zscore_{zscore_window}d',
            f'move_percentile_{percentile_window}d'
        ]
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute movement magnitude features for the current candle.
        
        Parameters:
        - candle: The current candle
        
        Returns:
        - List: [move_to_vol_ratio, move_zscore, move_percentile]
        """
        # Add current candle to history
        self.candles.append(candle)
        
        # Keep only what we need
        if len(self.candles) > self.percentile_window + 1:
            self.candles.pop(0)
        
        # Need at least vol_window + 1 candles to compute (need previous close)
        if len(self.candles) < self.vol_window + 1:
            self.output = [np.nan, np.nan, np.nan]
            return self.output
        
        # Compute today's return (log return)
        current_return = np.log(candle.close / self.candles[-2].close)
        
        # Store return in history
        self.return_history.append(current_return)
        if len(self.return_history) > self.percentile_window:
            self.return_history.pop(0)
        
        # Compute daily volatility
        daily_vol = self._compute_daily_volatility()
        
        # Compute move-to-vol ratio (KEY FEATURE)
        move_to_vol_ratio = abs(current_return) / (daily_vol + 1e-10)
        
        # Compute z-score (need enough history)
        if len(self.return_history) >= self.zscore_window:
            recent_returns = self.return_history[-self.zscore_window:]
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            move_zscore = (current_return - mean_return) / (std_return + 1e-10)
        else:
            move_zscore = np.nan
        
        # Compute percentile (need full window)
        if len(self.return_history) >= self.percentile_window:
            # Use absolute returns for percentile
            abs_returns = [abs(r) for r in self.return_history]
            move_percentile = stats.percentileofscore(abs_returns, abs(current_return))
        else:
            move_percentile = np.nan
        
        self.output = [move_to_vol_ratio, move_zscore, move_percentile]
        return self.output
    
    def _compute_daily_volatility(self) -> float:
        """
        Compute daily volatility for the volatility window.
        
        Returns:
        - float: Daily volatility
        """
        # Get candles for volatility calculation
        vol_candles = self.candles[-self.vol_window:]
        
        # Compute annualized volatility
        if self.method == 'garman_klass':
            annual_vol = self._garman_klass_volatility(vol_candles)
        elif self.method == 'parkinson':
            annual_vol = self._parkinson_volatility(vol_candles)
        elif self.method == 'close_to_close':
            annual_vol = self._close_to_close_volatility(vol_candles)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Convert to daily
        daily_vol = annual_vol / np.sqrt(252)
        
        return daily_vol
    
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
