from nodes import BiasNode
from utils.models import Candle
from utils.enums import Bias, Ticker, TimeFrame
from typing import List, Optional
import numpy as np
from collections import deque
import math

# Try to import Cython optimizations, fall back to pure Python if not available
try:
    from utils.cython_optimized import compute_ma_diff_fast
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    from scipy.stats import norm

class EWMACNode(BiasNode):
    """
    EWMACNode implements Robert Carver's Exponentially Weighted Moving Average Crossover 
    strategy with EWSD normalization to produce a scaled and capped forecast.
    
    The strategy calculates the difference between a faster EWMA (N-day span) and a slower 
    EWMA (4N-day span) of the back-adjusted price, normalizes this difference by the EWSD
    of price changes, scales it, and caps it at +/- 20.
    
    Default parameters are based on the EWMAC(16, 64) strategy which showed strong performance.
    
    Formula:
    1. Raw EWMAC Forecast: (EWMA(N, p_t) - EWMA(4N, p_t))
    2. EWSD (sigma_p): EWSD of price changes (p_t - p_{t-1}) using a 32-day span.
    3. Risk-Adjusted Forecast: Raw EWMAC Forecast / sigma_p
    4. Scaled & Capped Output: Max(Min(Risk-Adjusted Forecast * Scalar, 20), -20)
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, 
                 span_fast: int = 16, span_slow: int = 64, 
                 span_ewsd: int = 32, forecast_scalar: float = 4.10):
        """
        Initializes the EWMACNode bias node.
        
        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        - span_fast (int): N-day span for the faster EWMA (default: 16).
        - span_slow (int): 4N-day span for the slower EWMA (default: 64).
        - span_ewsd (int): Span for EWSD risk normalization (default: 32).
        - forecast_scalar (float): Scaling factor to bring the average absolute forecast to 10 (default: 4.10 for EWMAC16, 64).
        """
        super().__init__(ticker, tf)
        self.span_fast = span_fast
        self.span_slow = span_slow
        self.span_ewsd = span_ewsd
        self.forecast_scalar = forecast_scalar
        
        # Calculate EWMA lambdas: lambda = 2 / (N + 1)
        self.lambda_fast = 2.0 / (span_fast + 1.0)
        self.lambda_slow = 2.0 / (span_slow + 1.0)
        self.lambda_ewsd = 2.0 / (span_ewsd + 1.0)
        
        # Create unique feature names based on parameters
        self.columns = [
            f"ewmac_{span_fast}_{span_slow}",
            f"ewmac_{span_fast}_{span_slow}_bool"
        ]
        
        # Internal state for EWMA and EWSD
        self.ewma_fast: Optional[float] = None
        self.ewma_slow: Optional[float] = None
        self.ewsd_squared: Optional[float] = None # Stores sigma_p^2
        self.prev_price: Optional[float] = None
        
        # Need enough history for EWSD initialization (approx span_ewsd)
        self.ewsd_init_buffer = deque(maxlen=span_ewsd)
        self.front_bad = span_ewsd # Use EWSD span as the minimum required period for a meaningful EWSD estimate
        
        self.candle_count = 0
    
    def _initialize_ewma(self, price: float):
        """Initializes EWMA values with the first available price."""
        self.ewma_fast = price
        self.ewma_slow = price

    def _update_ewma(self, current_ewma: float, current_price: float, lambda_val: float) -> float:
        """Calculates the next EWMA value."""
        return lambda_val * current_price + (1.0 - lambda_val) * current_ewma

    def _compute_ewsd_init(self, price_change: float):
        """Builds buffer for initial EWSD estimate."""
        if price_change is not None:
            self.ewsd_init_buffer.append(price_change**2)
            
        if len(self.ewsd_init_buffer) == self.span_ewsd:
            # Initial EWSD^2 is the simple variance over the initial buffer period
            self.ewsd_squared = np.mean(self.ewsd_init_buffer)
        
    def _update_ewsd(self, price_change: float):
        """Updates EWSD using the EWSD formula (assuming mean price change is 0 for simplicity)."""
        if self.ewsd_squared is not None:
            self.ewsd_squared = self.lambda_ewsd * (price_change**2) + (1.0 - self.lambda_ewsd) * self.ewsd_squared
        
    def _compute_candle(self, candle: Candle) -> List:
        """
        Computes the EWMA crossover feature, normalized by EWSD, scaled, and capped.
        """
        self.candle_count += 1
        p_t = candle.close
        
        # 1. Calculate price change
        price_change = p_t - self.prev_price if self.prev_price is not None else 0.0
        
        # 2. Initialize EWMAs with first price
        if self.ewma_fast is None:
            self._initialize_ewma(p_t)
        
        # 3. Handle EWSD initialization (burn-in period)
        if self.ewsd_squared is None:
            self._compute_ewsd_init(price_change)
            self.ewma_fast = self._update_ewma(self.ewma_fast, p_t, self.lambda_fast)
            self.ewma_slow = self._update_ewma(self.ewma_slow, p_t, self.lambda_slow)
            self.prev_price = p_t
            
            # Not enough data for a robust forecast
            self.bias = Bias.NEUTRAL
            self.output = [0.0, False]
            return self.output
        
        # 4. Update EWMA and EWSD
        self.ewma_fast = self._update_ewma(self.ewma_fast, p_t, self.lambda_fast)
        self.ewma_slow = self._update_ewma(self.ewma_slow, p_t, self.lambda_slow)
        self._update_ewsd(price_change)
        
        # 5. Calculate Raw EWMAC Forecast
        raw_ewmac_forecast = self.ewma_fast - self.ewma_slow
        
        # 6. Normalize and calculate final forecast
        ewsd = math.sqrt(self.ewsd_squared)
        
        if ewsd > 1e-9: # Avoid division by zero
            # Risk-Adjusted Forecast
            risk_adjusted_forecast = raw_ewmac_forecast / ewsd
            
            # Scaled Forecast (to average absolute value of 10)
            scaled_forecast = risk_adjusted_forecast * self.forecast_scalar
            
            # Capped Forecast (at +/- 20)
            capped_forecast = max(min(scaled_forecast, 20.0), -20.0)
            
            output_value = capped_forecast
        else:
            output_value = 0.0
            
        # 7. Update previous price
        self.prev_price = p_t
        
        # 8. Update bias and store output
        if output_value > 0:
            self.bias = Bias.BULLISH
        elif output_value < 0:
            self.bias = Bias.BEARISH
        else:
            self.bias = Bias.NEUTRAL
        
        self.output = [output_value, bool(output_value > 0)]
        
        return self.output