from typing import List
import numpy as np
from numba import njit
from collections import deque
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame
from nodes import BiasNode

try:
    from utils.compute.fast_nodes import (
        CYTHON_NODES_AVAILABLE,
        compute_pct_change_fast,
        compute_ultimate_c_fast,
        smooth_ultimate_c_fast,
    )
except ImportError:
    CYTHON_NODES_AVAILABLE = False
    compute_pct_change_fast = None  # type: ignore[assignment]
    compute_ultimate_c_fast = None  # type: ignore[assignment]
    smooth_ultimate_c_fast = None  # type: ignore[assignment]


# ============================================================================
# NUMBA-COMPILED FALLBACKS (used when Cython not available)
# ============================================================================

@njit
def pct_change_1d(prev_close: float, curr_close: float) -> float:
    """
    Calculate 1-period percent change.
    
    Parameters:
    - prev_close: Previous close price
    - curr_close: Current close price
    
    Returns:
    - Percent change as a percentage (e.g., 1.5 for 1.5%)
    """
    if prev_close == 0.0:
        return 0.0
    return ((curr_close - prev_close) / prev_close) * 100.0


@njit
def rolling_max_1d(values: np.ndarray, window: int) -> float:
    """
    Compute rolling maximum over a window.
    
    Parameters:
    - values: Array of values (most recent last)
    - window: Window size for rolling max
    
    Returns:
    - Maximum value over the window
    """
    n = len(values)
    if n < window:
        return np.max(values) if n > 0 else 0.0
    
    # Get the most recent window values
    recent = values[n - window:]
    return np.max(recent)


@njit
def rolling_min_1d(values: np.ndarray, window: int) -> float:
    """
    Compute rolling minimum over a window.
    
    Parameters:
    - values: Array of values (most recent last)
    - window: Window size for rolling min
    
    Returns:
    - Minimum value over the window
    """
    n = len(values)
    if n < window:
        return np.min(values) if n > 0 else 0.0
    
    # Get the most recent window values
    recent = values[n - window:]
    return np.min(recent)


@njit
def rolling_mean_1d(values: np.ndarray, window: int) -> float:
    """
    Compute rolling mean over a window.
    
    Parameters:
    - values: Array of values (most recent last)
    - window: Window size for rolling mean
    
    Returns:
    - Mean value over the window
    """
    n = len(values)
    if n < window:
        return np.mean(values) if n > 0 else 0.0
    
    # Get the most recent window values
    recent = values[n - window:]
    return np.mean(recent)


@njit
def compute_ultimate_c(
    roc_values: np.ndarray,
    lookback: int,
    factor: float
) -> float:
    """
    Compute Ultimate C% value from ROC values.
    
    This implements the Ultimate C% algorithm:
    1. Calculate CaseyC_Short: normalized ROC over lookback period
    2. Calculate CaseyC_Med: normalized ROC over lookback * factor period
    3. Calculate CaseyC_Long: normalized ROC over lookback * factor * factor period
    4. Weighted combination: (short * factor² + med * factor + long) / (factor² + factor + 1)
    
    Note: Smoothing is applied separately after this computation.
    
    Parameters:
    - roc_values: Array of ROC values (most recent last)
    - lookback: Base lookback period
    - factor: Factor for medium and long periods
    
    Returns:
    - Ultimate C% value (0-100)
    """
    n = len(roc_values)
    
    if n == 0:
        return 50.0  # Neutral value
    
    # Current ROC value
    current_roc = roc_values[-1]
    
    # Calculate CaseyC_Short
    short_window = lookback
    if n >= short_window:
        high_short = rolling_max_1d(roc_values, short_window)
        low_short = rolling_min_1d(roc_values, short_window)
        if high_short - low_short == 0.0:
            casey_c_short = 50.0
        else:
            casey_c_short = ((current_roc - low_short) / (high_short - low_short)) * 100.0
    else:
        casey_c_short = 50.0
    
    # Calculate CaseyC_Med
    med_window = int(lookback * factor)
    if n >= med_window:
        high_med = rolling_max_1d(roc_values, med_window)
        low_med = rolling_min_1d(roc_values, med_window)
        if high_med - low_med == 0.0:
            casey_c_med = 50.0
        else:
            casey_c_med = ((current_roc - low_med) / (high_med - low_med)) * 100.0
    else:
        casey_c_med = 50.0
    
    # Calculate CaseyC_Long
    long_window = int(lookback * factor * factor)
    if n >= long_window:
        high_long = rolling_max_1d(roc_values, long_window)
        low_long = rolling_min_1d(roc_values, long_window)
        if high_long - low_long == 0.0:
            casey_c_long = 50.0
        else:
            casey_c_long = ((current_roc - low_long) / (high_long - low_long)) * 100.0
    else:
        casey_c_long = 50.0
    
    # Weighted combination
    factor_sq = factor * factor
    denominator = factor_sq + factor + 1.0
    ultimate_c = ((casey_c_short * factor_sq) + (casey_c_med * factor) + casey_c_long) / denominator
    
    return ultimate_c


@njit
def smooth_ultimate_c(ultimate_c_values: np.ndarray, smooth_lookback: int) -> float:
    """
    Smooth Ultimate C% values using rolling mean.
    
    Parameters:
    - ultimate_c_values: Array of Ultimate C% values (most recent last)
    - smooth_lookback: Smoothing period
    
    Returns:
    - Smoothed Ultimate C% value
    """
    return rolling_mean_1d(ultimate_c_values, smooth_lookback)


# ============================================================================
# PYTHON CLASS (Maintains state, delegates computation to Numba)
# ============================================================================

class UltimateC(BiasNode):
    """
    Ultimate C% Bias Node - Cython-accelerated when built, else Numba fallback.

    Computes the Ultimate C% indicator, which is a multi-timeframe momentum
    oscillator that combines short, medium, and long-term normalized ROC values.

    The algorithm:
    1. Calculates Rate of Change (ROC) as percent change
    2. Normalizes ROC over three different lookback periods (short, med, long)
    3. Combines the three normalized values using weighted average
    4. Smooths the result

    Ultimate C% oscillates between 0 and 100, similar to RSI:
    - Values above 70: Overbought
    - Values below 30: Oversold

    Performance: Uses Cython helpers from fast_nodes when built (pct_change,
    rolling max/min/mean, ultimate_c, smooth); otherwise Numba-compiled helpers.
    
    Parameters:
    - lookback: Base lookback period for short-term normalization (default: 2)
    - factor: Factor for medium and long periods (default: 2.0)
    - smooth_lookback: Smoothing period for final output (default: 2)
    """
    
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        lookback: int = 2,
        factor: float = 2.0,
        smooth_lookback: int = 2
    ):
        """
        Initialize Ultimate C% node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - lookback: Base lookback period (default: 5)
        - factor: Factor for medium/long periods (default: 1.0)
        - smooth_lookback: Smoothing period (default: 2)
        """
        super().__init__(ticker, tf)
        
        self.lookback = lookback
        self.factor = factor
        self.smooth_lookback = smooth_lookback
        
        # Standardized naming metadata
        self.module_name = 'ultimate_c'
        self.output_features = ['signal']
        self.params = {
            'lookback': lookback,
            'factor': factor,
            'smoothLookback': smooth_lookback
        }
        
        # Calculate maximum window size needed
        # We need: max(lookback, lookback*factor, lookback*factor*factor) ROC values
        # Plus smooth_lookback ultimate_c values for smoothing
        # To get max_window ROC values, we need max_window + 1 prices
        # To get smooth_lookback ultimate_c values, we need smooth_lookback more prices
        # Total: max_window + 1 + smooth_lookback = max_window + smooth_lookback + 1
        max_window = max(
            lookback,
            int(lookback * factor),
            int(lookback * factor * factor)
        )
        self.front_bad = max_window + smooth_lookback + 1
        
        # State management
        self.prev_close = 0.0
        self.roc_values = deque(maxlen=max_window * 2)  # Store ROC values
        self.ultimate_c_values = deque(maxlen=smooth_lookback + 10)  # Store Ultimate C% for smoothing
        self.n_prices = 0
        
        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute Ultimate C% for the given candle.
        
        Delegates heavy computation to Numba-compiled functions for optimal performance.
        Uses circular buffers (deques) to avoid expensive array append operations.
        
        Uses Cython from fast_nodes when available, else Numba helpers.
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the Ultimate C% value
        """
        curr_close = candle.close
        self.n_prices += 1
        
        # Calculate ROC (percent change)
        if self.n_prices > 1 and self.prev_close > 1e-10:
            if CYTHON_NODES_AVAILABLE and compute_pct_change_fast is not None:
                roc = compute_pct_change_fast(self.prev_close, curr_close)
            else:
                roc = pct_change_1d(self.prev_close, curr_close)
            self.roc_values.append(roc)
        else:
            # Not enough data or invalid price
            if self.n_prices < self.front_bad:
                self.prev_close = curr_close
                self.output.append(50.0)
                return [50.0]
            # If we have enough prices but prev_close is invalid, skip this ROC
            # but continue processing
        
        # Update previous close for next iteration
        self.prev_close = curr_close
        
        # Return neutral value until we have enough data
        # We need max_window ROC values to compute ultimate_c
        # Plus smooth_lookback ultimate_c values to smooth
        # To get smooth_lookback ultimate_c values, we need smooth_lookback ROC values beyond max_window
        # So we need: max_window + smooth_lookback ROC values total
        max_window = max(
            self.lookback,
            int(self.lookback * self.factor),
            int(self.lookback * self.factor * self.factor)
        )
        required_roc_count = max_window + self.smooth_lookback
        if len(self.roc_values) < required_roc_count:
            self.output.append(50.0)
            return [50.0]
        
        # Convert ROC values to numpy array
        roc_array = np.array(self.roc_values, dtype=np.float64)

        # Compute Ultimate C% (Cython when available, else Numba)
        if CYTHON_NODES_AVAILABLE and compute_ultimate_c_fast is not None:
            ultimate_c = compute_ultimate_c_fast(
                roc_array, self.lookback, self.factor
            )
        else:
            ultimate_c = compute_ultimate_c(
                roc_array, self.lookback, self.factor
            )

        # Store for smoothing
        self.ultimate_c_values.append(ultimate_c)

        # Smooth the result (Cython when available, else Numba)
        if len(self.ultimate_c_values) < self.smooth_lookback:
            smoothed_ultimate_c = ultimate_c
        else:
            ultimate_c_array = np.array(self.ultimate_c_values, dtype=np.float64)
            if CYTHON_NODES_AVAILABLE and smooth_ultimate_c_fast is not None:
                smoothed_ultimate_c = smooth_ultimate_c_fast(
                    ultimate_c_array, self.smooth_lookback
                )
            else:
                smoothed_ultimate_c = smooth_ultimate_c(
                    ultimate_c_array, self.smooth_lookback
                )
        
        self.output.append(smoothed_ultimate_c)
        return [smoothed_ultimate_c]
