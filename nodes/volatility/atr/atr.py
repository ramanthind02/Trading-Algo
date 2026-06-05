from nodes import BiasNode
from lib.core.models import Candle
from lib.core.enums import Bias, Ticker, TimeFrame
from typing import ClassVar, List, Optional
import numpy as np
from collections import deque

# Try to import Cython optimized version
try:
    from lib.compute.fast_nodes import compute_atr_fast, CYTHON_NODES_AVAILABLE
except ImportError:
    CYTHON_NODES_AVAILABLE = False

class ATRNode(BiasNode):
    """
    ATRNode is a bias node that computes the Average True Range (ATR) over a 252-period window.
    ATR measures market volatility by calculating the average of true ranges over the specified period.
    
    True Range is the maximum of:
    - Current High - Current Low
    - |Current High - Previous Close|
    - |Current Low - Previous Close|
    """
    lookback_param_names: ClassVar[frozenset[str]] = frozenset({"period"})
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, period: int = 252):
        """
        Initializes the ATRNode bias node.
        
        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        - period (int): The lookback period for ATR calculation (default: 252 for daily data).
        
        Returns: None
        """
        super().__init__(ticker, tf)
        self.period = period
        # Standardized naming metadata
        self.module_name = 'atr'
        self.output_features = ['atr', 'atrPct']
        self.params = {'period': period}
        
        if CYTHON_NODES_AVAILABLE:
            # Cython path: use circular buffer for 5-10x speedup
            self.prev_close: float = -1.0  # -1 indicates no previous close
            self.true_ranges = np.zeros(period, dtype=np.float64)
            self.buffer_idx = 0
            self.n_filled = 0
        else:
            # Fallback: use deque
            self.prev_close: Optional[float] = None
            self.true_ranges = deque(maxlen=period)
        
        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Computes the ATR for the current candle.
        
        Uses Cython-optimized computation if available (5-10x faster),
        otherwise falls back to standard implementation.
        
        Parameters:
        - candle (Candle): The candle to compute the ATR for
        
        Returns:
        - List: A list containing [atr, atr_pct] where atr_pct is ATR as percentage of close price
        """
        if CYTHON_NODES_AVAILABLE:
            # Fast Cython path (5-10x faster)
            atr, atr_pct, self.buffer_idx, self.n_filled = compute_atr_fast(
                candle.high,
                candle.low,
                candle.close,
                self.prev_close,
                self.true_ranges,
                self.buffer_idx,
                self.n_filled,
                self.period
            )
            self.prev_close = candle.close
        else:
            # Fallback to standard implementation
            if self.prev_close is not None:
                hl = candle.high - candle.low
                hc = abs(candle.high - self.prev_close)
                lc = abs(candle.low - self.prev_close)
                true_range = max(hl, hc, lc)
            else:
                true_range = candle.high - candle.low
            
            self.true_ranges.append(true_range)
            
            if len(self.true_ranges) > 0:
                atr = np.mean(self.true_ranges)
            else:
                atr = 0.0
            
            atr_pct = (atr / candle.close) * 100 if candle.close > 0 else 0.0
            self.prev_close = candle.close
        
        # Set bias to NEUTRAL (ATR doesn't indicate direction)
        self.bias = Bias.NEUTRAL
        
        # Store the output
        self.output = [atr, atr_pct]
        
        return self.output
