from typing import List, Optional, Dict, Any
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame
from nodes import BiasNode
from nodes.ewsd import EWSDNode
from collections import deque
from utils.compute.fast_nodes import compute_roc_fast


class ROC(BiasNode):
    """
    ROC (Rate of Change) Bias Node
    
    Computes the Rate of Change indicator, which is a normalized version of momentum.
    Also known as "Momentum" in technical analysis, but this implementation uses
    the percentage-based ROC formula.
    
    ROC(t) = (Close(t) - Close(t-x)) / Close(t-x) * 100
    
    where x is the lookback period.
    
    Optionally, ROC can be normalized by EWSD (Exponentially Weighted Standard Deviation)
    to create a volatility-adjusted measure:
    
    ROC_normalized(t) = ROC(t) / EWSD_daily_pct(t)
    
    The raw momentum (difference) can be computed as:
    Momentum(t) = Close(t) - Close(t-x)
    
    Parameters:
    - lookback: Period for ROC calculation (default: 10)
    - normalize_by_ewsd: Whether to normalize ROC by EWSD (default: False)
    - ewsd_params: Optional parameters for EWSD node (default: None, uses defaults)
    """
    
    def __init__(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        lookback: int = 10,
        normalize_by_ewsd: bool = False,
        ewsd_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ROC node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - lookback: ROC period (default: 10)
        - normalize_by_ewsd: Whether to normalize ROC by EWSD (default: False)
        - ewsd_params: Optional dict of parameters for EWSD node.
          Keys can include: lambda_short, long_run_window, blend_short_weight, blend_long_weight
          If None, uses EWSD defaults
        """
        super().__init__(ticker, tf)
        
        self.lookback = lookback
        self.normalize_by_ewsd = normalize_by_ewsd
        
        # Create EWSD node if normalization is enabled
        self.ewsd_node: Optional[EWSDNode] = None
        if normalize_by_ewsd:
            ewsd_params = ewsd_params or {}
            self.ewsd_node = EWSDNode(
                ticker=ticker,
                tf=tf,
                lambda_short=ewsd_params.get('lambda_short', 0.06061),
                long_run_window=ewsd_params.get('long_run_window', 252),
                blend_short_weight=ewsd_params.get('blend_short_weight', 0.7),
                blend_long_weight=ewsd_params.get('blend_long_weight', 0.3)
            )
        
        # Standardized naming metadata
        self.module_name = 'roc'
        self.output_features = ['signal']
        # Only include lookback in params for column naming
        # Don't include normalize_by_ewsd or ewsd_params to avoid confusion
        self.params = {
            'lookback': lookback
        }
        
        # Number of candles needed before we can compute valid output
        # We need lookback+1 candles: current + lookback historical
        # If normalizing by EWSD, we also need EWSD to be ready (it needs ~20 bars minimum)
        self.front_bad = lookback
        if normalize_by_ewsd:
            # EWSD needs at least 20 bars for long-run estimate
            self.front_bad = max(lookback, 20)
        
        # Store closing prices in a deque (circular buffer)
        # We need to keep lookback+1 prices to compare current with lookback periods ago
        # When buffer is full, it will have: [close(t-lookback), ..., close(t-1), close(t)]
        self.close_buffer = deque(maxlen=lookback + 1)
        self.n_prices = 0
        
        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute ROC for the given candle.
        
        If normalize_by_ewsd is True, the ROC will be divided by EWSD daily percentage
        to create a volatility-adjusted measure.
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the ROC value (as percentage) or normalized ROC (if normalize_by_ewsd=True)
        """
        curr_close = candle.close
        self.n_prices += 1
        
        # Return neutral value (0.0) until we have enough data
        if self.n_prices <= self.front_bad:
            # Still need to add to buffer for tracking
            self.close_buffer.append(curr_close)
            # Also update EWSD node if normalization is enabled
            if self.normalize_by_ewsd and self.ewsd_node is not None:
                self.ewsd_node.add_candle(candle)
            self.output.append(0.0)
            return [0.0]
        
        # Add current close to buffer (will automatically remove oldest if full)
        # When n_prices == lookback + 1, the deque will have lookback+1 items
        # and the oldest (index 0) will be the close from lookback periods ago
        self.close_buffer.append(curr_close)
        
        # Calculate ROC: (Close(t) - Close(t-x)) / Close(t-x) * 100
        # The deque automatically maintains the lookback period
        # The oldest value is at index 0, which is lookback periods ago
        past_close = self.close_buffer[0]
        
        # Use Cython-backed fast kernel for ROC computation
        # compute_roc_fast handles division by zero safely
        roc = compute_roc_fast(curr_close, past_close)
        
        # Normalize by EWSD if enabled
        if self.normalize_by_ewsd and self.ewsd_node is not None:
            # Get EWSD value (returns [ewsd_daily_pct, ewsd_annual_pct])
            ewsd_result = self.ewsd_node.add_candle(candle)
            if ewsd_result and len(ewsd_result) >= 1:
                ewsd_daily_pct = ewsd_result[0]  # Daily EWSD as percentage
                if ewsd_daily_pct != 0:
                    # Normalize: ROC / EWSD
                    # Both are percentages, so result is unitless (ratio)
                    roc = roc / ewsd_daily_pct
                else:
                    # Avoid division by zero
                    roc = 0.0
            else:
                # EWSD not ready yet, return 0
                roc = 0.0
        
        self.output.append(roc)
        return [roc]

