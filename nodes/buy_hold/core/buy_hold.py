from typing import ClassVar, List
from utils.core.models import Candle
from utils.core.enums import Ticker, TimeFrame
from nodes import BiasNode


class BuyHold(BiasNode):
    """
    Buy and Hold Bias Node
    
    A simple bias node that always outputs 1.0, representing a buy-and-hold strategy.
    This is useful for baseline comparisons and testing the vault system.
    
    The output is constant at 1.0 for all candles, meaning it always signals
    a long position (buy and hold).
    
    Parameters:
    - None (no parameters needed)
    """

    hardcoded_lookbacks: ClassVar[tuple[tuple[str, int], ...]] = (("startup_bars", 1),)
    
    def __init__(self, ticker: Ticker, tf: TimeFrame):
        """
        Initialize Buy and Hold node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        """
        super().__init__(ticker, tf)
        
        # Standardized naming metadata
        self.module_name = 'buy_hold'
        self.output_features = ['signal']
        self.params = {}
        
        # No warmup period needed - always outputs 1.0
        self.front_bad = 1
        
        # Define standardized columns
        self.ensure_standardized_columns()

        # Initialize cache after params are set
        self._init_cache_after_params()

    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute Buy and Hold signal for the given candle.
        
        Always returns 1.0, representing a constant buy-and-hold position.
        
        Parameters:
        - candle: The candle to process (not used, but required by interface)
        
        Returns:
        - List containing [1.0]
        """
        self.output.append(1.0)
        return [1.0]
