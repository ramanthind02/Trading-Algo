from typing import List
import numpy as np
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from nodes import BiasNode


class RandomFeature(BiasNode):
    """
    Random Feature Bias Node - For Testing Permutation Tests
    
    This node outputs random values uniformly distributed between -1 and 1.
    The random values are deterministically generated but with NO correlation to time.
    
    CRITICAL: Uses a counter-based approach to break time correlation:
    - Each candle gets a unique index (counter)
    - Random value is based on hash(counter) XOR hash(timestamp)
    - This ensures no correlation with time trends (e.g., NQ increasing over time)
    
    This serves as a negative control to verify that permutation tests correctly
    identify features with no predictive power.
    
    Expected behavior in permutation tests:
    - Should have p-value close to 1.0 (not significant)
    - Profit factor should be close to 1.0 (no edge)
    - Performance should not differ from permuted versions
    - NO correlation with time or market trends
    
    This is useful for:
    1. Validating that permutation tests work correctly
    2. Establishing baseline performance expectations
    3. Debugging feature selection pipelines
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame):
        """
        Initialize Random Feature node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        """
        super().__init__(ticker, tf)
        
        # No warmup period needed for random values
        self.front_bad = 0
        
        # Counter to track candle index (breaks time correlation)
        self.candle_counter = 0
        
        # Column name for output
        self.columns = ['random_feature']
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Generate a random value between -1 and 1 with NO time correlation.
        
        CRITICAL: Uses close price + counter + timestamp to ensure bar permutation works:
        - Close price ensures different OHLC data → different random values
        - Counter breaks sequential patterns
        - Timestamp provides base reproducibility
        - When bars are shuffled, close price at each timestamp changes → random values change
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing a random value in [-1, 1]
        """
        # Use close price + counter + timestamp to generate seed
        # This ensures bar permutation (which changes OHLC at each timestamp) produces different values
        timestamp_hash = int(candle.datetime.timestamp() * 1e6) % (2**31)
        counter_hash = hash(self.candle_counter) % (2**31)
        # CRITICAL: Include close price so shuffled bars produce different random values
        price_hash = hash(int(candle.close * 1e6)) % (2**31)
        
        # XOR all three to create final seed
        combined_seed = (timestamp_hash ^ counter_hash ^ price_hash) % (2**31)
        
        candle_rng = np.random.RandomState(combined_seed)
        random_value = candle_rng.uniform(-1.0, 1.0)
        
        # Increment counter for next candle
        self.candle_counter += 1
        
        self.output.append(random_value)
        return [random_value]
