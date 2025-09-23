from typing import List, Tuple
from utils.enums import Bias, TimeFrame, Ticker, BiasStrategy
from nodes.bias_nodes import BiasNode
from utils.models import Candle
import collections


class DonchianChannel(BiasNode):
    """
    Implements the Donchian Channel strategy as a bias node.
    
    The Donchian Channel strategy is always in the market and uses a single channel:
    - Long when price breaks above the highest high of the lookback period
    - Short when price breaks below the lowest low of the lookback period
    
    Unlike the Turtle Trading strategy, this implementation:
    1. Is always in the market (no flat state)
    2. Uses a single lookback period for both entry and exit
    3. Does not filter trades based on previous results
    """
    def __init__(self, ticker: Ticker, tf: TimeFrame, lookback: int):
        """
        Initializes the DonchianChannel bias node.

        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        - lookback (int): Number of periods to look back for channel calculation.

        Returns: None
        """
        super().__init__(BiasStrategy.DonchianChannel, ticker, tf)

        if not (lookback > 0):
            raise ValueError("Lookback period must be a positive integer.")

        self.lookback = lookback
        
        # We need enough history to calculate the channels plus one for the current candle
        self.required_history_len = lookback + 1
        self.candles_history = collections.deque(maxlen=self.required_history_len)
        
        # For tracking current position
        self.current_position = 1  # Start with long position (1=long, -1=short)
        
        self.columns = [f'donchian_{lookback}']

    def calculate_donchian_channel(self, lookback: int) -> Tuple[float, float]:
        """
        Calculates the Donchian channel (highest high and lowest low) over the specified lookback period.

        Parameters:
        - lookback (int): Number of periods to look back.

        Returns:
        - Tuple[float, float]: (highest_high, lowest_low) over the lookback period.
        """
        if len(self.candles_history) < lookback:
            return 0.0, 0.0
            
        # Get the most recent candles excluding the current one
        recent_candles = list(self.candles_history)[-(lookback+1):-1]
        
        if not recent_candles:
            return 0.0, 0.0
            
        highest_high = max(candle.high for candle in recent_candles)
        lowest_low = min(candle.low for candle in recent_candles)
        
        return highest_high, lowest_low

    def _compute_candle(self, candle: Candle) -> List[float]:
        """
        Responds to a new candle being added and computes the Donchian Channel bias.

        Parameters:
        - candle (Candle): The latest candle data.

        Returns:
        - List[float]: A list containing the computed bias value (1=long, -1=short).
        """
        self.candles_history.append(candle)

        if len(self.candles_history) <= self.lookback:
            return [float(self.current_position)]  # Return current position if not enough data

        # Calculate Donchian channel
        highest_high, lowest_low = self.calculate_donchian_channel(self.lookback)
        
        # Current price
        current_price = candle.close
        
        # Check for position changes
        if self.current_position == 1:  # Currently long
            # Switch to short if price breaks below the channel low
            if current_price < lowest_low:
                self.current_position = -1
        
        elif self.current_position == -1:  # Currently short
            # Switch to long if price breaks above the channel high
            if current_price > highest_high:
                self.current_position = 1
        
        # Return the current position as the bias
        return [float(self.current_position)]
