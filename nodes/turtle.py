from typing import List, Optional
from utils.enums import Bias, TimeFrame, Ticker, BiasStrategy
from nodes.bias_nodes import BiasNode
from utils.models import Candle
from nodes.bias_nodes.donchian_channel import DonchianChannel
import collections


class TurtleTrading(BiasNode):
    """
    Implements the Turtle Trading strategy as a bias node.
    
    The classic Turtle Trading strategy uses:
    - Entry: Breakout of highest high/lowest low of preceding 4 weeks
    - Stop: Breakout of lowest low/highest high of preceding 2 weeks
    - Filter: Only trade if previous signal resulted in a loss
    
    This implementation allows for configurable entry and stop lookback periods
    and works with any timeframe, not just weekly.
    """
    def __init__(self, ticker: Ticker, tf: TimeFrame, entry_lookback: int, stop_lookback: int):
        """
        Initializes the TurtleTrading bias node.

        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        - entry_lookback (int): Number of periods to look back for entry signals (classic: 4)
        - stop_lookback (int): Number of periods to look back for stop signals (classic: 2)

        Returns: None
        """
        super().__init__(BiasStrategy.TurtleTrading, ticker, tf)

        if not (entry_lookback > 0 and stop_lookback > 0):
            raise ValueError("Lookback periods must be positive integers.")

        self.entry_lookback = entry_lookback
        self.stop_lookback = stop_lookback
        
        # We need enough history to calculate the channels plus some extra for tracking positions
        # Max lookback + buffer for tracking previous trades
        self.required_history_len = max(entry_lookback, stop_lookback) + 10
        self.candles_history = collections.deque(maxlen=self.required_history_len)
        
        # For tracking current position and previous trade results
        self.current_position = 0  # 0=flat, 1=long, -1=short
        self.previous_trade_result = None  # None=no previous trade, True=win, False=loss
        self.entry_price = None  # Price at which position was entered
        
        self.columns = [f'turtle_{entry_lookback}_{stop_lookback}']

    def _calculate_donchian_channel(self, lookback: int) -> tuple[float, float]:
        """
        Calculates the Donchian channel (highest high and lowest low) over the specified lookback period.
        Uses the DonchianChannel class implementation.

        Parameters:
        - lookback (int): Number of periods to look back.

        Returns:
        - tuple[float, float]: (highest_high, lowest_low) over the lookback period.
        """
        # Create a temporary DonchianChannel instance with the same candles history
        temp_donchian = DonchianChannel(self.ticker, self.tf, lookback)
        temp_donchian.candles_history = self.candles_history.copy()
        
        # Use the DonchianChannel's method to calculate the channel
        return temp_donchian.calculate_donchian_channel(lookback)

    def _compute_candle(self, candle: Candle) -> List[float]:
        """
        Responds to a new candle being added and computes the Turtle Trading bias.

        Parameters:
        - candle (Candle): The latest candle data.

        Returns:
        - List[float]: A list containing the computed bias value (1=long, 0=flat, -1=short).
        """
        self.candles_history.append(candle)

        if len(self.candles_history) <= max(self.entry_lookback, self.stop_lookback):
            return [0.0]  # Not enough data yet to perform calculations

        # Calculate entry and stop channels
        entry_high, entry_low = self._calculate_donchian_channel(self.entry_lookback)
        stop_high, stop_low = self._calculate_donchian_channel(self.stop_lookback)
        
        # Current price
        current_price = candle.close
        
        # Previous position before any updates
        previous_position = self.current_position
        
        # Check for position changes
        if self.current_position == 0:  # Currently flat
            # Only enter new positions if previous trade was a loss or no previous trade
            if self.previous_trade_result is None or self.previous_trade_result is False:
                # Long entry: price breaks above entry high
                if current_price > entry_high:
                    self.current_position = 1
                    self.entry_price = current_price
                # Short entry: price breaks below entry low
                elif current_price < entry_low:
                    self.current_position = -1
                    self.entry_price = current_price
        
        elif self.current_position == 1:  # Currently long
            # Exit long if price breaks below stop low
            if current_price < stop_low:
                # Record trade result
                self.previous_trade_result = (current_price > self.entry_price) if self.entry_price else None
                self.current_position = 0
                self.entry_price = None
        
        elif self.current_position == -1:  # Currently short
            # Exit short if price breaks above stop high
            if current_price > stop_high:
                # Record trade result
                self.previous_trade_result = (current_price < self.entry_price) if self.entry_price else None
                self.current_position = 0
                self.entry_price = None
        
        # Return the current position as the bias
        return [float(self.current_position)]