from nodes import BiasNode
from utils.models import Candle
from utils.enums import Bias, Ticker, TimeFrame
from typing import List
import numpy as np

class ReturnNode(BiasNode):
    """
    ReturnNode is a bias node that outputs the return of the candle.
    This serves as a canary value to identify lookahead bias.
    
    The return is calculated as the percentage change between the close price
    and the open price of the candle.
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame):
        """
        Initializes the ReturnNode bias node.
        
        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        
        Returns: None
        """
        super().__init__(ticker, tf)
        # Standardized naming metadata
        self.module_name = 'prev_return'
        self.output_features = ['return', 'log_return', 'sign', 'is_bullish']
        self.params = {}
        # Define the columns attribute required by the MLManager
        self.columns = ['return', "log_return", "sign", "is_bullish"]
        # Define standardized columns
        self.ensure_standardized_columns()
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Computes the return of the candle as a percentage change
        
        Parameters:
        - candle (Candle): The candle to compute the return for
        
        Returns:
        - List: A list containing the return value as a percentage
        """
        # Calculate return as (close - open) / open * 100
        
        candle_return = ((candle.close - candle.open) / candle.open) * 100
        
        # Update bias based on return value
        if candle_return > 0:
            self.bias = Bias.BULLISH
        elif candle_return < 0:
            self.bias = Bias.BEARISH
        else:
            self.bias = Bias.NEUTRAL
        
        log_return = np.log(candle.close/candle.open)
        # Store the return value in output
        is_bullish = candle_return > 0
        self.output = [candle_return, log_return, np.sign(candle_return), is_bullish]
        
        return self.output
    