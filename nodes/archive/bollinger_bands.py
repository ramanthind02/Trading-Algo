from nodes import BiasNode
from utils.core.models import Candle
from utils.core.enums import Bias, Ticker, TimeFrame
from typing import List
import numpy as np
from collections import deque
from scipy.stats import norm


class BollingerBandsNode(BiasNode):
    """
    BollingerBandsNode computes the distance from the current price to Bollinger Band bands,
    normalized by standard deviation and scaled to be centered around 0.
    
    Bollinger Bands consist of:
    - Middle line: SMA (Simple Moving Average) of close prices
    - Upper band: SMA + (multiplier × standard deviation)
    - Lower band: SMA - (multiplier × standard deviation)
    
    This feature measures how far the current price is from the band edges, adjusted for
    volatility. The output is transformed through a normal CDF and scaled to [-50, 50] range.
    
    Formula:
    1. Compute SMA of close prices over ma_length period
    2. Compute standard deviation over ma_length period
    3. Upper band = SMA + (multiplier × std_dev)
    4. Lower band = SMA - (multiplier × std_dev)
    5. If price > upper: distance = (price - upper) / std_dev
       If price < lower: distance = (price - lower) / std_dev
       Else: distance = 0 (price within bands)
    6. Transform through normal CDF and scale: 100 * norm_cdf(compression * distance) - 50
    
    This centers the output around 0 with typical range of [-50, 50].
    Positive values indicate price above upper band, negative values indicate below lower band.
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, ma_length: int = 20, 
                 std_mult: float = 2.0, compression: float = 1.0):
        """
        Initializes the BollingerBandsNode bias node.
        
        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        - ma_length (int): The lookback period for SMA and std dev calculation (default: 20).
        - std_mult (float): Multiplier for standard deviation to create bands (default: 2.0).
        - compression (float): Compression factor for the output (default: 1.0).
                              Increase for more compression, decrease for less.
        
        Returns: None
        """
        super().__init__(ticker, tf)
        self.ma_length = ma_length
        self.std_mult = std_mult
        self.compression = compression
        
        # Storage for close prices
        self.close_prices = deque(maxlen=ma_length)
        
        # Track how many candles we've seen
        self.candle_count = 0
        self.front_bad = ma_length
        
        # Define the columns attribute required by the MLManager
        self.columns = [f'bollinger_{ma_length}_{std_mult}']
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Computes the Bollinger Bands feature for the current candle.
        
        Parameters:
        - candle (Candle): The candle to compute the feature for
        
        Returns:
        - List: A list containing the normalized distance from Bollinger bands centered around 0
        """
        self.candle_count += 1
        
        # Add current close to history
        self.close_prices.append(candle.close)
        
        # If we don't have enough data, return neutral value (0.0)
        if self.candle_count < self.front_bad:
            self.bias = Bias.NEUTRAL
            self.output = [0.0]
            return self.output
        
        # Compute SMA (Simple Moving Average)
        sma = np.mean(self.close_prices)
        
        # Compute standard deviation
        std_dev = np.std(self.close_prices, ddof=1)  # Use sample std dev (ddof=1)
        
        # Compute Bollinger Band bands
        upper_band = sma + (self.std_mult * std_dev)
        lower_band = sma - (self.std_mult * std_dev)
        
        # Compute distance from bands, normalized by standard deviation
        if std_dev > 0.0:
            if candle.close > upper_band:
                # Price above upper band (overbought)
                distance = (candle.close - upper_band) / std_dev
            elif candle.close < lower_band:
                # Price below lower band (oversold)
                distance = (candle.close - lower_band) / std_dev
            else:
                # Price within bands (neutral)
                distance = 0.0
            
            # Transform through normal CDF and scale to [-50, 50]
            output_value = 100.0 * norm.cdf(self.compression * distance) - 50.0
        else:
            output_value = 0.0
        
        # Update bias based on output value
        if output_value > 0:
            self.bias = Bias.BULLISH
        elif output_value < 0:
            self.bias = Bias.BEARISH
        else:
            self.bias = Bias.NEUTRAL
        
        # Store the output
        self.output = [output_value]
        
        return self.output