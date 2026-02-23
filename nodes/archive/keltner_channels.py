from nodes import BiasNode
from utils.core.models import Candle
from utils.core.enums import Bias, Ticker, TimeFrame
from typing import List, Optional
import numpy as np
from collections import deque
from scipy.stats import norm


class KeltnerChannelsNode(BiasNode):
    """
    KeltnerChannelsNode computes the distance from the current price to Keltner Channel bands,
    normalized by ATR and scaled to be centered around 0.
    
    Keltner Channels consist of:
    - Middle line: EMA of close prices
    - Upper band: EMA + (multiplier × ATR)
    - Lower band: EMA - (multiplier × ATR)
    
    This feature measures how far the current price is from the channel bands, adjusted for
    volatility. The output is transformed through a normal CDF and scaled to [-50, 50] range.
    
    Formula:
    1. Compute EMA of close prices over ema_length period
    2. Compute ATR over atr_length period
    3. Upper band = EMA + (multiplier × ATR)
    4. Lower band = EMA - (multiplier × ATR)
    5. If price > upper: distance = (price - upper) / ATR
       If price < lower: distance = (price - lower) / ATR
       Else: distance = 0 (price within bands)
    6. Transform through normal CDF and scale: 100 * norm_cdf(compression * distance) - 50
    
    This centers the output around 0 with typical range of [-50, 50].
    Positive values indicate price above upper band, negative values indicate below lower band.
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, ema_length: int = 20, 
                 atr_length: int = 20, multiplier: float = 2.0, compression: float = 1.0):
        """
        Initializes the KeltnerChannelsNode bias node.
        
        Parameters:
        - ticker (Ticker): The ticker symbol.
        - tf (TimeFrame): The timeframe of the candles.
        - ema_length (int): The lookback period for EMA calculation (default: 20).
        - atr_length (int): The period for ATR calculation (default: 20).
        - multiplier (float): Multiplier for ATR to create bands (default: 2.0).
        - compression (float): Compression factor for the output (default: 1.0).
                              Increase for more compression, decrease for less.
        
        Returns: None
        """
        super().__init__(ticker, tf)
        self.ema_length = ema_length
        self.atr_length = atr_length
        self.multiplier = multiplier
        self.compression = compression
        
        # EMA computation state
        self.ema_alpha = 2.0 / (ema_length + 1.0)
        self.ema: Optional[float] = None
        
        # Storage for ATR calculation
        self.prev_close: Optional[float] = None
        self.true_ranges = deque(maxlen=atr_length)
        
        # Track how many candles we've seen
        self.candle_count = 0
        self.front_bad = max(ema_length, atr_length)
        
        # Define the columns attribute required by the MLManager
        self.columns = [f'keltner_{ema_length}_{atr_length}_{multiplier}']
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Computes the Keltner Channels feature for the current candle.
        
        Parameters:
        - candle (Candle): The candle to compute the feature for
        
        Returns:
        - List: A list containing the normalized distance from Keltner bands centered around 0
        """
        self.candle_count += 1
        
        # Calculate True Range for ATR
        if self.prev_close is not None:
            hl = candle.high - candle.low
            hc = abs(candle.high - self.prev_close)
            lc = abs(candle.low - self.prev_close)
            true_range = max(hl, hc, lc)
        else:
            true_range = candle.high - candle.low
        
        self.true_ranges.append(true_range)
        
        # Update EMA
        if self.ema is None:
            # Initialize EMA with first close price
            self.ema = candle.close
        else:
            # Update EMA: EMA = alpha * close + (1 - alpha) * prev_EMA
            self.ema = self.ema_alpha * candle.close + (1.0 - self.ema_alpha) * self.ema
        
        # Update previous close
        self.prev_close = candle.close
        
        # If we don't have enough data, return neutral value (0.0)
        if self.candle_count <= self.front_bad:
            self.bias = Bias.NEUTRAL
            self.output = [0.0]
            return self.output
        
        # Compute ATR
        atr = np.mean(self.true_ranges) if len(self.true_ranges) > 0 else 0.0
        
        # Compute Keltner Channel bands
        upper_band = self.ema + (self.multiplier * atr)
        lower_band = self.ema - (self.multiplier * atr)
        
        # Compute distance from bands, normalized by ATR
        if atr > 0.0:
            if candle.close > upper_band:
                # Price above upper band (overbought)
                distance = (candle.close - upper_band) / atr
            elif candle.close < lower_band:
                # Price below lower band (oversold)
                distance = (candle.close - lower_band) / atr
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
