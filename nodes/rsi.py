from typing import List
from utils.models import Candle
from utils.enums import Ticker, TimeFrame
from nodes import BiasNode


class RSI(BiasNode):
    """
    RSI (Relative Strength Index) Bias Node
    
    Computes the standard RSI indicator using exponential moving average
    of up and down price movements.
    
    RSI = 100 * (average_gain) / (average_gain + average_loss)
    
    The RSI oscillates between 0 and 100, with values above 70 typically
    considered overbought and values below 30 considered oversold.
    
    Parameters:
    - lookback: Period for RSI calculation (default: 14)
    """
    
    def __init__(self, ticker: Ticker, tf: TimeFrame, lookback: int = 14):
        """
        Initialize RSI node
        
        Parameters:
        - ticker: The ticker symbol
        - tf: The timeframe
        - lookback: RSI period (default: 14)
        """
        super().__init__(ticker, tf)
        
        self.lookback = lookback
        
        # Number of candles needed before we can compute valid output
        self.front_bad = lookback
        
        # RSI computation state
        self.upsum = 1e-60  # Small value to avoid division by zero
        self.dnsum = 1e-60
        
        # Price history for RSI computation
        self.close_prices = []
        
        # Column name for output
        self.columns = [f'rsi_{lookback}']
    
    def _compute_candle(self, candle: Candle) -> List:
        """
        Compute RSI for the given candle
        
        Parameters:
        - candle: The candle to process
        
        Returns:
        - List containing the RSI value
        """
        self.close_prices.append(candle.close)
        n = len(self.close_prices)
        
        # Return neutral value (50.0) until we have enough data
        if n < self.front_bad:
            self.output.append(50.0)
            return [50.0]
        
        # Initialize RSI on the first valid computation
        if n == self.front_bad:
            # Initialize with simple average of gains and losses
            for i in range(1, self.front_bad):
                diff = self.close_prices[i] - self.close_prices[i-1]
                if diff > 0.0:
                    self.upsum += diff
                else:
                    self.dnsum -= diff
            
            # Convert to average
            self.upsum /= (self.lookback - 1)
            self.dnsum /= (self.lookback - 1)
        
        # Update RSI using exponential moving average
        diff = self.close_prices[-1] - self.close_prices[-2]
        
        if diff > 0.0:
            # Price went up
            self.upsum = ((self.lookback - 1) * self.upsum + diff) / self.lookback
            self.dnsum *= (self.lookback - 1.0) / self.lookback
        else:
            # Price went down
            self.dnsum = ((self.lookback - 1) * self.dnsum - diff) / self.lookback
            self.upsum *= (self.lookback - 1.0) / self.lookback
        
        # Compute RSI
        rsi = 100.0 * self.upsum / (self.upsum + self.dnsum)
        
        self.output.append(rsi)
        return [rsi]
