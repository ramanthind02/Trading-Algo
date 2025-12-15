"""
MT5 Data Connector for Live Forecasting

Adapted from samples to fetch live market data for forecasting.
Provides single candle fetching for real-time predictions.
"""

import os
import sys
from typing import Optional
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import Candle
from utils.logger import get_logger
from utils.enums import TimeFrame, Ticker
import utils.helpers as helpers

logger = get_logger(__name__)
NY_TZ = ZoneInfo("America/New_York")

class ForecastMT5DataConnector:
    """
    MT5 Data connector specialized for forecasting operations.
    
    Fetches single candles for real-time forecasting rather than 
    historical data batches.
    """
    
    def __init__(self):
        """Initialize MT5 connection for forecasting."""
        self.username = int(os.environ.get('MT5_USERNAME', '1511850895'))
        self.password = os.environ.get('MT5_PASSWORD', '@?v$vXjh4@!8$?')
        self.server = os.environ.get('MT5_SERVER', 'FTMO-Demo')
        self.path = 'C:/Program Files/FTMO MetaTrader 5/terminal64.exe'
        
        # MT5 timeframe mappings (only for available timeframes)
        self.timeframe_to_enum = {
            TimeFrame.D: mt5.TIMEFRAME_D1,
            TimeFrame.W: mt5.TIMEFRAME_W1,
            TimeFrame.M: mt5.TIMEFRAME_MN1
        }
        
        # Ticker to MT5 symbol mappings (only for existing tickers)
        self.ticker_to_symbol = {
            Ticker.ES: 'US500.cash',
            Ticker.NQ: 'US100.cash',
            Ticker.BP: 'GBPUSD',
            Ticker.EU: 'EURUSD',
            Ticker.GC: 'XAUUSD',
            Ticker.YM: 'US30.cash',
            # Additional mappings for forecasting
            'EURUSD': 'EURUSD',
            'GBPUSD': 'GBPUSD', 
            'US500': 'US500.cash',
            'US100': 'US100.cash'
        }
        
        # Initialize connection
        self.authenticate()
    
    def authenticate(self, first_try: bool = True) -> bool:
        """
        Authenticate with MT5 platform.
        
        Parameters
        ----------
        first_try : bool
            Whether this is the first attempt
            
        Returns
        -------
        bool
            True if authentication successful
        """
        try:
            if mt5.initialize(self.path, login=self.username, password=self.password, server=self.server):
                if mt5.login(login=self.username, password=self.password, server=self.server):
                    logger.info("MT5 authentication successful")
                    return True
                else:
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
            else:
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
        except Exception as e:
            logger.error(f"MT5 authentication exception: {e}")
        
        if first_try:
            logger.info('Retrying MT5 authentication...')
            return self.authenticate(first_try=False)
        
        return False
    
    def get_historical_candles(self, ticker: str, timeframe: TimeFrame, count: int = 100) -> list[Candle]:
        """
        Get multiple historical candles for a ticker and timeframe.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol (e.g., 'EURUSD', 'US500')
        timeframe : TimeFrame
            Timeframe for the candles
        count : int
            Number of candles to fetch (default 100)
            
        Returns
        -------
        list[Candle]
            List of historical candles, newest last
        """
        try:
            # Convert ticker to MT5 symbol
            if isinstance(ticker, Ticker):
                symbol = self.ticker_to_symbol.get(ticker)
            else:
                # Handle string tickers
                symbol = self.ticker_to_symbol.get(ticker, ticker)
            
            if symbol is None:
                logger.error(f"Unknown ticker: {ticker}")
                return []
            
            # Get MT5 timeframe
            mt5_timeframe = self.timeframe_to_enum.get(timeframe)
            if mt5_timeframe is None:
                logger.error(f"Unknown timeframe: {timeframe}")
                return []
            
            # Get current time and fetch candles
            now = datetime.now(NY_TZ)
            
            # Fetch requested number of candles
            rates = mt5.copy_rates_from(symbol, mt5_timeframe, helpers.convert_ny_time_to_ftmo_time(now), count)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No historical data received from MT5 for {ticker} {timeframe}")
                return []
            
            # Convert to Candle objects
            candles = []
            for candle_data in rates:
                candle_time = helpers.convert_ftmo_time_to_ny_time(candle_data['time'])
                
                candle = Candle(
                    open=float(candle_data['open']),
                    close=float(candle_data['close']),
                    high=float(candle_data['high']),
                    low=float(candle_data['low']),
                    volume=int(candle_data.get('tick_volume', 0)),
                    datetime=candle_time,
                    ticker=ticker if isinstance(ticker, Ticker) else Ticker[ticker] if hasattr(Ticker, ticker) else None,
                    tf=timeframe
                )
                candles.append(candle)
            
            logger.info(f"Retrieved {len(candles)} historical candles for {ticker} {timeframe}")
            return candles
            
        except Exception as e:
            logger.error(f"Error getting historical candles for {ticker} {timeframe}: {e}")
            # Try to re-authenticate once
            if self.authenticate():
                logger.info("Re-authenticated, retrying candle fetch...")
                return self.get_historical_candles(ticker, timeframe, count)
            return []
    
    def get_latest_candle(self, ticker: str, timeframe: TimeFrame) -> Optional[Candle]:
        """
        Get the most recent closed candle for a ticker and timeframe.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol (e.g., 'EURUSD', 'US500')
        timeframe : TimeFrame
            Timeframe for the candle
            
        Returns
        -------
        Optional[Candle]
            Latest closed candle, or None if error
        """
        try:
            # Convert ticker to MT5 symbol
            if isinstance(ticker, Ticker):
                symbol = self.ticker_to_symbol.get(ticker)
            else:
                # Handle string tickers
                symbol = self.ticker_to_symbol.get(ticker, ticker)
            
            if symbol is None:
                logger.error(f"Unknown ticker: {ticker}")
                return None
            
            # Get MT5 timeframe
            mt5_timeframe = self.timeframe_to_enum.get(timeframe)
            if mt5_timeframe is None:
                logger.error(f"Unknown timeframe: {timeframe}")
                return None
            
            # Get current time and fetch recent candles
            now = datetime.now(NY_TZ)
            
            # Fetch last 2 candles to ensure we get the most recent closed one
            rates = mt5.copy_rates_from(symbol, mt5_timeframe, helpers.convert_ny_time_to_ftmo_time(now), 2)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data received from MT5 for {ticker} {timeframe}")
                return None
            
            # Use the second-to-last candle (most recent closed)
            # The last candle might still be forming
            candle_data = rates[-2] if len(rates) > 1 else rates[-1]
            
            # Convert MT5 time to NY time
            candle_time = helpers.convert_ftmo_time_to_ny_time(candle_data['time'])
            
            # Create Candle object
            candle = Candle(
                open=float(candle_data['open']),
                close=float(candle_data['close']),
                high=float(candle_data['high']),
                low=float(candle_data['low']),
                volume=int(candle_data.get('tick_volume', 0)),
                datetime=candle_time,
                ticker=ticker if isinstance(ticker, Ticker) else Ticker[ticker] if hasattr(Ticker, ticker) else None,
                tf=timeframe
            )
            
            logger.debug(f"Retrieved candle for {ticker} {timeframe}: {candle_time}")
            return candle
            
        except Exception as e:
            logger.error(f"Error getting latest candle for {ticker} {timeframe}: {e}")
            # Try to re-authenticate once
            if self.authenticate():
                logger.info("Re-authenticated, retrying candle fetch...")
                return self.get_latest_candle(ticker, timeframe)
            return None
    
    def is_market_open(self, ticker: str = 'EURUSD') -> bool:
        """
        Check if the market is open for trading.
        
        Parameters
        ----------
        ticker : str
            Ticker to check (default EURUSD as forex is most active)
            
        Returns
        -------
        bool
            True if market appears to be open
        """
        try:
            # Try to get a recent daily candle (since we don't have M1)
            candle = self.get_latest_candle(ticker, TimeFrame.D)
            if candle is None:
                return False
            
            # Check if the candle is recent (within last day)
            now = datetime.now(NY_TZ)
            time_diff = now - candle.datetime
            
            # Market is considered open if we got a candle within last day
            return time_diff <= timedelta(days=1)
            
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
    
    def get_symbol_info(self, ticker: str) -> Optional[dict]:
        """
        Get symbol information from MT5.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
            
        Returns
        -------
        Optional[dict]
            Symbol info or None if error
        """
        try:
            symbol = self.ticker_to_symbol.get(ticker, ticker)
            info = mt5.symbol_info(symbol)
            
            if info is None:
                return None
            
            return {
                'symbol': symbol,
                'bid': info.bid,
                'ask': info.ask,
                'spread': info.spread,
                'digits': info.digits,
                'point': info.point
            }
            
        except Exception as e:
            logger.error(f"Error getting symbol info for {ticker}: {e}")
            return None
    
    def shutdown(self) -> None:
        """Shutdown MT5 connection."""
        try:
            mt5.shutdown()
            logger.info("MT5 connection closed")
        except Exception as e:
            logger.error(f"Error closing MT5 connection: {e}")
    
    def __del__(self):
        """Cleanup on object destruction."""
        self.shutdown()

# Utility function for easy access
def create_mt5_connector() -> ForecastMT5DataConnector:
    """
    Create and return an MT5 data connector instance.
    
    Returns
    -------
    ForecastMT5DataConnector
        Configured MT5 connector
    """
    return ForecastMT5DataConnector()
