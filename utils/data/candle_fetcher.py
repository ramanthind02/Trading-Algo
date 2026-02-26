import numba as nb
import numpy as np
import utils.core.helpers as helpers
from typing import List, Dict, Optional
from utils.core.enums import Ticker, TimeFrame


@nb.njit
def get_candle(data: np.ndarray, start_ts: int, method: str) -> Optional[Dict]:
    """
    Helper function using numba to retrieve candles from numpy data structure efficiently

    Parameters:
    - data (np.ndarray): two-dimensional numpy array containing all data for specific timeframe
    - start_ts (int): timestamp of candle to query for
    - method (str): method to use when searching for candle

    Returns:
        - Optional[Dict]: dictionary object with candle data or None if not found
    """
    if method not in ['exact', 'closest']:
        raise ValueError(f"Invalid method to fetch candle: {method}. Only 'exact' and 'closest' are supported.")

    idx = np.searchsorted(data['datetime'], start_ts)
    
    if idx == len(data['datetime']):
        return None
    if method == 'exact' and data['datetime'][idx] != start_ts:
        return None

    open = data['open'][idx]
    close = data['close'][idx]
    high = data['high'][idx]
    low = data['low'][idx]
    datetime = data['datetime'][idx]
    return open, close, high, low, datetime


class CandleFetcher:

    def __init__(self, ticker: Ticker, tfs: List[TimeFrame]):
        """
        Class to fetch candle data from numpy dataframes

        Parameters:
        - ticker (Ticker): ticker of the asset to fetch candles for
        - tfs (List[TimeFrame]): list of timeframes to load data for

        Returns: None
        """
        self.ticker = ticker
        self.data = {tf: helpers.load_numpy_data(self.ticker, tf) for tf in tfs}

    def get_candle(self, tf: TimeFrame, start_ts: int, method: str = 'exact') -> Optional[np.ndarray]:
        """
        Fetches candle for a specific timeframe and timestamp

        Parameters:
        - tf (TimeFrame): timeframe to search for candle in
        - start_ts (int): timestamp indicating candle open
        - method (str): method to search for candle, default is exact

        Returns:
        - Optional[np.ndarray]: numpy array with candle data or None if not found
        """
        result = get_candle(self.data[tf], start_ts, method)
        if result is None:
            return None
        open, close, high, low, datetime = result
        return np.array([(open, close, high, low, datetime)], 
                       dtype=[('open', np.float32), ('close', np.float32), 
                             ('high', np.float32), ('low', np.float32),
                             ('datetime', np.uint32)])[0]
