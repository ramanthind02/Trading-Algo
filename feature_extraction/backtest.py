import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

import numpy as np
from typing import List, Optional

from utils.core.enums import TimeFrame
from utils.data.candle_fetcher import CandleFetcher
from utils.core.models import Candle
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, timezone
# No order imports needed
from utils.core.logger import get_logger


logger = get_logger(__name__)


class Backtest:

    def __init__(
        self,
        name: str,
        data: np.ndarray,
        ml_manager: MLManager,
        candle_fetcher: CandleFetcher,
        evaluate_tf: TimeFrame = TimeFrame.D,
        fast_mode: bool = False,
    ):
        """
        Initializes a backtest with a strategy

        Parameters:
        - name (str): name to identify the backtest
        - data (np.ndarray): M1 data to backtest
        - ml_manager (MLManager): ml manager object
        - candle_fetcher (CandleFetcher): candle fetcher object to get candles
        - evaluate_tf (TimeFrame): timeframe to execute trades on
        - fast_mode (bool): Use FastCandle instead of Pydantic Candle (5-10x faster, no validation)


        Returns: None
        """
        self.name = name
        self.ml_manager = ml_manager
        self.data = data
        self.candle_fetcher = candle_fetcher
        self.tz = ZoneInfo('UTC')
        self.evaluate_tf = evaluate_tf
        self.fast_mode = fast_mode

    
    def run(self) -> None:
        """
        Runs the backtest

        Parameters: None

        Returns:
        - None
        """
        from tqdm import tqdm

        for idx in tqdm(range(len(self.data)), desc=f"Running backtest {self.name}", unit="candles"):
            exec_candle = self.data[idx]
            end_dt = datetime.fromtimestamp(exec_candle['datetime'], tz=timezone.utc) + timedelta(seconds=self.evaluate_tf.value)

            # Evaluate orders
            exec_ny_candle = self._apply_mapping(exec_candle)

            # Process each timeframe
            prev_candle = self.data[idx-1] if idx > 0 else None
            self._process_timeframes(end_dt, prev_candle)

            # Process evaluate_tf candle and place orders
            self.ml_manager.add_candle(exec_ny_candle, self.evaluate_tf)

        return 
    
    def _process_timeframes(self, end_dt: datetime, prev_candle: Optional[np.ndarray]) -> None:
        """
        Process timeframes from Monthly to M1, checking for new orders at period starts.
        Returns the most recent order found, if any.

        Parameters:
        - end_dt (datetime): Current candle end datetime being processed
        - prev_candle (Optional[np.ndarray]): Previous candle data, used for monthly boundary check

        Returns:
        - None
        """

        # Monthly
        if prev_candle is not None:
            prev_month = datetime.fromtimestamp(prev_candle['datetime']).month
            curr_month = datetime.fromtimestamp(prev_candle['datetime']).month
            if prev_month != curr_month:
                year = datetime.fromtimestamp(prev_candle['datetime']).year
                month = prev_month
                start_dt = datetime(year, month, 1)
                start_dt = start_dt.replace(tzinfo=self.tz).astimezone(timezone.utc)
                self._get_timeframe_order(TimeFrame.M, start_dt.timestamp(), method='closest')


        # Weekly
        if self._is_end_of_week(end_dt):
            start_dt = end_dt - timedelta(hours=((24*4)+16))
            start_dt = start_dt.replace(tzinfo=self.tz).astimezone(timezone.utc)
            self._get_timeframe_order(TimeFrame.W, start_dt.timestamp())


        # Daily
        if self._is_start_of_day(end_dt):
            start_dt = end_dt - timedelta(hours=24 if not self._is_end_of_week(end_dt) else 16)
            start_dt = start_dt.replace(tzinfo=self.tz).astimezone(timezone.utc)
            self._get_timeframe_order(TimeFrame.D, start_dt.timestamp())


    
    def _get_timeframe_order(self, timeframe: TimeFrame, start_ts: float, method: Optional[str] = 'closest') -> None:
        """
        Helper function to get order for a timeframe by fetching the candle and passing it to the middleman

        Parameters:
        - timeframe (TimeFrame): timeframe to get candle for
        - start_ts (float): timestamp of candle start time
        - method (Optional[str]): method to use when fetching candle, either 'exact' or 'closest'

        Returns:
        - None
        """
        if timeframe is not self.evaluate_tf:
            candle = self.candle_fetcher.get_candle(tf=timeframe, start_ts=start_ts, method=method)
            if candle is not None:
                mapped_candle = self._apply_mapping(candle)
                self.ml_manager.add_candle(candle=mapped_candle, tf=timeframe)

    def _apply_mapping(self, candle: np.ndarray):
        """
        Helper function to map candle array to dictionary

        Parameters:
        - candle (np.ndarray): candle in numpy array format

        Returns:
        - Candle or FastCandle: candle with references to original data
        """
        if self.fast_mode:
            # Use FastCandle for 5-10x speedup (no validation overhead)
            from utils.compute.fast_candle import FastCandle
            return FastCandle.from_numpy(
                candle,
                ticker=self.ml_manager.ticker,
                tf=self.evaluate_tf
            )
        else:
            # Use Pydantic Candle (with validation)
            return Candle(
                open=candle['open'],
                close=candle['close'],
                high=candle['high'],
                low=candle['low'],
                volume=candle['volume'] if 'volume' in candle else 0,
                datetime=datetime.fromtimestamp(candle['datetime'], tz=timezone.utc),
                ticker=self.ml_manager.ticker,
                tf=self.evaluate_tf
            )

    
    def _is_start_of_day(self, date: datetime) -> bool:
        """
        Helper function to check if the datetime is at the start of day

        Parameters:
        - date (datetime): date to check

        Returns:
        - bool: whether the date is start of day or not
        """
        return (date.hour == 0 and date.minute == 0 and date.second == 0) or self._is_end_of_week(date)

    def _is_end_of_week(self, date: datetime) -> bool:
        """
        Helper function to check if the datetime is end of week

        Parameters:
        - date (datetime): date to check

        Returns:
        - bool: whether the date is end of week or not
        """
        return date.weekday() == 4 and date.hour == 16 and date.minute == 0 and date.second == 0
    
    def _get_start_ts_from_end_dt(self, end_dt: datetime, tf: TimeFrame) -> int:
        """
        Helper function to turn end datetime of candle to start timestamp

        Parameters:
        - end_dt (datetime): end datetime of the candle
        - tf (TimeFrame): timeframe of the candle

        Returns:
        int: start timestamp of the candle 
        """
        start_dt = end_dt - timedelta(seconds=tf.value)
        start_dt = start_dt.replace(tzinfo=self.tz).astimezone(timezone.utc)
        return start_dt.timestamp()
    