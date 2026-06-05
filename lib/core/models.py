from uuid import UUID, uuid4
from zoneinfo import ZoneInfo
from typing import Dict
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, computed_field
from lib.core.enums import TimeFrame, Ticker

class Candle(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    ticker: Ticker
    tf: TimeFrame

    @computed_field
    @property
    def range(self) -> float:
        return self.high - self.low
    
    @computed_field
    @property
    def body_high(self) -> float:
        return max(self.open, self.close)

    @computed_field
    @property
    def body_low(self) -> float:
        return min(self.open, self.close)
    
    @computed_field
    @property
    def end_time(self) -> 'datetime':
        return self.datetime + timedelta(seconds=self.tf.value)
    
    def to_dict(self) -> Dict:
        return dict(
            id=self.id,
            datetime=self.datetime,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            ticker=self.ticker,
            tf=self.tf
        )

    def convert_for_mongo_db(self) -> Dict:
        return dict(
            timestamp=self.datetime.astimezone(ZoneInfo("UTC")),
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            ticker=self.ticker.name,
            timeframe=self.tf.name
        )
    
    @classmethod
    def from_row(cls, row: 'pd.Series') -> 'Candle':
        """
        Create Candle from pandas Series row.
        
        Expected columns: datetime, open, high, low, close, volume, ticker, timeframe
        
        Parameters
        ----------
        row : pd.Series
            DataFrame row with candle data
            
        Returns
        -------
        Candle
            Candle instance
        """
        import pandas as pd
        
        # Handle ticker (can be string or enum)
        ticker = row.get('ticker')
        if isinstance(ticker, str):
            ticker = Ticker[ticker]
        elif ticker is None:
            raise ValueError("ticker column is required")
        
        # Handle timeframe (can be string or enum)
        tf = row.get('timeframe')
        if isinstance(tf, str):
            tf = TimeFrame[tf]
        elif tf is None:
            raise ValueError("timeframe column is required")
        
        return cls(
            datetime=pd.to_datetime(row['datetime']),
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row.get('volume', 0)),
            ticker=ticker,
            tf=tf
        )

    @classmethod
    def from_row_fast(cls, row) -> 'Candle':
        """
        Create Candle from a named tuple (e.g. from df.itertuples(index=False)).

        Expected attributes: datetime, open, high, low, close, volume, ticker, timeframe

        Parameters
        ----------
        row : named tuple
            Row from DataFrame.itertuples(index=False)

        Returns
        -------
        Candle
            Candle instance
        """
        import pandas as pd
        ticker = getattr(row, 'ticker', None)
        if isinstance(ticker, str):
            ticker = Ticker[ticker]
        elif ticker is None:
            raise ValueError("ticker attribute is required")
        tf = getattr(row, 'timeframe', None) or getattr(row, 'tf', None)
        if isinstance(tf, str):
            tf = TimeFrame[tf]
        elif tf is None:
            raise ValueError("timeframe attribute is required")
        return cls(
            datetime=pd.to_datetime(getattr(row, 'datetime')),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(getattr(row, 'volume', 0)),
            ticker=ticker,
            tf=tf
        )