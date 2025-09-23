from uuid import UUID, uuid4
from zoneinfo import ZoneInfo
from typing import Dict
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, computed_field
from utils.enums import TimeFrame, Ticker

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