"""Canonical location for the legacy backtest runtime."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional
from zoneinfo import ZoneInfo

import numpy as np

from utils.core.enums import TimeFrame
from utils.core.models import Candle
from utils.data.candle_fetcher import CandleFetcher

if TYPE_CHECKING:
    from feature_extraction.ml_manager import MLManager


class Backtest:
    def __init__(
        self,
        name: str,
        data: np.ndarray,
        ml_manager: MLManager,
        candle_fetcher: CandleFetcher,
        evaluate_tf: TimeFrame = TimeFrame.D,
        fast_mode: bool = False,
    ) -> None:
        self.name = name
        self.ml_manager = ml_manager
        self.data = data
        self.candle_fetcher = candle_fetcher
        self.tz = ZoneInfo("UTC")
        self.evaluate_tf = evaluate_tf
        self.fast_mode = fast_mode

    def run(self) -> None:
        from tqdm import tqdm

        for idx in tqdm(range(len(self.data)), desc=f"Running backtest {self.name}", unit="candles"):
            exec_candle = self.data[idx]
            end_dt = datetime.fromtimestamp(exec_candle["datetime"], tz=timezone.utc) + timedelta(
                seconds=self.evaluate_tf.value
            )
            exec_ny_candle = self._apply_mapping(exec_candle)
            prev_candle = self.data[idx - 1] if idx > 0 else None
            self._process_timeframes(end_dt, prev_candle)
            self.ml_manager.add_candle(exec_ny_candle, self.evaluate_tf)

    def _process_timeframes(self, end_dt: datetime, prev_candle: Optional[np.ndarray]) -> None:
        if prev_candle is not None:
            prev_month = datetime.fromtimestamp(prev_candle["datetime"]).month
            curr_month = datetime.fromtimestamp(prev_candle["datetime"]).month
            if prev_month != curr_month:
                year = datetime.fromtimestamp(prev_candle["datetime"]).year
                month = prev_month
                start_dt = datetime(year, month, 1)
                start_dt = start_dt.replace(tzinfo=self.tz).astimezone(timezone.utc)
                self._get_timeframe_order(TimeFrame.M, start_dt.timestamp(), method="closest")

        if self._is_end_of_week(end_dt):
            start_dt = end_dt - timedelta(hours=((24 * 4) + 16))
            start_dt = start_dt.replace(tzinfo=self.tz).astimezone(timezone.utc)
            self._get_timeframe_order(TimeFrame.W, start_dt.timestamp())

        if self._is_start_of_day(end_dt):
            start_dt = end_dt - timedelta(hours=24 if not self._is_end_of_week(end_dt) else 16)
            start_dt = start_dt.replace(tzinfo=self.tz).astimezone(timezone.utc)
            self._get_timeframe_order(TimeFrame.D, start_dt.timestamp())

    def _get_timeframe_order(
        self,
        timeframe: TimeFrame,
        start_ts: float,
        method: Optional[str] = "closest",
    ) -> None:
        if timeframe is not self.evaluate_tf:
            candle = self.candle_fetcher.get_candle(tf=timeframe, start_ts=start_ts, method=method)
            if candle is not None:
                mapped_candle = self._apply_mapping(candle)
                self.ml_manager.add_candle(candle=mapped_candle, tf=timeframe)

    def _apply_mapping(self, candle: np.ndarray) -> Candle:
        if self.fast_mode:
            from utils.compute.fast_candle import FastCandle

            return FastCandle.from_numpy(
                candle,
                ticker=self.ml_manager.ticker,
                tf=self.evaluate_tf,
            )

        return Candle(
            open=candle["open"],
            close=candle["close"],
            high=candle["high"],
            low=candle["low"],
            volume=candle["volume"] if "volume" in candle else 0,
            datetime=datetime.fromtimestamp(candle["datetime"], tz=timezone.utc),
            ticker=self.ml_manager.ticker,
            tf=self.evaluate_tf,
        )

    def _is_start_of_day(self, date: datetime) -> bool:
        return (date.hour == 0 and date.minute == 0 and date.second == 0) or self._is_end_of_week(date)

    def _is_end_of_week(self, date: datetime) -> bool:
        return date.weekday() == 4 and date.hour == 16 and date.minute == 0 and date.second == 0

    def _get_start_ts_from_end_dt(self, end_dt: datetime, tf: TimeFrame) -> float:
        start_dt = end_dt - timedelta(seconds=tf.value)
        start_dt = start_dt.replace(tzinfo=self.tz).astimezone(timezone.utc)
        return start_dt.timestamp()


__all__ = ["Backtest"]
