from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle


def candle_row(candle: Candle, ticker: Ticker, timeframe: TimeFrame) -> dict[str, Any]:
    return {
        "datetime": candle.datetime,
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
        "ticker": ticker.name,
        "timeframe": timeframe,
    }


def candles_to_frame(
    candles: List[Candle],
    ticker: Ticker,
    timeframe: TimeFrame,
) -> pd.DataFrame:
    return pd.DataFrame([candle_row(candle, ticker, timeframe) for candle in candles])


def upsert_candle_frame(
    frame: pd.DataFrame,
    candle: Candle,
    ticker: Ticker,
    timeframe: TimeFrame,
) -> pd.DataFrame:
    base = frame.copy()
    if "datetime" not in base.columns:
        base = base.reset_index()
    if base.empty:
        return pd.DataFrame([candle_row(candle, ticker, timeframe)])

    base["datetime"] = pd.to_datetime(base["datetime"])
    filtered = base.loc[base["datetime"] != pd.Timestamp(candle.datetime)].copy()
    merged = pd.concat(
        [filtered, pd.DataFrame([candle_row(candle, ticker, timeframe)])],
        ignore_index=True,
    )
    return merged.sort_values("datetime").reset_index(drop=True)


@dataclass
class ForecastLiveInputs:
    mt5_connector: Any
    volatility_service: Any
    candle_buffers: Dict[tuple[Ticker, TimeFrame], List[Candle]]
    ml_managers: Dict[tuple[Ticker, TimeFrame], Any]
    ensembles: Dict[tuple[Ticker, TimeFrame], Any]
    lookback_candles: int
    tickers: List[Ticker]
    logger: logging.Logger

    def update_live_inputs_for_timeframe(self, timeframe: TimeFrame) -> None:
        """Refresh live candles, cross-ticker data, and daily volatility for one timeframe."""
        cross_pairs = self.get_cross_tickers()
        updated_cross_tickers: set[Ticker] = set()

        for ticker in self.tickers:
            try:
                if timeframe != TimeFrame.D:
                    self.refresh_daily_volatility(ticker)

                ml_manager_key = (ticker, timeframe)
                if ml_manager_key not in self.ml_managers:
                    self.logger.warning("No MLManager for %s %s", ticker.name, timeframe.name)
                    continue

                latest_candle = self.mt5_connector.get_latest_candle(ticker.value, timeframe)
                if latest_candle is None:
                    self.logger.warning("No candle data for %s", ticker.name)
                    continue

                self.add_candle(ticker, timeframe, latest_candle)

                if (ticker, timeframe) in cross_pairs:
                    self.upsert_cross_ticker_candle(ticker, timeframe, latest_candle)
                    updated_cross_tickers.add(ticker)
            except Exception as exc:
                self.logger.error("Error updating %s: %s", ticker.name, exc)

        self.update_cross_ticker_latest(timeframe, skip_tickers=updated_cross_tickers)

    def add_candle(self, ticker: Ticker, timeframe: TimeFrame, candle: Candle) -> None:
        """Insert a live candle into buffers, cache, MLManager, and daily vol state."""
        key = (ticker, timeframe)
        if key not in self.candle_buffers:
            self.candle_buffers[key] = []

        self.candle_buffers[key].append(candle)
        try:
            from utils.cache.central_cache import CentralCacheStore

            cache = CentralCacheStore.get_instance()
            cache.upsert_candles(
                ticker,
                timeframe,
                candles_to_frame(self.candle_buffers[key], ticker, timeframe),
            )
        except Exception as exc:
            self.logger.debug(
                "Central cache candle sync failed for %s %s: %s",
                ticker.name,
                timeframe.name,
                exc,
            )

        if len(self.candle_buffers[key]) > self.lookback_candles:
            self.candle_buffers[key] = self.candle_buffers[key][-self.lookback_candles :]

        ml_manager = self.ml_managers.get(key)
        if ml_manager is not None:
            ml_manager.add_candle(candle, timeframe)
            self.logger.debug("Added candle to %s %s: %s", ticker.name, timeframe.name, candle.datetime)

        if timeframe == TimeFrame.D:
            try:
                self.volatility_service.update_incremental(
                    pd.DataFrame(
                        [
                            {
                                "datetime": candle.datetime,
                                "ticker": ticker.name,
                                "close": candle.close,
                            }
                        ]
                    )
                )
            except Exception as exc:
                self.logger.error(
                    "Failed updating daily EWSD volatility for %s: %s",
                    ticker.name,
                    exc,
                )

    def refresh_daily_volatility(self, ticker: Ticker) -> None:
        """Refresh incremental daily EWSD state for a ticker using latest daily candle."""
        daily_candle = self.mt5_connector.get_latest_candle(ticker.value, TimeFrame.D)
        if daily_candle is None:
            return
        self.volatility_service.update_incremental(
            pd.DataFrame(
                [
                    {
                        "datetime": daily_candle.datetime,
                        "ticker": ticker.name,
                        "close": daily_candle.close,
                    }
                ]
            )
        )

    def get_cross_tickers(self) -> set[tuple[Ticker, TimeFrame]]:
        """Discover cross-tickers referenced in ensemble bias node params."""
        from utils.data.cross_ticker_store import extract_cross_ticker_names

        cross: set[tuple[Ticker, TimeFrame]] = set()
        for (_, timeframe), ensemble in self.ensembles.items():
            for spec in ensemble.get_required_bias_nodes():
                params = spec.get("params", {})
                for ct_name in extract_cross_ticker_names(params):
                    try:
                        cross.add((Ticker[ct_name], timeframe))
                    except KeyError:
                        self.logger.warning(
                            "Unknown cross ticker '%s' in bias node params; skipping.",
                            ct_name,
                        )
        return cross

    def load_cross_ticker_history(self) -> None:
        """Fetch historical cross-ticker candles and load them into the cache."""
        cross = self.get_cross_tickers()
        if not cross:
            return

        from utils.cache.central_cache import CentralCacheStore

        ct_store = CentralCacheStore.get_instance()
        for ct_ticker, timeframe in cross:
            try:
                buffered = self.candle_buffers.get((ct_ticker, timeframe), [])
                if buffered:
                    ct_store.set_candles(
                        ct_ticker,
                        timeframe,
                        candles_to_frame(buffered, ct_ticker, timeframe),
                    )
                    self.logger.info(
                        "Loaded cross-ticker %s from traded history (%s candles)",
                        ct_ticker.name,
                        len(buffered),
                    )
                    continue

                self.logger.info(
                    "Fetching cross-ticker history %s %s...",
                    ct_ticker.name,
                    timeframe.name,
                )
                candles = self.mt5_connector.get_historical_candles(
                    ct_ticker.value,
                    timeframe,
                    count=self.lookback_candles,
                )
                if candles:
                    ct_store.set_candles(
                        ct_ticker,
                        timeframe,
                        candles_to_frame(candles, ct_ticker, timeframe),
                    )
                    self.logger.info(
                        "Loaded cross-ticker %s (%s candles)",
                        ct_ticker.name,
                        len(candles),
                    )
            except Exception as exc:
                self.logger.warning(
                    "Failed to load cross-ticker %s: %s",
                    ct_ticker.name,
                    exc,
                )

    def upsert_cross_ticker_candle(self, ticker: Ticker, timeframe: TimeFrame, candle: Candle) -> None:
        """Insert or replace the latest candle for a cross-ticker in the cache."""
        from utils.cache.central_cache import CentralCacheStore
        from utils.cache.central_cache_errors import ArtifactMissingError

        ct_store = CentralCacheStore.get_instance()
        try:
            existing_frame = ct_store.query_candles(ticker, timeframe).reset_index()
        except ArtifactMissingError:
            existing_frame = candles_to_frame(
                self.candle_buffers.get((ticker, timeframe), []),
                ticker,
                timeframe,
            )
        merged_frame = upsert_candle_frame(existing_frame, candle, ticker, timeframe)
        ct_store.upsert_candles(ticker, timeframe, merged_frame)

    def update_cross_ticker_latest(
        self,
        timeframe: TimeFrame,
        skip_tickers: Optional[set[Ticker]] = None,
    ) -> None:
        """Fetch latest candle for each cross-ticker and update the cache."""
        cross = self.get_cross_tickers()
        if not cross:
            return

        skip_tickers = skip_tickers or set()
        for ct_ticker, cross_timeframe in cross:
            if cross_timeframe != timeframe or ct_ticker in skip_tickers:
                continue
            try:
                candle = self.mt5_connector.get_latest_candle(ct_ticker.value, cross_timeframe)
                if candle is not None:
                    self.upsert_cross_ticker_candle(ct_ticker, cross_timeframe, candle)
            except Exception as exc:
                self.logger.warning(
                    "Failed to update cross-ticker %s: %s",
                    ct_ticker.name,
                    exc,
                )
