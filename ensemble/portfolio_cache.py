from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from utils.cache.central_cache import CentralCacheStore
from utils.cache.central_cache_errors import ArtifactMissingError, CacheCoverageError
from utils.cache.central_cache_models import ArtifactDescriptor, ArtifactScope, CacheRequest
from utils.core.enums import TimeFrame, Ticker


def _coerce_ticker(value: str | Ticker) -> Ticker:
    if isinstance(value, Ticker):
        return value
    return Ticker[str(value)]


def _clamp_range_to_coverage(
    *,
    coverage_start: datetime | None,
    coverage_end: datetime | None,
    start: datetime,
    end: datetime,
    module_name: str,
    ticker: Ticker | None,
    timeframe: TimeFrame | None,
) -> tuple[datetime, datetime]:
    effective_start = max(pd.Timestamp(start), pd.Timestamp(coverage_start or start))
    effective_end = min(pd.Timestamp(end), pd.Timestamp(coverage_end or end))
    if effective_start > effective_end:
        raise CacheCoverageError(
            module_name=module_name,
            ticker=ticker,
            timeframe=timeframe,
            start=start,
            end=end,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
        )
    return effective_start.to_pydatetime(), effective_end.to_pydatetime()


@dataclass(frozen=True)
class PortfolioCacheQuery:
    """Cache-native request for portfolio fit/predict operations."""

    tickers: tuple[str, ...]
    start: datetime
    end: datetime
    timeframes: tuple[TimeFrame, ...]
    volatility_timeframe: TimeFrame = TimeFrame.D
    scope: ArtifactScope = ArtifactScope.LIVE
    grid: tuple[datetime, ...] = ()

    def for_timeframe(self, timeframe: TimeFrame) -> "PortfolioCacheQuery":
        return PortfolioCacheQuery(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            timeframes=(timeframe,),
            volatility_timeframe=self.volatility_timeframe,
            scope=self.scope,
            grid=self.grid,
        )


def _query_candles_from_cache(
    query: PortfolioCacheQuery,
    timeframe: TimeFrame,
) -> pd.DataFrame:
    store = CentralCacheStore.get_instance()
    frames = []
    for ticker in query.tickers:
        ticker_enum = _coerce_ticker(ticker)
        record = store.describe_candle(ticker_enum, timeframe)
        if record is None:
            raise ArtifactMissingError(
                module_name="candles",
                ticker=ticker_enum,
                timeframe=timeframe,
                requested_at=query.end,
                reason="Candles are not loaded",
            )
        effective_start, effective_end = _clamp_range_to_coverage(
            coverage_start=record.coverage.start,
            coverage_end=record.coverage.end,
            start=query.start,
            end=query.end,
            module_name="candles",
            ticker=ticker_enum,
            timeframe=timeframe,
        )
        frame = store.query_candles(
            ticker_enum,
            timeframe,
            start=effective_start,
            end=effective_end,
        ).reset_index()
        if "ticker" not in frame.columns:
            frame["ticker"] = ticker_enum.name
        if "timeframe" not in frame.columns:
            frame["timeframe"] = timeframe
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "datetime"]).reset_index(drop=True)


def _query_volatility_from_cache(query: PortfolioCacheQuery) -> pd.DataFrame:
    store = CentralCacheStore.get_instance()
    frames: list[pd.DataFrame] = []
    for ticker in query.tickers:
        ticker_enum = _coerce_ticker(ticker)
        descriptor = ArtifactDescriptor(
            family="bias",
            ticker=ticker_enum,
            timeframe=query.volatility_timeframe,
            module_name="ewsd",
            params={"long_run_window": 2520},
            scope=query.scope,
            artifact_name="ewsd",
        )
        try:
            record = store.describe_artifact(descriptor)
            if record is None:
                raise ArtifactMissingError(
                    module_name="ewsd",
                    ticker=ticker_enum,
                    timeframe=query.volatility_timeframe,
                    requested_at=query.end,
                    reason="EWSD volatility is missing from the central cache",
                )
            effective_start, effective_end = _clamp_range_to_coverage(
                coverage_start=record.coverage.start,
                coverage_end=record.coverage.end,
                start=query.start,
                end=query.end,
                module_name="ewsd",
                ticker=ticker_enum,
                timeframe=query.volatility_timeframe,
            )
            frame = store.read_artifact(
                descriptor,
                request=CacheRequest(start=effective_start, end=effective_end),
            )
        except ArtifactMissingError as exc:
            raise ArtifactMissingError(
                module_name="ewsd",
                ticker=ticker_enum,
                timeframe=query.volatility_timeframe,
                requested_at=query.end,
                reason="EWSD volatility is missing from the central cache",
            ) from exc
        frames.append(
            frame.reset_index().rename(columns={"index": "datetime"}).assign(
                ticker=ticker_enum.name
            )
        )
    if not frames:
        raise ArtifactMissingError(
            module_name="ewsd",
            ticker=None,
            timeframe=query.volatility_timeframe,
            requested_at=query.end,
            reason="EWSD volatility is missing from the central cache",
        )
    volatility_df = pd.concat(frames, ignore_index=True)
    if "ewsd_annual_vol" not in volatility_df.columns and "close" in volatility_df.columns:
        volatility_df = volatility_df.rename(columns={"close": "ewsd_annual_vol"})
    return volatility_df
