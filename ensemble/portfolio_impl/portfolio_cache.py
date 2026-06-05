from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd

from lib.cache.runtime.central_cache import CentralCacheStore
from lib.cache.runtime.central_cache_errors import ArtifactMissingError, CacheCoverageError
from lib.cache.runtime.central_cache_models import ArtifactDescriptor, ArtifactScope, CacheRequest
from lib.core.enums import TimeFrame, Ticker


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


def _merge_daily_candle_overlay(
    base: pd.DataFrame,
    overlay: pd.DataFrame | None,
    *,
    allowed_tickers: Iterable[str],
) -> pd.DataFrame:
    """Append/replace same (ticker, datetime) rows with overlay (session-only daily bars)."""
    if overlay is None or overlay.empty:
        return base
    allowed = {str(t) for t in allowed_tickers}
    ov = overlay.copy()
    if "ticker" in ov.columns:
        ov = ov.loc[ov["ticker"].astype(str).isin(allowed)]
    if ov.empty:
        return base
    if "timeframe" in ov.columns:
        ov = ov.loc[ov["timeframe"] == TimeFrame.D]
    if ov.empty:
        return base
    combined = pd.concat([base, ov], ignore_index=True)
    combined = combined.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["ticker", "datetime"], keep="last")
    return combined


def _tail_daily_rows_by_distinct_dates(frame: pd.DataFrame, max_bars: int) -> pd.DataFrame:
    """Keep rows whose datetimes fall in the last ``max_bars`` distinct calendar dates (daily grid)."""
    if max_bars <= 0 or frame.empty:
        return frame
    dates = sorted(pd.to_datetime(frame["datetime"]).dt.normalize().unique())
    if len(dates) <= max_bars:
        return frame
    keep = set(dates[-max_bars:])
    dt_norm = pd.to_datetime(frame["datetime"]).dt.normalize()
    return frame.loc[dt_norm.isin(keep)].sort_values(["ticker", "datetime"]).reset_index(drop=True)


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
    daily_candle_overlay: pd.DataFrame | None = None
    """Optional daily OHLCV rows merged after store reads (not persisted). Used for session-only bars."""

    prediction_daily_max_bars: int = 0
    """If > 0, daily candle frames are trimmed to this many trailing distinct dates (monthly unchanged)."""

    volatility_history_start: datetime | None = None
    """Optional earlier start for EWSD reads so predict windows can forward-fill from fit history."""

    def effective_volatility_start(self) -> datetime:
        return self.volatility_history_start if self.volatility_history_start is not None else self.start

    def for_timeframe(self, timeframe: TimeFrame) -> "PortfolioCacheQuery":
        return PortfolioCacheQuery(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            timeframes=(timeframe,),
            volatility_timeframe=self.volatility_timeframe,
            scope=self.scope,
            grid=self.grid,
            daily_candle_overlay=self.daily_candle_overlay,
            prediction_daily_max_bars=self.prediction_daily_max_bars,
            volatility_history_start=self.volatility_history_start,
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
    merged = pd.concat(frames, ignore_index=True).sort_values(["ticker", "datetime"]).reset_index(drop=True)
    if timeframe == TimeFrame.D:
        merged = _merge_daily_candle_overlay(
            merged,
            query.daily_candle_overlay,
            allowed_tickers=query.tickers,
        )
        merged = _tail_daily_rows_by_distinct_dates(merged, query.prediction_daily_max_bars)
    return merged


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
                start=query.effective_volatility_start(),
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
