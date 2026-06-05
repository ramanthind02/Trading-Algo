from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .bias_node_cache import BiasNodeCache, CacheMissError
from .cache_paths import (
    default_candle_cache_dir,
    default_central_cache_dir,
    default_live_artifact_cache_dir,
    default_research_artifact_cache_dir,
)
from .central_cache_errors import (
    ArtifactMissingError,
    CacheCoverageError,
)
from .central_cache_models import (
    ArtifactDescriptor,
    ArtifactRecord,
    ArtifactScope,
    CacheRequest,
    CoverageWindow,
    LookupMode,
)
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle

logger = logging.getLogger(__name__)


def _normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        normalized = df.copy()
        if normalized.index.name != "datetime":
            normalized.index.name = "datetime"
        return normalized

    normalized = df.copy()
    if "datetime" in normalized.columns and not isinstance(normalized.index, pd.DatetimeIndex):
        normalized = normalized.set_index("datetime")
    if not isinstance(normalized.index, pd.DatetimeIndex):
        normalized.index = pd.to_datetime(normalized.index)
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized = normalized.sort_index()
    normalized.index.name = "datetime"
    return normalized


def _frame_coverage(frame: pd.DataFrame) -> CoverageWindow:
    if len(frame) == 0:
        return CoverageWindow(start=None, end=None)
    return CoverageWindow(
        start=frame.index.min().to_pydatetime(),
        end=frame.index.max().to_pydatetime(),
    )


def _coerce_timeframe_value(value: Any, fallback: TimeFrame) -> TimeFrame:
    if isinstance(value, TimeFrame):
        return value
    if isinstance(value, str):
        cleaned = value.replace("TimeFrame.", "")
        try:
            return TimeFrame[cleaned]
        except KeyError:
            return fallback
    return fallback


def _normalize_candle_frame(
    candles: pd.DataFrame,
    ticker: Ticker,
    timeframe: TimeFrame,
) -> pd.DataFrame:
    normalized = _normalize_datetime_index(candles)
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    normalized["ticker"] = ticker.name
    normalized["timeframe"] = timeframe
    return normalized


def _datetime_index_values_equal(left: pd.Index, right: pd.Index) -> bool:
    """True when datetimes match, ignoring stored resolution (ns vs us, etc.)."""
    if len(left) != len(right):
        return False
    left_dti = pd.DatetimeIndex(left)
    right_dti = pd.DatetimeIndex(right)
    try:
        left_ns = left_dti.as_unit("ns")
        right_ns = right_dti.as_unit("ns")
    except (AttributeError, TypeError, ValueError):
        left_ns, right_ns = left_dti, right_dti
    return bool(left_ns.equals(right_ns))


def _candle_frame_semantically_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Compare normalized candle frames without strict pandas dtype identity."""
    if len(left) != len(right):
        return False
    if not _datetime_index_values_equal(left.index, right.index):
        return False
    numeric_cols = [
        c
        for c in ("open", "high", "low", "close", "volume")
        if c in left.columns and c in right.columns
    ]
    if not numeric_cols:
        return False
    for col in numeric_cols:
        l64 = np.asarray(left[col], dtype=np.float64)
        r64 = np.asarray(right[col], dtype=np.float64)
        if not np.allclose(l64, r64, rtol=1e-9, atol=1e-12, equal_nan=True):
            return False
    return True


class CentralCacheStore:
    """Simplified facade for candle and bias-artifact reads/writes.

    Artifacts are present or absent — no lifecycle states, no dependency
    graph, no revision tracking. Coverage is derived from the parquet index.
    All bias artifacts are stored under a single ``artifacts/live/`` tree
    regardless of the ``ArtifactDescriptor.scope`` field (scope is kept on
    the descriptor for downstream consumers like the materialization layer).
    """

    _instance: ClassVar[Optional["CentralCacheStore"]] = None

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_central_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.candle_cache_dir = (
            self.cache_dir / "candles"
            if cache_dir is not None
            else default_candle_cache_dir()
        )
        self.live_artifact_cache_dir = (
            self.cache_dir / "artifacts" / "live"
            if cache_dir is not None
            else default_live_artifact_cache_dir()
        )
        self.research_artifact_cache_dir = (
            self.cache_dir / "artifacts" / "research"
            if cache_dir is not None
            else default_research_artifact_cache_dir()
        )
        self.candle_cache_dir.mkdir(parents=True, exist_ok=True)
        self.live_artifact_cache_dir.mkdir(parents=True, exist_ok=True)
        self.research_artifact_cache_dir.mkdir(parents=True, exist_ok=True)
        self._candle_frames: Dict[tuple[Ticker, TimeFrame], pd.DataFrame] = {}
        self._candle_records: Dict[tuple[Ticker, TimeFrame], ArtifactRecord] = {}
        self._artifact_frames: Dict[ArtifactDescriptor, pd.DataFrame] = {}
        self._artifact_records: Dict[ArtifactDescriptor, ArtifactRecord] = {}

    @classmethod
    def get_instance(cls) -> "CentralCacheStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        from .live_cache_refresh import reset_live_cache_refresh_orchestrator

        if cls._instance is not None:
            cls._instance.clear()
        cls._instance = None
        reset_live_cache_refresh_orchestrator()

    def clear(self) -> None:
        self._candle_frames.clear()
        self._candle_records.clear()
        self._artifact_frames.clear()
        self._artifact_records.clear()

    def clear_candles(self, purge_persisted: bool = False) -> None:
        self._candle_frames.clear()
        self._candle_records.clear()
        if purge_persisted and self.candle_cache_dir.exists():
            shutil.rmtree(self.candle_cache_dir, ignore_errors=True)
            self.candle_cache_dir.mkdir(parents=True, exist_ok=True)

    def loaded_tickers(self) -> list[Ticker]:
        return sorted({ticker for ticker, _ in self._candle_frames.keys()}, key=lambda item: item.name)

    def is_candle_loaded(self, ticker: Ticker, timeframe: TimeFrame) -> bool:
        return (
            self._source_key(ticker, timeframe) in self._candle_frames
            or self._candle_path(ticker, timeframe).exists()
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _source_key(self, ticker: Ticker, timeframe: TimeFrame) -> tuple[Ticker, TimeFrame]:
        return ticker, timeframe

    def _candle_path(self, ticker: Ticker, timeframe: TimeFrame) -> Path:
        return self.candle_cache_dir / ticker.name / f"{timeframe.name}.parquet"

    def _artifact_cache_dir(self) -> Path:
        return self.live_artifact_cache_dir

    def _node_cache(self, descriptor: ArtifactDescriptor) -> BiasNodeCache:
        if descriptor.module_name is None or descriptor.ticker is None or descriptor.timeframe is None:
            raise ValueError("Node artifacts require module_name, ticker, and timeframe.")
        return BiasNodeCache(
            module_name=descriptor.module_name,
            params=dict(descriptor.params),
            ticker=descriptor.ticker,
            tf=descriptor.timeframe,
            cache_dir=str(self._artifact_cache_dir()),
        )

    def _persist_dataframe(self, path: Path, frame: pd.DataFrame) -> None:
        serialized = frame.copy()
        if "ticker" in serialized.columns:
            serialized["ticker"] = serialized["ticker"].map(
                lambda value: value.name if isinstance(value, Ticker) else value
            )
        if "timeframe" in serialized.columns:
            serialized["timeframe"] = serialized["timeframe"].map(
                lambda value: value.name if isinstance(value, TimeFrame) else value
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized.to_parquet(path, index=True)

    def _range_slice(
        self,
        frame: pd.DataFrame,
        *,
        module_name: str,
        ticker: Optional[Ticker],
        timeframe: Optional[TimeFrame],
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> pd.DataFrame:
        coverage = _frame_coverage(frame)
        if start is not None and coverage.start is not None and pd.Timestamp(start) < frame.index.min():
            raise CacheCoverageError(
                module_name=module_name,
                ticker=ticker,
                timeframe=timeframe,
                start=start,
                end=end,
                coverage_start=coverage.start,
                coverage_end=coverage.end,
            )
        if end is not None and coverage.end is not None and pd.Timestamp(end) > frame.index.max():
            raise CacheCoverageError(
                module_name=module_name,
                ticker=ticker,
                timeframe=timeframe,
                start=start,
                end=end,
                coverage_start=coverage.start,
                coverage_end=coverage.end,
            )

        sliced = frame
        if start is not None:
            sliced = sliced[sliced.index >= pd.Timestamp(start)]
        if end is not None:
            sliced = sliced[sliced.index <= pd.Timestamp(end)]
        if sliced.empty:
            raise CacheCoverageError(
                module_name=module_name,
                ticker=ticker,
                timeframe=timeframe,
                start=start,
                end=end,
                coverage_start=coverage.start,
                coverage_end=coverage.end,
            )
        return sliced.copy()

    # ------------------------------------------------------------------
    # Candle storage
    # ------------------------------------------------------------------

    def _read_candles_from_disk(
        self,
        ticker: Ticker,
        timeframe: TimeFrame,
    ) -> pd.DataFrame | None:
        key = self._source_key(ticker, timeframe)
        path = self._candle_path(ticker, timeframe)
        if not path.exists():
            return None

        loaded = _normalize_candle_frame(pd.read_parquet(path), ticker, timeframe)
        record = ArtifactRecord(
            descriptor=ArtifactDescriptor(
                family="candles",
                ticker=ticker,
                timeframe=timeframe,
            ),
            coverage=_frame_coverage(loaded),
        )
        self._candle_frames[key] = loaded
        self._candle_records[key] = record
        return loaded

    def set_candles(
        self,
        ticker: Ticker,
        timeframe: TimeFrame,
        candles: pd.DataFrame,
        scope: ArtifactScope = ArtifactScope.LIVE,
    ) -> None:
        """Replace the persisted candle snapshot for a ticker/timeframe."""
        normalized = _normalize_candle_frame(candles, ticker, timeframe)
        key = self._source_key(ticker, timeframe)
        existing = self._candle_frames.get(key)
        if existing is None:
            existing = self._read_candles_from_disk(ticker, timeframe)
        if existing is not None and _candle_frame_semantically_equal(existing, normalized):
            return

        self._candle_frames[key] = normalized
        record = ArtifactRecord(
            descriptor=ArtifactDescriptor(
                family="candles",
                ticker=ticker,
                timeframe=timeframe,
                scope=scope,
            ),
            coverage=_frame_coverage(normalized),
        )
        self._candle_records[key] = record
        self._persist_dataframe(self._candle_path(ticker, timeframe), normalized)
        self._notify_live_refresh(ticker, timeframe, scope)

    def upsert_candles(
        self,
        ticker: Ticker,
        timeframe: TimeFrame,
        candles: pd.DataFrame,
        scope: ArtifactScope = ArtifactScope.LIVE,
    ) -> None:
        """Merge candle updates into the persisted runtime cache."""
        normalized = _normalize_candle_frame(candles, ticker, timeframe)
        key = self._source_key(ticker, timeframe)
        existing = self._candle_frames.get(key)
        if existing is None:
            existing = self._read_candles_from_disk(ticker, timeframe)

        if existing is None or existing.empty:
            merged = normalized
        else:
            merged = pd.concat([existing, normalized], axis=0)
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()

        if existing is not None and _candle_frame_semantically_equal(existing, merged):
            return

        self._candle_frames[key] = merged
        record = ArtifactRecord(
            descriptor=ArtifactDescriptor(
                family="candles",
                ticker=ticker,
                timeframe=timeframe,
                scope=scope,
            ),
            coverage=_frame_coverage(merged),
        )
        self._candle_records[key] = record
        self._persist_dataframe(self._candle_path(ticker, timeframe), merged)
        self._notify_live_refresh(ticker, timeframe, scope)

    def query_candle(
        self,
        ticker: Ticker,
        timeframe: TimeFrame,
        dt: datetime,
        lookup_mode: LookupMode = LookupMode.EXACT,
    ) -> Candle:
        frame = self._candle_frames.get(self._source_key(ticker, timeframe))
        if frame is None:
            frame = self._read_candles_from_disk(ticker, timeframe)
        if frame is None or frame.empty:
            raise ArtifactMissingError(
                module_name="candles",
                ticker=ticker,
                timeframe=timeframe,
                requested_at=dt,
                reason="Candles are not loaded",
            )

        lookup = pd.Timestamp(dt)
        if lookup.tzinfo is not None:
            lookup = lookup.tz_convert(None)

        if lookup_mode is LookupMode.EXACT:
            if lookup not in frame.index:
                raise ArtifactMissingError(
                    module_name="candles",
                    ticker=ticker,
                    timeframe=timeframe,
                    requested_at=dt,
                    reason="Exact candle is missing",
                )
            row = frame.loc[lookup]
        else:
            position = frame.index.searchsorted(lookup, side="right") - 1
            if position < 0:
                raise ArtifactMissingError(
                    module_name="candles",
                    ticker=ticker,
                    timeframe=timeframe,
                    requested_at=dt,
                    reason="No as-of candle available",
                )
            row = frame.iloc[position]

        return Candle(
            datetime=pd.Timestamp(row.name if hasattr(row, "name") else lookup).to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0)),
            ticker=ticker,
            tf=timeframe,
        )

    def query_candles(
        self,
        ticker: Ticker,
        timeframe: TimeFrame,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        frame = self._candle_frames.get(self._source_key(ticker, timeframe))
        if frame is None:
            frame = self._read_candles_from_disk(ticker, timeframe)
        if frame is None or frame.empty:
            raise ArtifactMissingError(
                module_name="candles",
                ticker=ticker,
                timeframe=timeframe,
                reason="Candles are not loaded",
            )

        return self._range_slice(
            frame,
            module_name="candles",
            ticker=ticker,
            timeframe=timeframe,
            start=start,
            end=end,
        )

    def describe_candle(self, ticker: Ticker, timeframe: TimeFrame) -> ArtifactRecord | None:
        key = self._source_key(ticker, timeframe)
        record = self._candle_records.get(key)
        if record is None:
            self._read_candles_from_disk(ticker, timeframe)
            record = self._candle_records.get(key)
        return record

    # ------------------------------------------------------------------
    # Artifact storage
    # ------------------------------------------------------------------

    def _read_artifact_from_disk(self, descriptor: ArtifactDescriptor) -> Optional[pd.DataFrame]:
        if descriptor.module_name is None or descriptor.ticker is None or descriptor.timeframe is None:
            return None
        existing = self._artifact_frames.get(descriptor)
        if existing is not None:
            return existing
        cache = self._node_cache(descriptor)
        if not cache.exists():
            return None
        try:
            loaded = _normalize_datetime_index(cache.load())
        except CacheMissError:
            return None
        record = ArtifactRecord(
            descriptor=descriptor,
            coverage=_frame_coverage(loaded),
        )
        self._artifact_frames[descriptor] = loaded
        self._artifact_records[descriptor] = record
        return loaded

    def write_artifact(
        self,
        descriptor: ArtifactDescriptor,
        data: pd.DataFrame,
        depends_on: Sequence[tuple[Ticker, TimeFrame]] = (),
        source_revision: int = 0,
    ) -> None:
        """Persist a bias artifact. ``depends_on`` and ``source_revision`` are
        accepted for API compatibility but ignored — there is no dependency
        graph or revision tracking in the simplified cache."""
        normalized = _normalize_datetime_index(data)
        if descriptor.module_name is not None and descriptor.ticker is not None and descriptor.timeframe is not None:
            cache = self._node_cache(descriptor)
            cache.save(normalized)
        self._artifact_frames[descriptor] = normalized
        self._artifact_records[descriptor] = ArtifactRecord(
            descriptor=descriptor,
            coverage=_frame_coverage(normalized),
        )

    def read_artifact(
        self,
        descriptor: ArtifactDescriptor,
        request: Optional[CacheRequest] = None,
        lookup_mode: LookupMode = LookupMode.EXACT,
    ) -> pd.DataFrame:
        frame = self._artifact_frames.get(descriptor)
        if frame is None:
            frame = self._read_artifact_from_disk(descriptor)
        if frame is None or frame.empty:
            raise ArtifactMissingError(
                module_name=descriptor.module_name or descriptor.family,
                ticker=descriptor.ticker,
                timeframe=descriptor.timeframe,
                requested_at=request.exact_dt if request is not None else None,
                reason="Artifact is not loaded",
            )

        if request is None:
            return frame.copy()
        if request.exact_dt is not None:
            return self._read_artifact_exact(frame, descriptor, request.exact_dt, lookup_mode)
        if request.as_of_dt is not None:
            return self._read_artifact_as_of(frame, descriptor, request.as_of_dt)
        if request.start is not None or request.end is not None:
            return self._read_artifact_range(frame, descriptor, request.start, request.end)
        return frame.copy()

    def _read_artifact_exact(
        self,
        frame: pd.DataFrame,
        descriptor: ArtifactDescriptor,
        dt: datetime,
        lookup_mode: LookupMode,
    ) -> pd.DataFrame:
        lookup = pd.Timestamp(dt)
        if lookup.tzinfo is not None:
            lookup = lookup.tz_convert(None)
        if lookup_mode is LookupMode.EXACT:
            if lookup not in frame.index:
                raise ArtifactMissingError(
                    module_name=descriptor.module_name or descriptor.family,
                    ticker=descriptor.ticker,
                    timeframe=descriptor.timeframe,
                    requested_at=dt,
                    reason="Exact artifact row is missing",
                )
            return frame.loc[[lookup]].copy()
        position = frame.index.searchsorted(lookup, side="right") - 1
        if position < 0:
            raise ArtifactMissingError(
                module_name=descriptor.module_name or descriptor.family,
                ticker=descriptor.ticker,
                timeframe=descriptor.timeframe,
                requested_at=dt,
                reason="No as-of artifact row available",
            )
        return frame.iloc[[position]].copy()

    def _read_artifact_as_of(
        self,
        frame: pd.DataFrame,
        descriptor: ArtifactDescriptor,
        dt: datetime,
    ) -> pd.DataFrame:
        return self._read_artifact_exact(frame, descriptor, dt, LookupMode.AS_OF)

    def _read_artifact_range(
        self,
        frame: pd.DataFrame,
        descriptor: ArtifactDescriptor,
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> pd.DataFrame:
        return self._range_slice(
            frame,
            module_name=descriptor.module_name or descriptor.family,
            ticker=descriptor.ticker,
            timeframe=descriptor.timeframe,
            start=start,
            end=end,
        )

    def describe_artifact(self, descriptor: ArtifactDescriptor) -> ArtifactRecord | None:
        cached = self._artifact_records.get(descriptor)
        if cached is not None:
            return cached
        cache = self._node_cache(descriptor) if (
            descriptor.module_name is not None
            and descriptor.ticker is not None
            and descriptor.timeframe is not None
        ) else None
        if cache is None or not cache.exists():
            return None
        try:
            loaded = _normalize_datetime_index(cache.load())
        except CacheMissError:
            return None
        record = ArtifactRecord(
            descriptor=descriptor,
            coverage=_frame_coverage(loaded),
        )
        self._artifact_frames[descriptor] = loaded
        self._artifact_records[descriptor] = record
        return record

    def list_artifacts(self, scope: Optional[ArtifactScope] = None) -> list[ArtifactRecord]:
        records = list(self._artifact_records.values())
        if scope is not None:
            records = [r for r in records if r.descriptor.scope is scope]
        return sorted(records, key=lambda r: r.descriptor.cache_key())

    # ------------------------------------------------------------------
    # Live refresh notification
    # ------------------------------------------------------------------

    def _notify_live_refresh(
        self,
        ticker: Ticker,
        timeframe: TimeFrame,
        scope: ArtifactScope,
    ) -> None:
        if scope is not ArtifactScope.LIVE:
            return
        try:
            from .live_cache_refresh import get_live_cache_refresh_orchestrator

            get_live_cache_refresh_orchestrator().note_candle_update(
                ticker=ticker,
                timeframe=timeframe,
                scope=scope,
            )
        except Exception as exc:
            logger.debug(
                "Live cache refresh notification failed for %s/%s: %s",
                ticker.name,
                timeframe.name,
                exc,
            )
