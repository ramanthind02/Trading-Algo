from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from dataclasses import replace
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
    ArtifactLifecycleError,
    ArtifactMissingError,
    CacheCoverageError,
    SourceRevisionConflictError,
)
from .central_cache_models import (
    ArtifactDescriptor,
    ArtifactLifecycleState,
    ArtifactRecord,
    ArtifactScope,
    CacheRequest,
    CoverageWindow,
    LookupMode,
)
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle

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


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.meta.json")


def _serialize_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _deserialize_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


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


def _coerce_ticker_value(value: Any, fallback: Ticker) -> str:
    if value is None:
        return fallback.name
    if isinstance(value, Ticker):
        return value.name
    return str(value)


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
    """True when datetimes match, ignoring stored resolution (ns vs us, etc.).

    Parquet / pyarrow may restore a coarser ``datetime64`` unit than the in-memory
    frame from ``load_source_candles``. Plain ``Index.equals`` then returns False,
    ``set_candles`` rewrites, and ``mark_dependents_stale`` forces a full bias refresh.
    """
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
    """Compare normalized candle frames without strict pandas dtype identity.

    ``DataFrame.equals`` is often false after a parquet round-trip (float32 vs float64,
    tiny float noise). Treating those as unchanged avoids rewriting candles and calling
    ``mark_dependents_stale``, which would otherwise force a full bias-artifact rebuild
    on every ``bootstrap_source_candles`` + ``set_candles`` cycle.
    """
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
    """Small facade for candles and artifact-backed reads/writes.

    Writable runtime state lives under a dedicated cache root and never shares
    a directory with repository-backed source candles in ``data/ohlc_data``.
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
        self._dependencies: dict[tuple[Ticker, TimeFrame], set[ArtifactDescriptor]] = defaultdict(set)
        self._dependency_index_loaded: set[tuple[Ticker, TimeFrame]] = set()

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
        self._dependencies.clear()
        self._dependency_index_loaded.clear()

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

    def _scope_cache_dir(self, scope: ArtifactScope) -> Path:
        if scope is ArtifactScope.RESEARCH:
            return self.research_artifact_cache_dir
        return self.live_artifact_cache_dir

    def _node_cache(self, descriptor: ArtifactDescriptor) -> BiasNodeCache:
        if descriptor.module_name is None or descriptor.ticker is None or descriptor.timeframe is None:
            raise ValueError("Node artifacts require module_name, ticker, and timeframe.")
        return BiasNodeCache(
            module_name=descriptor.module_name,
            params=dict(descriptor.params),
            ticker=descriptor.ticker,
            tf=descriptor.timeframe,
            cache_dir=str(self._scope_cache_dir(descriptor.scope)),
        )

    def _source_key(self, ticker: Ticker, timeframe: TimeFrame) -> tuple[Ticker, TimeFrame]:
        return ticker, timeframe

    def _candle_path(self, ticker: Ticker, timeframe: TimeFrame) -> Path:
        return self.candle_cache_dir / ticker.name / f"{timeframe.name}.parquet"

    def _artifact_path(self, descriptor: ArtifactDescriptor) -> Path | None:
        if descriptor.module_name is None or descriptor.ticker is None or descriptor.timeframe is None:
            return None
        return Path(self._node_cache(descriptor).cache_path)

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

    def _serialize_descriptor(self, descriptor: ArtifactDescriptor) -> dict[str, Any]:
        return {
            "family": descriptor.family,
            "ticker": descriptor.ticker.name if descriptor.ticker is not None else None,
            "timeframe": descriptor.timeframe.name if descriptor.timeframe is not None else None,
            "module_name": descriptor.module_name,
            "params": dict(descriptor.params),
            "scope": descriptor.scope.value,
            "artifact_name": descriptor.artifact_name,
        }

    def _deserialize_descriptor(self, payload: dict[str, Any]) -> ArtifactDescriptor:
        ticker_name = payload.get("ticker")
        timeframe_name = payload.get("timeframe")
        scope_value = payload.get("scope", ArtifactScope.LIVE.value)
        return ArtifactDescriptor(
            family=payload["family"],
            ticker=Ticker[ticker_name] if ticker_name is not None else None,
            timeframe=TimeFrame[timeframe_name] if timeframe_name is not None else None,
            module_name=payload.get("module_name"),
            params=payload.get("params", {}),
            scope=ArtifactScope(scope_value),
            artifact_name=payload.get("artifact_name"),
        )

    def _serialize_record(self, record: ArtifactRecord) -> dict[str, Any]:
        return {
            "descriptor": self._serialize_descriptor(record.descriptor),
            "coverage": {
                "start": _serialize_datetime(record.coverage.start),
                "end": _serialize_datetime(record.coverage.end),
            },
            "lifecycle_state": record.lifecycle_state.value,
            "revision": record.revision,
            "source_revision": record.source_revision,
            "depends_on": [list(item) for item in record.depends_on],
        }

    def _deserialize_record(self, payload: dict[str, Any]) -> ArtifactRecord:
        coverage = payload.get("coverage", {})
        depends_on = tuple(tuple(item) for item in payload.get("depends_on", []))
        return ArtifactRecord(
            descriptor=self._deserialize_descriptor(payload["descriptor"]),
            coverage=CoverageWindow(
                start=_deserialize_datetime(coverage.get("start")),
                end=_deserialize_datetime(coverage.get("end")),
            ),
            lifecycle_state=ArtifactLifecycleState(payload["lifecycle_state"]),
            revision=int(payload["revision"]),
            source_revision=int(payload.get("source_revision", 0)),
            depends_on=depends_on,
        )

    def _persist_record(self, path: Path, record: ArtifactRecord) -> None:
        metadata_path = _metadata_path(path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(self._serialize_record(record), sort_keys=True, indent=2, default=str),
            encoding="utf-8",
        )

    def _load_record(self, path: Path) -> ArtifactRecord | None:
        metadata_path = _metadata_path(path)
        if not metadata_path.exists():
            return None
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return self._deserialize_record(payload)

    def _load_record_from_metadata_path(self, metadata_path: Path) -> ArtifactRecord | None:
        if not metadata_path.exists():
            return None
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return self._deserialize_record(payload)

    def _register_dependencies(
        self,
        descriptor: ArtifactDescriptor,
        depends_on: tuple[tuple[str, str], ...],
    ) -> None:
        for ticker_name, timeframe_name in depends_on:
            try:
                ticker = Ticker[ticker_name]
                timeframe = TimeFrame[timeframe_name]
            except KeyError:
                logger.warning(
                    "Skipping invalid dependency while loading cache metadata: %s/%s",
                    ticker_name,
                    timeframe_name,
                )
                continue
            self._dependencies[self._source_key(ticker, timeframe)].add(descriptor)

    def _current_source_revision(
        self,
        depends_on: Sequence[tuple[Ticker, TimeFrame]],
    ) -> int:
        revisions = []
        for ticker, timeframe in depends_on:
            key = self._source_key(ticker, timeframe)
            if key not in self._candle_records:
                self._read_candles_from_disk(ticker, timeframe)
            record = self._candle_records.get(key)
            if record is not None:
                revisions.append(record.revision)
        return max(revisions, default=0)

    def _load_dependency_index(self, ticker: Ticker, timeframe: TimeFrame) -> None:
        key = self._source_key(ticker, timeframe)
        if key in self._dependency_index_loaded:
            return

        target = (ticker.name, timeframe.name)
        for root in (self.live_artifact_cache_dir, self.research_artifact_cache_dir):
            if not root.exists():
                continue
            for metadata_path in root.rglob("*.meta.json"):
                record = self._load_record_from_metadata_path(metadata_path)
                if record is None or target not in record.depends_on:
                    continue
                self._artifact_records.setdefault(record.descriptor, record)
                self._register_dependencies(record.descriptor, record.depends_on)
        self._dependency_index_loaded.add(key)

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
        record = self._load_record(path)
        if record is None:
            record = ArtifactRecord(
                descriptor=ArtifactDescriptor(
                    family="candles",
                    ticker=ticker,
                    timeframe=timeframe,
                    scope=ArtifactScope.LIVE,
                ),
                coverage=_frame_coverage(loaded),
                lifecycle_state=ArtifactLifecycleState.FRESH,
                revision=1,
            )

        self._candle_frames[key] = loaded
        self._candle_records[key] = record
        return loaded

    def _range_slice(
        self,
        frame: pd.DataFrame,
        *,
        module_name: str,
        ticker: Ticker | None,
        timeframe: TimeFrame | None,
        start: datetime | None,
        end: datetime | None,
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

        revision = 1 if key not in self._candle_records else self._candle_records[key].revision + 1
        self._candle_frames[key] = normalized
        descriptor = ArtifactDescriptor(
            family="candles",
            ticker=ticker,
            timeframe=timeframe,
            scope=scope,
        )
        record = ArtifactRecord(
            descriptor=descriptor,
            coverage=_frame_coverage(normalized),
            lifecycle_state=ArtifactLifecycleState.FRESH,
            revision=revision,
        )
        self._candle_records[key] = record
        self._persist_dataframe(self._candle_path(ticker, timeframe), normalized)
        self._persist_record(self._candle_path(ticker, timeframe), record)
        self.mark_dependents_stale(ticker, timeframe)
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

        revision = 1 if key not in self._candle_records else self._candle_records[key].revision + 1
        self._candle_frames[key] = merged
        descriptor = ArtifactDescriptor(
            family="candles",
            ticker=ticker,
            timeframe=timeframe,
            scope=scope,
        )
        record = ArtifactRecord(
            descriptor=descriptor,
            coverage=_frame_coverage(merged),
            lifecycle_state=ArtifactLifecycleState.FRESH,
            revision=revision,
        )
        self._candle_records[key] = record
        self._persist_dataframe(self._candle_path(ticker, timeframe), merged)
        self._persist_record(self._candle_path(ticker, timeframe), record)
        self.mark_dependents_stale(ticker, timeframe)
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

    def write_artifact(
        self,
        descriptor: ArtifactDescriptor,
        data: pd.DataFrame,
        depends_on: Sequence[tuple[Ticker, TimeFrame]] = (),
        source_revision: int = 0,
    ) -> None:
        normalized = _normalize_datetime_index(data)
        cache_dir = self._scope_cache_dir(descriptor.scope)
        existing = self._artifact_frames.get(descriptor)
        if existing is None:
            existing = self._read_artifact_from_disk(descriptor)
        existing_record = self._artifact_records.get(descriptor)
        current_source_revision = self._current_source_revision(depends_on)
        resolved_source_revision = source_revision or current_source_revision

        if current_source_revision != 0 and resolved_source_revision < current_source_revision:
            raise SourceRevisionConflictError(
                module_name=descriptor.module_name or descriptor.family,
                ticker=descriptor.ticker,
                timeframe=descriptor.timeframe,
                expected_revision=current_source_revision,
                actual_revision=resolved_source_revision,
            )
        if (
            existing_record is not None
            and existing_record.source_revision != 0
            and resolved_source_revision != 0
            and resolved_source_revision < existing_record.source_revision
        ):
            raise SourceRevisionConflictError(
                module_name=descriptor.module_name or descriptor.family,
                ticker=descriptor.ticker,
                timeframe=descriptor.timeframe,
                expected_revision=existing_record.source_revision,
                actual_revision=resolved_source_revision,
            )

        if descriptor.module_name is not None and descriptor.ticker is not None and descriptor.timeframe is not None:
            cache = BiasNodeCache(
                module_name=descriptor.module_name,
                params=dict(descriptor.params),
                ticker=descriptor.ticker,
                tf=descriptor.timeframe,
                cache_dir=str(cache_dir),
            )
            cache.save(normalized)

        revision = 1 if existing is None else self._artifact_records[descriptor].revision + 1
        self._artifact_frames[descriptor] = normalized
        record = ArtifactRecord(
            descriptor=descriptor,
            coverage=_frame_coverage(normalized),
            lifecycle_state=ArtifactLifecycleState.FRESH,
            revision=revision,
            source_revision=resolved_source_revision,
            depends_on=tuple((ticker.name, timeframe.name) for ticker, timeframe in depends_on),
        )
        self._artifact_records[descriptor] = record
        self._remove_artifact_dependencies(descriptor)
        if descriptor_path := self._artifact_path(descriptor):
            self._persist_record(descriptor_path, record)
        self._register_dependencies(descriptor, record.depends_on)

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
        record = self._artifact_records.get(descriptor)
        if record is not None and record.lifecycle_state is not ArtifactLifecycleState.FRESH:
            path = None
            if descriptor.module_name is not None and descriptor.ticker is not None and descriptor.timeframe is not None:
                path = self._node_cache(descriptor).cache_path
            raise ArtifactLifecycleError(
                key=descriptor.cache_key(),
                lifecycle_state=record.lifecycle_state.value,
                path=path,
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

    def _read_artifact_from_disk(self, descriptor: ArtifactDescriptor) -> Optional[pd.DataFrame]:
        if descriptor.module_name is None or descriptor.ticker is None or descriptor.timeframe is None:
            return None
        existing_frame = self._artifact_frames.get(descriptor)
        if existing_frame is not None:
            return existing_frame
        cache = self._node_cache(descriptor)
        if not cache.exists():
            return None
        try:
            loaded = _normalize_datetime_index(cache.load())
        except CacheMissError:
            # Unreadable or truncated artifact was removed by BiasNodeCache.load().
            return None
        descriptor_path = Path(cache.cache_path)
        disk_meta = self._load_record(descriptor_path)
        coverage = _frame_coverage(loaded)
        if disk_meta is not None:
            record = replace(disk_meta, descriptor=descriptor, coverage=coverage)
        elif (mem_rec := self._artifact_records.get(descriptor)) is not None:
            record = replace(mem_rec, descriptor=descriptor, coverage=coverage)
        else:
            record = ArtifactRecord(
                descriptor=descriptor,
                coverage=coverage,
                lifecycle_state=ArtifactLifecycleState.FRESH,
                revision=1,
            )
        self._artifact_frames[descriptor] = loaded
        self._artifact_records[descriptor] = record
        self._register_dependencies(descriptor, record.depends_on)
        return loaded

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

    def mark_dependents_stale(self, ticker: Ticker, timeframe: TimeFrame) -> None:
        self._load_dependency_index(ticker, timeframe)
        dependents = self._dependencies.get(self._source_key(ticker, timeframe), set())
        for descriptor in dependents:
            record = self._artifact_records.get(descriptor)
            if record is None:
                continue
            stale_record = ArtifactRecord(
                descriptor=record.descriptor,
                coverage=record.coverage,
                lifecycle_state=ArtifactLifecycleState.STALE,
                revision=record.revision,
                source_revision=record.source_revision,
                depends_on=record.depends_on,
            )
            self._artifact_records[descriptor] = stale_record
            if descriptor_path := self._artifact_path(descriptor):
                self._persist_record(descriptor_path, stale_record)

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

    def _remove_artifact_dependencies(self, descriptor: ArtifactDescriptor) -> None:
        dependency_keys = [
            source_key
            for source_key, dependents in self._dependencies.items()
            if descriptor in dependents
        ]
        for source_key in dependency_keys:
            dependents = self._dependencies[source_key]
            dependents.discard(descriptor)
            if not dependents:
                self._dependencies.pop(source_key, None)

    def prune_research_artifacts(self) -> list[ArtifactDescriptor]:
        removed: list[ArtifactDescriptor] = []
        for descriptor in list(self._artifact_frames.keys()):
            if descriptor.scope is not ArtifactScope.RESEARCH:
                continue
            removed.append(descriptor)
            self._artifact_frames.pop(descriptor, None)
            self._artifact_records.pop(descriptor, None)
            self._remove_artifact_dependencies(descriptor)
            if descriptor.module_name is None or descriptor.ticker is None or descriptor.timeframe is None:
                continue
            cache = self._node_cache(descriptor)
            if cache.exists():
                cache_path = Path(cache.cache_path)
                if cache_path.exists():
                    cache_path.unlink()
                metadata_path = _metadata_path(cache_path)
                if metadata_path.exists():
                    metadata_path.unlink()
        research_dir = self._scope_cache_dir(ArtifactScope.RESEARCH)
        if research_dir.exists() and not any(research_dir.iterdir()):
            shutil.rmtree(research_dir, ignore_errors=True)
        return removed

    def promote_artifact(
        self,
        descriptor: ArtifactDescriptor,
        target_scope: ArtifactScope = ArtifactScope.LIVE,
    ) -> ArtifactDescriptor:
        frame = self.read_artifact(descriptor)
        record = self.describe_artifact(descriptor)
        promoted = ArtifactDescriptor(
            family=descriptor.family,
            ticker=descriptor.ticker,
            timeframe=descriptor.timeframe,
            module_name=descriptor.module_name,
            params=dict(descriptor.params),
            scope=target_scope,
            artifact_name=descriptor.artifact_name,
        )
        depends_on: tuple[tuple[Ticker, TimeFrame], ...] = ()
        source_revision = 0
        if record is not None:
            depends_on = tuple(
                (Ticker[ticker_name], TimeFrame[timeframe_name])
                for ticker_name, timeframe_name in record.depends_on
            )
            source_revision = record.source_revision
        self.write_artifact(
            promoted,
            frame,
            depends_on=depends_on,
            source_revision=source_revision,
        )
        return promoted

    def get_artifact_record(self, descriptor: ArtifactDescriptor) -> ArtifactRecord:
        record = self.describe_artifact(descriptor)
        if record is None:
            raise ArtifactMissingError(
                module_name=descriptor.module_name or descriptor.family,
                ticker=descriptor.ticker,
                timeframe=descriptor.timeframe,
                reason="Artifact record is not loaded",
            )
        return record

    def describe_candle(self, ticker: Ticker, timeframe: TimeFrame) -> ArtifactRecord | None:
        key = self._source_key(ticker, timeframe)
        record = self._candle_records.get(key)
        if record is None:
            self._read_candles_from_disk(ticker, timeframe)
            record = self._candle_records.get(key)
        return record

    def describe_artifact(self, descriptor: ArtifactDescriptor) -> ArtifactRecord | None:
        cached = self._artifact_records.get(descriptor)
        if cached is not None:
            return cached
        if descriptor.module_name is None or descriptor.ticker is None or descriptor.timeframe is None:
            self._read_artifact_from_disk(descriptor)
            return self._artifact_records.get(descriptor)
        cache = self._node_cache(descriptor)
        if not cache.exists():
            return None
        descriptor_path = Path(cache.cache_path)
        meta = self._load_record(descriptor_path)
        if meta is not None:
            # Fast path for coverage checks (e.g. ensure_bias_cache_coverage): sidecar JSON
            # has lifecycle + coverage; avoid reading the full Parquet until read_artifact.
            record = replace(meta, descriptor=descriptor)
            self._artifact_records[descriptor] = record
            self._register_dependencies(descriptor, record.depends_on)
            return record
        self._read_artifact_from_disk(descriptor)
        return self._artifact_records.get(descriptor)

    def list_artifacts(self, scope: ArtifactScope | None = None) -> list[ArtifactRecord]:
        records = list(self._artifact_records.values())
        if scope is not None:
            records = [record for record in records if record.descriptor.scope is scope]
        return sorted(records, key=lambda record: record.descriptor.cache_key())

    def prune_scope(self, scope: ArtifactScope) -> list[ArtifactDescriptor]:
        if scope is ArtifactScope.RESEARCH:
            return self.prune_research_artifacts()
        removed: list[ArtifactDescriptor] = []
        for descriptor in list(self._artifact_frames.keys()):
            if descriptor.scope is not scope:
                continue
            removed.append(descriptor)
            self._artifact_frames.pop(descriptor, None)
            self._artifact_records.pop(descriptor, None)
            self._remove_artifact_dependencies(descriptor)
            if descriptor.module_name is None or descriptor.ticker is None or descriptor.timeframe is None:
                continue
            cache = self._node_cache(descriptor)
            cache_path = Path(cache.cache_path)
            if cache_path.exists():
                cache_path.unlink()
            metadata_path = _metadata_path(cache_path)
            if metadata_path.exists():
                metadata_path.unlink()
        return removed


CentralCache = CentralCacheStore
