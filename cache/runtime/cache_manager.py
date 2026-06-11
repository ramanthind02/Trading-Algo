"""
Cache Manager - ensures central-cache coverage for bias-node artifacts.

The CacheManager bootstraps source candles into the central cache and
(re)builds the bias-node / EWSD artifacts required by vault ensembles for a
requested date window. Canonical entry points:

- ``bootstrap_source_candles`` - load repository OHLC into the central cache.
- ``ensure_bias_cache_coverage`` - build/validate artifacts for explicit specs.
- ``ensure_vault_cache_coverage`` - same, driven by a vault ensemble's specs.

Usage:
    manager = CacheManager()
    manager.ensure_vault_cache_coverage(
        ['vault/D/buy_hold/buy_hold_long'],
        start_date=datetime(2010, 1, 1),
        end_date=datetime(2024, 12, 31),
    )
"""

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from .cache_paths import (
    default_live_artifact_cache_dir,
    default_source_candle_dir,
)
from lib.core.enums import Ticker, TimeFrame
from lib.core.models import Candle

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Orchestrates cache population for bias nodes.

    This class manages the creation and population of bias node caches.
    It supports:
    - Concurrent cache population with configurable worker count
    - Progress tracking and reporting
    - Integration with vault ensembles
    - Incremental updates (only compute missing data)

    Parameters
    ----------
    cache_dir : str, optional
        Root cache directory. Defaults to the runtime cache tree under
        ``.cache/trading_algo/central_cache/artifacts/live``.
    candle_dir : str, optional
        Directory containing source candle parquet files.
        Defaults to repository-backed ``data/ohlc_data``.

    Examples
    --------
    >>> manager = CacheManager()
    >>> summary = manager.ensure_vault_cache_coverage(
    ...     ['vault/D/buy_hold/buy_hold_long'],
    ...     start_date=datetime(2020, 1, 1),
    ...     end_date=datetime(2024, 12, 31),
    ... )
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        candle_dir: Optional[str] = None
    ):
        if cache_dir is None:
            cache_dir = str(default_live_artifact_cache_dir())
        if candle_dir is None:
            candle_dir = str(default_source_candle_dir())

        self.cache_dir = cache_dir
        self.candle_dir = candle_dir

        # Ensure directories exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def _central_cache_root(self) -> Path:
        cache_path = Path(self.cache_dir)
        if cache_path.name == "live" and cache_path.parent.name == "artifacts":
            return cache_path.parents[1]
        return cache_path

    def _central_cache_store(self) -> "CentralCacheStore":
        from .central_cache import CentralCacheStore

        root = self._central_cache_root()
        store = CentralCacheStore._instance  # type: ignore[attr-defined]
        if store is None or store.cache_dir != root:
            store = CentralCacheStore(cache_dir=str(root))
            CentralCacheStore._instance = store  # type: ignore[attr-defined]
        live_artifact_dir = Path(self.cache_dir)
        if store.live_artifact_cache_dir != live_artifact_dir:
            store.live_artifact_cache_dir = live_artifact_dir
            store.live_artifact_cache_dir.mkdir(parents=True, exist_ok=True)
        return store

    def find_source_candle_path(
        self,
        ticker: Ticker,
        tf: TimeFrame,
    ) -> Optional[Path]:
        """Return the discovered repository-backed candle file for a ticker/timeframe."""
        return self._find_candle_path(ticker, tf)

    def load_source_candles(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Public wrapper around source-candle loading for cache ingest paths."""
        return self._load_candles(ticker, tf, start_date=start_date, end_date=end_date)

    def _candidate_candle_paths(
        self,
        ticker: Ticker,
        tf: TimeFrame,
    ) -> tuple[Path, ...]:
        ticker_str = ticker.name if hasattr(ticker, "name") else str(ticker)
        tf_str = tf.name if hasattr(tf, "name") else str(tf)
        candle_root = Path(self.candle_dir)
        return (
            candle_root / f"{ticker_str}_{tf_str}.parquet",
            candle_root / tf_str / f"{ticker_str}.parquet",
            candle_root / ticker_str / f"{tf_str}.parquet",
            candle_root / f"{ticker_str}.parquet",
            candle_root / ticker_str / f"{tf_str}_{ticker_str}.parquet",
        )

    def _find_candle_path(
        self,
        ticker: Ticker,
        tf: TimeFrame,
    ) -> Optional[Path]:
        return next((path for path in self._candidate_candle_paths(ticker, tf) if path.exists()), None)

    def _load_candles(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load candles from parquet files.

        Parameters
        ----------
        ticker : Ticker
            Instrument ticker
        tf : TimeFrame
            Timeframe
        start_date : datetime, optional
            Start date filter
        end_date : datetime, optional
            End date filter

        Returns
        -------
        pd.DataFrame
            Candles DataFrame with columns: datetime, open, high, low, close, volume, ticker, timeframe

        Raises
        ------
        FileNotFoundError
            If candle file does not exist
        """
        from lib.core.research_feed import LEGACY_FEED, feed_for_ticker

        _feed = feed_for_ticker(ticker)
        if _feed != LEGACY_FEED:
            return self._load_alt_candles(ticker, tf, _feed, start_date, end_date)

        ticker_str = ticker.name if hasattr(ticker, 'name') else str(ticker)
        tf_str = tf.name if hasattr(tf, 'name') else str(tf)
        possible_paths = self._candidate_candle_paths(ticker, tf)
        candle_path = self._find_candle_path(ticker, tf)

        if candle_path is None:
            raise FileNotFoundError(
                f"Candle file not found for {ticker_str}/{tf_str}. "
                f"Searched: {[str(p) for p in possible_paths]}"
            )

        # Load candles
        df = pd.read_parquet(candle_path)

        # Ensure datetime column is datetime type. Accept both the legacy
        # (``datetime`` column) and current (``date``-named index/column, written
        # by data_platform.providers.norgate.migrate) parquet schemas, mirroring
        # data_platform.loaders._normalize_loaded_frame.
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'datetime'})
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif df.index.name in ('datetime', 'date') or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            col = (
                'datetime' if 'datetime' in df.columns
                else 'date' if 'date' in df.columns
                else df.columns[0]
            )
            df = df.rename(columns={col: 'datetime'})
            df['datetime'] = pd.to_datetime(df['datetime'])

        # Add ticker and timeframe columns if missing
        if 'ticker' not in df.columns:
            df['ticker'] = ticker_str
        if 'timeframe' not in df.columns:
            df['timeframe'] = tf

        # Filter by date range
        if start_date is not None:
            df = df[df['datetime'] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df['datetime'] <= pd.to_datetime(end_date)]

        # Sort by datetime
        df = df.sort_values('datetime')

        return df

    def _load_alt_candles(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        feed: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load candles from a non-futures research feed (cfd / spliced).

        Returns the same shape as :meth:`_load_candles`
        (datetime, open, high, low, close, volume, ticker, timeframe), keyed by
        the canonical (vault) ticker via ``brokers.resolve``.
        """
        from data_platform.providers.mt5.cfd_candles import load_research_candles_raw

        ticker_str = ticker.name if hasattr(ticker, 'name') else str(ticker)
        df = load_research_candles_raw(ticker, tf, feed).copy()
        df['ticker'] = ticker_str
        df['timeframe'] = tf
        if start_date is not None:
            df = df[df['datetime'] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df['datetime'] <= pd.to_datetime(end_date)]
        return df.sort_values('datetime').reset_index(drop=True)

    def get_available_date_range(
        self,
        tickers: List[Ticker],
        timeframes: List[TimeFrame],
    ) -> Optional[tuple[datetime, datetime]]:
        """Return (min_date, max_date) across all OHLC data for the given tickers and timeframes.

        Used when populating cache so the invariant "always use all possible data" is satisfied:
        cache population uses this full range instead of config.start/end.

        Returns
        -------
        (datetime, datetime) or None
            Global min and max datetime across all parquet files. None if no data found.
        """
        all_mins: List[datetime] = []
        all_maxes: List[datetime] = []
        for ticker in tickers:
            for tf in timeframes:
                candle_path = self._find_candle_path(ticker, tf)
                if candle_path is None:
                    continue
                try:
                    df = pd.read_parquet(candle_path)
                    if "datetime" in df.columns:
                        dts = pd.to_datetime(df["datetime"])
                    elif isinstance(df.index, pd.DatetimeIndex):
                        dts = df.index
                    else:
                        dts = pd.to_datetime(df.reset_index().iloc[:, 0])
                    if len(dts) == 0:
                        continue
                    all_mins.append(pd.Timestamp(dts.min()).to_pydatetime())
                    all_maxes.append(pd.Timestamp(dts.max()).to_pydatetime())
                except Exception as e:
                    logger.warning("Could not read date range from %s: %s", candle_path, e)
        if not all_mins or not all_maxes:
            return None
        return (min(all_mins), max(all_maxes))

    def get_available_date_range_per_ticker(
        self,
        tickers: List[Ticker],
        timeframes: List[TimeFrame],
    ) -> Dict[Ticker, Tuple[datetime, datetime]]:
        """Return (min_date, max_date) per ticker from OHLC parquet files.

        Only tickers that have at least one parquet file with data are included.
        Used to filter config.tickers to those that cover a requested date range.
        """
        result: Dict[Ticker, tuple] = {}
        for ticker in tickers:
            ticker_mins: List[datetime] = []
            ticker_maxes: List[datetime] = []
            for tf in timeframes:
                candle_path = self._find_candle_path(ticker, tf)
                if candle_path is None:
                    continue
                try:
                    df = pd.read_parquet(candle_path)
                    if "datetime" in df.columns:
                        dts = pd.to_datetime(df["datetime"])
                    elif isinstance(df.index, pd.DatetimeIndex):
                        dts = df.index
                    else:
                        dts = pd.to_datetime(df.reset_index().iloc[:, 0])
                    if len(dts) == 0:
                        continue
                    ticker_mins.append(pd.Timestamp(dts.min()).to_pydatetime())
                    ticker_maxes.append(pd.Timestamp(dts.max()).to_pydatetime())
                except Exception as e:
                    logger.warning("Could not read date range from %s: %s", candle_path, e)
            if ticker_mins and ticker_maxes:
                result[ticker] = (min(ticker_mins), max(ticker_maxes))
        return result

    def bootstrap_source_candles(
        self,
        tickers: Sequence[Ticker] | None = None,
        timeframes: Sequence[TimeFrame] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        reset_existing: bool = False,
    ) -> dict[str, Any]:
        """Load repository-backed candles into the runtime cache explicitly.

        Rows are **merged** via ``upsert_candles`` (not a full replace) so any
        newer history already in cache (e.g. from Interactive Brokers) is kept
        for dates beyond the parquet file end, while overlapping dates prefer the
        latest write per timestamp.
        """
        requested_tickers = list(tickers) if tickers is not None else list(Ticker)
        requested_timeframes = (
            list(timeframes)
            if timeframes is not None
            else [TimeFrame.D, TimeFrame.W, TimeFrame.M]
        )

        store = self._central_cache_store()
        if reset_existing:
            store.clear_candles(purge_persisted=True)
        from .live_cache_refresh import get_live_cache_refresh_orchestrator

        refresh_orchestrator = get_live_cache_refresh_orchestrator()
        refresh_orchestrator.begin_batch()

        details: list[dict[str, Any]] = []
        success = 0
        failed = 0

        series_pairs: list[tuple[Ticker, TimeFrame]] = [
            (ticker, timeframe)
            for ticker in requested_tickers
            for timeframe in requested_timeframes
        ]

        try:
            bar = tqdm(
                series_pairs,
                desc="Bootstrap OHLC → cache",
                unit="series",
            )
            for ticker, timeframe in bar:
                try:
                    candles_df = self.load_source_candles(
                        ticker,
                        timeframe,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if candles_df.empty:
                        raise ValueError("No source candles found in requested range")
                    # Merge into any existing cache (e.g. Interactive Brokers upserts) so
                    # repo parquets extend history without discarding newer bars past the file end.
                    store.upsert_candles(ticker, timeframe, candles_df)
                    details.append(
                        {
                            "ticker": ticker.name,
                            "tf": timeframe.name,
                            "status": "success",
                            "rows": len(candles_df),
                        }
                    )
                    success += 1
                except Exception as exc:
                    details.append(
                        {
                            "ticker": ticker.name,
                            "tf": timeframe.name,
                            "status": "failed",
                            "message": str(exc),
                        }
                    )
                    failed += 1
        finally:
            refresh_orchestrator.end_batch()

        return {
            "total": len(details),
            "success": success,
            "failed": failed,
            "details": details,
        }

    def _compute_bias_node_output_from_node(
        self,
        bias_node,
        candles_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute bias node output using an existing bias node instance.

        Parameters
        ----------
        bias_node : BiasNode
            Existing bias node instance
        candles_df : pd.DataFrame
            Candles to stream through the bias node

        Returns
        -------
        pd.DataFrame
            DataFrame with datetime index and output columns
        """
        # Stream candles and collect output
        datetimes = []
        outputs = []

        for _, row in candles_df.iterrows():
            candle = Candle.from_row(row)
            result = bias_node.add_candle(candle)
            datetimes.append(candle.datetime)
            outputs.append(result)

        # Build output DataFrame
        # Most bias nodes return [value] or [value, bool_value]
        if outputs and len(outputs[0]) == 1:
            df = pd.DataFrame({
                'datetime': datetimes,
                'value': [o[0] for o in outputs]
            })
        elif outputs and len(outputs[0]) == 2:
            df = pd.DataFrame({
                'datetime': datetimes,
                'value': [o[0] for o in outputs],
                'value_bool': [o[1] for o in outputs]
            })
        else:
            # Handle multi-output nodes
            n_outputs = len(outputs[0]) if outputs else 0
            data = {'datetime': datetimes}
            for i in range(n_outputs):
                col_name = f'value_{i}' if i > 0 else 'value'
                data[col_name] = [o[i] for o in outputs]
            df = pd.DataFrame(data)

        # Set datetime as index
        df = df.set_index('datetime')

        return df

    def _artifact_descriptor(
        self,
        module_name: str,
        params: Dict[str, Any],
        ticker: Ticker,
        tf: TimeFrame,
        *,
        scope: Optional["ArtifactScope"] = None,
    ) -> "ArtifactDescriptor":
        from .central_cache_models import ArtifactDescriptor, ArtifactScope
        from lib.core.helpers import create_bias_node

        resolved_scope = ArtifactScope.LIVE if scope is None else scope
        resolved_module_name = module_name
        resolved_params = dict(params)
        if module_name and module_name != "ewsd":
            bias_node = create_bias_node(module_name, ticker, tf, params)
            resolved_module_name = getattr(bias_node, "module_name", None) or module_name
            normalized_params = getattr(bias_node, "params", None)
            resolved_params = dict(normalized_params) if normalized_params else dict(params)

        return ArtifactDescriptor(
            family="bias",
            ticker=ticker,
            timeframe=tf,
            module_name=resolved_module_name,
            params=resolved_params,
            scope=resolved_scope,
            artifact_name=resolved_module_name,
        )

    def _coverage_spans_window(
        self,
        coverage_start: Optional[datetime],
        coverage_end: Optional[datetime],
        start_date: datetime,
        end_date: datetime,
    ) -> bool:
        if coverage_start is None or coverage_end is None:
            return False
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        return pd.Timestamp(coverage_start) <= start_ts and pd.Timestamp(coverage_end) >= end_ts

    def _refresh_status_for_descriptor(
        self,
        descriptor: "ArtifactDescriptor",
        start_date: datetime,
        end_date: datetime,
    ) -> tuple[bool, str]:
        store = self._central_cache_store()
        record = store.describe_artifact(descriptor)
        if record is None:
            return True, "missing"
        if not self._coverage_spans_window(
            record.coverage.start,
            record.coverage.end,
            start_date,
            end_date,
        ):
            return True, "out_of_range"
        return False, "fresh"

    def _boundary_gap_tolerance(self, timeframe: TimeFrame) -> pd.Timedelta:
        """Allow small boundary gaps caused by market calendars and coarse bar closes."""
        if timeframe in {TimeFrame.H1, TimeFrame.H4}:
            return pd.Timedelta(days=1)
        if timeframe is TimeFrame.D:
            return pd.Timedelta(days=3)
        if timeframe is TimeFrame.W:
            return pd.Timedelta(days=8)
        if timeframe is TimeFrame.M:
            return pd.Timedelta(days=31)
        return pd.Timedelta(0)

    def _coverage_supports_requested_boundary(
        self,
        timeframe: TimeFrame,
        requested_at: datetime,
        coverage_at: datetime,
        *,
        boundary: str,
    ) -> bool:
        """Return whether a coverage edge can satisfy a requested boundary."""
        requested_ts = pd.Timestamp(requested_at)
        coverage_ts = pd.Timestamp(coverage_at)
        tolerance = self._boundary_gap_tolerance(timeframe)

        if boundary == "start":
            if coverage_ts <= requested_ts:
                return True
            return coverage_ts - requested_ts <= tolerance
        if boundary == "end":
            if coverage_ts >= requested_ts:
                return True
            return requested_ts - coverage_ts <= tolerance
        raise ValueError(f"Unknown boundary '{boundary}'")

    def _require_exact_window_for_dependencies(
        self,
        depends_on: Sequence[tuple[Ticker, TimeFrame]],
        start_date: Optional[datetime],
        end_date: datetime,
    ) -> tuple[datetime, datetime]:
        """Intersect ``[start_date, end_date]`` with candle coverage for each dependency.

        Earlier implementations required candle coverage edges to match requested boundaries
        within a small tolerance, which failed when OHLC history starts later for one ticker
        (e.g. RTY from 2005 while config asks from 2000). We now clip to the **overlap** of
        all dependency coverages so bias artifacts can build on available data.

        ``start_date=None`` means "use the earliest available candle coverage" so that
        every artifact is always built with its full warmup history.
        """
        from .central_cache_errors import ArtifactMissingError, CacheCoverageError

        store = self._central_cache_store()
        if not depends_on:
            if start_date is None:
                raise ArtifactMissingError(
                    module_name="candles",
                    ticker=None,
                    timeframe=None,
                    reason="start_date=None requires at least one dependency to determine coverage start",
                )
            return start_date, end_date

        requested_end = pd.Timestamp(end_date)
        # None → use candle coverage start (determined below by clamping)
        effective_start = pd.Timestamp(start_date) if start_date is not None else None
        effective_end = requested_end

        for dep_ticker, dep_tf in depends_on:
            record = store.describe_candle(dep_ticker, dep_tf)
            if record is None or record.coverage.start is None or record.coverage.end is None:
                raise ArtifactMissingError(
                    module_name="candles",
                    ticker=dep_ticker,
                    timeframe=dep_tf,
                    reason="Dependency candles are not loaded",
                )
            cov_s = pd.Timestamp(record.coverage.start)
            cov_e = pd.Timestamp(record.coverage.end)
            effective_start = cov_s if effective_start is None else max(effective_start, cov_s)
            effective_end = min(effective_end, cov_e)

        if effective_start is None:
            raise ArtifactMissingError(
                module_name="candles",
                ticker=depends_on[0][0],
                timeframe=depends_on[0][1],
                reason="Could not determine effective start: no candle coverage found for any dependency",
            )

        if effective_start > effective_end:
            first_ticker, first_tf = depends_on[0]
            raise CacheCoverageError(
                module_name="candles",
                ticker=first_ticker,
                timeframe=first_tf,
                start=start_date,
                end=end_date,
                requested_range=(start_date, end_date),
                available_range=(
                    effective_start.to_pydatetime(),
                    effective_end.to_pydatetime(),
                ),
                message="No overlapping candle coverage for requested window across dependencies",
            )

        return effective_start.to_pydatetime(), effective_end.to_pydatetime()

    def _compute_artifact_from_central_cache(
        self,
        module_name: str,
        params: Dict[str, Any],
        ticker: Ticker,
        tf: TimeFrame,
        start_date: datetime,
        end_date: datetime,
        cold_rebuild_candle_count: int,
    ) -> pd.DataFrame:
        from lib.compute.daily_ewsd_volatility import compute_daily_ewsd_volatility
        from lib.core.helpers import create_fresh_bias_node

        store = self._central_cache_store()
        candles_frame = store.query_candles(
            ticker,
            tf,
            end=end_date,
        )
        start_ts = pd.Timestamp(start_date)
        candles_df = self._select_cold_rebuild_candles(
            candles_frame,
            requested_start=start_date,
            cold_rebuild_candle_count=cold_rebuild_candle_count,
        ).reset_index()

        if module_name == "ewsd":
            volatility_df = compute_daily_ewsd_volatility(candles_df)
            output_df = volatility_df.set_index("datetime")[["ewsd_annual_vol"]]
            return output_df.loc[
                (output_df.index >= start_ts) & (output_df.index <= pd.Timestamp(end_date))
            ].copy()

        bias_node = create_fresh_bias_node(module_name, ticker, tf, params)
        output_df = self._compute_bias_node_output_from_node(bias_node, candles_df)
        return output_df.loc[
            (output_df.index >= start_ts) & (output_df.index <= pd.Timestamp(end_date))
        ].copy()

    def _cold_rebuild_candle_count_for_spec(
        self,
        module_name: str,
        params: Dict[str, Any],
        ticker: Ticker,
        tf: TimeFrame,
    ) -> int:
        from lib.core.helpers import create_fresh_bias_node

        if module_name == "ewsd":
            long_run_window = int(params.get("long_run_window", 2520))
            if long_run_window <= 0:
                return 1
            buffer_bars = max(1, math.ceil(long_run_window * 0.2))
            return long_run_window + buffer_bars

        bias_node = create_fresh_bias_node(module_name, ticker, tf, params)
        return bias_node.cold_rebuild_candle_count()

    def _select_cold_rebuild_candles(
        self,
        candles_frame: pd.DataFrame,
        *,
        requested_start: datetime,
        cold_rebuild_candle_count: int,
    ) -> pd.DataFrame:
        if candles_frame.empty:
            return candles_frame.copy()

        start_position = candles_frame.index.searchsorted(
            pd.Timestamp(requested_start),
            side="left",
        )
        warmup_position = max(
            0,
            start_position - max(cold_rebuild_candle_count - 1, 0),
        )
        return candles_frame.iloc[warmup_position:].copy()

    def _resolve_cold_rebuild_start(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        requested_start: datetime,
        requested_end: datetime,
        cold_rebuild_candle_count: int,
    ) -> datetime:
        store = self._central_cache_store()
        candles_frame = store.query_candles(
            ticker,
            tf,
            end=requested_end,
        )
        if candles_frame.empty:
            return requested_start

        rebuild_frame = self._select_cold_rebuild_candles(
            candles_frame,
            requested_start=requested_start,
            cold_rebuild_candle_count=cold_rebuild_candle_count,
        )
        return pd.Timestamp(rebuild_frame.index[0]).to_pydatetime()

    def _resolved_source_revision(
        self,
        descriptor: Any,
        depends_on: Sequence[tuple[Ticker, TimeFrame]],
    ) -> int:
        """Source-revision tag stored alongside a rebuilt bias artifact.

        The simplified central cache does **no** revision tracking (see
        ``CentralCacheStore.write_artifact`` — ``source_revision`` is accepted but
        inert), so this returns ``0`` to match the default / serial-rebuild path.
        It exists so the parallel-rebuild path's ``write_artifact`` call resolves;
        wire real provenance here if revision-based invalidation is added.
        """
        return 0

    def ensure_bias_cache_coverage(
        self,
        bias_node_specs: Sequence[Dict[str, Any]],
        tickers: Sequence[Ticker],
        start_date: Optional[datetime],
        end_date: datetime,
        refresh_mode: str = "missing_stale_only",
        include_daily_ewsd: bool = True,
        artifact_scope: Optional["ArtifactScope"] = None,
        *,
        max_workers: int = 4,
    ) -> Dict[str, Any]:
        """Ensure central-cache coverage for explicit bias-node specs and tickers.

        ``start_date=None`` means "full available history" — the cache will be
        built from the earliest available candle date for each dependency ticker.
        This guarantees every indicator has its complete warmup window and the
        cache is always the source of truth for predictions.

        Parameters
        ----------
        artifact_scope :
            When set (e.g. ``ArtifactScope.RESEARCH``), read/write artifacts under the
            research subtree instead of ``artifacts/live``. Defaults to live scope.
        max_workers :
            Parallel threads used to **compute** artifacts that need a rebuild (each
            task streams candles through the bias node). Writes remain sequential.
            Set to ``1`` to disable parallelism (matches legacy single-thread behavior).
        """
        from .central_cache_models import ArtifactScope as _ArtifactScope

        resolved_scope: _ArtifactScope = (
            _ArtifactScope.LIVE if artifact_scope is None else artifact_scope
        )
        allowed_refresh_modes = {"missing_stale_only", "always_rebuild", "validate_only"}
        if refresh_mode not in allowed_refresh_modes:
            raise ValueError(
                f"refresh_mode must be one of {sorted(allowed_refresh_modes)}, "
                f"got '{refresh_mode}'"
            )

        requested_tickers = sorted(set(tickers), key=lambda item: item.name)
        source_timeframes: set[TimeFrame] = {TimeFrame.D} if include_daily_ewsd else set()
        artifact_tasks: dict[tuple[str, str, str, str], dict[str, Any]] = {}

        for spec in bias_node_specs:
            module_name = str(spec["module_name"])
            params = dict(spec.get("params", {}))
            timeframes = tuple(
                TimeFrame[tf] if isinstance(tf, str) else tf
                for tf in spec.get("timeframes", [TimeFrame.D])
            )
            source_timeframes.update(timeframes)
            for ticker in requested_tickers:
                for tf in timeframes:
                    task_key = (
                        module_name,
                        ticker.name,
                        tf.name,
                        repr(sorted(params.items(), key=lambda item: item[0])),
                    )
                    if task_key in artifact_tasks:
                        continue
                    artifact_tasks[task_key] = {
                        "module_name": module_name,
                        "params": params,
                        "ticker": ticker,
                        "tf": tf,
                        "depends_on": ((ticker, tf),),
                        "cold_rebuild_candle_count": self._cold_rebuild_candle_count_for_spec(
                            module_name,
                            params,
                            ticker,
                            tf,
                        ),
                        "descriptor": self._artifact_descriptor(
                            module_name,
                            params,
                            ticker,
                            tf,
                            scope=resolved_scope,
                        ),
                    }

        if include_daily_ewsd:
            for ticker in requested_tickers:
                for tf in sorted(source_timeframes, key=lambda item: item.name):
                    long_run_window = 10 * tf.bars_per_year
                    task_key = (
                        "ewsd",
                        ticker.name,
                        tf.name,
                        f"long_run_window={long_run_window}",
                    )
                    artifact_tasks[task_key] = {
                        "module_name": "ewsd",
                        "params": {"long_run_window": long_run_window},
                        "ticker": ticker,
                        "tf": tf,
                        "depends_on": ((ticker, tf),),
                        "cold_rebuild_candle_count": self._cold_rebuild_candle_count_for_spec(
                            "ewsd",
                            {"long_run_window": long_run_window},
                            ticker,
                            tf,
                        ),
                        "descriptor": self._artifact_descriptor(
                            "ewsd",
                            {"long_run_window": long_run_window},
                            ticker,
                            tf,
                            scope=resolved_scope,
                        ),
                    }

        compatibility_bootstrap_summary = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "details": [],
            "status": "not_requested",
        }

        store = self._central_cache_store()
        details: list[dict[str, Any]] = []
        rebuilt = 0
        validated = 0
        failed = 0

        task_list = list(artifact_tasks.values())
        rebuild_queue: list[
            tuple[
                dict[str, Any],
                datetime,
                datetime,
                datetime,
                str,
            ]
        ] = []

        for task in tqdm(
            task_list,
            desc="Bias / EWSD cache (check)",
            unit="task",
        ):
            descriptor = task["descriptor"]
            try:
                effective_start, effective_end = self._require_exact_window_for_dependencies(
                    task["depends_on"],
                    start_date,
                    end_date,
                )
                cold_rebuild_start = self._resolve_cold_rebuild_start(
                    task["ticker"],
                    task["tf"],
                    effective_start,
                    effective_end,
                    task["cold_rebuild_candle_count"],
                )
                self._require_exact_window_for_dependencies(
                    task["depends_on"],
                    cold_rebuild_start,
                    effective_end,
                )
                needs_refresh, reason = self._refresh_status_for_descriptor(
                    descriptor,
                    effective_start,
                    effective_end,
                )
                if refresh_mode == "always_rebuild":
                    needs_refresh = True
                    reason = "forced_rebuild"
            except Exception as exc:
                details.append(
                    {
                        "module_name": task["module_name"],
                        "ticker": task["ticker"].name,
                        "tf": task["tf"].name,
                        "status": "failed",
                        "reason": "coverage_unavailable",
                        "message": str(exc),
                    }
                )
                failed += 1
                continue

            if not needs_refresh:
                details.append(
                    {
                        "module_name": task["module_name"],
                        "ticker": task["ticker"].name,
                        "tf": task["tf"].name,
                        "status": "fresh",
                        "cold_rebuild_start": cold_rebuild_start.isoformat(),
                        "cold_rebuild_candle_count": task["cold_rebuild_candle_count"],
                        "effective_start": effective_start.isoformat(),
                        "effective_end": effective_end.isoformat(),
                    }
                )
                validated += 1
                continue

            if refresh_mode == "validate_only":
                details.append(
                    {
                        "module_name": task["module_name"],
                        "ticker": task["ticker"].name,
                        "tf": task["tf"].name,
                        "status": "missing",
                        "reason": reason,
                        "cold_rebuild_start": cold_rebuild_start.isoformat(),
                        "cold_rebuild_candle_count": task["cold_rebuild_candle_count"],
                        "effective_start": effective_start.isoformat(),
                        "effective_end": effective_end.isoformat(),
                    }
                )
                failed += 1
                continue

            rebuild_queue.append(
                (task, effective_start, effective_end, cold_rebuild_start, reason)
            )

        workers = max(1, min(max_workers, len(rebuild_queue), (os.cpu_count() or 4)))

        def _run_rebuild(
            item: tuple[
                dict[str, Any],
                datetime,
                datetime,
                datetime,
                str,
            ],
        ) -> tuple[dict[str, Any], pd.DataFrame]:
            t, eff_s, eff_e, _cold_s, _reason = item
            df = self._compute_artifact_from_central_cache(
                t["module_name"],
                t["params"],
                t["ticker"],
                t["tf"],
                eff_s,
                eff_e,
                t["cold_rebuild_candle_count"],
            )
            return (t, df)

        if workers == 1:
            for item in tqdm(
                rebuild_queue,
                desc="Bias / EWSD artifacts (rebuild)",
                unit="task",
            ):
                task, effective_start, effective_end, cold_rebuild_start, reason = item
                descriptor = task["descriptor"]
                try:
                    _, artifact_df = _run_rebuild(item)
                    store.write_artifact(descriptor, artifact_df)
                    details.append(
                        {
                            "module_name": task["module_name"],
                            "ticker": task["ticker"].name,
                            "tf": task["tf"].name,
                            "status": "rebuilt",
                            "reason": reason,
                            "rows": len(artifact_df),
                            "cold_rebuild_start": cold_rebuild_start.isoformat(),
                            "cold_rebuild_candle_count": task["cold_rebuild_candle_count"],
                            "effective_start": effective_start.isoformat(),
                            "effective_end": effective_end.isoformat(),
                        }
                    )
                    rebuilt += 1
                except Exception as exc:
                    logger.error(
                        "Failed to refresh artifact %s/%s/%s: %s",
                        task["module_name"],
                        task["ticker"].name,
                        task["tf"].name,
                        exc,
                        exc_info=True,
                    )
                    details.append(
                        {
                            "module_name": task["module_name"],
                            "ticker": task["ticker"].name,
                            "tf": task["tf"].name,
                            "status": "failed",
                            "reason": reason,
                            "message": str(exc),
                            "cold_rebuild_start": cold_rebuild_start.isoformat(),
                            "cold_rebuild_candle_count": task["cold_rebuild_candle_count"],
                            "effective_start": effective_start.isoformat(),
                            "effective_end": effective_end.isoformat(),
                        }
                    )
                    failed += 1
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_item = {
                    executor.submit(_run_rebuild, item): item for item in rebuild_queue
                }
                for future in tqdm(
                    as_completed(future_to_item),
                    total=len(rebuild_queue),
                    desc="Bias / EWSD artifacts (rebuild)",
                    unit="task",
                ):
                    item = future_to_item[future]
                    task, effective_start, effective_end, cold_rebuild_start, reason = item
                    descriptor = task["descriptor"]
                    try:
                        _, artifact_df = future.result()
                        store.write_artifact(
                            descriptor,
                            artifact_df,
                            depends_on=task["depends_on"],
                            source_revision=self._resolved_source_revision(
                                descriptor,
                                task["depends_on"],
                            ),
                        )
                        details.append(
                            {
                                "module_name": task["module_name"],
                                "ticker": task["ticker"].name,
                                "tf": task["tf"].name,
                                "status": "rebuilt",
                                "reason": reason,
                                "rows": len(artifact_df),
                                "cold_rebuild_start": cold_rebuild_start.isoformat(),
                                "cold_rebuild_candle_count": task["cold_rebuild_candle_count"],
                                "effective_start": effective_start.isoformat(),
                                "effective_end": effective_end.isoformat(),
                            }
                        )
                        rebuilt += 1
                    except Exception as exc:
                        logger.error(
                            "Failed to refresh artifact %s/%s/%s: %s",
                            task["module_name"],
                            task["ticker"].name,
                            task["tf"].name,
                            exc,
                            exc_info=True,
                        )
                        details.append(
                            {
                                "module_name": task["module_name"],
                                "ticker": task["ticker"].name,
                                "tf": task["tf"].name,
                                "status": "failed",
                                "reason": reason,
                                "message": str(exc),
                                "cold_rebuild_start": cold_rebuild_start.isoformat(),
                                "cold_rebuild_candle_count": task["cold_rebuild_candle_count"],
                                "effective_start": effective_start.isoformat(),
                                "effective_end": effective_end.isoformat(),
                            }
                        )
                        failed += 1

        return {
            "refresh_mode": refresh_mode,
            "bootstrap": compatibility_bootstrap_summary,
            "ingested": compatibility_bootstrap_summary,
            "total_tasks": len(artifact_tasks),
            "rebuilt": rebuilt,
            "validated": validated,
            "failed": failed,
            "requested_tickers": [ticker.name for ticker in requested_tickers],
            "timeframes": [tf.name for tf in sorted(source_timeframes)],
            "details": details,
        }

    def ensure_vault_cache_coverage(
        self,
        vault_ensemble_dirs: Sequence[str],
        start_date: Optional[datetime],
        end_date: datetime,
        refresh_mode: str = "missing_stale_only",
    ) -> Dict[str, Any]:
        """
        Ensure central-cache coverage for all artifacts required by vault ensembles.

        The current vault layout is feature-file based, so this method reads
        specs from ``features/*.json`` rather than the older control-file path.
        It validates and refreshes only the required artifacts for the requested
        date window using candles already present in the central cache.
        """
        from ensemble.vault_manager import get_bias_node_specs, get_ensemble_tickers

        portfolio_tickers: set[Ticker] = set()
        bias_node_specs: list[dict[str, Any]] = []

        for ensemble_dir in vault_ensemble_dirs:
            ensemble_tickers = get_ensemble_tickers(ensemble_dir)
            portfolio_tickers.update(ensemble_tickers)
            bias_node_specs.extend(get_bias_node_specs(ensemble_dir))

        summary = self.ensure_bias_cache_coverage(
            bias_node_specs=bias_node_specs,
            tickers=sorted(portfolio_tickers),
            start_date=start_date,
            end_date=end_date,
            refresh_mode=refresh_mode,
            include_daily_ewsd=True,
        )
        summary["portfolio_tickers"] = [ticker.name for ticker in sorted(portfolio_tickers)]
        return summary
