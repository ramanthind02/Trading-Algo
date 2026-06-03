"""
Cache Manager - Orchestrates cache population for bias nodes.

This module provides the CacheManager class for populating bias node
caches with concurrent workers. It can populate caches for:
- Individual bias node specifications
- All bias nodes in a vault ensemble

Usage:
    # Programmatic API
    manager = CacheManager()
    manager.populate_cache(
        bias_node_specs=[
            {'module_name': 'rsi', 'params': {'lookback': 14}, 'timeframes': ['D']},
            {'module_name': 'ewmac', 'params': {'spanFast': 16, 'spanSlow': 64}, 'timeframes': ['D']},
        ],
        tickers=[Ticker.ES, Ticker.NQ],
        start_date=datetime(2010, 1, 1),
        end_date=datetime(2024, 12, 31)
    )

    # CLI
    python -m utils.cache.runtime.cache_manager --vault vault/D/buy_hold_long --start 2010-01-01 --end 2024-12-31

Author: Trading Research Team
Date: 2025-01-07
"""

import argparse
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from .bias_node_cache import BiasNodeCache
from .cache_paths import (
    default_live_artifact_cache_dir,
    default_source_candle_dir,
)
from utils.core.enums import Ticker, TimeFrame
from utils.core.models import Candle

logger = logging.getLogger(__name__)

# Required for volatility-scaled targets (log_return_ewsd).
# Volatility normalization is EWSD-only and always uses daily settings.
def get_auxiliary_specs_for_timeframe(tf: TimeFrame) -> list[dict]:
    """Return EWSD auxiliary specs for volatility-scaled targets."""
    _ = tf
    return [
        {"module_name": "ewsd", "params": {"long_run_window": 2520}},
    ]


REQUIRED_AUXILIARY_SPECS = get_auxiliary_specs_for_timeframe(TimeFrame.D)


def _spec_matches(spec: Dict[str, Any], module_name: str, params: Dict[str, Any]) -> bool:
    """Return True if spec has the given module_name and params."""
    if spec.get("module_name") != module_name:
        return False
    return spec.get("params", {}) == params


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
    >>> results = manager.populate_cache(
    ...     bias_node_specs=[
    ...         {'module_name': 'rsi', 'params': {'lookback': 14}, 'timeframes': ['D']},
    ...     ],
    ...     tickers=[Ticker.ES],
    ...     start_date=datetime(2020, 1, 1),
    ...     end_date=datetime(2024, 12, 31)
    ... )
    >>> print(results)
    {'total': 1, 'success': 1, 'failed': 0, 'skipped': 0}
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

        # Ensure datetime column is datetime type
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif df.index.name == 'datetime' or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
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

    def _compute_bias_node_output(
        self,
        module_name: str,
        params: Dict[str, Any],
        ticker: Ticker,
        tf: TimeFrame,
        candles_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute bias node output by streaming candles.

        Parameters
        ----------
        module_name : str
            Bias node module name
        params : Dict[str, Any]
            Bias node parameters
        ticker : Ticker
            Instrument ticker
        tf : TimeFrame
            Timeframe
        candles_df : pd.DataFrame
            Candles to stream through the bias node

        Returns
        -------
        pd.DataFrame
            DataFrame with datetime index and output columns
        """
        from utils.core.helpers import create_fresh_bias_node

        # Create bias node instance
        bias_node = create_fresh_bias_node(module_name, ticker, tf, params)

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

    def _populate_single_cache(
        self,
        module_name: str,
        params: Dict[str, Any],
        ticker: Ticker,
        tf: TimeFrame,
        start_date: datetime,
        end_date: datetime,
        overwrite: bool = True
    ) -> Dict[str, Any]:
        """
        Populate cache for a single bias node configuration.

        Parameters
        ----------
        module_name : str
            Bias node module name
        params : Dict[str, Any]
            Bias node parameters
        ticker : Ticker
            Instrument ticker
        tf : TimeFrame
            Timeframe
        start_date : datetime
            Start date for cache
        end_date : datetime
            End date for cache
        overwrite : bool
            Whether to overwrite existing cache

        Returns
        -------
        Dict[str, Any]
            Result dict with status and details
        """
        from .central_cache import CentralCacheStore
        from .central_cache_models import ArtifactLifecycleState
        from utils.core.helpers import create_fresh_bias_node

        ticker_str = ticker.name if hasattr(ticker, 'name') else str(ticker)
        tf_str = tf.name if hasattr(tf, 'name') else str(tf)

        # Initialize result with basic info (may be updated after bias node creation)
        result = {
            'module_name': module_name,
            'params': params,
            'ticker': ticker_str,
            'tf': tf_str,
            'cache_path': None,
            'status': 'unknown'
        }

        try:
            # Create bias node to get its actual module_name and params
            # (bias nodes may normalize/rename these)
            bias_node = create_fresh_bias_node(module_name, ticker, tf, params)
            actual_module_name = getattr(bias_node, 'module_name', None) or module_name
            normalized_params = getattr(bias_node, 'params', None)
            actual_params = dict(normalized_params) if normalized_params else dict(params)
            if module_name == "ewsd":
                actual_module_name = "ewsd"
                actual_params = dict(params)

            cache = BiasNodeCache(
                module_name=actual_module_name,
                params=actual_params,
                ticker=ticker,
                tf=tf,
                cache_dir=self.cache_dir
            )

            # Update result with actual values
            result['module_name'] = actual_module_name
            result['params'] = actual_params
            result['cache_path'] = cache.cache_path
            cold_rebuild_candle_count = self._cold_rebuild_candle_count_for_spec(
                actual_module_name,
                actual_params,
                ticker,
                tf,
            )

            descriptor = self._artifact_descriptor(actual_module_name, actual_params, ticker, tf)
            store = self._central_cache_store()
            record = store.describe_artifact(descriptor)

            # Skip only fresh, in-range artifacts. Stale or partial artifacts
            # must be rebuilt even when overwrite=False.
            if (
                cache.exists()
                and not overwrite
                and record is not None
                and record.lifecycle_state is ArtifactLifecycleState.FRESH
                and self._coverage_spans_window(
                    record.coverage.start,
                    record.coverage.end,
                    start_date,
                    end_date,
                )
            ):
                result['status'] = 'skipped'
                result['message'] = 'Cache exists, artifact is fresh, and overwrite=False'
                return result

            # Load candles
            candles_df = self._load_candles(ticker, tf, None, end_date)

            if candles_df.empty:
                result['status'] = 'failed'
                result['message'] = 'No candles found for date range'
                return result

            candles_df = candles_df.set_index("datetime").sort_index()
            rebuild_df = self._select_cold_rebuild_candles(
                candles_df,
                requested_start=start_date,
                cold_rebuild_candle_count=cold_rebuild_candle_count,
            ).reset_index()

            result['candle_count'] = len(rebuild_df)

            # Compute bias node output (reuse the bias node we created)
            output_df = self._compute_bias_node_output_from_node(
                bias_node, rebuild_df
            )
            output_df = output_df.loc[
                (output_df.index >= pd.Timestamp(start_date))
                & (output_df.index <= pd.Timestamp(end_date))
            ].copy()

            # Save through the central cache so parquet payload and metadata stay in sync.
            depends_on = self._depends_on_for_spec(ticker, tf, actual_params)
            store.write_artifact(
                descriptor,
                output_df,
                depends_on=depends_on,
                source_revision=self._resolved_source_revision(descriptor, depends_on),
            )

            result['status'] = 'success'
            result['row_count'] = len(output_df)
            result['date_range'] = (
                output_df.index.min().isoformat(),
                output_df.index.max().isoformat()
            )

        except FileNotFoundError as e:
            result['status'] = 'failed'
            result['message'] = str(e)
            logger.error(f"Failed to populate cache: {e}")

        except Exception as e:
            result['status'] = 'failed'
            result['message'] = str(e)
            logger.error(f"Error populating cache for {module_name}/{ticker_str}: {e}", exc_info=True)

        return result

    def populate_cache(
        self,
        bias_node_specs: List[Dict[str, Any]],
        tickers: List[Ticker],
        start_date: datetime,
        end_date: datetime,
        max_workers: int = 4,
        overwrite_existing: bool = True,
        show_progress: bool = True,
        timeframe: TimeFrame = TimeFrame.D,
    ) -> Dict[str, Any]:
        """
        Populate caches for multiple bias node specifications.

        Parameters
        ----------
        bias_node_specs : List[Dict]
            List of bias node specifications. Each dict should have:
            - module_name: str
            - params: Dict[str, Any]
            - timeframes: List[TimeFrame] or List[str]
        tickers : List[Ticker]
            List of tickers to cache
        start_date : datetime
            Start date for cache
        end_date : datetime
            End date for cache
        max_workers : int
            Number of concurrent workers
        overwrite_existing : bool
            Whether to overwrite existing caches
        show_progress : bool
            Whether to print progress
        timeframe : TimeFrame
            Active research timeframe (auxiliary EWSD cache remains daily).

        Returns
        -------
        Dict[str, Any]
            Summary with total, success, failed, skipped counts and details
        """
        # Collect all timeframes used by the requested specs
        all_timeframes: List[TimeFrame] = []
        for spec in bias_node_specs:
            timeframes = spec.get("timeframes", [TimeFrame.D])
            for tf in timeframes:
                t = TimeFrame[tf] if isinstance(tf, str) else tf
                if t not in all_timeframes:
                    all_timeframes.append(t)
        if not all_timeframes:
            all_timeframes = [TimeFrame.D]

        # Ensure required auxiliary bias nodes (EWSD-only) are included.
        specs_to_use = list(bias_node_specs)
        for aux in get_auxiliary_specs_for_timeframe(timeframe):
            if not any(
                _spec_matches(spec, aux["module_name"], aux["params"])
                for spec in bias_node_specs
            ):
                specs_to_use.append({
                    "module_name": aux["module_name"],
                    "params": aux["params"],
                    "timeframes": [TimeFrame.D],
                })
                if show_progress:
                    print(f"  Adding required auxiliary: {aux['module_name']} (for log_return_ewsd)")

        # Build list of (module, params, ticker, tf) combinations
        tasks = []
        for spec in specs_to_use:
            module_name = spec['module_name']
            params = spec.get('params', {})
            timeframes = spec.get('timeframes', [TimeFrame.D])

            # Normalize timeframes to TimeFrame enums
            normalized_tfs = []
            for tf in timeframes:
                if isinstance(tf, str):
                    normalized_tfs.append(TimeFrame[tf])
                else:
                    normalized_tfs.append(tf)

            for ticker in tickers:
                for tf in normalized_tfs:
                    tasks.append((module_name, params, ticker, tf))

        if show_progress:
            print(f"Populating cache for {len(tasks)} combinations...")
            print(f"  Bias node specs: {len(specs_to_use)} (including required ewsd if needed)")
            print(f"  Tickers: {len(tickers)}")
            print(f"  Date range: {start_date} to {end_date}")
            print(f"  Max workers: {max_workers}")

        # Execute with concurrent workers
        results = []
        if max_workers == 1:
            # Sequential execution
            for i, (module_name, params, ticker, tf) in enumerate(tasks):
                if show_progress:
                    print(f"  [{i+1}/{len(tasks)}] {module_name} / {ticker.name if hasattr(ticker, 'name') else ticker} / {tf.name if hasattr(tf, 'name') else tf}")
                result = self._populate_single_cache(
                    module_name, params, ticker, tf,
                    start_date, end_date, overwrite_existing
                )
                results.append(result)
        else:
            # Concurrent execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for module_name, params, ticker, tf in tasks:
                    future = executor.submit(
                        self._populate_single_cache,
                        module_name, params, ticker, tf,
                        start_date, end_date, overwrite_existing
                    )
                    futures[future] = (module_name, ticker, tf)

                for i, future in enumerate(as_completed(futures)):
                    module_name, ticker, tf = futures[future]
                    if show_progress:
                        ticker_str = ticker.name if hasattr(ticker, 'name') else str(ticker)
                        tf_str = tf.name if hasattr(tf, 'name') else str(tf)
                        print(f"  [{i+1}/{len(tasks)}] {module_name} / {ticker_str} / {tf_str}")
                    results.append(future.result())

        # Summarize results
        summary = {
            'total': len(results),
            'success': sum(1 for r in results if r['status'] == 'success'),
            'failed': sum(1 for r in results if r['status'] == 'failed'),
            'skipped': sum(1 for r in results if r['status'] == 'skipped'),
            'details': results
        }

        if show_progress:
            print(f"\nCache population complete:")
            print(f"  Success: {summary['success']}/{summary['total']}")
            print(f"  Failed: {summary['failed']}/{summary['total']}")
            print(f"  Skipped: {summary['skipped']}/{summary['total']}")

            # Show failures
            failures = [r for r in results if r['status'] == 'failed']
            if failures:
                print("\nFailures:")
                for f in failures[:5]:  # Show first 5
                    print(f"  - {f['module_name']}/{f['ticker']}/{f['tf']}: {f.get('message', 'Unknown error')}")
                if len(failures) > 5:
                    print(f"  ... and {len(failures) - 5} more")

        return summary

    def populate_cache_for_vault(
        self,
        vault_ensemble_dir: str,
        start_date: datetime,
        end_date: datetime,
        max_workers: int = 4,
        overwrite_existing: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Populate caches for all bias nodes in a vault ensemble.

        Reads the control file from the vault ensemble directory and
        extracts all required bias node specifications.

        Parameters
        ----------
        vault_ensemble_dir : str
            Path to vault ensemble directory (e.g., 'vault/D/buy_hold_long')
        start_date : datetime
            Start date for cache
        end_date : datetime
            End date for cache
        max_workers : int
            Number of concurrent workers
        overwrite_existing : bool
            Whether to overwrite existing caches
        show_progress : bool
            Whether to print progress

        Returns
        -------
        Dict[str, Any]
            Summary with total, success, failed, skipped counts and details
        """
        from ensemble.ensemble_utils import parse_control_file

        # Find control file
        ensemble_path = Path(vault_ensemble_dir)
        control_file = None

        for pattern in ['control_file*.json', 'ensemble_*.json', '*.json']:
            matches = list(ensemble_path.glob(pattern))
            if matches:
                control_file = matches[0]
                break

        if control_file is None:
            raise FileNotFoundError(
                f"No control file found in {vault_ensemble_dir}"
            )

        if show_progress:
            print(f"Loading control file: {control_file}")

        # Parse control file
        config = parse_control_file(str(control_file))

        # Extract bias node specs from base models
        bias_node_specs = []
        tickers_set = set()

        for model_config in config.get('base_models', []):
            # Get bias node spec
            bias_node_spec = model_config.get('bias_node_spec')
            if bias_node_spec:
                # Convert timeframe strings to TimeFrame enums
                timeframes = bias_node_spec.get('timeframes', ['D'])
                normalized_tfs = []
                for tf in timeframes:
                    if isinstance(tf, str):
                        try:
                            normalized_tfs.append(TimeFrame[tf])
                        except KeyError:
                            logger.warning(f"Unknown timeframe: {tf}")
                    elif isinstance(tf, TimeFrame):
                        normalized_tfs.append(tf)

                bias_node_specs.append({
                    'module_name': bias_node_spec['module_name'],
                    'params': bias_node_spec.get('params', {}),
                    'timeframes': normalized_tfs
                })

            # Get tickers
            model_tickers = model_config.get('tickers', [])
            for ticker in model_tickers:
                if isinstance(ticker, str):
                    try:
                        tickers_set.add(Ticker[ticker])
                    except KeyError:
                        logger.warning(f"Unknown ticker: {ticker}")
                elif isinstance(ticker, Ticker):
                    tickers_set.add(ticker)

        # Also check config-level tickers
        # Note: Config-level tickers may include defaults that don't match our Ticker enum,
        # so we silently ignore invalid entries here (unlike ensemble-level which warns)
        config_tickers = config.get('tickers', [])
        for ticker in config_tickers:
            if isinstance(ticker, str):
                try:
                    tickers_set.add(Ticker[ticker])
                except KeyError:
                    pass  # Silently ignore unknown tickers from config defaults
            elif isinstance(ticker, Ticker):
                tickers_set.add(ticker)

        tickers = list(tickers_set)

        if not bias_node_specs:
            raise ValueError(f"No bias node specs found in {control_file}")

        if not tickers:
            raise ValueError(f"No tickers found in {control_file}")

        if show_progress:
            print(f"Found {len(bias_node_specs)} bias node spec(s)")
            print(f"Found {len(tickers)} ticker(s): {[t.name for t in tickers]}")

        # Populate caches
        return self.populate_cache(
            bias_node_specs=bias_node_specs,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            max_workers=max_workers,
            overwrite_existing=overwrite_existing,
            show_progress=show_progress
        )

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
        from utils.core.helpers import create_bias_node

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
        from .central_cache import CentralCacheStore
        from .central_cache_models import ArtifactLifecycleState

        store = self._central_cache_store()
        record = store.describe_artifact(descriptor)
        if record is None:
            return True, "missing"
        if record.lifecycle_state is not ArtifactLifecycleState.FRESH:
            return True, record.lifecycle_state.value.lower()
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
        from utils.compute.daily_ewsd_volatility import compute_daily_ewsd_volatility
        from utils.core.helpers import create_fresh_bias_node

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
        from utils.core.helpers import create_fresh_bias_node

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

    def _depends_on_for_spec(
        self,
        ticker: Ticker,
        tf: TimeFrame,
        params: Dict[str, Any],
    ) -> tuple[tuple[Ticker, TimeFrame], ...]:
        from .cross_ticker_store import extract_cross_ticker_names

        dependencies: list[tuple[Ticker, TimeFrame]] = [(ticker, tf)]
        for cross_name in sorted(extract_cross_ticker_names(params)):
            if cross_name not in Ticker.__members__:
                logger.warning("Unknown cross ticker in cache refresh params: %s", cross_name)
                continue
            dependency = (Ticker[cross_name], tf)
            if dependency not in dependencies:
                dependencies.append(dependency)
        return tuple(dependencies)

    def _resolved_source_revision(
        self,
        descriptor: "ArtifactDescriptor",
        depends_on: Sequence[tuple[Ticker, TimeFrame]],
    ) -> int:
        """Return a monotonic source revision for artifact rebuilds.

        Candle revision counters are local cache metadata, not source-of-truth market
        data versions. After cache reinitialization it is possible for dependency candle
        revisions to be lower than an already-persisted artifact revision even when the
        underlying candle payload is unchanged. In that case we preserve the higher
        persisted artifact source revision so stale artifacts can still be rebuilt.
        """
        store = self._central_cache_store()
        current_dependency_revision = max(
            (
                record.revision
                for ticker, timeframe in depends_on
                if (record := store.describe_candle(ticker, timeframe)) is not None
            ),
            default=0,
        )
        existing_record = store.describe_artifact(descriptor)
        existing_source_revision = 0 if existing_record is None else existing_record.source_revision
        return max(current_dependency_revision, existing_source_revision)

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
        dependency_tickers: set[Ticker] = set()
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
                    depends_on = self._depends_on_for_spec(ticker, tf, params)
                    dependency_tickers.update(dep_ticker for dep_ticker, _ in depends_on)
                    task_key = (
                        module_name,
                        ticker.name,
                        tf.name,
                        repr(sorted(params.items(), key=lambda item: item[0])),
                    )
                    existing_task = artifact_tasks.get(task_key)
                    task_payload = {
                        "module_name": module_name,
                        "params": params,
                        "ticker": ticker,
                        "tf": tf,
                        "depends_on": depends_on,
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
                    if existing_task is None:
                        artifact_tasks[task_key] = task_payload
                        continue
                    existing_task["depends_on"] = tuple(
                        dict.fromkeys(existing_task["depends_on"] + depends_on)
                    )

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

        source_tickers = sorted(
            set(requested_tickers).union(dependency_tickers),
            key=lambda item: item.name,
        )
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
            "dependency_tickers": [ticker.name for ticker in sorted(dependency_tickers)],
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

    def list_caches(
        self,
        module_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all existing caches.

        Parameters
        ----------
        module_name : str, optional
            Filter by module name

        Returns
        -------
        List[Dict[str, Any]]
            List of cache metadata dicts
        """
        cache_path = Path(self.cache_dir)
        caches = []

        if module_name:
            search_dirs = [cache_path / module_name]
        else:
            search_dirs = [d for d in cache_path.iterdir() if d.is_dir()]

        for module_dir in search_dirs:
            if not module_dir.exists():
                continue

            for parquet_file in module_dir.glob('*.parquet'):
                # Parse filename to extract ticker, tf, params
                filename = parquet_file.stem
                parts = filename.split('_')

                if len(parts) >= 2:
                    ticker = parts[0]
                    tf = parts[1]
                    params_suffix = '_'.join(parts[2:]) if len(parts) > 2 else ''
                else:
                    ticker = filename
                    tf = 'unknown'
                    params_suffix = ''

                # Get file stats
                stats = parquet_file.stat()

                caches.append({
                    'module_name': module_dir.name,
                    'ticker': ticker,
                    'tf': tf,
                    'params_suffix': params_suffix,
                    'path': str(parquet_file),
                    'file_size_mb': stats.st_size / (1024 * 1024),
                    'modified_time': datetime.fromtimestamp(stats.st_mtime)
                })

        return caches

    def clear_caches(
        self,
        module_name: Optional[str] = None,
        ticker: Optional[str] = None,
        confirm: bool = False
    ) -> int:
        """
        Clear cache files.

        Parameters
        ----------
        module_name : str, optional
            Only clear caches for this module
        ticker : str, optional
            Only clear caches for this ticker
        confirm : bool
            Must be True to actually delete files

        Returns
        -------
        int
            Number of files deleted
        """
        if not confirm:
            caches = self.list_caches(module_name)
            if ticker:
                caches = [c for c in caches if c['ticker'] == ticker]
            print(f"Would delete {len(caches)} cache file(s). Pass confirm=True to delete.")
            return 0

        deleted = 0
        cache_path = Path(self.cache_dir)

        if module_name:
            search_dirs = [cache_path / module_name]
        else:
            search_dirs = [d for d in cache_path.iterdir() if d.is_dir()]

        for module_dir in search_dirs:
            if not module_dir.exists():
                continue

            for parquet_file in module_dir.glob('*.parquet'):
                if ticker:
                    filename = parquet_file.stem
                    if not filename.startswith(ticker + '_'):
                        continue

                parquet_file.unlink()
                deleted += 1
                logger.info(f"Deleted cache: {parquet_file}")

        return deleted


def main():
    """CLI entry point for cache manager."""
    parser = argparse.ArgumentParser(
        description='Populate bias node caches for vectorized backtesting'
    )

    parser.add_argument(
        '--vault',
        type=str,
        help='Path to vault ensemble directory (e.g., vault/D/buy_hold_long)'
    )

    parser.add_argument(
        '--module',
        type=str,
        help='Single module name to cache (e.g., rsi)'
    )

    parser.add_argument(
        '--tickers',
        type=str,
        nargs='+',
        default=['ES', 'NQ'],
        help='Tickers to cache (default: ES NQ)'
    )

    parser.add_argument(
        '--timeframes',
        type=str,
        nargs='+',
        default=['D'],
        help='Timeframes to cache (default: D)'
    )

    parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of concurrent workers (default: 4)'
    )

    parser.add_argument(
        '--no-overwrite',
        action='store_true',
        help='Skip existing caches'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List existing caches'
    )

    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear caches (requires --confirm)'
    )

    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Confirm destructive operations'
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    manager = CacheManager()

    # Handle list command
    if args.list:
        caches = manager.list_caches(args.module)
        print(f"\nFound {len(caches)} cache file(s):\n")
        for cache in caches:
            print(f"  {cache['module_name']}/{cache['ticker']}_{cache['tf']}")
            print(f"    Path: {cache['path']}")
            print(f"    Size: {cache['file_size_mb']:.2f} MB")
            print(f"    Modified: {cache['modified_time']}")
        return

    # Handle clear command
    if args.clear:
        deleted = manager.clear_caches(
            module_name=args.module,
            ticker=args.tickers[0] if args.tickers else None,
            confirm=args.confirm
        )
        print(f"Deleted {deleted} cache file(s)")
        return

    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')

    # Handle vault mode
    if args.vault:
        manager.populate_cache_for_vault(
            vault_ensemble_dir=args.vault,
            start_date=start_date,
            end_date=end_date,
            max_workers=args.workers,
            overwrite_existing=not args.no_overwrite
        )
        return

    # Handle single module mode
    if args.module:
        # Parse tickers
        tickers = []
        for ticker_str in args.tickers:
            try:
                tickers.append(Ticker[ticker_str])
            except KeyError:
                print(f"Warning: Unknown ticker '{ticker_str}', skipping")

        # Parse timeframes
        timeframes = []
        for tf_str in args.timeframes:
            try:
                timeframes.append(TimeFrame[tf_str])
            except KeyError:
                print(f"Warning: Unknown timeframe '{tf_str}', skipping")

        if not tickers or not timeframes:
            print("Error: No valid tickers or timeframes specified")
            return

        manager.populate_cache(
            bias_node_specs=[{
                'module_name': args.module,
                'params': {},
                'timeframes': timeframes
            }],
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            max_workers=args.workers,
            overwrite_existing=not args.no_overwrite
        )
        return

    print("Error: Either --vault or --module must be specified")
    parser.print_help()


if __name__ == '__main__':
    main()
