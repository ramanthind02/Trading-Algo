"""
Bias Node Cache - Persistent cache for bias node outputs.

This module provides a caching layer for bias node outputs to enable
vectorized backtesting. Instead of streaming candles one-by-one through
bias nodes (211k Python function calls per backtest), we can compute
outputs once and cache them as parquet files for instant retrieval.

Cache path structure:
    cache/{module_name}/{ticker}_{tf}_{params_hash}.parquet

Author: Trading Research Team
Date: 2025-01-07
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from utils.core.enums import Ticker, TimeFrame

logger = logging.getLogger(__name__)


class CacheMissError(Exception):
    """
    Raised when cache is required but missing or incomplete.

    This error includes detailed information about what cache was
    expected to help with debugging and cache population.
    """

    def __init__(
        self,
        module_name: str,
        params: Dict[str, Any],
        ticker: Ticker,
        tf: TimeFrame,
        date_range: Optional[tuple] = None,
        cache_path: Optional[str] = None,
        reason: str = "Cache file does not exist"
    ):
        self.module_name = module_name
        self.params = params
        self.ticker = ticker
        self.tf = tf
        self.date_range = date_range
        self.cache_path = cache_path
        self.reason = reason

        # Build informative message
        ticker_str = ticker.name if hasattr(ticker, 'name') else str(ticker)
        tf_str = tf.name if hasattr(tf, 'name') else str(tf)
        params_str = json.dumps(params, sort_keys=True)

        msg = (
            f"Cache miss for {module_name} ({ticker_str}, {tf_str}).\n"
            f"  Params: {params_str}\n"
            f"  Reason: {reason}"
        )

        if date_range:
            msg += f"\n  Date range: {date_range[0]} to {date_range[1]}"

        if cache_path:
            msg += f"\n  Expected path: {cache_path}"

        msg += "\n  Run cache_manager.populate_cache() to generate the cache."

        super().__init__(msg)


class BiasNodeCache:
    """
    Cache for bias node outputs stored as parquet files.

    Provides persistent storage for bias node outputs to enable
    vectorized backtesting. Cache files are organized by:
    - module_name: The bias node type (e.g., 'rsi', 'ewmac')
    - ticker: The instrument ticker
    - tf: The timeframe
    - params: Hyperparameters (hashed for filename)

    Cache path format:
        cache/{module_name}/{ticker}_{tf}_{params_hash}.parquet

    Parquet schema:
        - datetime (index): Candle timestamps
        - value (float64): Primary output value
        - Additional columns for multi-output nodes (e.g., signalBool)

    Parameters
    ----------
    module_name : str
        Name of the bias node module (e.g., 'rsi', 'ewmac')
    params : Dict[str, Any]
        Bias node hyperparameters
    ticker : Ticker
        Instrument ticker
    tf : TimeFrame
        Timeframe for the data
    cache_dir : str, optional
        Root cache directory. Defaults to 'cache' in project root.

    Examples
    --------
    >>> cache = BiasNodeCache('rsi', {'lookback': 14}, Ticker.ES, TimeFrame.D)
    >>> if cache.exists():
    ...     data = cache.load()
    ...     values = cache.get_values(start_date, end_date)
    ... else:
    ...     # Compute and save
    ...     cache.save(computed_data)
    """

    def __init__(
        self,
        module_name: str,
        params: Dict[str, Any],
        ticker: Ticker,
        tf: TimeFrame,
        cache_dir: Optional[str] = None
    ):
        self.module_name = module_name
        self.params = params
        self.ticker = ticker
        self.tf = tf

        # Normalize ticker and timeframe to strings
        self.ticker_str = ticker.name if hasattr(ticker, 'name') else str(ticker)
        self.tf_str = tf.name if hasattr(tf, 'name') else str(tf)

        # Set cache directory (default to 'cache' in project root)
        if cache_dir is None:
            # Get project root (parent of utils directory)
            project_root = Path(__file__).parent.parent
            cache_dir = str(project_root / 'cache')
        self.cache_dir = cache_dir

        # Build cache path
        self._cache_path = self._build_cache_path()

        # Cached data (loaded lazily)
        self._data: Optional[pd.DataFrame] = None
        # Warning suppression for partial coverage
        self._warned_partial_start: bool = False
        self._warned_partial_end: bool = False

    def _warn_partial_coverage(
        self,
        cache_start: pd.Timestamp,
        cache_end: pd.Timestamp,
        start: Optional[datetime],
        end: Optional[datetime]
    ) -> None:
        if start is not None and not self._warned_partial_start:
            if cache_start > start + pd.Timedelta(days=1):
                logger.warning(
                    f"Partial cache coverage for {self.module_name} ({self.ticker_str}, {self.tf_str}). "
                    f"Cache starts at {cache_start}, requested {start}. "
                    f"Proceeding with available data."
                )
                self._warned_partial_start = True
        
        if end is not None and not self._warned_partial_end:
            if cache_end < end - pd.Timedelta(days=1):
                logger.warning(
                    f"Partial cache coverage for {self.module_name} ({self.ticker_str}, {self.tf_str}). "
                    f"Cache ends at {cache_end}, requested {end}. "
                    f"Proceeding with available data."
                )
                self._warned_partial_end = True

    @staticmethod
    def _hash_params(params: Dict[str, Any]) -> str:
        """
        Generate a deterministic hash for parameters.

        Uses JSON serialization with sorted keys to ensure
        consistent ordering, then SHA256 hash truncated to 8 chars.

        Parameters
        ----------
        params : Dict[str, Any]
            Hyperparameters to hash

        Returns
        -------
        str
            8-character hash string
        """
        if not params:
            return "default"

        # Serialize params to JSON with sorted keys for deterministic ordering
        params_str = json.dumps(params, sort_keys=True, default=str)

        # SHA256 hash, truncated to 8 characters
        hash_obj = hashlib.sha256(params_str.encode())
        return hash_obj.hexdigest()[:8]

    def _build_params_suffix(self) -> str:
        """
        Build a human-readable params suffix for the filename.

        For simple params (few keys, short values), creates a readable
        suffix like 'lookback_14'. For complex params, falls back to hash.

        Returns
        -------
        str
            Params suffix for filename
        """
        if not self.params:
            return ""

        # For simple params, create readable suffix
        # e.g., {'lookback': 14} -> 'lookback_14'
        # e.g., {'spanFast': 16, 'spanSlow': 64} -> 'spanFast_16_spanSlow_64'
        parts = []
        for key, value in sorted(self.params.items()):
            # Only include if key and value are simple
            if isinstance(value, (int, float, str, bool)):
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 10:
                    value_str = value_str[:10]
                parts.append(f"{key}_{value_str}")

        # If params are too complex or all values are non-simple types
        # (lists, dicts), fall back to hash to avoid filename collisions
        suffix = "_".join(parts)
        if len(suffix) > 60 or (not parts and self.params):
            # Hash when: suffix is too long OR params exist but none were simple
            suffix = self._hash_params(self.params)

        return suffix

    def _build_cache_path(self) -> Path:
        """
        Build the full cache file path.

        Format: cache/{module_name}/{ticker}_{tf}_{params_suffix}.parquet

        Returns
        -------
        Path
            Full path to cache file
        """
        # Build filename
        params_suffix = self._build_params_suffix()
        if params_suffix:
            filename = f"{self.ticker_str}_{self.tf_str}_{params_suffix}.parquet"
        else:
            filename = f"{self.ticker_str}_{self.tf_str}.parquet"

        # Build full path
        cache_path = Path(self.cache_dir) / self.module_name / filename

        return cache_path

    @property
    def cache_path(self) -> str:
        """Get the cache file path as string."""
        return str(self._cache_path)

    def exists(self) -> bool:
        """
        Check if cache file exists.

        Returns
        -------
        bool
            True if cache file exists
        """
        return self._cache_path.exists()

    def load(self) -> pd.DataFrame:
        """
        Load cached data from parquet file.

        Returns
        -------
        pd.DataFrame
            Cached data with datetime index

        Raises
        ------
        CacheMissError
            If cache file does not exist
        FileNotFoundError
            If cache file cannot be read
        """
        if not self.exists():
            raise CacheMissError(
                module_name=self.module_name,
                params=self.params,
                ticker=self.ticker,
                tf=self.tf,
                cache_path=self.cache_path,
                reason="Cache file does not exist"
            )

        try:
            self._data = pd.read_parquet(self._cache_path)

            # Ensure datetime index
            if 'datetime' in self._data.columns:
                self._data = self._data.set_index('datetime')

            # Ensure index is datetime type
            if not isinstance(self._data.index, pd.DatetimeIndex):
                self._data.index = pd.to_datetime(self._data.index)

            logger.debug(
                f"Loaded cache for {self.module_name} ({self.ticker_str}, {self.tf_str}): "
                f"{len(self._data)} rows, {self._data.index.min()} to {self._data.index.max()}"
            )

            return self._data

        except Exception as e:
            raise FileNotFoundError(
                f"Failed to read cache file {self.cache_path}: {e}"
            )

    def save(self, data: Union[pd.Series, pd.DataFrame]) -> None:
        """
        Save data to cache as parquet file.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Data to cache. If Series, will be converted to DataFrame
            with column name 'value'. Must have datetime index.

        Raises
        ------
        ValueError
            If data does not have datetime index
        """
        # Convert Series to DataFrame
        if isinstance(data, pd.Series):
            df = data.to_frame(name='value')
        else:
            df = data.copy()

        # Validate datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                raise ValueError(
                    "Cache data must have datetime index. "
                    f"Got index type: {type(df.index)}"
                )

        # Ensure index is named 'datetime' for parquet
        df.index.name = 'datetime'

        # Create directory if needed
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to parquet
        df.to_parquet(self._cache_path, index=True)

        # Update cached data
        self._data = df

        logger.info(
            f"Saved cache for {self.module_name} ({self.ticker_str}, {self.tf_str}): "
            f"{len(df)} rows, {df.index.min()} to {df.index.max()}"
        )

    def get_values(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        require_cache: bool = True
    ) -> pd.Series:
        """
        Get cached values for a date range.

        This is the primary API for vectorized access to cached data.
        Returns only the primary 'value' column as a Series.

        Parameters
        ----------
        start : datetime, optional
            Start of date range (inclusive). If None, uses earliest available.
        end : datetime, optional
            End of date range (inclusive). If None, uses latest available.
        require_cache : bool, default=True
            If True, raises CacheMissError if cache is missing or incomplete.
            If False, returns None when cache is missing.

        Returns
        -------
        pd.Series or None
            Cached values for the date range, or None if cache missing
            and require_cache=False.

        Raises
        ------
        CacheMissError
            If require_cache=True and cache is missing or incomplete
        """
        # Load data if not already loaded
        if self._data is None:
            if not self.exists():
                if require_cache:
                    raise CacheMissError(
                        module_name=self.module_name,
                        params=self.params,
                        ticker=self.ticker,
                        tf=self.tf,
                        date_range=(start, end),
                        cache_path=self.cache_path,
                        reason="Cache file does not exist"
                    )
                else:
                    logger.warning(
                        f"Cache miss for {self.module_name} ({self.ticker_str}, {self.tf_str}). "
                        f"Returning None."
                    )
                    return None

            self.load()

        # Get the primary value column
        if 'value' in self._data.columns:
            values = self._data['value']
        elif len(self._data.columns) == 1:
            values = self._data.iloc[:, 0]
        else:
            # For multi-output nodes, use first column
            values = self._data.iloc[:, 0]

        # Filter by date range
        if start is not None:
            start = pd.to_datetime(start)
            values = values[values.index >= start]

        if end is not None:
            end = pd.to_datetime(end)
            values = values[values.index <= end]

        # Check for incomplete cache (requested range not fully covered)
        if require_cache and start is not None and end is not None:
            if values.empty:
                raise CacheMissError(
                    module_name=self.module_name,
                    params=self.params,
                    ticker=self.ticker,
                    tf=self.tf,
                    date_range=(start, end),
                    cache_path=self.cache_path,
                    reason=f"No cached data in range {start} to {end}"
                )
            
            # Check if range is fully covered (allow 1 day tolerance for edge cases)
            cache_start = values.index.min()
            cache_end = values.index.max()
            
            self._warn_partial_coverage(cache_start, cache_end, start, end)

        return values

    def get_dataframe(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        require_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get cached data as DataFrame (for multi-output nodes).

        Similar to get_values() but returns all columns for nodes
        with multiple outputs (e.g., EWMAC with signal and signalBool).

        Parameters
        ----------
        start : datetime, optional
            Start of date range (inclusive)
        end : datetime, optional
            End of date range (inclusive)
        require_cache : bool, default=True
            If True, raises CacheMissError if cache is missing

        Returns
        -------
        pd.DataFrame or None
            All cached columns for the date range
        """
        # Load data if not already loaded
        if self._data is None:
            if not self.exists():
                if require_cache:
                    raise CacheMissError(
                        module_name=self.module_name,
                        params=self.params,
                        ticker=self.ticker,
                        tf=self.tf,
                        date_range=(start, end),
                        cache_path=self.cache_path,
                        reason="Cache file does not exist"
                    )
                else:
                    return None

            self.load()

        df = self._data.copy()

        # Filter by date range
        if start is not None:
            start = pd.to_datetime(start)
            df = df[df.index >= start]
        
        if end is not None:
            end = pd.to_datetime(end)
            df = df[df.index <= end]
        
        if require_cache and start is not None and end is not None and not df.empty:
            cache_start = df.index.min()
            cache_end = df.index.max()
            self._warn_partial_coverage(cache_start, cache_end, start, end)
        
        return df

    def invalidate(self) -> bool:
        """
        Delete the cache file.

        Returns
        -------
        bool
            True if file was deleted, False if it didn't exist
        """
        if self.exists():
            self._cache_path.unlink()
            self._data = None
            logger.info(f"Invalidated cache: {self.cache_path}")
            return True
        return False

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get cache metadata without loading full data.

        Returns
        -------
        Dict[str, Any]
            Metadata including file size, date range, row count
        """
        if not self.exists():
            return {
                'exists': False,
                'path': self.cache_path
            }

        # Get file stats
        file_size = self._cache_path.stat().st_size
        modified_time = datetime.fromtimestamp(self._cache_path.stat().st_mtime)

        # Load data to get date range (if not already loaded)
        if self._data is None:
            self.load()

        return {
            'exists': True,
            'path': self.cache_path,
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'modified_time': modified_time,
            'row_count': len(self._data),
            'date_range_start': self._data.index.min(),
            'date_range_end': self._data.index.max(),
            'columns': list(self._data.columns)
        }

    def __repr__(self) -> str:
        """String representation."""
        exists_str = "exists" if self.exists() else "missing"
        return (
            f"BiasNodeCache({self.module_name}, {self.ticker_str}, "
            f"{self.tf_str}, {exists_str})"
        )
