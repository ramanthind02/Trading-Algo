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
    python -m utils.cache_manager --vault vault/D/buy_hold_long --start 2010-01-01 --end 2024-12-31

Author: Trading Research Team
Date: 2025-01-07
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.bias_node_cache import BiasNodeCache
from utils.enums import Ticker, TimeFrame
from utils.models import Candle

logger = logging.getLogger(__name__)

# Required for volatility-scaled targets (log_return_atr, log_return_ewsd).
# extract_features_with_forward_returns always requests these when building targets.
REQUIRED_AUXILIARY_SPECS = [
    {"module_name": "atr", "params": {"period": 252}},
    {"module_name": "ewsd", "params": {}},
]


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
        Root cache directory. Defaults to 'cache' in project root.
    candle_dir : str, optional
        Directory containing candle parquet files.
        Defaults to 'candles' in project root.

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
        # Set default directories
        project_root = Path(__file__).parent.parent

        if cache_dir is None:
            cache_dir = str(project_root / 'cache')
        if candle_dir is None:
            candle_dir = str(project_root / 'candles')

        self.cache_dir = cache_dir
        self.candle_dir = candle_dir

        # Ensure directories exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

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

        # Try different file naming conventions
        possible_paths = [
            Path(self.candle_dir) / f"{ticker_str}_{tf_str}.parquet",
            Path(self.candle_dir) / tf_str / f"{ticker_str}.parquet",
            Path(self.candle_dir) / ticker_str / f"{tf_str}.parquet",
            Path(self.candle_dir) / f"{ticker_str}.parquet",
            # Also check data/ohlc_data format: {ticker}/{tf}_{ticker}.parquet
            Path(self.candle_dir) / ticker_str / f"{tf_str}_{ticker_str}.parquet",
        ]

        candle_path = None
        for path in possible_paths:
            if path.exists():
                candle_path = path
                break

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
        from utils.helpers import create_bias_node

        # Create bias node instance
        bias_node = create_bias_node(module_name, ticker, tf, params)

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
        from utils.helpers import create_bias_node

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
            bias_node = create_bias_node(module_name, ticker, tf, params)
            actual_module_name = getattr(bias_node, 'module_name', module_name)
            actual_params = getattr(bias_node, 'params', params)

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

            # Check if cache exists and skip if not overwriting
            if cache.exists() and not overwrite:
                result['status'] = 'skipped'
                result['message'] = 'Cache exists and overwrite=False'
                return result

            # Load candles
            candles_df = self._load_candles(ticker, tf, start_date, end_date)

            if candles_df.empty:
                result['status'] = 'failed'
                result['message'] = 'No candles found for date range'
                return result

            result['candle_count'] = len(candles_df)

            # Compute bias node output (reuse the bias node we created)
            output_df = self._compute_bias_node_output_from_node(
                bias_node, candles_df
            )

            # Save to cache
            cache.save(output_df)

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
        show_progress: bool = True
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

        # Ensure required auxiliary bias nodes (atr, ewsd) are included for volatility-scaled targets
        specs_to_use = list(bias_node_specs)
        for aux in REQUIRED_AUXILIARY_SPECS:
            if not any(
                _spec_matches(spec, aux["module_name"], aux["params"])
                for spec in bias_node_specs
            ):
                specs_to_use.append({
                    "module_name": aux["module_name"],
                    "params": aux["params"],
                    "timeframes": all_timeframes,
                })
                if show_progress:
                    print(f"  Adding required auxiliary: {aux['module_name']} (for log_return_atr/log_return_ewsd)")

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
            print(f"  Bias node specs: {len(specs_to_use)} (including required atr/ewsd if needed)")
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
        default=4,
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
