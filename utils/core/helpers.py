
import os
import gc
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from utils.core.enums import TimeFrame, Ticker
from datetime import datetime, timezone
from utils.core.logger import get_logger

logger = get_logger(__name__)


def is_dst(dt: datetime) -> bool:
    """Check if datetime is in daylight saving time."""
    from datetime import timedelta
    return dt.dst() != timedelta(0)


def convert_ftmo_time_to_ny_time(timestamp: int) -> datetime:
    """
    Convert FTMO server timestamp to New York time.
    
    Parameters
    ----------
    timestamp : int
        Unix timestamp from FTMO server
        
    Returns
    -------
    datetime
        Datetime in NY timezone
    """
    from zoneinfo import ZoneInfo
    from datetime import timedelta
    
    dt_gmt2 = datetime.fromtimestamp(timestamp, tz=ZoneInfo("UTC"))
    dt_ny_plus_offset = dt_gmt2.astimezone(ZoneInfo("America/New_York"))
    hours_offset = 3 if is_dst(dt_ny_plus_offset) else 2
    dt_ny = dt_ny_plus_offset - timedelta(hours=hours_offset)
    return dt_ny


def convert_ny_time_to_ftmo_time(dt_ny: datetime) -> int:
    """
    Convert New York time to FTMO server time timestamp.
    
    Parameters
    ----------
    dt_ny : datetime
        Datetime in NY timezone
        
    Returns
    -------
    int
        Unix timestamp for FTMO server
    """
    from datetime import timedelta
    
    hours_offset = 3 if is_dst(dt_ny) else 2
    dt_ny = dt_ny + timedelta(hours=hours_offset)
    return int(dt_ny.timestamp())


def get_next_ftmo_midnight_time() -> tuple:
    """
    Get today's and tomorrow's midnight timestamps in FTMO time.
    
    Returns
    -------
    tuple
        (today_midnight_timestamp, tomorrow_midnight_timestamp)
    """
    from zoneinfo import ZoneInfo
    from datetime import timedelta
    
    cet_now = datetime.now(ZoneInfo('Europe/Berlin'))
    today_midnight = cet_now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_midnight = today_midnight.replace(tzinfo=ZoneInfo('UTC'))
    tomorrow_midnight = today_midnight + timedelta(days=1)
    
    return (
        int(today_midnight.timestamp()),
        int(tomorrow_midnight.timestamp())
    )


def load_data(ticker: Ticker, timeframe: TimeFrame, start: datetime = datetime(1990, 1, 1), end: datetime = datetime(2025, 12, 30)) -> pd.DataFrame:
    """Load OHLC data from parquet files using pandas for efficient reading.

    Args:
        ticker: The ticker symbol to load data for.
        timeframe: The timeframe of the OHLC data.
        start: The start datetime to filter from, defaults to Jan 1, 1990.
        end: The end datetime to filter to, defaults to Dec 1, 2025.

    Returns:
        pd.DataFrame: DataFrame containing the OHLC data

    Raises:
        FileNotFoundError: If the parquet file does not exist.
    """
    # Project root: repo root (parent of utils/), not utils/ itself
    _helpers_path = Path(__file__).resolve()
    project_root = next(
        (p for p in _helpers_path.parents if (p / "pyproject.toml").exists()),
        _helpers_path.parents[2],
    )
    base_dir = project_root / "data" / "ohlc_data"
    file_path = base_dir / ticker.name / f"{timeframe.name}_{ticker.name}.parquet"

    if file_path.exists():
        # Read parquet file with pandas (fastparquet engine for PyPy compatibility)
        df = pd.read_parquet(file_path, engine='fastparquet')
        
        # Convert datetime column to proper datetime type
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Filter by date range
        mask = (df['datetime'] >= start) & (df['datetime'] <= end)
        df = df[mask]
        
        # Create proper timestamp index
        df['timestamp'] = df['datetime'].astype('int64') // 10**9
        df.set_index('timestamp', inplace=True)
        
        # Sort by timestamp index
        df.sort_index(inplace=True)
        
        return df
    raise FileNotFoundError(f"File {file_path} does not exist")


def load_data_multi_ticker(
    tickers: List[Ticker],
    timeframe: TimeFrame,
    start: datetime = datetime(1990, 1, 1),
    end: datetime = datetime(2025, 12, 30),
    use_millisecond_offset: bool = False
) -> pd.DataFrame:
    """
    Load OHLC data from multiple tickers and append rows with ticker column.
    
    Parameters
    ----------
    tickers : List[Ticker]
        List of ticker symbols to load data for
    timeframe : TimeFrame
        The timeframe of the OHLC data
    start : datetime, default=datetime(1990, 1, 1)
        Start datetime to filter from
    end : datetime, default=datetime(2025, 12, 30)
        End datetime to filter to
    use_millisecond_offset : bool, default=False
        Deprecated. Ignored. Primary key is (datetime, ticker); no offsets applied.
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all tickers' data. Includes:
        - All OHLC columns (datetime, open, high, low, close, volume)
        - 'ticker' column identifying the ticker for each row
        - 'timeframe' column (same for all rows)
        - Rows keyed by (datetime, ticker); datetime is bar time (no per-ticker offset)
        
    Examples
    --------
    >>> from utils.core.enums import Ticker, TimeFrame
    >>> from datetime import datetime
    >>> 
    >>> # Load multiple tickers
    >>> df = load_data_multi_ticker(
    ...     tickers=[Ticker.ES, Ticker.NQ, Ticker.YM],
    ...     timeframe=TimeFrame.D,
    ...     start=datetime(2020, 1, 1),
    ...     end=datetime(2024, 12, 31)
    ... )
    >>> 
    >>> # DataFrame has ticker column
    >>> print(df['ticker'].unique())  # ['ES', 'NQ', 'YM']
    >>> print(df.columns)  # ['datetime', 'open', 'high', 'low', 'close', 'volume', 'ticker', 'timeframe']
    """
    all_dfs = []
    
    for ticker in tickers:
        # Load data for this ticker
        ticker_df = load_data(ticker, timeframe, start=start, end=end)
        
        # Reset index to get timestamp as column (we'll use datetime as index)
        ticker_df = ticker_df.reset_index()
        
        # Add ticker column (primary key is (datetime, ticker); no millisecond offset)
        ticker_df['ticker'] = ticker
        ticker_df['timeframe'] = timeframe
        all_dfs.append(ticker_df)
    
    # Concatenate all tickers
    combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    
    # Sort by datetime
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
    
    return combined_df
    

def load_numpy_data(ticker: Ticker, timeframe: TimeFrame, start: datetime = datetime(1990, 1, 1, 0, 0, 0), end: datetime = datetime.now()) -> np.ndarray:
    # Load DataFrame
    df = load_data(ticker, timeframe, start, end)

    try:
        # Convert 'datetime' column to Unix timestamps (in seconds)
        df['datetime'] = (df['datetime'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')

        # Pre-allocate numpy array
        data = np.zeros(len(df), dtype=[
            ('open', np.float32),
            ('close', np.float32),
            ('high', np.float32),
            ('low', np.float32),
            ('datetime', np.uint32)
        ])

        # Copy values
        data['open'] = df['open']
        data['close'] = df['close']
        data['high'] = df['high']
        data['low'] = df['low']
        data['datetime'] = df['datetime']

        return data
    
    finally:
        # Explicitly delete DataFrame and force garbage collection
        del df
        gc.collect()


def _normalize_module_base_name(module_name: str) -> str:
    """Normalize incoming module names to a flat base module token."""
    normalized = module_name.replace('.py', '').strip()
    base_module_match = re.match(r'^([a-z_]+)(?:_\d+.*)?$', normalized)
    if base_module_match:
        return base_module_match.group(1)
    return normalized


def _resolve_bias_node_import_path(base_module_name: str) -> str:
    """Resolve module import path using taxonomy map, then recursive search."""
    from nodes._taxonomy import CANONICAL_MODULE_IMPORTS

    canonical_path = CANONICAL_MODULE_IMPORTS.get(base_module_name)
    if canonical_path:
        return canonical_path

    nodes_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "nodes"
    matches = [
        candidate
        for candidate in nodes_root.rglob(f"{base_module_name}.py")
        if candidate.name != "__init__.py" and "archive" not in candidate.parts
    ]

    if not matches:
        raise ValueError(f"Could not find module file recursively for: {base_module_name}")

    if len(matches) > 1:
        candidate_paths = sorted(str(candidate.relative_to(nodes_root)) for candidate in matches)
        raise ValueError(
            f"Ambiguous module resolution for '{base_module_name}'. Candidates: {candidate_paths}"
        )

    relative_module_path = matches[0].relative_to(nodes_root).with_suffix("")
    return f"nodes.{'.'.join(relative_module_path.parts)}"



def _resolve_bias_node_class(module_name: str) -> type[Any]:
    """Resolve the concrete bias-node class for *module_name*."""
    import importlib
    import inspect

    base_module_name = _normalize_module_base_name(module_name)
    full_module_name = _resolve_bias_node_import_path(base_module_name)

    # Import the resolved module path
    try:
        module = importlib.import_module(full_module_name)
    except ImportError as e:
        raise ImportError(f"Error importing module {full_module_name}: {e}")
    
    # Find the main class in the module
    # Strategy: Look for classes that are DEFINED in this module (not imported)
    # This prevents finding imported classes like RSI when we want RSISignalNode
    main_class = None
    for name, obj in inspect.getmembers(module):
        # Skip the abstract BiasNode class
        if name == 'BiasNode':
            continue
        # Only consider classes that are actually defined in this module
        if inspect.isclass(obj) and obj.__module__ == full_module_name:
            if hasattr(obj, 'get_instance') and callable(getattr(obj, 'get_instance')):
                main_class = obj
                break
    
    if main_class is None:
        raise ValueError(f"Could not find a suitable class in module {module_name}")

    return main_class


def _instantiate_bias_node(
    module_name: str,
    ticker: Ticker,
    tf: TimeFrame,
    params: Dict,
    *,
    use_singleton: bool,
) -> Any:
    """Instantiate a bias node, optionally bypassing class singletons."""
    # Special handling for TimeSeriesFeatureNode
    if module_name == 'ts_feature' or module_name == 'tsFeature':
        from nodes.ts_feature import TimeSeriesFeatureNode
        from types import FunctionType

        wrapped_module = params.get('wrapped_module')
        wrapped_params = params.get('wrapped_params', {})
        lookback = params.get('lookback')
        transformation = params.get('transformation')
        transformation_args = params.get('transformation_args', {})

        if not wrapped_module or not lookback or not transformation:
            raise ValueError("TimeSeriesFeatureNode requires 'wrapped_module', 'lookback', and 'transformation' in params")

        wrapped_node = _instantiate_bias_node(
            wrapped_module,
            ticker,
            tf,
            wrapped_params,
            use_singleton=use_singleton,
        )

        if isinstance(transformation, (FunctionType, type(lambda: None))):
            transformation_func = transformation
            transformation_name = getattr(transformation, '__name__', 'transformation')
        elif isinstance(transformation, str):
            transformation_func = _get_functime_function(transformation)
            transformation_name = transformation
        else:
            raise ValueError(f"transformation must be a function or string name, got {type(transformation)}")

        if use_singleton and hasattr(TimeSeriesFeatureNode, 'get_instance'):
            return TimeSeriesFeatureNode.get_instance(
                ticker, tf, wrapped_node, lookback, transformation_func,
                transformation_name, transformation_args
            )
        return TimeSeriesFeatureNode(
            ticker, tf, wrapped_node, lookback, transformation_func,
            transformation_name, transformation_args
        )

    main_class = _resolve_bias_node_class(module_name)

    try:
        if use_singleton and hasattr(main_class, 'get_instance') and callable(getattr(main_class, 'get_instance')):
            return main_class.get_instance(ticker, tf, **params)
        return main_class(ticker, tf, **params)
    except Exception as e:
        raise RuntimeError(f"Error instantiating class from {module_name}: {e}")


def create_bias_node(module_name: str, ticker: Ticker, tf: TimeFrame, params: Dict) -> Any:
    """
    Creates a bias node by directly specifying the module name

    Parameters:
    - module_name (str): Name of the module file (without .py extension) containing the bias node class
    - ticker (Ticker): Ticker symbol
    - tf (TimeFrame): Timeframe
    - params (Dict): Hyperparameters for the bias node

    Returns:
    - BiasNode: The appropriate bias node instance
    """
    return _instantiate_bias_node(
        module_name,
        ticker,
        tf,
        params,
        use_singleton=True,
    )


def create_fresh_bias_node(module_name: str, ticker: Ticker, tf: TimeFrame, params: Dict) -> Any:
    """Create a fresh bias-node instance that bypasses singleton reuse."""
    return _instantiate_bias_node(
        module_name,
        ticker,
        tf,
        params,
        use_singleton=False,
    )


def create_filtered_bias_node(
    module_name: str,
    ticker: 'Ticker',
    tf: 'TimeFrame',
    params: Dict,
    filter_specs: 'Sequence',
    *,
    neutral_value: float = 0.0,
) -> Any:
    """Create a bias node optionally wrapped with a filter chain.

    If *filter_specs* is empty the raw node is returned unchanged.
    Mirrors :func:`create_bias_node` but adds filter support.

    Parameters
    ----------
    module_name, ticker, tf, params
        Forwarded to :func:`create_bias_node`.
    filter_specs
        Zero or more :class:`filters.FilterSpec` instances defining the
        filter chain.
    neutral_value
        Value emitted when a filter blocks (default ``0.0``).
    """
    base_node = create_bias_node(module_name, ticker, tf, params)
    if not filter_specs:
        return base_node
    from filters import create_filter
    from nodes.filtered import FilteredBiasNode
    filters = [create_filter(spec) for spec in filter_specs]
    return FilteredBiasNode.get_instance(
        base_node, tuple(filters), neutral_value=neutral_value,
    )




# ============================================================================
# Standardized Feature Column Naming Utilities
# Format: module_name_feature_name_tf_param1_param1_value_param2_param2_value
# - All snake_case (matching Python file names in nodes/)
# - No ticker in names
# - TimeFrame token uses TimeFrame.name (e.g., D, H1, M15)
# - Params sorted alphabetically by parameter name
# Examples:
#   - rsi_signal_D_lookback_14
#   - ma_diff_signal_D_lookback_50
#   - momentum_signal_W_lookback_20
# ============================================================================

def _to_snake_case(token: str) -> str:
    """Convert lowerCamelCase to snake_case; preserve existing snake_case."""
    if not isinstance(token, str):
        return str(token)
    # If it already contains underscores, assume it's already snake_case
    if '_' in token:
        return token.lower()
    # Convert camelCase to snake_case
    import re
    # Insert underscore before uppercase letters (except at start)
    result = re.sub(r'(?<!^)(?=[A-Z])', '_', token)
    return result.lower()


def _to_camel_case(token: str) -> str:
    """Convert snake_case to camelCase; preserve existing camelCase."""
    if not isinstance(token, str):
        return str(token)
    # If already camelCase (has uppercase after first char and no underscores), return as-is
    if '_' not in token and '-' not in token and any(ch.isupper() for ch in token[1:]):
        return token
    # Convert snake_case or kebab-case to camelCase
    token = token.replace('-', '_')
    parts = [p for p in token.split('_') if p]
    if not parts:
        return ''
    head = parts[0].lower()
    tail = ''.join(p.capitalize() for p in parts[1:])
    return head + tail


def build_feature_column_name(
    module: str,
    feature: str,
    tf: TimeFrame,
    params: Dict[str, Any]
) -> str:
    """
    Build standardized feature column name.
    
    Module and feature names use snake_case, parameter names use camelCase
    to avoid parser confusion with multi-part parameter names.

    Args:
        module: Module name (e.g., 'rsi', 'ma_diff')
        feature: Feature token for this node's output (e.g., 'signal', 'atr_pct')
        tf: TimeFrame enum
        params: Parameter dict; sorted alphabetically by key

    Returns:
        str: module_feature_tf_param1_val1_param2_val2
             (module/feature in snake_case, params in camelCase)
    """
    # Module and feature use snake_case
    module_tok = _to_snake_case(module)
    feature_tok = _to_snake_case(feature)
    name_parts: List[str] = [module_tok, feature_tok, tf.name]
    # Parameter names use camelCase to avoid underscore confusion
    if params:
        for key in sorted(params.keys()):
            name_parts.append(_to_camel_case(str(key)))
            val = params[key]
            if isinstance(val, (list, tuple)):
                name_parts.append("_".join(str(v) for v in val))
            else:
                name_parts.append(str(val))
    return '_'.join(name_parts)


def _get_functime_function(function_name: str):
    """
    Get a functime feature extraction function by name.
    
    This function first tries to import from utils.core.functime (the local functime module),
    then falls back to other possible locations. Users can also import functime
    functions and pass them directly as function objects.
    
    Parameters:
    - function_name: Name of the functime function (e.g., 'mean_abs_change')
    
    Returns:
    - The functime function callable
    
    Note: If functime functions are in a custom module, users should
    import and pass them directly rather than as strings.
    """
    import importlib
    import sys
    
    # First try utils.core.functime (the local functime module)
    # Handle import errors gracefully - functime might have optional dependencies
    try:
        from utils.core import functime
        if hasattr(functime, function_name):
            func = getattr(functime, function_name)
            # Verify it's callable (it's a function, not just an attribute)
            if callable(func):
                return func
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        # Import might fail due to missing optional dependencies (e.g., scipy)
        # But some functions might still be available if imported before the error
        pass
    except Exception as e:
        # Other errors - log but continue trying other methods
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Error importing utils.core.functime: {e}")
    
    # Try importing from functime package or user-provided module
    # Check common locations and also look in sys.modules for already-imported modules
    
    possible_modules = [
        'utils.core.functime',  # Explicitly try again
        'machine_learning.feature_engineering.functime',
        'functime',
        'functime.feature_extractor',
    ]
    
    # Check already-imported modules
    for module_name, module in sys.modules.items():
        if 'functime' in module_name.lower() and hasattr(module, function_name):
            return getattr(module, function_name)
    
    # Then try importing from possible locations
    for module_path in possible_modules:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, function_name):
                return getattr(module, function_name)
        except (ImportError, ModuleNotFoundError):
            continue
    
    # If not found, raise error with helpful message
    raise ValueError(
        f"Could not find functime function '{function_name}'. "
        f"Available functions in utils.core.functime include: mean_abs_change, mean_change, "
        f"autocorrelation, number_crossings, linear_trend, and many more. "
        f"Please ensure the function name is correct, or import the function and pass it "
        f"directly as a function object. "
        f"Tried modules: {possible_modules}"
    )


def _parse_filter_suffix(segment: str) -> Dict[str, Any]:
    """Parse a single filter suffix segment like ``vol_atrPeriod_14_regime_high``.

    Returns ``{'filter_name': 'vol', 'params': {'atrPeriod': 14, 'regime': 'high'}}``.
    """
    tokens = segment.split('_')
    filter_name = tokens[0] if tokens else ''
    params: Dict[str, Any] = {}
    i = 1
    while i + 1 < len(tokens):
        key = _to_camel_case(tokens[i])
        val_raw = tokens[i + 1]
        val: Any = val_raw
        try:
            val = int(val_raw)
        except Exception:
            try:
                val = float(val_raw)
            except Exception:
                pass
        params[key] = val
        i += 2
    return {'filter_name': filter_name, 'params': params}


def parse_feature_column_name(name: str) -> Dict[str, Any]:
    """
    Parse standardized feature column name into components.
    
    Module and feature names use snake_case, parameter names use camelCase.
    Expected format: {module}_{feature}_{tf}_{param}_{value}_{param}_{value}...
    Optionally with filter suffixes: ...{base}__f_{filter}_{params}__f_{filter}_{params}
    
    Example: rsi_signal_D_lookback_14
    Example: cumulative_rsi_signal_D_avgPeriod_2_lookback_2
    Example: rsi_signal_D_lookback_14__f_vol_atrPeriod_14_regime_high
    
    Handles multi-word module names (ma_diff, cumulative_rsi, ts_feature, etc.)
    by checking against known modules first.
    
    Parameter names are in camelCase (e.g., 'avgPeriod' not 'avg_period')
    to avoid parser confusion with underscores.

    Returns dict with keys: { 'module', 'feature', 'tf', 'params', 'filters' }
    - tf is the TimeFrame enum if name matches, else raw string
    - params keys are in camelCase (converted from snake_case if needed)
    - params values are auto-converted to int/float when possible
    - filters is a list of dicts with 'filter_name' and 'params' (empty if no filters)
    """
    if not isinstance(name, str):
        name = str(name)
    
    # Split off filter suffixes (separated by __f_)
    filter_parts: List[Dict[str, Any]] = []
    base_name = name
    if '__f_' in name:
        segments = name.split('__f_')
        base_name = segments[0]
        for seg in segments[1:]:
            filter_parts.append(_parse_filter_suffix(seg))
    name = base_name
    
    # Known multi-word module names (in order of length, longest first to match greedily)
    # All use snake_case to match Python file names
    known_modules = [
        'cumulative_rsi',
        'consec_momentum',
        'ts_feature',
        'ma_diff',
        'simple_ma',
        'momentum',
        'ewmac',
        'ewsd',
        'cmma',
        'rsi',
        'roc',
        'atr',
    ]
    
    # Try to find matching module name at start of feature column name
    module = None
    remainder = name
    for known_module in known_modules:
        if name.startswith(known_module + '_'):
            module = known_module
            remainder = name[len(known_module) + 1:]  # +1 for the underscore
            break
    
    # If no known module matched, fall back to first token
    if module is None:
        tokens = name.split('_')
        if len(tokens) < 3:
            return {'module': None, 'feature': None, 'tf': None, 'params': {}, 'filters': filter_parts}
        module = tokens[0]
        remainder = '_'.join(tokens[1:])
    
    # Parse remainder: feature_tf_param_value_param_value...
    tokens = remainder.split('_')
    if len(tokens) < 2:
        return {'module': module, 'feature': None, 'tf': None, 'params': {}, 'filters': filter_parts}
    
    feature = tokens[0]
    tf_token = tokens[1]
    
    # Try to convert timeframe token to TimeFrame enum
    try:
        tf = TimeFrame[tf_token]
    except Exception:
        tf = tf_token

    # Parse remaining tokens as key-value pairs for params
    # Parameter names are in camelCase, but we need to handle both formats
    params: Dict[str, Any] = {}
    i = 2
    while i + 1 < len(tokens):
        key_raw = tokens[i]
        val_raw = tokens[i + 1]
        
        # Convert parameter name to camelCase if it's in snake_case
        # This handles backward compatibility and ensures consistency
        key = _to_camel_case(key_raw)
        
        # Try to coerce to int, then float
        val: Any = val_raw
        try:
            val = int(val_raw)
        except Exception:
            try:
                val = float(val_raw)
            except Exception:
                val = val_raw
        params[key] = val
        i += 2

    return {
        'module': module,
        'feature': feature,
        'tf': tf,
        'params': params,
        'filters': filter_parts,
    }
def align_candles_with_features(
    candles_df: pd.DataFrame,
    features_df: pd.DataFrame,
    datetime_col: str = 'datetime'
) -> pd.DataFrame:
    """
    Align candles DataFrame with features DataFrame by datetime.
    
    This is a standardized method for ensuring candles and features have
    matching datetime values for proper alignment in BaseModel.fit().
    
    Parameters
    ----------
    candles_df : pd.DataFrame
        Candles DataFrame. Can have datetime as index or column.
        If column, must have 'datetime' column.
    features_df : pd.DataFrame
        Features DataFrame with datetime index (timezone-aware UTC)
    datetime_col : str, default='datetime'
        Name of datetime column if not using index
        
    Returns
    -------
    pd.DataFrame
        Aligned candles DataFrame with:
        - datetime as column (for BaseModel.fit compatibility)
        - Columns: datetime, open, high, low, close, volume, ticker, timeframe
        - Only rows that match features_df.index
    """
    # Make a copy to avoid modifying original
    aligned = candles_df.copy()
    
    # If datetime is a column, temporarily set it as index for alignment
    datetime_is_column = datetime_col in aligned.columns
    if datetime_is_column:
        aligned.set_index(datetime_col, inplace=True)
    
    # Ensure index is datetime type
    if not isinstance(aligned.index, pd.DatetimeIndex):
        aligned.index = pd.to_datetime(aligned.index)
    
    # Ensure timezone-aware (UTC) to match features
    if aligned.index.tz is None:
        aligned.index = aligned.index.tz_localize('UTC')
    else:
        aligned.index = aligned.index.tz_convert('UTC')
    
    # Align with features index (inner join - only matching datetimes)
    aligned = aligned.reindex(features_df.index)
    
    # Drop rows with NaN in required columns
    aligned = aligned.dropna(subset=[col for col in ['open', 'high', 'low', 'close'] if col in aligned.columns], how='all')
    
    # Reset index to get datetime as column (BaseModel.fit expects datetime column)
    aligned = aligned.reset_index()
    if 'index' in aligned.columns:
        aligned.rename(columns={'index': datetime_col}, inplace=True)
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'ticker', 'timeframe']
    for col in required_cols:
        if col not in aligned.columns:
            if col == 'volume':
                aligned[col] = 0.0
            elif col in ['ticker', 'timeframe']:
                # These should be set by caller, but provide defaults
                pass
    
    return aligned


def get_ticker_list() -> List[Ticker]:
    return [
        Ticker.ES,   # CONTINUOUS E-MINI S&P 500 CONTRACT
        Ticker.NQ,   # CONTINUOUS E-MINI NASDAQ 100 CONTRACT
        Ticker.YM,   # CONTINUOUS E-MINI DOW JONES $5 CONTRACT
        Ticker.RTY,  # CONTINUOUS E-MINI RUSSELL 2000 CONTRACT

        # Energy
        Ticker.CL,   # CONTINUOUS CRUDE OIL CONTRACT
        Ticker.NG,   # CONTINUOUS NATURAL GAS CONTRACT
        Ticker.HO,   # CONTINUOUS NEW YORK HARBOR ULSD CONTRACT

        # Metals
        Ticker.GC,   # CONTINUOUS GOLD CONTRACT
        Ticker.HG,   # CONTINUOUS COPPER CONTRACT
        Ticker.SI,   # CONTINUOUS SILVER CONTRACT
        Ticker.PL,   # CONTINUOUS PLATINUM CONTRACT

        # Currencies (FX)
        Ticker.EU,   # CONTINUOUS EURO FX CONTRACT
        Ticker.JY,   # CONTINUOUS JAPANESE YEN CONTRACT
        Ticker.BP,   # CONTINUOUS BRITISH POUND CONTRACT
        Ticker.CD,   # CONTINUOUS CANADIAN DOLLAR CONTRACT
        Ticker.SF,   # CONTINUOUS SWISS FRANC CONTRACT

        # Agricultural (Food Grains)
        Ticker.C,    # CONTINUOUS CORN CONTRACT
        Ticker.S,    # CONTINUOUS SOYBEANS CONTRACT
        Ticker.W,    # CONTINUOUS WHEAT CONTRACT

        Ticker.GF,   # CONTINUOUS FEEDER CATTLE CONTRACT

        # Fixed Income
        Ticker.TY,   # CONTINUOUS 10 YR US TREASURY NOTE CONTRACT
        Ticker.FV,   # CONTINUOUS 5 YR US TREASURY NOTE CONTRACT
        Ticker.US,   # CONTINUOUS 30 YR US TREASURY BOND CONTRACT
        Ticker.TU,   # CONTINUOUS 2 YR US TREASURY NOTE CONTRACT
    ]


def spearman_rho(var1: pd.Series, var2: pd.Series) -> float:
    """
    Compute Spearman Rho correlation coefficient between two pandas Series.
    
    This implementation follows the classical algorithm with tie correction,
    providing accurate correlation values even when ties are present in the data.
    The algorithm:
    1. Ranks both variables independently
    2. Computes tie corrections for each variable
    3. Calculates correlation using the corrected rank differences
    
    Args:
        var1: First pandas Series
        var2: Second pandas Series
        
    Returns:
        float: Spearman Rho correlation coefficient in range [-1, 1]
        
    Raises:
        ValueError: If series have different lengths or contain all NaN values
        
    Example:
        >>> s1 = pd.Series([1, 2, 3, 4, 5])
        >>> s2 = pd.Series([5, 6, 7, 8, 7])
        >>> rho = spearman_rho(s1, s2)
    """
    # Remove NaN values and align the series
    x_vals = var1.values.copy()
    y_vals = var2.values.copy()
    n = len(x_vals)
    
    if n < 2:
        raise ValueError("Need at least 2 valid data points to compute correlation")
    
    # Helper function to compute ranks with tie correction
    def rank_with_tie_correction(arr):
        """
        Rank array and compute tie correction factor.
        Returns ranks and tie correction sum.
        """
        # Sort indices to get ordering
        sorted_indices = np.argsort(arr)
        sorted_arr = arr[sorted_indices]
        
        # Initialize ranks array
        ranks = np.empty(n, dtype=np.float64)
        tie_correction = 0.0
        
        j = 0
        while j < n:
            val = sorted_arr[j]
            # Find all ties
            k = j + 1
            while k < n and sorted_arr[k] == val:
                k += 1
            
            # Number of tied values
            ntied = k - j
            
            # Tie correction: sum of (ties^3 - ties)
            tie_correction += ntied * ntied * ntied - ntied
            
            # Average rank for tied values (1-indexed, so +1)
            rank = 0.5 * (j + k + 1.0)
            
            # Assign average rank to all tied positions
            for idx in range(j, k):
                ranks[sorted_indices[idx]] = rank
            
            j = k
        
        return ranks, tie_correction
    
    # Compute ranks and tie corrections for both variables
    x_ranks, x_tie_correc = rank_with_tie_correction(x_vals)
    y_ranks, y_tie_correc = rank_with_tie_correction(y_vals)
    
    # Final computations
    dn = float(n)
    ssx = (dn * dn * dn - dn - x_tie_correc) / 12.0
    ssy = (dn * dn * dn - dn - y_tie_correc) / 12.0
    
    # Compute squared rank differences
    rank_diff = x_ranks - y_ranks
    rankerr = np.sum(rank_diff * rank_diff)
    
    # Compute Spearman Rho with tie correction
    denominator = np.sqrt(ssx * ssy + 1.0e-20)  # Small epsilon to avoid division by zero
    rho = 0.5 * (ssx + ssy - rankerr) / denominator
    
    return rho
