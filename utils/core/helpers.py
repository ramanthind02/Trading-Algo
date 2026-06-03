
import os
import gc
from enum import Enum
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from utils.core.enums import TimeFrame, Ticker
from datetime import datetime, timezone
from utils.core.logger import get_logger

logger = get_logger(__name__)

# In-process cache for OHLC parquet reads (same ticker, timeframe, range, file).
# Feature research and permutation call ``load_data`` / ``load_data_multi_ticker`` many
# times per run with identical arguments; the bias-node parquet cache is separate.
_LOAD_DATA_CACHE: dict[tuple[str, int, int, int], pd.DataFrame] = {}


def _load_data_cache_key(
    file_path: Path,
    start: datetime,
    end: datetime,
    *,
    mtime_ns: int,
) -> tuple[str, int, int, int]:
    resolved = str(file_path.resolve())
    return (
        resolved,
        int(pd.Timestamp(start).value),
        int(pd.Timestamp(end).value),
        mtime_ns,
    )


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
        mtime_ns = file_path.stat().st_mtime_ns
        cache_key = _load_data_cache_key(file_path, start, end, mtime_ns=mtime_ns)
        cached = _LOAD_DATA_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy()

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

        _LOAD_DATA_CACHE[cache_key] = df.copy()
        return df.copy()
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


def _normalize_module_base_name(module_name: str) -> str:
    """Normalize incoming module names to a flat base module token."""
    normalized = module_name.replace('.py', '').strip()
    base_module_match = re.match(r'^([a-z_]+)(?:_\d+.*)?$', normalized)
    if base_module_match:
        return base_module_match.group(1)
    return normalized


def _resolve_bias_node_import_path(base_module_name: str) -> str:
    """Resolve module import path using the explicit taxonomy registry."""
    from nodes._taxonomy import CANONICAL_MODULE_IMPORTS

    canonical_path = CANONICAL_MODULE_IMPORTS.get(base_module_name)
    if canonical_path:
        return canonical_path

    known_modules = ", ".join(sorted(CANONICAL_MODULE_IMPORTS))
    raise ValueError(
        f"Bias node '{base_module_name}' is not registered in nodes._taxonomy.CANONICAL_MODULE_IMPORTS. "
        f"Known modules: {known_modules}"
    )



def _resolve_bias_node_class(module_name: str) -> type[Any]:
    """Resolve the concrete bias-node class for *module_name*."""
    import importlib

    base_module_name = _normalize_module_base_name(module_name)
    full_module_name = _resolve_bias_node_import_path(base_module_name)
    from nodes._taxonomy import CANONICAL_MODULE_CLASSES

    class_name = CANONICAL_MODULE_CLASSES.get(base_module_name)
    if class_name is None:
        known_modules = ", ".join(sorted(CANONICAL_MODULE_CLASSES))
        raise ValueError(
            f"Bias node '{base_module_name}' does not have a registered class in "
            f"nodes._taxonomy.CANONICAL_MODULE_CLASSES. Known modules: {known_modules}"
        )

    # Import the resolved module path
    try:
        module = importlib.import_module(full_module_name)
    except ImportError as e:
        raise ImportError(f"Error importing module {full_module_name}: {e}")

    main_class = getattr(module, class_name, None)
    if main_class is None:
        raise ValueError(
            f"Could not find registered class '{class_name}' in module {full_module_name}"
        )

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


def _format_param_value_for_feature_name(val: object) -> str:
    """Flatten param values for column/file stem names (no nested repr / OS-invalid chars)."""
    if isinstance(val, Enum):
        return str(val.value)
    if isinstance(val, dict):
        parts: List[str] = []
        for k in sorted(val.keys()):
            parts.append(_to_camel_case(str(k)))
            parts.append(_format_param_value_for_feature_name(val[k]))
        return "_".join(parts)
    if isinstance(val, (list, tuple)):
        return "_".join(_format_param_value_for_feature_name(v) for v in val)
    return str(val)


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
            name_parts.append(_format_param_value_for_feature_name(val))
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
        'envelope_reversion_signal',
        'casey_percent_c_signal',
        'donchian_breakout_signal',
        'rebalancing_flow',
        'percent_b_signal',
        'cumulative_rsi_signal',
        'zscore_rsi_signal',
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
