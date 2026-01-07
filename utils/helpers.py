
import os
import gc
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from utils.enums import TimeFrame, Ticker
from datetime import datetime, timezone
from utils.logger import get_logger

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
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    base_dir = os.path.join(root_dir, 'data', 'ohlc_data')
    file_path = os.path.join(base_dir, ticker.name, f"{timeframe.name}_{ticker.name}.parquet")

    if os.path.exists(file_path):
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
    else:
        raise FileNotFoundError(f"File {file_path} does not exist")
    

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
    import importlib
    import os
    import inspect
    import re
    from pathlib import Path
    
    # Special handling for TimeSeriesFeatureNode
    if module_name == 'ts_feature' or module_name == 'tsFeature':
        from nodes.ts_feature import TimeSeriesFeatureNode
        from types import FunctionType
        
        # Extract wrapped node parameters
        wrapped_module = params.get('wrapped_module')
        wrapped_params = params.get('wrapped_params', {})
        lookback = params.get('lookback')
        transformation = params.get('transformation')  # Can be function or string name
        transformation_args = params.get('transformation_args', {})
        
        if not wrapped_module or not lookback or not transformation:
            raise ValueError("TimeSeriesFeatureNode requires 'wrapped_module', 'lookback', and 'transformation' in params")
        
        # Create wrapped node
        wrapped_node = create_bias_node(wrapped_module, ticker, tf, wrapped_params)
        
        # Get transformation function and name
        if isinstance(transformation, (FunctionType, type(lambda: None))):
            # Transformation is already a function
            transformation_func = transformation
            transformation_name = getattr(transformation, '__name__', 'transformation')
        elif isinstance(transformation, str):
            # Transformation is a string name - look it up
            transformation_func = _get_functime_function(transformation)
            transformation_name = transformation
        else:
            raise ValueError(f"transformation must be a function or string name, got {type(transformation)}")
        
        # Create TimeSeriesFeatureNode
        if hasattr(TimeSeriesFeatureNode, 'get_instance'):
            return TimeSeriesFeatureNode.get_instance(
                ticker, tf, wrapped_node, lookback, transformation_func, 
                transformation_name, transformation_args
            )
        else:
            return TimeSeriesFeatureNode(
                ticker, tf, wrapped_node, lookback, transformation_func,
                transformation_name, transformation_args
            )
    
    # Extract the base module name (e.g., 'ma_diff' from 'ma_diff_50_D')
    # This pattern matches the base module name before any underscore followed by numbers
    base_module_match = re.match(r'^([a-z_]+)(?:_\d+.*)?$', module_name)
    if base_module_match:
        base_module_name = base_module_match.group(1)
    else:
        base_module_name = module_name
    
    # Make sure we're using the module name without extension
    base_module_name = base_module_name.replace('.py', '')
    
    # Try to find the module in the nodes directory
    base_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "nodes"
    potential_file = base_path / f"{base_module_name}.py"
    
    if not potential_file.exists():
        raise ValueError(f"Could not find module file: {potential_file}")
    
    # Store the original module name for later use
    original_module_name = module_name
    
    # Import the module using the base module name
    full_module_name = f"nodes.{base_module_name}"
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
    
    # Create and return an instance of the bias node
    try:
        if hasattr(main_class, 'get_instance') and callable(getattr(main_class, 'get_instance')):
            return main_class.get_instance(ticker, tf, **params)
        else:
            # Fall back to direct instantiation if get_instance is not available
            return main_class(ticker, tf, **params)
    except Exception as e:
        raise RuntimeError(f"Error instantiating class from {module_name}: {e}")


def get_bias_nodes(
    ticker: Ticker,
    bias_node_specs: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Tuple[List[TimeFrame], Dict]]:
    """
    Creates a dictionary of bias strategies in the format expected by MLManager.
    
    Instead of hardcoding bias nodes, this function now accepts a list of bias node
    specifications and formats them for MLManager.
    
    Parameters:
    - ticker (Ticker): The ticker symbol to create bias nodes for
    - bias_node_specs (Optional[List[Dict[str, Any]]]): List of bias node specifications.
        Each spec should be a dict with:
        - 'module_name' (str): Name of the bias node module (e.g., 'rsi', 'atr')
        - 'timeframes' (List[TimeFrame]): List of timeframes to use
        - 'params' (Dict): Parameters for the bias node
        - 'strategy_key' (Optional[str]): Custom key, auto-generated if not provided
        
        If None, returns default bias nodes for backwards compatibility.
    
    Returns:
    - Dict[str, Tuple[List[TimeFrame], Dict]]: A dictionary mapping strategy names to their timeframes and parameters
    
    Examples:
    --------
    >>> # Single RSI feature with lookback=14
    >>> specs = [{
    ...     'module_name': 'rsi',
    ...     'timeframes': [TimeFrame.D],
    ...     'params': {'lookback': 14},
    ...     'strategy_key': 'rsi_14'
    ... }]
    >>> bias_nodes = get_bias_nodes(Ticker.SPY, bias_node_specs=specs)
    
    >>> # Multiple ATR features with different periods
    >>> specs = [
    ...     {'module_name': 'atr', 'timeframes': [TimeFrame.D], 'params': {'period': 20}},
    ...     {'module_name': 'atr', 'timeframes': [TimeFrame.D], 'params': {'period': 50}},
    ... ]
    >>> bias_nodes = get_bias_nodes(Ticker.SPY, bias_node_specs=specs)
    """
    bias_strategies = {}
    
    # If no specs provided, return empty dict (fully dynamic - no defaults)
    if bias_node_specs is None:
        return bias_strategies
    
    # Always ensure ATR 252 is included for target normalization
    # Check if ATR is already in the specs
    has_atr_252 = False
    for spec in bias_node_specs:
        if spec.get('module_name') == 'atr' and spec.get('params', {}).get('period') == 252:
            has_atr_252 = True
            break
    
    # Add ATR 252 if not present
    if not has_atr_252:
        atr_spec = {
            'module_name': 'atr',
            'timeframes': [TimeFrame.D],
            'params': {'period': 252},
            'strategy_key': 'atr_252'
        }
        # Add ATR at the beginning so it's available for other features
        bias_node_specs = [atr_spec] + list(bias_node_specs)
    
    # Always ensure EWSD is included for volatility estimation (Robert Carver methodology)
    # Check if EWSD is already in the specs
    has_ewsd = False
    for spec in bias_node_specs:
        if spec.get('module_name') == 'ewsd':
            has_ewsd = True
            break
    
    # Add EWSD if not present
    if not has_ewsd:
        ewsd_spec = {
            'module_name': 'ewsd',
            'timeframes': [TimeFrame.D],
            'params': {
                'lambda_short': 0.06061,      # 32-day span (Carver's preferred)
                'long_run_window': 252,        # 1 year for long-run estimate
                'blend_short_weight': 0.7,     # 70% short-run
                'blend_long_weight': 0.3       # 30% long-run
            },
            'strategy_key': 'ewsd_252'
        }
        # Add EWSD at the beginning alongside ATR
        bias_node_specs = [ewsd_spec] + list(bias_node_specs)
    
    # Process each bias node specification
    for spec in bias_node_specs:
        module_name = spec.get('module_name')
        timeframes = spec.get('timeframes', [TimeFrame.D])
        params = spec.get('params', {}).copy()  # Make a copy to avoid mutating original
        
        # Ensure module_name is in params for create_bias_node to access
        # This is especially important for ts_feature which needs to know the actual module name
        if 'module_name' not in params:
            params['_module_name'] = module_name
        
        # Generate strategy key if not provided
        if 'strategy_key' in spec:
            strategy_key = str(spec['strategy_key'])  # Ensure it's always a string
        else:
            # Auto-generate key from module name and params
            strategy_key = module_name
            if params:
                # For ts_feature, create a simpler key that doesn't include nested dicts
                if module_name == 'ts_feature' or module_name == 'tsFeature':
                    # For ts_feature, use wrapped_module + transformation + lookback for key
                    wrapped_mod = params.get('wrapped_module', 'unknown')
                    transform = params.get('transformation', 'unknown')
                    lookback = params.get('lookback', 'unknown')
                    # Ensure all values are strings (handle function objects, etc.)
                    wrapped_mod = str(wrapped_mod) if wrapped_mod != 'unknown' else 'unknown'
                    # If transformation is a function, get its name
                    if callable(transform):
                        transform = getattr(transform, '__name__', str(transform))
                    else:
                        transform = str(transform) if transform != 'unknown' else 'unknown'
                    lookback = str(lookback) if lookback != 'unknown' else 'unknown'
                    strategy_key = f"{module_name}_{wrapped_mod}_{transform}_{lookback}"
                else:
                    # For other modules, add param values to key for uniqueness
                    # Filter out complex types (dicts, lists) from key generation
                    simple_params = {k: v for k, v in params.items() 
                                   if not isinstance(v, (dict, list)) and k != '_module_name'}
                    if simple_params:
                        # Convert all values to strings and sort for consistent key generation
                        param_values = [str(v) for v in simple_params.values()]
                        param_str = '_'.join(sorted(param_values))
                        strategy_key = f"{module_name}_{param_str}"
        
        # Ensure strategy_key is always a string (safety check)
        strategy_key = str(strategy_key) if strategy_key else str(module_name)
        
        bias_strategies[strategy_key] = (timeframes, params)
    
    return bias_strategies


def create_ml_manager(
    ticker: Ticker, 
    base_tf: TimeFrame = TimeFrame.D, 
    build_matrix: bool = True,
    feature_filter: List[str] = None,
    bias_node_specs: Optional[List[Dict[str, Any]]] = None
):
    """
    Creates an ML Manager instance with bias nodes configured.
    
    This function instantiates an MLManager with bias nodes from the get_bias_nodes function.
    Optionally filters to only include nodes needed for specified features for performance.
    
    Parameters:
    - ticker (Ticker): The ticker symbol to create the ML manager for
    - base_tf (TimeFrame): Base timeframe for the ML model, defaults to daily
    - build_matrix (bool): Whether to construct a matrix for training, defaults to False
    - feature_filter (List[str]): Optional list of feature names to filter bias strategies.
                                  If provided, only bias strategies needed for these features
                                  will be included. This significantly improves performance
                                  when only testing a subset of features.
    - bias_node_specs (Optional[List[Dict[str, Any]]]): List of bias node specifications.
                                  If provided, these specific bias nodes will be used.
                                  If None, default bias nodes will be used.
                                  See get_bias_nodes() for specification format.
    
    Returns:
    - MLManager: An instantiated ML manager with bias nodes configured
    """
    from feature_extraction.ml_manager import MLManager
    
    # Get bias strategies dictionary using our get_bias_nodes function
    bias_strategies = get_bias_nodes(ticker, bias_node_specs=bias_node_specs)
    
    # Filter bias strategies if feature_filter is provided
    if feature_filter is not None and len(feature_filter) > 0:
        filtered_strategies = {}
        
        # Extract unique strategy prefixes from feature names
        # Feature names typically follow pattern: {module_name}_{params}_{tf}_{output_name}
        # Strategy keys in bias_strategies are like: "atr_252", "ma_diff_50", "rsi_14", etc.
        needed_strategies = set()
        
        for feature_name in feature_filter:
            # Try to match feature name to strategy keys
            # Feature names can be complex, e.g., "atr_252_D_atr_pct_252"
            # Strategy key would be "atr_252"
            
            # Try exact match first
            if feature_name in bias_strategies:
                needed_strategies.add(feature_name)
                continue
            
            # Try to find the strategy key that this feature belongs to
            # by checking if the strategy_key appears at the start of the feature_name
            for strategy_key in bias_strategies.keys():
                # Check if feature starts with strategy key followed by underscore or end
                # This handles cases like:
                # - "atr_252" matches "atr_252_D_atr_pct_252"
                # - "ma_diff_50" matches "ma_diff_50_D_ma_diff_50_252"
                if feature_name.startswith(strategy_key + '_') or feature_name == strategy_key:
                    needed_strategies.add(strategy_key)
                    break
        
        # Keep only the needed strategies
        for strategy_key in needed_strategies:
            if strategy_key in bias_strategies:
                filtered_strategies[strategy_key] = bias_strategies[strategy_key]
        
        # Use filtered strategies if we found any matches, otherwise use all
        # (better to compute too much than too little)
        if len(filtered_strategies) > 0:
            bias_strategies = filtered_strategies
        else:
            # No matches found - log warning but use all strategies to be safe
            print(f"Warning: Could not match any of {len(feature_filter)} features to bias strategies, using all strategies")
    
    # Create the ML Manager
    ml_manager = MLManager(
        bias_strategies=bias_strategies,
        ticker=ticker,
        base_tf=base_tf,
        build_matrix=build_matrix
    )
    
    return ml_manager


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
            name_parts.append(str(params[key]))
    return '_'.join(name_parts)


def _get_functime_function(function_name: str):
    """
    Get a functime feature extraction function by name.
    
    This function first tries to import from utils.functime (the local functime module),
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
    
    # First try utils.functime (the local functime module)
    # Handle import errors gracefully - functime might have optional dependencies
    try:
        from utils import functime
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
        logger.debug(f"Error importing utils.functime: {e}")
    
    # Try importing from functime package or user-provided module
    # Check common locations and also look in sys.modules for already-imported modules
    
    possible_modules = [
        'utils.functime',  # Explicitly try again
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
        f"Available functions in utils.functime include: mean_abs_change, mean_change, "
        f"autocorrelation, number_crossings, linear_trend, and many more. "
        f"Please ensure the function name is correct, or import the function and pass it "
        f"directly as a function object. "
        f"Tried modules: {possible_modules}"
    )


def compute_ts_features_from_ml_manager(ml_manager) -> pd.DataFrame:
    """
    Post-process matrix_df to compute time series features for TimeSeriesFeatureNode instances.
    
    This function should be called after backtest completes but before matrix_df is accessed.
    It identifies TimeSeriesFeatureNode instances, computes their features, and updates the matrix.
    
    Parameters:
    - ml_manager: MLManager instance with bias nodes
    
    Returns:
    - Updated DataFrame with computed TS features
    """
    from nodes.ts_feature import TimeSeriesFeatureNode
    
    # Ensure buffer is flushed
    ml_manager._flush_matrix_buffer()
    
    # Get the current matrix
    matrix_df = ml_manager.matrix.copy()
    
    if len(matrix_df) == 0:
        return matrix_df
    
    # Find all TimeSeriesFeatureNode instances and map to column indices
    ts_node_column_map = {}  # Maps column name -> (ts_node, base_column_index)
    
    column_idx = 0
    for idx, (tf, bias_node) in enumerate(ml_manager.bias_nodes):
        if isinstance(bias_node, TimeSeriesFeatureNode):
            # Get the column names that this TS node should produce
            ts_output_names = bias_node.get_column_names()
            
            # Get base column names from wrapped node (for reference)
            try:
                base_names = bias_node.wrapped_node.get_column_names()
            except:
                base_names = getattr(bias_node.wrapped_node, 'columns', [])
            
            # Map each TS output column name to the TS node
            for i, ts_output_name in enumerate(ts_output_names):
                # Find the corresponding column in ml_manager.columns
                if column_idx < len(ml_manager.columns):
                    current_col_name = ml_manager.columns[column_idx]
                    ts_node_column_map[current_col_name] = (bias_node, i)
                    column_idx += 1
        else:
            # Regular node - skip its columns
            try:
                col_names = bias_node.get_column_names() if hasattr(bias_node, 'get_column_names') else getattr(bias_node, 'columns', [])
                column_idx += len(col_names) if col_names else 1
            except:
                column_idx += 1
    
    if not ts_node_column_map:
        # No TS feature nodes, return matrix as-is
        return matrix_df
    
    # Group columns by TS node for efficient processing
    nodes_to_process = {}
    for col_name, (ts_node, base_idx) in ts_node_column_map.items():
        if ts_node not in nodes_to_process:
            nodes_to_process[ts_node] = {'columns': [], 'indices': []}
        nodes_to_process[ts_node]['columns'].append(col_name)
        nodes_to_process[ts_node]['indices'].append(base_idx)
    
    # Compute features for each TimeSeriesFeatureNode
    for ts_node, info in nodes_to_process.items():
        try:
            # Compute features from stored data
            computed_features = ts_node.compute_features_from_stored_data()
            
            # Get output feature names in correct order
            output_names = ts_node.get_column_names()
            
            # Update each column that belongs to this TS node
            for col_name in info['columns']:
                # Find the corresponding output feature
                base_idx = info['indices'][info['columns'].index(col_name)]
                
                # The keys in computed_features are from ts_node.output_features (simple names)
                # not from get_column_names() (standardized names)
                # So we need to use output_features[base_idx] to look up
                feature_values = None
                
                if hasattr(ts_node, 'output_features') and base_idx < len(ts_node.output_features):
                    # This is the key that compute_features_from_stored_data() uses
                    output_feature_name = ts_node.output_features[base_idx]
                    if output_feature_name in computed_features:
                        feature_values = computed_features[output_feature_name]
                
                # Fallback: try base column name
                if feature_values is None and hasattr(ts_node, 'base_column_names') and base_idx < len(ts_node.base_column_names):
                    base_col_name = ts_node.base_column_names[base_idx]
                    if base_col_name in computed_features:
                        feature_values = computed_features[base_col_name]
                
                if feature_values is None:
                    # Debug: print available keys
                    available_keys = list(computed_features.keys())[:5]  # First 5 keys for debugging
                    tried_names = []
                    if hasattr(ts_node, 'output_features') and base_idx < len(ts_node.output_features):
                        tried_names.append(f"output_features[{base_idx}]={ts_node.output_features[base_idx]}")
                    if hasattr(ts_node, 'base_column_names') and base_idx < len(ts_node.base_column_names):
                        tried_names.append(f"base_column_names[{base_idx}]={ts_node.base_column_names[base_idx]}")
                    logger.warning(f"Could not find computed features for {col_name} (tried: {', '.join(tried_names)}). Available keys (first 5): {available_keys}")
                    continue
                    
                # Align with matrix rows
                if len(feature_values) == len(matrix_df):
                    matrix_df[col_name] = feature_values
                elif len(feature_values) < len(matrix_df):
                    # Pad with NaN at the beginning (since we need lookback period)
                    padded = [np.nan] * (len(matrix_df) - len(feature_values)) + feature_values
                    matrix_df[col_name] = padded
                else:
                    # Truncate from the beginning (take the last len(matrix_df) values)
                    matrix_df[col_name] = feature_values[-len(matrix_df):]
                        
        except Exception as e:
            logger.warning(f"Error computing TS features for node {ts_node.name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            continue
    
    return matrix_df


def parse_feature_column_name(name: str) -> Dict[str, Any]:
    """
    Parse standardized feature column name into components.
    
    Module and feature names use snake_case, parameter names use camelCase.
    Expected format: {module}_{feature}_{tf}_{param}_{value}_{param}_{value}...
    Example: rsi_signal_D_lookback_14
    Example: cumulative_rsi_signal_D_avgPeriod_2_lookback_2
    
    Handles multi-word module names (ma_diff, cumulative_rsi, ts_feature, etc.)
    by checking against known modules first.
    
    Parameter names are in camelCase (e.g., 'avgPeriod' not 'avg_period')
    to avoid parser confusion with underscores.

    Returns dict with keys: { 'module', 'feature', 'tf', 'params' }
    - tf is the TimeFrame enum if name matches, else raw string
    - params keys are in camelCase (converted from snake_case if needed)
    - params values are auto-converted to int/float when possible
    """
    if not isinstance(name, str):
        name = str(name)
    
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
            return {'module': None, 'feature': None, 'tf': None, 'params': {}}
        module = tokens[0]
        remainder = '_'.join(tokens[1:])
    
    # Parse remainder: feature_tf_param_value_param_value...
    tokens = remainder.split('_')
    if len(tokens) < 2:
        return {'module': module, 'feature': None, 'tf': None, 'params': {}}
    
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
        'params': params
    }
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