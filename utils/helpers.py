
import os
import gc
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from utils.enums import TimeFrame, Ticker
from datetime import datetime


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
    
    # Extract the base module name (e.g., 'donchian_channel' from 'donchian_channel_10_D')
    # This pattern matches the base module name before any underscore followed by numbers
    base_module_match = re.match(r'^([a-zA-Z_]+)(?:_\d+.*)?$', module_name)
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
    # Strategy: Look for classes that have a get_instance method
    main_class = None
    for name, obj in inspect.getmembers(module):
        # Skip the abstract BiasNode class
        if name == 'BiasNode':
            continue
        if inspect.isclass(obj) and hasattr(obj, 'get_instance') and callable(getattr(obj, 'get_instance')):
            main_class = obj
            break
    
    # If we couldn't find a class with get_instance, try to find any class defined in the module
    if main_class is None:
        for name, obj in inspect.getmembers(module):
            # Skip the abstract BiasNode class
            if name == 'BiasNode':
                continue
            if inspect.isclass(obj) and obj.__module__ == full_module_name:
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


def get_bias_nodes(ticker: Ticker) -> Dict[str, Tuple[List[TimeFrame], Dict]]:
    """
    Creates a dictionary of bias strategies in the format expected by MLManager.
    
    This function defines bias strategies with their timeframes and parameters:
    - Bollinger Bands for daily timeframe
    - Donchian Channels for weekly and daily timeframes with various lookback periods
    - Moving Average Difference for weekly and daily timeframes with various short/long lookback combinations
    - Turtle Trading for weekly timeframe
    
    Parameters:
    - ticker (Ticker): The ticker symbol to create bias nodes for
    
    Returns:
    - Dict[str, Tuple[List[TimeFrame], Dict]]: A dictionary mapping strategy names to their timeframes and parameters
    """
    bias_strategies = {}
    
    # Bollinger Bands - Daily
    bias_strategies["bollinger_band"] = ([TimeFrame.D], {"ma_length": 80, "std_mult": 1})
    
    # Donchian Channel - Weekly and Daily with different lookback periods
    # For Weekly: 4, 10, 20, 50
    # For Daily: 5, 10, 20, 40, 80, 160
    # Note: The MLManager will handle creating separate instances for each lookback period
    weekly_lookbacks = [4, 10, 20, 50]
    daily_lookbacks = [5, 10, 20, 40, 80, 160]
    
    for lookback in weekly_lookbacks:
        strategy_key = f"donchian_channel_{lookback}"
        bias_strategies[strategy_key] = ([TimeFrame.W], {"lookback": lookback})
    
    for lookback in daily_lookbacks:
        strategy_key = f"donchian_channel_{lookback}"
        bias_strategies[strategy_key] = ([TimeFrame.D], {"lookback": lookback})
    
    # Moving Average Difference - Weekly and Daily with different short/long lookback combinations
    ma_pairs = [
        (2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256),
        (5, 50), (10, 100), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128)
    ]
    
    for tf in [TimeFrame.W, TimeFrame.D]:
        for short_lookback, long_lookback in ma_pairs:
            strategy_key = f"moving_avg_diff_{short_lookback}_{long_lookback}"
            bias_strategies[strategy_key] = ([tf], {
                "short_lookback": short_lookback,
                "long_lookback": long_lookback
            })
    
    # Turtle Trading - Weekly
    # Note: Using 'lookback' parameter to match DonchianChannel's expected parameter name
    bias_strategies["turtle"] = ([TimeFrame.W], {"lookback": 4})

    bias_strategies['prev_return'] = ([TimeFrame.D], {})

    # Detrended RSI
    bias_strategies['detrended_rsi'] = ([TimeFrame.D], {"short_length": 2, "long_length": 20, "lookback": 252})
    
    # RSI with different lookback periods
    # Note: Each needs a unique key to avoid overwriting
    rsi_lookbacks = [2, 5, 10, 14]
    for lookback in rsi_lookbacks:
        strategy_key = f"rsi_{lookback}"
        bias_strategies[strategy_key] = ([TimeFrame.D], {"lookback": lookback})
    

    
    return bias_strategies


def create_ml_manager(ticker: Ticker, base_tf: TimeFrame = TimeFrame.D, build_matrix: bool = True):
    """
    Creates an ML Manager instance with all the bias nodes configured.
    
    This function instantiates an MLManager with all the bias nodes from the get_bias_nodes function,
    formatted in the way MLManager expects.
    
    Parameters:
    - ticker (Ticker): The ticker symbol to create the ML manager for
    - base_tf (TimeFrame): Base timeframe for the ML model, defaults to daily
    - build_matrix (bool): Whether to construct a matrix for training, defaults to False
    
    Returns:
    - MLManager: An instantiated ML manager with all bias nodes configured
    """
    from feature_extraction.ml_manager import MLManager
    
    # Get bias strategies dictionary using our get_bias_nodes function
    bias_strategies = get_bias_nodes(ticker)
    
    # Create the ML Manager
    ml_manager = MLManager(
        bias_strategies=bias_strategies,
        ticker=ticker,
        base_tf=base_tf,
        build_matrix=build_matrix
    )
    
    return ml_manager
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