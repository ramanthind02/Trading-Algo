
import os
import gc
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from utils.enums import TimeFrame, Ticker
from datetime import datetime



def load_data(ticker: Ticker, timeframe: TimeFrame, start: datetime = datetime(1990, 1, 1), end: datetime = datetime(2025, 12, 30)) -> pd.DataFrame:
    """Load OHLC data from parquet files using Dask for efficient reading.

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
    import dask.dataframe as dd
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    base_dir = os.path.join(root_dir, 'data', 'ohlc_data')
    file_path = os.path.join(base_dir, ticker.name, f"{timeframe.name}_{ticker.name}.parquet")

    if os.path.exists(file_path):
        # Read parquet file with Dask
        ddf = dd.read_parquet(file_path)
        
        # Convert datetime column to proper datetime type
        ddf['datetime'] = dd.to_datetime(ddf['datetime'])
        
        # Filter by date range
        mask = (ddf['datetime'] >= start) & (ddf['datetime'] <= end)
        ddf = ddf[mask]
        
        # Convert to pandas DataFrame
        df = ddf.compute()
        
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
    from pathlib import Path
    
    # Make sure we're using the module name without extension
    module_name = module_name.replace('.py', '')
    
    # Try to find the module in the nodes directory
    base_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "nodes"
    potential_file = base_path / f"{module_name}.py"
    
    if not potential_file.exists():
        raise ValueError(f"Could not find module file: {potential_file}")
    
    # Import the module
    full_module_name = f"nodes.{module_name}"
    try:
        module = importlib.import_module(full_module_name)
    except ImportError as e:
        raise ImportError(f"Error importing module {full_module_name}: {e}")
    
    # Find the main class in the module
    # Strategy: Look for classes that have a get_instance method
    main_class = None
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and hasattr(obj, 'get_instance') and callable(getattr(obj, 'get_instance')):
            main_class = obj
            break
    
    # If we couldn't find a class with get_instance, try to find any class defined in the module
    if main_class is None:
        for name, obj in inspect.getmembers(module):
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
        strategy_key = f"donchian_channel_{lookback}_W"
        bias_strategies[strategy_key] = ([TimeFrame.W], {"lookback": lookback})
    
    for lookback in daily_lookbacks:
        strategy_key = f"donchian_channel_{lookback}_D"
        bias_strategies[strategy_key] = ([TimeFrame.D], {"lookback": lookback})
    
    # Moving Average Difference - Weekly and Daily with different short/long lookback combinations
    ma_pairs = [
        (2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256),
        (5, 50), (10, 100), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128)
    ]
    
    for tf in [TimeFrame.W, TimeFrame.D]:
        for short_lookback, long_lookback in ma_pairs:
            strategy_key = f"moving_avg_diff_{short_lookback}_{long_lookback}_{tf.name}"
            bias_strategies[strategy_key] = ([tf], {
                "short_lookback": short_lookback,
                "long_lookback": long_lookback
            })
    
    # Turtle Trading - Weekly
    bias_strategies["turtle"] = ([TimeFrame.W], {"entry_lookback": 4, "stop_lookback": 2})
    
    return bias_strategies


def create_ml_manager(ticker: Ticker, base_tf: TimeFrame = TimeFrame.D, build_matrix: bool = False):
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