"""Feature Extraction Module

This module provides feature extraction functionality for bias nodes.

**UNIVERSAL EXTRACTION FUNCTION (Recommended)**
------------------------------------------------
Use `extract_features_from_bias_node()` as the single source of truth for all feature extraction.

It supports:
- Single or multiple tickers
- Single or multiple bias nodes
- Returns (features_df, targets_df) with explicit target columns:
  * raw_return: (close/open - 1)
  * log_return: log(close/open)
  * log_return_atr: log_return normalized by ATR
  * log_return_ewsd: log_return normalized by EWSD

Examples:
    # Single ticker, single node (simple API)
    features_df, targets_df = extract_features_from_bias_node(
        ticker=Ticker.SPY,
        module_name='rsi',
        params={'lookback': 14}
    )
    
    # Multiple tickers, multiple nodes (advanced API)
    features_df, targets_df = extract_features_from_bias_node(
        ticker=[Ticker.ES, Ticker.NQ],
        bias_node_specs=[
            {'module_name': 'rsi', 'timeframes': [TimeFrame.D], 'params': {'lookback': 14}},
            {'module_name': 'atr', 'timeframes': [TimeFrame.D], 'params': {'lookback': 252}}
        ]
    )

"""

from datetime import datetime, timezone
from utils.enums import TimeFrame, Ticker
from feature_extraction.backtest import Backtest
import utils.helpers as helpers
from utils.candle_fetcher import CandleFetcher
import pandas as pd
import numpy as np
from typing import List


def get_backtester(name: str, start: datetime, end: datetime, middleman):
    """
    Create a backtester instance with the specified parameters.
    
    Parameters
    ----------
    name : str
        Name identifier for the backtest
    start : datetime
        Start date for the data
    end : datetime
        End date for the data
    middleman : object
        Middleman object for the backtest
        
    Returns
    -------
    Backtest
        Configured backtest instance
    """
    data = helpers.load_numpy_data(middleman.ticker, TimeFrame.D, start=start, end=end)
    candle_fetcher = CandleFetcher(
        ticker=middleman.ticker, 
        tfs=[TimeFrame.W, TimeFrame.M]
    )
    return Backtest(name, data, middleman, candle_fetcher)


def extract_bias(
        ticker: Ticker,
        start: datetime = datetime(1990, 1, 1),
        end: datetime = datetime.now(),
        feature_filter: list = None,
        bias_node_specs: list = None,
        bar_data: np.ndarray = None
) -> tuple[pd.DataFrame, dict]:
    """
    Process a single timeframe for bias feature extraction.
    
    Parameters
    ----------
    ticker : Ticker
        Ticker symbol to process
    start : datetime
        Start date for the data
    end : datetime
        End date for the data
    feature_filter : list, optional
        List of feature names to extract. If provided, only nodes needed
        for these features will be computed, significantly improving performance.
    bias_node_specs : list, optional
        List of bias node specifications. Each spec should be a dict with:
        - 'module_name' (str): Name of the bias node module (e.g., 'rsi', 'atr')
        - 'timeframes' (List[TimeFrame]): List of timeframes to use
        - 'params' (Dict): Parameters for the bias node
        - 'strategy_key' (Optional[str]): Custom key, auto-generated if not provided
        If None, default bias nodes will be used.
    bar_data : np.ndarray, optional
        Pre-loaded bar data as numpy structured array with dtype:
        [('open', 'close', 'high', 'low', 'datetime')]
        If provided, this data will be used instead of loading from disk.
        This is useful for permutation testing where bars have been shuffled.
        
    Returns
    -------
    tuple[pd.DataFrame, dict]
        Tuple containing:
        - Processed and normalized features dataframe
        - Columns dictionary for feature identification
    """
    # Build middleman and run backtest
    # Pass feature_filter to only compute needed nodes
    ml_manager = helpers.create_ml_manager(
        ticker, 
        feature_filter=feature_filter,
        bias_node_specs=bias_node_specs
    )
    
    # Load data or use provided bar_data
    if bar_data is None:
        # Normal path: load data from disk
        data = helpers.load_numpy_data(ticker, TimeFrame.D, start=start, end=end)
    else:
        # Permutation path: use provided bar data
        data = bar_data
    
    # Create candle fetcher for multi-timeframe support
    candle_fetcher = CandleFetcher(
        ticker=ticker, 
        tfs=[TimeFrame.W, TimeFrame.M]
    )
    
    # Create and run backtest
    features_backtest = Backtest("features", data, ml_manager, candle_fetcher)
    features_backtest.run()
    
    # Extract features and clear middleman
    # Use matrix_df property to ensure buffer is flushed
    features = ml_manager.matrix_df
    columns = ml_manager.columns
    
    return features, columns



def compute_target_columns(price_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all four target columns from price and feature data.
    
    This centralizes target calculation logic to avoid duplication.
    
    Parameters
    ----------
    price_df : pd.DataFrame
        Price data with OHLCV columns
    features_df : pd.DataFrame
        Features dataframe (must contain ATR and EWSD columns)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with four target columns:
        - raw_return: (close/open - 1)
        - log_return: log(close/open)
        - log_return_atr: log_return normalized by ATR
        - log_return_ewsd: log_return normalized by EWSD
    """
    # Calculate raw return
    raw_return = (price_df['close'] / price_df['open']) - 1
    
    # Calculate log return
    log_return = np.log(price_df['close'] / price_df['open'])
    
    # Find ATR and EWSD columns in features
    atr_cols = [col for col in features_df.columns if 'atr_252_D_atr_pct' in col]
    ewsd_cols = [col for col in features_df.columns if 'ewsd_252_D_ewsd_daily_pct' in col]
    
    # Normalize by ATR (use safe division to prevent extreme outliers)
    if atr_cols:
        atr_col = atr_cols[0]
        MIN_ATR = 0.0001  # Minimum ATR threshold
        log_return_atr = log_return / np.maximum(features_df[atr_col], MIN_ATR)
    else:
        log_return_atr = log_return.copy()
    
    # Normalize by EWSD (use safe division)
    if ewsd_cols:
        ewsd_col = ewsd_cols[0]
        MIN_EWSD = 0.0001  # Minimum EWSD threshold (0.01% daily vol)
        # EWSD is already in percentage, so divide by 100 to get decimal
        ewsd_decimal = features_df[ewsd_col] / 100.0
        log_return_ewsd = log_return / np.maximum(ewsd_decimal, MIN_EWSD)
    else:
        log_return_ewsd = log_return.copy()
    
    # Create target dataframe
    targets_df = pd.DataFrame({
        'raw_return': raw_return,
        'log_return': log_return,
        'log_return_atr': log_return_atr,
        'log_return_ewsd': log_return_ewsd
    }, index=price_df.index)
    
    return targets_df


def extract_features_from_bias_node(
    ticker,
    module_name: str = None,
    params: dict = None,
    timeframes: list = None,
    start: datetime = None,
    end: datetime = None,
    strategy_key: str = None,
    use_millisecond_offset: bool = True,
    bias_node_specs: list = None
):
    """
    **UNIVERSAL FEATURE EXTRACTION FUNCTION**
    
    Extract features and targets from one or more bias nodes for single or multiple tickers.
    This is the single source of truth for all feature extraction in the codebase.
    
    Returns features and four explicit target columns:
    - raw_return: (close/open - 1)
    - log_return: log(close/open)
    - log_return_atr: log_return normalized by ATR
    - log_return_ewsd: log_return normalized by EWSD
    
    Parameters
    ----------
    ticker : Ticker or List[Ticker]
        Single ticker or list of tickers
    module_name : str, optional
        Name of a single bias node module (e.g., 'rsi', 'atr', 'cmma').
        Use this for simple single-node extraction. Mutually exclusive with bias_node_specs.
    params : dict, optional
        Parameters for the bias node (only used with module_name)
    timeframes : list, optional
        List of TimeFrame objects (only used with module_name). Defaults to [TimeFrame.D]
    start : datetime, optional
        Start date for extraction. Defaults to datetime(1990, 1, 1)
    end : datetime, optional
        End date for extraction. Defaults to datetime.now()
    strategy_key : str, optional
        Custom strategy key (only used with module_name). If None, auto-generated
    use_millisecond_offset : bool, default=True
        For multi-ticker: add millisecond offsets to avoid duplicate indices
    bias_node_specs : list of dict, optional
        List of bias node specifications for extracting multiple nodes at once.
        Each spec should be a dict with:
        - 'module_name' (str): Name of the bias node module
        - 'timeframes' (List[TimeFrame]): List of timeframes to use
        - 'params' (dict): Parameters for the bias node
        - 'strategy_key' (str, optional): Custom key, auto-generated if not provided
        Mutually exclusive with module_name.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (features_df, targets_df) - Features and four target columns with aligned indices
        
    Examples
    --------
    >>> # Single ticker, single bias node (simple API)
    >>> features_df, targets_df = extract_features_from_bias_node(
    ...     ticker=Ticker.SPY,
    ...     module_name='rsi',
    ...     params={'lookback': 14}
    ... )
    >>> 
    >>> # Multiple tickers, single bias node
    >>> features_df, targets_df = extract_features_from_bias_node(
    ...     ticker=[Ticker.ES, Ticker.NQ],
    ...     module_name='cmma',
    ...     params={'lookback': 20, 'atr_length': 252}
    ... )
    >>> 
    >>> # Single ticker, multiple bias nodes (advanced API)
    >>> features_df, targets_df = extract_features_from_bias_node(
    ...     ticker=Ticker.SPY,
    ...     bias_node_specs=[
    ...         {'module_name': 'rsi', 'timeframes': [TimeFrame.D], 'params': {'lookback': 14}},
    ...         {'module_name': 'atr', 'timeframes': [TimeFrame.D], 'params': {'lookback': 252}},
    ...         {'module_name': 'cmma', 'timeframes': [TimeFrame.D], 'params': {'lookback': 20}}
    ...     ]
    ... )
    >>> 
    >>> # Multiple tickers, multiple bias nodes (full power!)
    >>> features_df, targets_df = extract_features_from_bias_node(
    ...     ticker=[Ticker.ES, Ticker.NQ, Ticker.YM],
    ...     bias_node_specs=[
    ...         {'module_name': 'rsi', 'timeframes': [TimeFrame.D], 'params': {'lookback': 14}},
    ...         {'module_name': 'volatility_regime', 'timeframes': [TimeFrame.D], 'params': {'window': 252}}
    ...     ]
    ... )
    """
    # Validate inputs
    if module_name is not None and bias_node_specs is not None:
        raise ValueError(
            "Cannot specify both 'module_name' and 'bias_node_specs'. "
            "Use 'module_name' for single node extraction or 'bias_node_specs' for multiple nodes."
        )
    
    if module_name is None and bias_node_specs is None:
        raise ValueError(
            "Must specify either 'module_name' (for single node) or 'bias_node_specs' (for multiple nodes)"
        )
    
    # Set defaults
    if start is None:
        start = datetime(1990, 1, 1)
    if end is None:
        end = datetime.now()
    
    # Convert simple API (module_name) to advanced API (bias_node_specs)
    if module_name is not None:
        if params is None:
            params = {}
        if timeframes is None:
            timeframes = [TimeFrame.D]
        
        bias_node_specs = [{
            'module_name': module_name,
            'timeframes': timeframes,
            'params': params
        }]
        if strategy_key:
            bias_node_specs[0]['strategy_key'] = strategy_key
    
    # Handle single ticker vs multiple tickers
    from utils.enums import Ticker as TickerEnum
    if isinstance(ticker, TickerEnum):
        tickers = [ticker]
    else:
        tickers = ticker
    
    # Extract features
    if len(tickers) == 1:
        # Single ticker extraction
        features_df, columns = extract_bias(
            ticker=tickers[0],
            start=start,
            end=end,
            bias_node_specs=bias_node_specs
        )
        
        # Load price data
        price_df = helpers.load_data(tickers[0], TimeFrame.D)
        price_df.set_index('datetime', inplace=True)
        price_df.index = price_df.index.tz_localize('UTC')
        
        # Align price data with features
        price_df = price_df.reindex(features_df.index)
        
        # Compute all target columns
        targets_df = compute_target_columns(price_df, features_df)
        
        # Add ticker column for identification
        features_df['ticker'] = tickers[0].name
        targets_df['ticker'] = tickers[0].name
        
    else:
        # Multi-ticker extraction
        all_features_dfs = []
        all_targets_dfs = []
        
        for ticker_idx, single_ticker in enumerate(tickers):
            # Extract features for this ticker
            ticker_features_df, columns = extract_bias(
                ticker=single_ticker,
                start=start,
                end=end,
                bias_node_specs=bias_node_specs
            )
            
            # Load price data for this ticker
            ticker_price_data = helpers.load_data(single_ticker, TimeFrame.D)
            ticker_price_data.set_index('datetime', inplace=True)
            ticker_price_data.index = ticker_price_data.index.tz_localize('UTC')
            
            # Add millisecond offset to avoid duplicate datetime indices
            if use_millisecond_offset and ticker_idx > 0:
                offset = pd.Timedelta(milliseconds=ticker_idx)
                ticker_features_df.index = ticker_features_df.index + offset
                ticker_price_data.index = ticker_price_data.index + offset
            
            # Align price data with features
            ticker_price_data = ticker_price_data.reindex(ticker_features_df.index)
            
            # Compute all target columns for this ticker
            ticker_targets_df = compute_target_columns(ticker_price_data, ticker_features_df)
            
            # Add ticker column for identification
            ticker_features_df['ticker'] = single_ticker.name
            ticker_targets_df['ticker'] = single_ticker.name
            
            all_features_dfs.append(ticker_features_df)
            all_targets_dfs.append(ticker_targets_df)
        
        # Concatenate all tickers' data
        features_df = pd.concat(all_features_dfs, axis=0).sort_index()
        targets_df = pd.concat(all_targets_dfs, axis=0).sort_index()
    
    return features_df, targets_df