"""Simplified Feature Extraction Module

Direct feature extraction by iterating over candle dataframes.
Extracts one bias node at a time with parameter grid exploration.

**SINGLE BIAS NODE EXTRACTION**
-------------------------------
Extract features for a single bias node with multiple parameter combinations.
Columns = parameter combinations, Rows = timestamps.

Examples:
    # Single parameter set
    features_df, targets_df = extract_features(
        module_name='rsi',
        params={'lookback': 14},
        ticker=Ticker.SPY
    )
    
    # Multiple parameter sets (grid search)
    features_df, targets_df = extract_features(
        module_name='rsi',
        params={'lookback': [14, 21, 28]},
        ticker=Ticker.SPY
    )
    
    # Multiple parameters with grid search
    features_df, targets_df = extract_features(
        module_name='cmma',
        params={'lookback': [20, 50], 'atr_length': [252]},
        ticker=Ticker.SPY
    )
"""

from datetime import datetime
from utils.enums import TimeFrame, Ticker
import utils.helpers as helpers
from utils.models import Candle
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Tuple
from itertools import product


def _expand_param_grid(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand parameter grid to list of parameter dicts."""
    if not isinstance(params, dict):
        return [{}]
    
    # Check if any values are lists (grid search)
    has_lists = any(isinstance(v, list) for v in params.values())
    
    if not has_lists:
        return [params]
    
    # Grid search: expand all combinations
    keys = list(params.keys())
    values = [v if isinstance(v, list) else [v] for v in params.values()]
    
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _compute_targets(price_df: pd.DataFrame, atr_col: str = None, ewsd_col: str = None) -> pd.DataFrame:
    """Compute target columns from price data."""
    raw_return = (price_df['close'] / price_df['open']) - 1
    log_return = np.log(price_df['close'] / price_df['open'])
    
    log_return_atr = log_return.copy()
    if atr_col and atr_col in price_df.columns:
        log_return_atr = log_return / np.maximum(price_df[atr_col], 0.0001)
    
    log_return_ewsd = log_return.copy()
    if ewsd_col and ewsd_col in price_df.columns:
        ewsd_decimal = price_df[ewsd_col] / 100.0
        log_return_ewsd = log_return / np.maximum(ewsd_decimal, 0.0001)
    
    return pd.DataFrame({
        'raw_return': raw_return,
        'log_return': log_return,
        'log_return_atr': log_return_atr,
        'log_return_ewsd': log_return_ewsd
    }, index=price_df.index)


def extract_features(
    module_name: str,
    params: Dict[str, Any],
    ticker: Ticker,
    start: datetime = None,
    end: datetime = None,
    timeframes: List[TimeFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract features for a single bias node with parameter grid exploration.
    
    Parameters
    ----------
    module_name : str
        Name of the bias node module (e.g., 'rsi', 'atr', 'cmma')
    params : Dict[str, Any]
        Parameters for the bias node. Supports grid search:
        - Single value: {'lookback': 14}
        - List of values: {'lookback': [14, 21, 28]} -> creates columns for each
        - Multiple params: {'lookback': [14, 21], 'period': 252} -> all combinations
    ticker : Ticker
        Ticker to extract features for
    start : datetime, optional
        Start date. Defaults to datetime(1990, 1, 1)
    end : datetime, optional
        End date. Defaults to datetime.now()
    timeframes : List[TimeFrame], optional
        Timeframes to use. Defaults to [TimeFrame.D]
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (features_df, targets_df)
        - features_df: Columns = parameter combinations, Rows = timestamps
        - targets_df: Target columns (raw_return, log_return, etc.)
        
    Examples
    --------
    >>> # Single parameter
    >>> features_df, targets_df = extract_features(
    ...     module_name='rsi',
    ...     params={'lookback': 14},
    ...     ticker=Ticker.SPY
    ... )
    >>> 
    >>> # Parameter grid
    >>> features_df, targets_df = extract_features(
    ...     module_name='rsi',
    ...     params={'lookback': [14, 21, 28]},
    ...     ticker=Ticker.SPY
    ... )
    """
    if start is None:
        start = datetime(1990, 1, 1)
    if end is None:
        end = datetime.now()
    if timeframes is None:
        timeframes = [TimeFrame.D]
    
    # Load price data
    price_df = helpers.load_data(ticker, TimeFrame.D, start=start, end=end)
    price_df.set_index('datetime', inplace=True)
    price_df.index = price_df.index.tz_localize('UTC')
    
    # Expand parameter grid
    param_combos = _expand_param_grid(params)
    
    # Create bias nodes for each parameter combination
    bias_nodes = []
    column_names = []
    
    for param_combo in param_combos:
        for tf in timeframes:
            bias_node = helpers.create_bias_node(module_name, ticker, tf, param_combo)
            bias_nodes.append(bias_node)
            
            # Get column names for this node
            node_cols = bias_node.get_column_names() if hasattr(bias_node, 'get_column_names') else getattr(bias_node, 'columns', [])
            if not node_cols:
                node_cols = [f"{module_name}_{tf.name}"]
            column_names.extend(node_cols)
    
    # Initialize feature storage
    n_rows = len(price_df)
    n_cols = len(column_names)
    feature_data = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    
    # Map nodes to column ranges
    node_to_cols = {}
    col_idx = 0
    for bias_node in bias_nodes:
        node_cols = bias_node.get_column_names() if hasattr(bias_node, 'get_column_names') else getattr(bias_node, 'columns', [])
        if not node_cols:
            node_cols = [f"{bias_node.module_name}_{bias_node.tf.name}"]
        n_node_cols = len(node_cols)
        node_to_cols[bias_node] = (col_idx, col_idx + n_node_cols)
        col_idx += n_node_cols
    
    # Extract features by iterating over candles
    for idx, (dt, row) in enumerate(price_df.iterrows()):
        candle = Candle(
            datetime=dt,
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row.get('volume', 0)),
            ticker=ticker,
            tf=TimeFrame.D
        )
        
        for bias_node in bias_nodes:
            if bias_node.tf == TimeFrame.D:
                values = bias_node.add_candle(candle)
                start_col, end_col = node_to_cols[bias_node]
                
                for i in range(min(len(values), end_col - start_col)):
                    val = values[i]
                    if hasattr(val, 'value'):
                        val = val.value
                    feature_data[idx, start_col + i] = float(val) if val is not None else np.nan
    
    # Create features DataFrame
    features_df = pd.DataFrame(feature_data, index=price_df.index, columns=column_names)
    
    # Compute targets (need ATR/EWSD if available)
    atr_col = next((col for col in features_df.columns if 'atr' in col.lower() and '252' in col), None)
    ewsd_col = next((col for col in features_df.columns if 'ewsd' in col.lower()), None)
    
    # Add ATR/EWSD to price_df for target computation if found
    if atr_col:
        price_df[atr_col] = features_df[atr_col]
    if ewsd_col:
        price_df[ewsd_col] = features_df[ewsd_col]
    
    targets_df = _compute_targets(price_df, atr_col, ewsd_col)
    
    return features_df, targets_df
