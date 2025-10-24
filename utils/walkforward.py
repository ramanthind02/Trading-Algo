"""
Abstract Walk-Forward and Rolling Window Analysis Utilities

This module provides reusable utilities for time-based rolling window analysis,
including walk-forward validation and rolling EDA.

Author: Trading Research Team
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Callable, List, Tuple, Optional, Any, Dict
import warnings


def generate_walkforward_splits(
    df: pd.DataFrame,
    train_start: datetime,
    train_end: datetime,
    test_step: int,
    num_steps: int,
    min_train_samples: int = 100,
    min_test_samples: int = 10
) -> List[Dict[str, Any]]:
    """
    Generate walk-forward train/test splits for time series data.
    
    This creates a rolling window where:
    1. Train on a fixed window of historical data
    2. Test on the next period immediately after training
    3. Roll forward and repeat
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    train_start : datetime
        Start date for initial training period
    train_end : datetime
        End date for initial training period
    test_step : int
        Number of days for test period (also the roll-forward step size)
    num_steps : int
        Number of walk-forward steps to perform
    min_train_samples : int, default=100
        Minimum samples required in training set
    min_test_samples : int, default=10
        Minimum samples required in test set
        
    Returns
    -------
    List[Dict[str, Any]]
        List of split dictionaries, each containing:
        - 'step': Step number
        - 'train_start': Training start date
        - 'train_end': Training end date
        - 'test_start': Test start date
        - 'test_end': Test end date
        - 'train_mask': Boolean mask for training data
        - 'test_mask': Boolean mask for test data
        - 'train_size': Number of training samples
        - 'test_size': Number of test samples
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    # Ensure datetime parameters are timezone-aware if DataFrame index is
    if df.index.tz is not None:
        if train_start.tzinfo is None:
            train_start = train_start.replace(tzinfo=df.index.tz)
        if train_end.tzinfo is None:
            train_end = train_end.replace(tzinfo=df.index.tz)
    
    splits = []
    
    for step in range(num_steps):
        # Calculate date ranges for this step
        current_train_start = train_start + timedelta(days=step * test_step)
        current_train_end = train_end + timedelta(days=step * test_step)
        current_test_start = current_train_end
        current_test_end = current_test_start + timedelta(days=test_step)
        
        # Create masks
        train_mask = (df.index >= current_train_start) & (df.index < current_train_end)
        test_mask = (df.index >= current_test_start) & (df.index < current_test_end)
        
        train_size = train_mask.sum()
        test_size = test_mask.sum()
        
        # Skip if insufficient data
        if train_size < min_train_samples or test_size < min_test_samples:
            continue
        
        splits.append({
            'step': step,
            'train_start': current_train_start,
            'train_end': current_train_end,
            'test_start': current_test_start,
            'test_end': current_test_end,
            'train_mask': train_mask,
            'test_mask': test_mask,
            'train_size': train_size,
            'test_size': test_size
        })
    
    return splits


def generate_rolling_windows(
    df: pd.DataFrame,
    window_size: int,
    step_size: int = None,
    min_samples: int = 100
) -> List[Dict[str, Any]]:
    """
    Generate rolling windows for time series data.
    
    Unlike walk-forward which has separate train/test, this creates
    overlapping windows for rolling analysis (e.g., rolling deciles).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    window_size : int
        Number of days in each window
    step_size : int, optional
        Number of days to roll forward (default: window_size, non-overlapping)
    min_samples : int, default=100
        Minimum samples required in each window
        
    Returns
    -------
    List[Dict[str, Any]]
        List of window dictionaries, each containing:
        - 'window_num': Window number
        - 'start_date': Window start date
        - 'end_date': Window end date
        - 'mask': Boolean mask for window data
        - 'size': Number of samples in window
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    if step_size is None:
        step_size = window_size
    
    # Get date range
    start_date = df.index.min()
    end_date = df.index.max()
    
    windows = []
    window_num = 0
    current_start = start_date
    
    while current_start < end_date:
        current_end = current_start + timedelta(days=window_size)
        
        # Cap end date at actual data end (prevent windows extending into future)
        current_end = min(current_end, end_date)
        
        # Create mask
        mask = (df.index >= current_start) & (df.index < current_end)
        size = mask.sum()
        
        # Skip if insufficient data
        if size < min_samples:
            current_start += timedelta(days=step_size)
            continue
        
        windows.append({
            'window_num': window_num,
            'start_date': current_start,
            'end_date': current_end,
            'mask': mask,
            'size': size
        })
        
        window_num += 1
        current_start += timedelta(days=step_size)
    
    return windows


def apply_function_to_walkforward(
    df: pd.DataFrame,
    func: Callable,
    train_start: datetime,
    train_end: datetime,
    test_step: int,
    num_steps: int,
    columns: List[str] = None,
    min_train_samples: int = 100,
    min_test_samples: int = 10,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Apply a function to each walk-forward split.
    
    This is a generic utility that applies a user-defined function to
    train and test data in each walk-forward step.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    func : Callable
        Function to apply. Should accept (train_data, test_data, split_info)
        and return a dictionary of results
    train_start : datetime
        Start date for initial training period
    train_end : datetime
        End date for initial training period
    test_step : int
        Number of days for test period
    num_steps : int
        Number of walk-forward steps
    columns : List[str], optional
        Columns to pass to function (default: all columns)
    min_train_samples : int, default=100
        Minimum samples in training set
    min_test_samples : int, default=10
        Minimum samples in test set
    verbose : bool, default=False
        Print progress
        
    Returns
    -------
    List[Dict[str, Any]]
        List of results from each step
    """
    # Generate splits
    splits = generate_walkforward_splits(
        df, train_start, train_end, test_step, num_steps,
        min_train_samples, min_test_samples
    )
    
    if verbose:
        print(f"Generated {len(splits)} valid walk-forward splits")
    
    # Select columns
    if columns is None:
        data = df
    else:
        data = df[columns]
    
    # Apply function to each split
    results = []
    for split in splits:
        if verbose:
            print(f"Processing step {split['step']}: "
                  f"train={split['train_size']}, test={split['test_size']}")
        
        train_data = data[split['train_mask']]
        test_data = data[split['test_mask']]
        
        try:
            result = func(train_data, test_data, split)
            result['step'] = split['step']
            result['train_start'] = split['train_start']
            result['train_end'] = split['train_end']
            result['test_start'] = split['test_start']
            result['test_end'] = split['test_end']
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"  Warning: Step {split['step']} failed: {e}")
            continue
    
    return results


def apply_function_to_rolling_windows(
    df: pd.DataFrame,
    func: Callable,
    window_size: int,
    step_size: int = None,
    columns: List[str] = None,
    min_samples: int = 100,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Apply a function to each rolling window.
    
    This is a generic utility that applies a user-defined function to
    data in each rolling window.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    func : Callable
        Function to apply. Should accept (window_data, window_info)
        and return a dictionary of results
    window_size : int
        Number of days in each window
    step_size : int, optional
        Number of days to roll forward
    columns : List[str], optional
        Columns to pass to function (default: all columns)
    min_samples : int, default=100
        Minimum samples in each window
    verbose : bool, default=False
        Print progress
        
    Returns
    -------
    List[Dict[str, Any]]
        List of results from each window
    """
    # Generate windows
    windows = generate_rolling_windows(
        df, window_size, step_size, min_samples
    )
    
    if verbose:
        print(f"Generated {len(windows)} rolling windows")
    
    # Select columns
    if columns is None:
        data = df
    else:
        data = df[columns]
    
    # Apply function to each window
    results = []
    for window in windows:
        if verbose:
            print(f"Processing window {window['window_num']}: "
                  f"{window['start_date'].date()} to {window['end_date'].date()}, "
                  f"n={window['size']}")
        
        window_data = data[window['mask']]
        
        try:
            result = func(window_data, window)
            result['window_num'] = window['window_num']
            result['start_date'] = window['start_date']
            result['end_date'] = window['end_date']
            result['window_size'] = window['size']
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"  Warning: Window {window['window_num']} failed: {e}")
            continue
    
    return results
