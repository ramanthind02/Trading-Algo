"""
Walk-Forward Model and Splitter

This module provides a clean, single-responsibility implementation of walk-forward validation:
1. WalkForwardSplitter: Splits datetime-indexed data into train/test folds
2. WalkForwardModel: Wraps a BaseModel to perform walk-forward evaluation

Key Design Principles:
- Single Responsibility: Splitter ONLY splits data, Model ONLY evaluates
- No metric computation: That's delegated to ObjectiveMetric classes
- No logging/stats: That's delegated to caller
- Clean interface: Returns simple dicts with results
- Reusable: Splitter can be used independently

Author: Trading Research Team
Date: 2025-10-24
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from datetime import datetime, timedelta
from feature_selection.base_models.base_model import BinningModelBase


class WalkForwardSplitter:
    """
    Walk-forward data splitter.
    
    This class has ONE responsibility: split datetime-indexed data into
    train/test folds for walk-forward validation.
    
    It does NOT:
    - Fit models
    - Compute metrics
    - Log results
    - Store statistics
    
    It ONLY returns a list of (train_indices, test_indices) tuples.
    
    Parameters
    ----------
    train_start : datetime
        Start date for initial training window
    train_end : datetime
        End date for initial training window
    test_step : int
        Number of days for test period (also the roll-forward step)
    num_steps : int
        Number of walk-forward steps to perform
        
    Examples
    --------
    >>> splitter = WalkForwardSplitter(
    ...     train_start=datetime(2010, 1, 1),
    ...     train_end=datetime(2015, 1, 1),
    ...     test_step=252,
    ...     num_steps=5
    ... )
    >>> 
    >>> # Get splits for datetime-indexed data
    >>> splits = splitter.split(df.index)
    >>> 
    >>> for train_idx, test_idx in splits:
    ...     X_train, y_train = df.iloc[train_idx], target.iloc[train_idx]
    ...     X_test, y_test = df.iloc[test_idx], target.iloc[test_idx]
    ...     # ... fit and evaluate
    """
    
    def __init__(
        self,
        train_start: datetime,
        train_end: datetime,
        test_step: int,
        num_steps: int
    ):
        """
        Initialize walk-forward splitter.
        
        Args:
            train_start: Start date for initial training window
            train_end: End date for initial training window
            test_step: Number of days for test period
            num_steps: Number of walk-forward steps
        """
        self.train_start = train_start
        self.train_end = train_end
        self.test_step = test_step
        self.num_steps = num_steps
    
    def split(self, datetime_index: pd.DatetimeIndex) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Split datetime-indexed data into train/test folds.
        
        Args:
            datetime_index: DatetimeIndex from the data to split
            
        Returns:
            List of (train_indices, test_indices) tuples, one per step
            
        Examples:
        --------
        >>> splits = splitter.split(df.index)
        >>> len(splits)  # Number of valid steps
        5
        >>> train_idx, test_idx = splits[0]
        >>> len(train_idx), len(test_idx)
        (1260, 252)  # ~5 years train, 1 year test
        """
        # Handle timezone compatibility
        if datetime_index.tz is not None:
            if self.train_start.tzinfo is None:
                train_start = self.train_start.replace(tzinfo=datetime_index.tz)
                train_end = self.train_end.replace(tzinfo=datetime_index.tz)
            else:
                train_start = self.train_start
                train_end = self.train_end
        else:
            if self.train_start.tzinfo is not None:
                train_start = self.train_start.replace(tzinfo=None)
                train_end = self.train_end.replace(tzinfo=None)
            else:
                train_start = self.train_start
                train_end = self.train_end
        
        splits = []
        
        for step in range(self.num_steps):
            # Calculate date ranges for this step
            current_train_start = train_start + timedelta(days=step * self.test_step)
            current_train_end = train_end + timedelta(days=step * self.test_step)
            current_test_start = current_train_end
            current_test_end = current_test_start + timedelta(days=self.test_step)
            
            # Get indices for train and test
            train_mask = (datetime_index >= current_train_start) & (datetime_index < current_train_end)
            test_mask = (datetime_index >= current_test_start) & (datetime_index < current_test_end)
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            # Only include splits with sufficient data
            if len(train_indices) >= 100 and len(test_indices) >= 10:
                splits.append((train_indices, test_indices))
        
        return splits
    
    def get_split_info(self, datetime_index: pd.DatetimeIndex) -> List[Dict]:
        """
        Get detailed information about each split.
        
        This is useful for debugging or logging, but is separate from the
        core split() method to maintain single responsibility.
        
        Args:
            datetime_index: DatetimeIndex from the data
            
        Returns:
            List of dicts with split information
        """
        splits = self.split(datetime_index)
        split_info = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            train_dates = datetime_index[train_idx]
            test_dates = datetime_index[test_idx]
            
            split_info.append({
                'step': i,
                'train_start': train_dates.min(),
                'train_end': train_dates.max(),
                'test_start': test_dates.min(),
                'test_end': test_dates.max(),
                'n_train': len(train_idx),
                'n_test': len(test_idx)
            })
        
        return split_info


class WalkForwardModel(BinningModelBase):
    """
    Walk-forward wrapper for BaseModel.
    
    This class wraps any BaseModel and evaluates it using walk-forward validation.
    It delegates:
    - Data splitting: to WalkForwardSplitter
    - Model fitting/prediction: to wrapped BaseModel
    - Metric computation: to ObjectiveMetric
    
    It has ONE responsibility: orchestrate walk-forward evaluation and return
    aggregated out-of-sample results.
    
    Parameters
    ----------
    base_model : BinningModelBase
        The model to wrap (e.g., ContinuousBinningModel, DecisionTreeBinningModel)
    train_start : datetime
        Start date for initial training window
    train_end : datetime
        End date for initial training window
    test_step : int
        Number of days for test period
    num_steps : int
        Number of walk-forward steps
    min_test_samples : int, default=10
        Minimum test samples required per step
        
    Attributes
    ----------
    base_model : BinningModelBase
        The wrapped model
    splitter : WalkForwardSplitter
        The data splitter
    oos_results_ : List[Dict]
        Out-of-sample results for each step (set after compute_objective_metric)
        
    Examples
    --------
    >>> from feature_selection.base_models import ContinuousBinningModel
    >>> from metrics.performance import SortinoRatio
    >>> 
    >>> # Create walk-forward wrapper
    >>> base_model = ContinuousBinningModel(n_bins=3)
    >>> wf_model = WalkForwardModel(
    ...     base_model=base_model,
    ...     train_start=datetime(2010, 1, 1),
    ...     train_end=datetime(2015, 1, 1),
    ...     test_step=252,
    ...     num_steps=5
    ... )
    >>> 
    >>> # Compute OOS metric
    >>> metric = SortinoRatio(annualization_factor=252)
    >>> oos_sortino = wf_model.compute_objective_metric(
    ...     feature_data=X,
    ...     target_data=y,
    ...     objective_metric=metric,
    ...     strategy='long'
    ... )
    >>> 
    >>> # Access per-step results
    >>> for step_result in wf_model.oos_results_:
    ...     print(f"Step {step_result['step']}: {step_result['oos_metric']:.2f}")
    """
    
    def __init__(
        self,
        base_model: BinningModelBase,
        train_start: datetime,
        train_end: datetime,
        test_step: int,
        num_steps: int,
        min_test_samples: int = 10
    ):
        """
        Initialize walk-forward model.
        
        Args:
            base_model: Model to wrap
            train_start: Start date for initial training window
            train_end: End date for initial training window
            test_step: Number of days for test period
            num_steps: Number of walk-forward steps
            min_test_samples: Minimum test samples required
        """
        # Initialize parent with base model's parameters
        super().__init__(
            n_bins=base_model.n_bins,
            selection_metric=base_model.selection_metric
        )
        
        self.base_model = base_model
        self.splitter = WalkForwardSplitter(
            train_start=train_start,
            train_end=train_end,
            test_step=test_step,
            num_steps=num_steps
        )
        self.min_test_samples = min_test_samples
        
        # Results storage (populated after compute_objective_metric)
        self.oos_results_ = None
    
    def _create_bins(self, feature_data: pd.Series, target_data: pd.Series) -> pd.Series:
        """
        Not used for walk-forward model (delegates to base_model).
        
        This method is required by BaseModel interface but not used directly.
        """
        return self.base_model._create_bins(feature_data, target_data)
    
    def fit(self, feature_data: pd.Series, target_data: pd.Series) -> 'WalkForwardModel':
        """
        Not used for walk-forward model.
        
        Walk-forward model doesn't have a single "fit" - it fits on each
        training fold during compute_objective_metric().
        
        Raises:
            NotImplementedError: Always, use compute_objective_metric() instead
        """
        raise NotImplementedError(
            "WalkForwardModel doesn't support fit(). "
            "Use compute_objective_metric() which performs walk-forward evaluation."
        )
    
    def predict(self, feature_data: pd.Series, strategy: str = 'long') -> pd.Series:
        """
        Not used for walk-forward model.
        
        Walk-forward model doesn't have a single "predict" - it predicts on each
        test fold during compute_objective_metric().
        
        Raises:
            NotImplementedError: Always, use compute_objective_metric() instead
        """
        raise NotImplementedError(
            "WalkForwardModel doesn't support predict(). "
            "Use compute_objective_metric() which performs walk-forward evaluation."
        )
    
    def compute_objective_metric(
        self,
        feature_data: pd.Series,
        target_data: pd.Series,
        objective_metric: 'ObjectiveMetric',
        strategy: str = 'long'
    ) -> float:
        """
        Perform walk-forward evaluation and return aggregated OOS metric.
        
        This method:
        1. Splits data into train/test folds
        2. For each fold:
            - Fits base_model on training data
            - Predicts on test data
            - Collects OOS returns
        3. Aggregates all OOS returns
        4. Computes objective metric on aggregated OOS returns
        
        Args:
            feature_data: Feature values (datetime-indexed)
            target_data: Target values (datetime-indexed)
            objective_metric: Metric to compute (e.g., SortinoRatio)
            strategy: 'long' or 'short'
            
        Returns:
            Aggregated out-of-sample objective metric value
        """
        # Ensure data is aligned and has datetime index
        if not isinstance(feature_data.index, pd.DatetimeIndex):
            raise ValueError("feature_data must have DatetimeIndex")
        if not isinstance(target_data.index, pd.DatetimeIndex):
            raise ValueError("target_data must have DatetimeIndex")
        
        # Get splits
        splits = self.splitter.split(feature_data.index)
        
        if len(splits) == 0:
            raise ValueError("No valid walk-forward splits found")
        
        # Collect OOS returns and per-step results
        all_oos_returns = []
        self.oos_results_ = []
        
        for step, (train_idx, test_idx) in enumerate(splits):
            # Get train/test data
            X_train = feature_data.iloc[train_idx]
            y_train = target_data.iloc[train_idx]
            X_test = feature_data.iloc[test_idx]
            y_test = target_data.iloc[test_idx]
            
            # Fit base model on training data
            self.base_model.fit(X_train, y_train)
            
            # Predict on test data
            test_signals = self.base_model.predict(X_test, strategy=strategy)
            
            # Get OOS returns for selected signals
            oos_returns = y_test[test_signals == 1]
            
            # Only include if sufficient samples
            if len(oos_returns) >= self.min_test_samples:
                all_oos_returns.append(oos_returns)
                
                # Store per-step results (for debugging/analysis)
                step_metric = objective_metric.compute(oos_returns)
                self.oos_results_.append({
                    'step': step,
                    'n_trades': len(oos_returns),
                    'oos_metric': step_metric,
                    'train_start': X_train.index.min(),
                    'train_end': X_train.index.max(),
                    'test_start': X_test.index.min(),
                    'test_end': X_test.index.max()
                })
        
        # Aggregate all OOS returns
        if len(all_oos_returns) == 0:
            return 0.0
        
        aggregated_oos_returns = pd.concat(all_oos_returns, axis=0)
        
        # Compute objective metric on aggregated OOS returns
        oos_metric = objective_metric.compute(aggregated_oos_returns)
        
        return oos_metric
    
    def get_oos_results(self) -> List[Dict]:
        """
        Get per-step out-of-sample results.
        
        Returns:
            List of dicts with results for each step
            
        Raises:
            ValueError: If compute_objective_metric hasn't been called yet
        """
        if self.oos_results_ is None:
            raise ValueError(
                "No results available. Call compute_objective_metric() first."
            )
        return self.oos_results_
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WalkForwardModel("
            f"base_model={self.base_model.__class__.__name__}, "
            f"num_steps={self.splitter.num_steps})"
        )


# =============================================================================
# Utility Functions for Walk-Forward and Rolling Window Analysis
# =============================================================================

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
        
    Examples
    --------
    >>> def compute_metrics(train_data, test_data, split_info):
    ...     # Your custom analysis here
    ...     return {'metric': 0.5}
    >>> 
    >>> results = apply_function_to_walkforward(
    ...     df=data,
    ...     func=compute_metrics,
    ...     train_start=datetime(2010, 1, 1),
    ...     train_end=datetime(2015, 1, 1),
    ...     test_step=252,
    ...     num_steps=5
    ... )
    """
    # Use WalkForwardSplitter for data splitting
    splitter = WalkForwardSplitter(
        train_start=train_start,
        train_end=train_end,
        test_step=test_step,
        num_steps=num_steps
    )
    
    # Get splits
    index_splits = splitter.split(df.index)
    
    if verbose:
        print(f"Generated {len(index_splits)} valid walk-forward splits")
    
    # Select columns
    if columns is None:
        data = df
    else:
        data = df[columns]
    
    # Apply function to each split
    results = []
    for step, (train_idx, test_idx) in enumerate(index_splits):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # Check minimum samples
        if len(train_idx) < min_train_samples or len(test_idx) < min_test_samples:
            continue
        
        if verbose:
            print(f"Processing step {step}: "
                  f"train={len(train_idx)}, test={len(test_idx)}")
        
        # Create split info
        split_info = {
            'step': step,
            'train_start': train_data.index.min(),
            'train_end': train_data.index.max(),
            'test_start': test_data.index.min(),
            'test_end': test_data.index.max(),
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        }
        
        try:
            result = func(train_data, test_data, split_info)
            result.update(split_info)  # Add split info to result
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"  Warning: Step {step} failed: {e}")
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
        
    Examples
    --------
    >>> def analyze_window(window_data, window_info):
    ...     # Your custom analysis here
    ...     return {'mean': window_data['value'].mean()}
    >>> 
    >>> results = apply_function_to_rolling_windows(
    ...     df=data,
    ...     func=analyze_window,
    ...     window_size=252,
    ...     step_size=63
    ... )
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