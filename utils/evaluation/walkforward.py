"""
Abstract Walk-Forward and Rolling Window Analysis Utilities

This module provides reusable utilities for time-based rolling window analysis,
including walk-forward validation and rolling EDA.

Author: Trading Research Team
Date: 2025-10-18
"""

# Import all functions from canonical module
from feature_selection.walkforward.walkforward_model import (
    WalkForwardSplitter,
    generate_rolling_windows,
    apply_function_to_walkforward,
    apply_function_to_rolling_windows
)

from datetime import datetime
from typing import List, Dict, Any
import pandas as pd



# =============================================================================
# Backward Compatibility Wrapper
# =============================================================================

def generate_walkforward_splits(
    df: pd.DataFrame,
    train_start: datetime,
    train_end: datetime,
    test_step: int,
    num_steps: int,
    min_train_samples: int = 100,
    min_test_samples: int = 10
) -> List[Dict[Any, Any]]:
    """
    DEPRECATED: Use WalkForwardSplitter from feature_selection.walkforward.walkforward_model instead.
    
    This function is kept for backward compatibility only.
    """
    import warnings
    warnings.warn(
        "generate_walkforward_splits() is deprecated. "
        "Use WalkForwardSplitter from feature_selection.walkforward.walkforward_model instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    # Use canonical WalkForwardSplitter
    splitter = WalkForwardSplitter(
        train_start=train_start,
        train_end=train_end,
        test_step=test_step,
        num_steps=num_steps
    )
    
    # Get splits as (train_indices, test_indices) tuples
    index_splits = splitter.split(df.index)
    
    # Convert to old API format (with masks and metadata) for backward compatibility
    splits = []
    for step, (train_idx, test_idx) in enumerate(index_splits):
        train_dates = df.index[train_idx]
        test_dates = df.index[test_idx]
        
        # Create boolean masks
        train_mask = pd.Series(False, index=df.index)
        train_mask.iloc[train_idx] = True
        test_mask = pd.Series(False, index=df.index)
        test_mask.iloc[test_idx] = True
        
        # Check minimum samples
        if len(train_idx) < min_train_samples or len(test_idx) < min_test_samples:
            continue
        
        splits.append({
            'step': step,
            'train_start': train_dates.min(),
            'train_end': train_dates.max(),
            'test_start': test_dates.min(),
            'test_end': test_dates.max(),
            'train_mask': train_mask,
            'test_mask': test_mask,
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        })
    
    return splits


# All other functions are directly imported from canonical module
# No need to redefine them here
__all__ = [
    'WalkForwardSplitter',
    'generate_walkforward_splits',
    'generate_rolling_windows',
    'apply_function_to_walkforward',
    'apply_function_to_rolling_windows'
]
