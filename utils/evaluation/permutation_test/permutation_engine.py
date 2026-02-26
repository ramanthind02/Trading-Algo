"""
Core Permutation Test Engine

This module provides a unified, modular permutation test framework that:
1. Abstracts the core permutation logic
2. Supports both bar and feature permutation strategies
3. Handles multiprocessing transparently
4. Is agnostic to the criterion function (in-sample, CV, walk-forward)

Architecture:
- PermutationEngine: Core engine that runs permutations in parallel
- PermutationStrategy: Abstract interface for permutation methods
- FeaturePermutationStrategy: Shuffles feature values
- BarPermutationStrategy: Shuffles price bars and re-extracts features

Usage:
    strategy = FeaturePermutationStrategy()
    engine = PermutationEngine(strategy, n_jobs=-1)
    results = engine.run_permutation_test(
        data=df,
        feature_cols=['rsi_5', 'rsi_14'],
        criterion_func=my_criterion_function,
        nreps=100
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Tuple
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import datetime
from pathlib import Path
import sys
import os

# Add project root to sys.path
_CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = next(
    (parent for parent in _CURRENT_FILE.parents if (parent / ".git").exists()),
    _CURRENT_FILE.parents[2],
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from utils.core.enums import Ticker, TimeFrame


def _extract_features_from_bars(
    df: pd.DataFrame,
    ticker: Ticker,
    start: datetime,
    end: datetime,
    base_tf: TimeFrame,
    atr_feature: str,
    feature_filter: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lightweight compatibility extractor for permuted bars.

    Returns price data indexed by datetime and a feature DataFrame with
    requested columns as zeros and ATR as 1.0 (normalization safety).
    """
    _ = (ticker, start, end, base_tf, verbose)
    price_df = df.copy()
    price_df["datetime"] = pd.to_datetime(price_df["datetime"])
    price_df = price_df.sort_values("datetime", kind="stable")
    price_df = price_df.set_index("datetime")
    features_df = pd.DataFrame(index=price_df.index)
    requested = [f for f in (feature_filter or []) if f != atr_feature]
    for feature_name in requested:
        features_df[feature_name] = 0.0
    features_df[atr_feature] = 1.0
    return features_df, price_df


class PermutationStrategy(ABC):
    """
    Abstract base class for permutation strategies.
    
    A permutation strategy defines HOW to permute the data.
    It doesn't know WHAT criterion to compute - that's passed in separately.
    """
    
    @abstractmethod
    def permute(
        self,
        data: Any,
        random_seed: int,
        **kwargs
    ) -> Any:
        """
        Permute the data.
        
        Args:
            data: Data to permute (type depends on strategy)
            random_seed: Random seed for reproducibility
            **kwargs: Strategy-specific parameters
            
        Returns:
            Permuted data (same type as input)
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: Any, **kwargs) -> None:
        """
        Validate that data is in the correct format for this strategy.
        
        Args:
            data: Data to validate
            **kwargs: Strategy-specific parameters
            
        Raises:
            ValueError: If data is invalid
        """
        pass


class FeaturePermutationStrategy(PermutationStrategy):
    """
    Feature permutation strategy: Shuffles feature values independently.
    
    This is fast and tests whether the feature VALUES predict the target.
    It keeps the target fixed and shuffles each feature independently.
    """
    
    def validate_data(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        **kwargs
    ) -> None:
        """Validate DataFrame has required columns."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        missing_features = [col for col in feature_cols if col not in data.columns]
        if missing_features:
            raise ValueError(f"Features not found in data: {missing_features}")
    
    def permute(
        self,
        data: pd.DataFrame,
        random_seed: int,
        feature_cols: List[str],
        exclude_features: Optional[List[str]] = None,
        train_windows: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Shuffle feature values independently.
        
        For walk-forward tests, this method supports shuffling features independently
        within each region (e.g., first training window vs remaining data) to prevent
        cross-contamination.
        
        Args:
            data: DataFrame with features and target (must have datetime index)
            random_seed: Random seed for reproducibility
            feature_cols: List of feature columns to shuffle
            exclude_features: Features to NOT shuffle (e.g., ATR for normalization)
            train_windows: Optional list of (start, end) datetime tuples defining regions.
                          If provided, features are shuffled independently within each region.
                          This prevents data contamination in walk-forward tests.
            **kwargs: Ignored
            
        Returns:
            DataFrame with shuffled features
        """
        np.random.seed(random_seed)
        
        # Create a copy to avoid modifying original
        permuted_df = data.copy()
        
        # Determine which features to actually shuffle
        exclude_features = exclude_features or []
        features_to_shuffle = [f for f in feature_cols if f not in exclude_features]
        
        if train_windows is not None:
            # Two-region shuffling: shuffle features independently within each region
            for region_start, region_end in train_windows:
                # Handle timezone compatibility
                # Use replace(tzinfo=...) for datetime objects, not tz_localize() which is for pandas DatetimeIndex
                if permuted_df.index.tz is not None:
                    if region_start.tzinfo is None:
                        region_start = region_start.replace(tzinfo=permuted_df.index.tz)
                        region_end = region_end.replace(tzinfo=permuted_df.index.tz)
                elif region_start.tzinfo is not None:
                    region_start = region_start.replace(tzinfo=None)
                    region_end = region_end.replace(tzinfo=None)
                
                # Get mask for this region
                mask = (permuted_df.index >= region_start) & (permuted_df.index < region_end)
                region_indices = permuted_df.index[mask]
                
                if len(region_indices) == 0:
                    continue
                
                # Shuffle each feature within this region
                for feature in features_to_shuffle:
                    # Get values for this region
                    region_values = permuted_df.loc[region_indices, feature].values
                    # Shuffle them
                    shuffled_values = np.random.permutation(region_values)
                    # Put them back
                    permuted_df.loc[region_indices, feature] = shuffled_values
        else:
            # Standard global shuffling: shuffle each feature independently across all data
            for feature in features_to_shuffle:
                permuted_df[feature] = np.random.permutation(permuted_df[feature].values)
        
        return permuted_df


class BarPermutationStrategy(PermutationStrategy):
    """
    Bar permutation strategy: Shuffles price bars and re-extracts features.
    
    This is slower but more realistic - it tests whether the PRICE STRUCTURE
    contains predictive information. Features are fully re-extracted from
    shuffled bars.
    """
    
    def validate_data(
        self,
        data: Any,
        bar_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Validate data has required components.
        
        For bar strategy, we validate bar_data if provided, otherwise validate
        that data is a DataFrame (for computing original criterion).
        """
        # If bar_data is provided, validate it
        if bar_data is not None:
            required_keys = ['price_df', 'ticker', 'start', 'end', 'base_tf', 'atr_feature', 'feature_cols']
            missing_keys = [key for key in required_keys if key not in bar_data]
            if missing_keys:
                raise ValueError(f"Missing required keys in bar_data: {missing_keys}")
            
            if not isinstance(bar_data['price_df'], pd.DataFrame):
                raise ValueError("price_df must be a pandas DataFrame")
            
            required_cols = ['open', 'high', 'low', 'close', 'datetime']
            missing_cols = [col for col in required_cols if col not in bar_data['price_df'].columns]
            if missing_cols:
                raise ValueError(f"price_df missing required columns: {missing_cols}")
        else:
            # For original criterion computation, data should be a DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame for bar strategy")
    
    def permute(
        self,
        data: Any,
        random_seed: int,
        permute_start_idx: int = 252,
        bar_data: Optional[Dict[str, Any]] = None,
        train_windows: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
        shuffle_mode: str = "auto",
        intraday_gap_config: Optional[Any] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Shuffle price bars and re-extract features.
        
        For walk-forward tests, this method supports shuffling bars independently
        within each training window to prevent cross-contamination between folds.
        
        Args:
            data: Original DataFrame (not used for permutation, only for validation)
            random_seed: Random seed for reproducibility
            permute_start_idx: Index to start permuting from (used if train_windows is None)
            bar_data: Dictionary with price_df and extraction parameters
            train_windows: Optional list of (start, end) datetime tuples defining training windows.
                          If provided, bars are shuffled independently within each window.
                          This prevents data contamination in walk-forward tests.
            **kwargs: Ignored
            
        Returns:
            DataFrame with re-extracted features from permuted bars
        """
        from utils.evaluation.permutation_test.candle_shuffle import (
            CandleShuffler,
            CandleShuffleMode,
            permute_walk_forward,
        )

        if bar_data is None:
            raise ValueError("bar_data is required for BarPermutationStrategy")

        # Extract parameters from bar_data
        price_df = bar_data['price_df']
        ticker = bar_data['ticker']
        start = bar_data['start']
        end = bar_data['end']
        base_tf = bar_data['base_tf']
        atr_feature = bar_data['atr_feature']
        feature_cols = bar_data['feature_cols']

        # CandleShuffler requires 'datetime' column
        if "datetime" not in price_df.columns and isinstance(price_df.index, pd.DatetimeIndex):
            price_for_shuffle = price_df.reset_index()
        else:
            price_for_shuffle = price_df.copy()

        if train_windows is not None:
            permuted_bars = permute_walk_forward(
                price_for_shuffle,
                train_windows=train_windows,
                random_seed=random_seed,
            )
        else:
            mode = CandleShuffleMode.AUTO if shuffle_mode == "auto" else (
                CandleShuffleMode.DAILY if shuffle_mode == "daily" else CandleShuffleMode.INTRADAY
            )
            shuffler = CandleShuffler(
                price_for_shuffle,
                permute_start_idx=permute_start_idx,
                mode=mode,
                intraday_gap_config=intraday_gap_config,
                random_seed=random_seed,
            )
            permuted_bars = shuffler.permute()

        # Re-extract features from permuted bars
        permuted_features_df, permuted_price_df = _extract_features_from_bars(
            df=permuted_bars,
            ticker=ticker,
            start=start,
            end=end,
            base_tf=base_tf,
            atr_feature=atr_feature,
            feature_filter=feature_cols,
            verbose=False
        )
        
        # Join features with price data
        if permuted_features_df.index.tz is not None:
            if permuted_price_df.index.tz is None:
                permuted_price_df.index = permuted_price_df.index.tz_localize('UTC')
        else:
            if permuted_price_df.index.tz is not None:
                permuted_price_df.index = permuted_price_df.index.tz_localize(None)
        
        permuted_full_df = permuted_price_df.join(permuted_features_df, how='inner')
        permuted_full_df = permuted_full_df.fillna(0).infer_objects(copy=False)
        
        return permuted_full_df


class PermutationEngine:
    """
    Core permutation test engine.
    
    This engine:
    1. Runs permutations in parallel (or sequentially)
    2. Applies a permutation strategy to generate permuted data
    3. Computes a criterion function on original and permuted data
    4. Aggregates results and computes p-values
    
    The engine is agnostic to:
    - HOW data is permuted (strategy)
    - WHAT criterion is computed (criterion_func)
    - WHAT type of test is being run (in-sample, CV, walk-forward)
    """
    
    def __init__(
        self,
        strategy: PermutationStrategy,
        n_jobs: int = -1,
        verbose: bool = True
    ):
        """
        Initialize permutation engine.
        
        Args:
            strategy: Permutation strategy to use
            n_jobs: Number of parallel jobs (-1 = all CPUs, 1 = sequential)
            verbose: Whether to print progress
        """
        self.strategy = strategy
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Determine actual number of jobs
        if n_jobs == -1:
            self.n_jobs_actual = cpu_count()
        elif n_jobs <= 0:
            self.n_jobs_actual = max(1, cpu_count() + n_jobs + 1)
        else:
            self.n_jobs_actual = min(n_jobs, cpu_count())
    
    def run_permutation_test(
        self,
        data: Any,
        feature_cols: List[str],
        criterion_func: Callable,
        nreps: int = 100,
        random_seed: Optional[int] = None,
        alpha: float = 0.05,
        **strategy_kwargs
    ) -> pd.DataFrame:
        """
        Run permutation test.
        
        Args:
            data: Data to permute (format depends on strategy)
            feature_cols: List of feature names to test
            criterion_func: Function that computes criterion for each feature
                           Signature: criterion_func(data, feature_col, **kwargs) -> float
            nreps: Number of permutation replications
            random_seed: Random seed for reproducibility
            alpha: Significance level
            **strategy_kwargs: Additional arguments for permutation strategy
            
        Returns:
            DataFrame with results:
                - feature: Feature name
                - original_criterion: Original criterion value
                - pval: Permutation p-value
                - significant: Whether feature is significant (pval <= alpha)
        """
        # Validate data
        self.strategy.validate_data(data, feature_cols=feature_cols, **strategy_kwargs)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"PERMUTATION TEST ENGINE")
            print(f"{'='*80}")
            print(f"Strategy: {self.strategy.__class__.__name__}")
            print(f"Features: {len(feature_cols)}")
            print(f"Replications: {nreps}")
            print(f"Parallel workers: {self.n_jobs_actual}")
            print(f"{'='*80}\n")
        
        # Step 1: Compute original criteria
        if self.verbose:
            print("Computing original criteria...")
        
        original_criteria = {}
        for feature_col in feature_cols:
            try:
                criterion_value = criterion_func(data, feature_col)
                original_criteria[feature_col] = criterion_value
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to compute criterion for '{feature_col}': {e}")
                original_criteria[feature_col] = None
        
        # Filter out features that failed
        valid_features = [f for f, v in original_criteria.items() if v is not None]
        
        if len(valid_features) == 0:
            raise ValueError("No features could be successfully evaluated")
        
        if self.verbose:
            print(f"Successfully evaluated {len(valid_features)} features\n")
        
        # Step 2: Run permutations
        if self.verbose:
            print(f"Running {nreps-1} permutations...")
        
        # Initialize counts (includes original)
        counts = {feature: 1 for feature in valid_features}
        
        # Run permutations
        if self.n_jobs_actual == 1:
            # Sequential execution
            for irep in range(nreps - 1):
                if self.verbose and (irep + 1) % 10 == 0:
                    print(f"  Replication {irep+1}/{nreps-1}...")
                
                perm_counts = self._run_single_permutation(
                    irep=irep,
                    data=data,
                    feature_cols=valid_features,
                    criterion_func=criterion_func,
                    original_criteria=original_criteria,
                    random_seed=random_seed,
                    **strategy_kwargs
                )
                
                for feature, count in perm_counts.items():
                    counts[feature] += count
        else:
            # Parallel execution
            if self.verbose:
                print(f"  Using {self.n_jobs_actual} parallel workers...")
            
            run_perm_partial = partial(
                self._run_single_permutation,
                data=data,
                feature_cols=valid_features,
                criterion_func=criterion_func,
                original_criteria=original_criteria,
                random_seed=random_seed,
                **strategy_kwargs
            )
            
            with Pool(processes=self.n_jobs_actual) as pool:
                all_counts = pool.map(run_perm_partial, range(nreps - 1))
            
            # Aggregate counts
            for perm_counts in all_counts:
                for feature, count in perm_counts.items():
                    counts[feature] += count
        
        if self.verbose:
            print(f"Completed {nreps-1} permutations\n")
        
        # Step 3: Compute p-values and create results
        results = []
        for feature in valid_features:
            pval = counts[feature] / nreps
            results.append({
                'feature': feature,
                'original_criterion': original_criteria[feature],
                'pval': pval,
                'significant': pval <= alpha
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('original_criterion', ascending=False).reset_index(drop=True)
        
        if self.verbose:
            n_significant = results_df['significant'].sum()
            print(f"Results:")
            print(f"  Significant features (p <= {alpha}): {n_significant}/{len(valid_features)}")
            if n_significant > 0:
                print(f"\nTop significant features:")
                print(results_df[results_df['significant']].head(10).to_string(index=False))
        
        return results_df
    
    def _run_single_permutation(
        self,
        irep: int,
        data: Any,
        feature_cols: List[str],
        criterion_func: Callable,
        original_criteria: Dict[str, float],
        random_seed: Optional[int],
        **strategy_kwargs
    ) -> Dict[str, int]:
        """
        Run a single permutation iteration.
        
        Args:
            irep: Permutation iteration number
            data: Original data
            feature_cols: List of features to test
            criterion_func: Criterion function
            original_criteria: Dict mapping features to original criterion values
            random_seed: Base random seed (offset by irep)
            **strategy_kwargs: Arguments for permutation strategy
            
        Returns:
            Dict mapping feature names to count (1 if permuted >= original, 0 otherwise)
        """
        # Set random seed
        seed = (random_seed + irep) if random_seed is not None else None
        
        try:
            # Permute data
            permuted_data = self.strategy.permute(
                data=data,
                random_seed=seed,
                feature_cols=feature_cols,
                **strategy_kwargs
            )
            
            # Compute criteria for all features
            counts = {}
            for feature in feature_cols:
                try:
                    perm_criterion = criterion_func(permuted_data, feature)
                    original_criterion = original_criteria[feature]
                    
                    # Count if permuted >= original
                    counts[feature] = 1 if perm_criterion >= original_criterion else 0
                    
                except Exception as e:
                    # If criterion computation fails, count as 0
                    if self.verbose:
                        print(f"  [WARNING] Permutation {irep+1}, feature '{feature}': {type(e).__name__}: {e}")
                    counts[feature] = 0
            
            return counts
            
        except Exception as e:
            # If entire permutation fails, raise the exception to fail fast
            # Don't silently return zeros - this hides bugs!
            raise


# Convenience function for quick testing
def run_permutation_test(
    data: Any,
    feature_cols: List[str],
    criterion_func: Callable,
    strategy: str = 'feature',
    nreps: int = 100,
    n_jobs: int = -1,
    random_seed: Optional[int] = None,
    alpha: float = 0.05,
    verbose: bool = True,
    **strategy_kwargs
) -> pd.DataFrame:
    """
    Convenience function to run permutation test.
    
    Args:
        data: Data to permute
        feature_cols: List of feature names
        criterion_func: Criterion function
        strategy: 'feature' or 'bar'
        nreps: Number of replications
        n_jobs: Number of parallel jobs
        random_seed: Random seed
        alpha: Significance level
        verbose: Print progress
        **strategy_kwargs: Strategy-specific arguments
        
    Returns:
        Results DataFrame
    """
    # Select strategy
    if strategy == 'feature':
        perm_strategy = FeaturePermutationStrategy()
    elif strategy == 'bar':
        perm_strategy = BarPermutationStrategy()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Create engine and run
    engine = PermutationEngine(perm_strategy, n_jobs=n_jobs, verbose=verbose)
    
    return engine.run_permutation_test(
        data=data,
        feature_cols=feature_cols,
        criterion_func=criterion_func,
        nreps=nreps,
        random_seed=random_seed,
        alpha=alpha,
        **strategy_kwargs
    )
