"""
Feature Validator - Walkforward Testing Framework

This module provides a comprehensive walkforward testing framework for validating
trading features and portfolios. It reuses existing infrastructure (Portfolio,
PortfolioTester, WalkForwardSplitter) to maintain DRY principles.

Key features:
- Feature-level testing: Test individual features or feature variations
- Portfolio-level testing: Test full portfolios using walkforward validation
- Permutation testing: Validate features using two-region shuffling
- Rule-based feature selection: Select best variation using objective metric

Author: Trading Research Team
Date: 2025-01-26
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from datetime import datetime
from copy import deepcopy

from feature_selection.walkforward.walkforward_model import WalkForwardSplitter
from ensemble.portfolio import Portfolio
from ensemble.portfolio_tester import PortfolioTester, calculate_log_returns_from_candles
from utils.permutation_test.permutation_engine import PermutationEngine, FeaturePermutationStrategy

logger = logging.getLogger(__name__)


class FeatureValidator:
    """
    Walkforward portfolio validation framework.
    
    Validates portfolios using walkforward analysis by:
    1. Splitting data into train/test folds using WalkForwardSplitter
    2. For each fold:
       a. Copy the portfolio
       b. Fit portfolio on training data
       c. Generate predictions on test data
       d. Calculate returns and metrics using PortfolioTester
    3. Aggregate results across all folds
    
    The validator is agnostic to portfolio structure - it works with any Portfolio
    instance (single base model or full multi-ensemble portfolio). Users are
    responsible for constructing the portfolio beforehand.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Features dataframe with feature columns
        Columns: feature columns + optional 'ticker' column
        Must have datetime index matching targets_df
    targets_df : pd.DataFrame
        Targets dataframe with target columns
        Columns: raw_return, log_return, log_return_atr, log_return_ewsd, optional 'ticker'
        Must have datetime index matching features_df
    metadata : Optional[Dict[str, Any]], default=None
        Optional metadata about the features
        
    Attributes
    ----------
    features_df : pd.DataFrame
        Features dataframe
    targets_df : pd.DataFrame
        Targets dataframe
    feature_names : List[str]
        List of feature column names (excluding 'ticker')
    n_features : int
        Number of features
    n_samples : int
        Number of samples
    date_range : Tuple[datetime, datetime]
        Date range of the data
    metadata : Dict[str, Any]
        Feature metadata
    results : Dict[str, Any]
        Dictionary storing all test results
        
    Examples
    --------
    >>> # Create portfolio (single-feature or multi-ensemble)
    >>> portfolio = Portfolio(ensembles=[my_ensemble], ...)
    >>> 
    >>> # Initialize validator
    >>> validator = FeatureValidator(features_df, targets_df)
    >>> 
    >>> # Run walkforward test
    >>> fold_results, summary_df, aggregate_metrics = validator.walkforward_test(
    ...     portfolio=portfolio,
    ...     candles_df=candles_df,
    ...     train_start=datetime(2010, 1, 1),
    ...     train_end=datetime(2015, 1, 1)
    ... )
    """
    
    def __init__(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize FeatureValidator with features and targets dataframes."""
        # Validate inputs
        if not isinstance(features_df, pd.DataFrame):
            raise TypeError("features_df must be a pandas DataFrame")
        if not isinstance(targets_df, pd.DataFrame):
            raise TypeError("targets_df must be a pandas DataFrame")
        
        # Check that indices match
        if not features_df.index.equals(targets_df.index):
            raise ValueError(
                "features_df and targets_df must have matching indices. "
                "Use FeatureExtractor to ensure proper alignment."
            )
        
        # Store dataframes
        self.features_df = features_df
        self.targets_df = targets_df
        self.metadata = metadata or {}
        self.results = {}
        
        # Extract feature names (exclude 'ticker' column)
        self.feature_names = [
            col for col in features_df.columns 
            if col != 'ticker'
        ]
        self.n_features = len(self.feature_names)
        
        # Store basic statistics
        self.n_samples = len(features_df)
        self.date_range = (features_df.index.min(), features_df.index.max())
        
        # Check if multi-ticker
        self.has_ticker = 'ticker' in features_df.columns
        if self.has_ticker:
            self.tickers = features_df['ticker'].unique().tolist()
            self.n_tickers = len(self.tickers)
        else:
            self.tickers = None
            self.n_tickers = 1
        
        # Create combined dataframe for internal use
        self.df = pd.concat([features_df, targets_df], axis=1)
        # Remove duplicate columns if any
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]
    
    def walkforward_test(
        self,
        portfolio: Portfolio,
        candles_df: pd.DataFrame,
        train_start: datetime,
        train_end: datetime,
        test_step: int = 252,
        num_steps: int = 10,
        target_col: str = 'log_return',
        verbose: bool = True
    ) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, float]]:
        """
        Perform walkforward validation on a portfolio.
        
        The validator is agnostic to portfolio structure - it works with any Portfolio
        instance (single base model or full multi-ensemble portfolio). For each fold,
        it copies the portfolio, fits on training data, predicts on test data, and
        computes metrics.
        
        Parameters
        ----------
        portfolio : Portfolio
            Portfolio instance to test (required). Can be:
            - Minimal portfolio with single ensemble + single base model
            - Full portfolio with multiple ensembles and custom configurations
        candles_df : pd.DataFrame
            Candles DataFrame with OHLC data (required for fitting/prediction)
        train_start : datetime
            Start date for initial training window
        train_end : datetime
            End date for initial training window
        test_step : int, default=252
            Number of days for test period (~1 year)
        num_steps : int, default=10
            Number of walkforward steps
        target_col : str, default='log_return'
            Target column to use (only for validation, not used in fitting)
        verbose : bool, default=True
            Print progress
            
        Returns
        -------
        Tuple[List[Dict], pd.DataFrame, Dict]
            (fold_results, summary_df, aggregate_metrics)
            - fold_results: List of per-fold result dictionaries
            - summary_df: Summary DataFrame with one row per fold
            - aggregate_metrics: Aggregated metrics across all folds
        """
        # Validate inputs
        if not isinstance(portfolio, Portfolio):
            raise TypeError("portfolio must be a Portfolio instance")
        
        if candles_df is None or not isinstance(candles_df, pd.DataFrame):
            raise ValueError("candles_df must be a pandas DataFrame")
        
        # Validate target column
        if target_col not in self.targets_df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found. "
                f"Available: {list(self.targets_df.columns)}"
            )
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"WALKFORWARD VALIDATION TEST")
            print(f"{'='*70}")
            print(f"Portfolio: {len(portfolio.ensembles)} ensemble(s)")
            print(f"Target: {target_col}")
            print(f"Steps: {num_steps}")
            print(f"{'='*70}")
        
        # Create splitter
        splitter = WalkForwardSplitter(
            train_start=train_start,
            train_end=train_end,
            test_step=test_step,
            num_steps=num_steps
        )
        
        # Get splits
        splits = splitter.split(self.df.index)
        
        if len(splits) == 0:
            raise ValueError("No valid walkforward splits found")
        
        if verbose:
            print(f"\nGenerated {len(splits)} valid walkforward splits")
        
        # Process each fold
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            if verbose:
                print(f"\n{'='*70}")
                print(f"Fold {fold + 1}/{len(splits)}")
                print(f"{'='*70}")
            
            # Get train/test dates
            train_dates = self.df.index[train_idx]
            test_dates = self.df.index[test_idx]
            
            if verbose:
                print(f"Train: {train_dates.min().date()} to {train_dates.max().date()} (n={len(train_idx)})")
                print(f"Test: {test_dates.min().date()} to {test_dates.max().date()} (n={len(test_idx)})")
            
            try:
                # Copy portfolio to avoid modifying original
                fold_portfolio = self._copy_portfolio(portfolio)
                
                # Get train/test candles
                train_candles = self._filter_candles(candles_df, train_dates)
                test_candles = self._filter_candles(candles_df, test_dates)
                
                if verbose:
                    print(f"Train candles: {len(train_candles)} rows")
                    print(f"Test candles: {len(test_candles)} rows")
                
                # Fit portfolio on training data
                train_targets = calculate_log_returns_from_candles(train_candles)
                fold_portfolio.fit_from_candles(train_candles, train_targets)
                
                if verbose:
                    print(f"✓ Portfolio fitted")
                
                # Generate predictions on test data
                positions_df = fold_portfolio.predict_from_candles(test_candles)
                
                if verbose:
                    print(f"✓ Predictions generated: {len(positions_df)} positions")
                
                # Calculate returns and metrics
                tester = PortfolioTester(fold_portfolio)
                tester.positions_df = positions_df
                strategy_returns = tester.calculate_strategy_returns(test_candles)
                baseline_returns = tester.calculate_baseline_returns(test_candles)
                
                if verbose:
                    print(f"✓ Returns calculated: {len(strategy_returns)} days")
                
                # Extract metrics
                metrics = self._extract_metrics_from_returns(
                    strategy_returns, baseline_returns
                )
                
                if verbose:
                    print(f"✓ Metrics computed: Sharpe={metrics.get('sharpe_ratio', 0):.4f}")
                
                # Store fold results
                fold_results.append({
                    'fold': fold,
                    'train_start': train_dates.min(),
                    'train_end': train_dates.max(),
                    'test_start': test_dates.min(),
                    'test_end': test_dates.max(),
                    'n_train': len(train_idx),
                    'n_test': len(test_idx),
                    'positions_df': positions_df,
                    'strategy_returns': strategy_returns,
                    'baseline_returns': baseline_returns,
                    'metrics': metrics
                })
                
            except Exception as e:
                logger.error(f"Error in fold {fold}: {e}", exc_info=True)
                if verbose:
                    print(f"✗ Fold {fold + 1} failed: {e}")
                continue
        
        if len(fold_results) == 0:
            raise ValueError("All folds failed. Check data and parameters.")
        
        # Create summary DataFrame
        summary_rows = []
        for fold_result in fold_results:
            summary_rows.append({
                'fold': fold_result['fold'],
                'train_start': fold_result['train_start'],
                'train_end': fold_result['train_end'],
                'test_start': fold_result['test_start'],
                'test_end': fold_result['test_end'],
                'n_train': fold_result['n_train'],
                'n_test': fold_result['n_test'],
                **fold_result['metrics']
            })
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Compute aggregate metrics
        all_strategy_returns = pd.concat([r['strategy_returns'] for r in fold_results])
        all_baseline_returns = pd.concat([r['baseline_returns'] for r in fold_results])
        aggregate_metrics = self._extract_metrics_from_returns(
            all_strategy_returns, all_baseline_returns
        )
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"AGGREGATE RESULTS")
            print(f"{'='*70}")
            print(f"Total folds: {len(fold_results)}")
            print(f"Total trading days: {len(all_strategy_returns)}")
            for key, value in aggregate_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.4f}")
            print(f"{'='*70}")
        
        # Store results
        self.results['walkforward_test'] = {
            'fold_results': fold_results,
            'summary_df': summary_df,
            'aggregate_metrics': aggregate_metrics,
            'train_start': train_start,
            'train_end': train_end,
            'test_step': test_step,
            'num_steps': num_steps
        }
        
        return fold_results, summary_df, aggregate_metrics
    
    def walkforward_permutation_test(
        self,
        portfolio: Portfolio,
        candles_df: pd.DataFrame,
        train_start: datetime,
        train_end: datetime,
        test_step: int = 252,
        num_steps: int = 10,
        target_col: str = 'log_return',
        nreps: int = 100,
        alpha: float = 0.05,
        random_seed: Optional[int] = 42,
        n_jobs: int = -1,
        shuffle_target: bool = False,
        exclude_features: Optional[List[str]] = None,
        verbose: bool = True,
        **walkforward_kwargs
    ) -> pd.DataFrame:
        """
        Perform walkforward permutation test.
        
        Compares walkforward performance on real data vs permuted data.
        Uses two-region shuffling: first training window vs remaining data.
        
        Parameters
        ----------
        portfolio : Portfolio
            Portfolio instance to test (required)
        candles_df : pd.DataFrame
            Candles DataFrame with OHLC data
        train_start : datetime
            Start date for initial training window
        train_end : datetime
            End date for initial training window
        test_step : int, default=252
            Number of days for test period
        num_steps : int, default=10
            Number of walkforward steps
        target_col : str, default='log_return'
            Target column to use
        nreps : int, default=100
            Number of permutation replications
        alpha : float, default=0.05
            Significance level
        random_seed : int, optional
            Random seed for reproducibility
        n_jobs : int, default=-1
            Number of parallel jobs (-1 = all CPUs)
        shuffle_target : bool, default=False
            If True, shuffle target values instead of features
        exclude_features : List[str], optional
            Features to NOT shuffle (e.g., ATR for normalization)
        verbose : bool, default=True
            Print progress
        **walkforward_kwargs
            Additional arguments passed to walkforward_test()
            
        Returns
        -------
        pd.DataFrame
            Results with columns:
            - feature: Feature/portfolio identifier
            - original_criterion: Original walkforward metric value
            - pval: Permutation p-value
            - significant: Boolean (pval <= alpha)
            - n_valid_permutations: Number of successful permutations
        """
        # Validate inputs
        if not isinstance(portfolio, Portfolio):
            raise TypeError("portfolio must be a Portfolio instance")
        
        if candles_df is None or not isinstance(candles_df, pd.DataFrame):
            raise ValueError("candles_df must be a pandas DataFrame")
        
        # Validate target column
        if target_col not in self.targets_df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found. "
                f"Available: {list(self.targets_df.columns)}"
            )
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"WALKFORWARD PERMUTATION TEST")
            print(f"{'='*70}")
            print(f"Portfolio: {len(portfolio.ensembles)} ensemble(s)")
            print(f"Shuffle target: {shuffle_target}")
            print(f"Replications: {nreps}")
            print(f"Alpha: {alpha}")
            print(f"Parallel workers: {n_jobs if n_jobs != -1 else 'all'}")
            print(f"{'='*70}")
        
        # Prepare combined DataFrame
        combined_df = pd.concat([self.features_df, self.targets_df], axis=1)
        combined_df = combined_df.dropna()
        
        # Determine data end date
        data_end = combined_df.index.max()
        
        # Create two-region train_windows for permutation
        train_windows = [
            (train_start, train_end),  # First training window
            (train_end, data_end)      # All remaining data
        ]
        
        if verbose:
            print(f"\nTwo-region permutation:")
            print(f"  Region 1: {train_start.date()} to {train_end.date()}")
            print(f"  Region 2: {train_end.date()} to {data_end.date()}")
        
        # Determine what to shuffle
        if shuffle_target:
            shuffle_cols = [target_col]
            exclude_features = None
        else:
            # Shuffle all features (portfolio will use what it needs)
            shuffle_cols = self.feature_names
        
        # Create criterion function
        criterion_func = self._create_walkforward_criterion_func(
            train_start=train_start,
            train_end=train_end,
            test_step=test_step,
            num_steps=num_steps,
            target_col=target_col,
            portfolio=portfolio,
            candles_df=candles_df,
            verbose=verbose,
            **walkforward_kwargs
        )
        
        # Create permutation strategy and engine
        strategy = FeaturePermutationStrategy()
        engine = PermutationEngine(strategy, n_jobs=n_jobs, verbose=verbose)
        
        # Run permutation test
        results = engine.run_permutation_test(
            data=combined_df,
            feature_cols=shuffle_cols,
            criterion_func=criterion_func,
            nreps=nreps,
            random_seed=random_seed,
            alpha=alpha,
            target_col=target_col,
            train_windows=train_windows,
            exclude_features=exclude_features
        )
        
        # Add metadata
        results['permutation_mode'] = 'walkforward_two_region'
        results['shuffle_target'] = shuffle_target
        results['n_valid_permutations'] = nreps
        
        # Store results
        self.results['walkforward_permutation_test'] = {
            'results': results,
            'nreps': nreps,
            'alpha': alpha,
            'shuffle_target': shuffle_target
        }
        
        if verbose:
            n_significant = results['significant'].sum()
            print(f"\n{'='*70}")
            print(f"PERMUTATION TEST SUMMARY")
            print(f"{'='*70}")
            print(f"Significant (p <= {alpha}): {n_significant}/{len(results)}")
            if n_significant > 0:
                sig_results = results[results['significant']].copy()
                sig_results = sig_results.sort_values('original_criterion', ascending=False)
                print(f"\nSignificant features:")
                print(sig_results[['feature', 'original_criterion', 'pval']].to_string(index=False))
            print(f"{'='*70}")
        
        return results
    
    def _copy_portfolio(self, portfolio: Portfolio) -> Portfolio:
        """
        Create a copy of portfolio to avoid modifying original.
        
        Note: We create a shallow copy of the portfolio structure.
        The portfolio will be re-fitted anyway, so we don't need deep copy of models.
        """
        # Create new Portfolio with same configuration
        new_portfolio = Portfolio(
            ensembles=portfolio.ensembles.copy(),  # Shallow copy of ensembles list
            trading_timeframe=portfolio.trading_timeframe,
            target_volatility=portfolio.target_volatility,
            max_position_pct=portfolio.max_position_pct,
            weight_layer=portfolio.weight_layer,  # Share weight layer
            instrument_weights=portfolio.instrument_weights,
            idm_max=portfolio.idm_max
        )
        
        return new_portfolio
    
    def _filter_candles(
        self,
        candles_df: pd.DataFrame,
        date_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Filter candles DataFrame to date range.
        
        Parameters
        ----------
        candles_df : pd.DataFrame
            Full candles DataFrame
        date_index : pd.DatetimeIndex
            Date index to filter to
            
        Returns
        -------
        pd.DataFrame
            Filtered candles
        """
        # Convert datetime column to datetime if needed
        candles = candles_df.copy()
        candles['datetime'] = pd.to_datetime(candles['datetime'])
        
        # Get date range
        min_date = date_index.min()
        max_date = date_index.max()
        
        # Handle timezone compatibility
        if candles['datetime'].dt.tz is not None:
            if min_date.tzinfo is None:
                min_date = min_date.replace(tzinfo=candles['datetime'].dt.tz[0])
                max_date = max_date.replace(tzinfo=candles['datetime'].dt.tz[0])
        else:
            if min_date.tzinfo is not None:
                min_date = min_date.replace(tzinfo=None)
                max_date = max_date.replace(tzinfo=None)
        
        # Filter
        mask = (candles['datetime'] >= min_date) & (candles['datetime'] <= max_date)
        return candles[mask].copy()
    
    def _extract_metrics_from_returns(
        self,
        strategy_returns: pd.Series,
        baseline_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Extract metrics from strategy and baseline returns.
        
        Uses quantstats internally via generate_tearsheet with mode='metrics'.
        """
        # Import quantstats for metrics calculation
        try:
            import quantstats as qs
            
            # Calculate metrics
            metrics = {}
            
            # Sharpe ratio
            metrics['sharpe_ratio'] = qs.stats.sharpe(strategy_returns, rf=0)
            
            # Sortino ratio
            metrics['sortino_ratio'] = qs.stats.sortino(strategy_returns, rf=0)
            
            # Total return
            metrics['total_return'] = qs.stats.comp(strategy_returns)
            
            # Max drawdown
            metrics['max_drawdown'] = qs.stats.max_drawdown(strategy_returns)
            
            # CAGR
            metrics['cagr'] = qs.stats.cagr(strategy_returns)
            
            # Calmar ratio
            metrics['calmar_ratio'] = qs.stats.calmar(strategy_returns)
            
            # Win rate
            metrics['win_rate'] = (strategy_returns > 0).sum() / len(strategy_returns) if len(strategy_returns) > 0 else 0.0
            
            # Number of trades
            metrics['n_trades'] = len(strategy_returns)
            
            # Replace NaN/inf with 0.0
            for key in metrics:
                if not isinstance(metrics[key], (int, float)) or np.isnan(metrics[key]) or np.isinf(metrics[key]):
                    metrics[key] = 0.0
            
            return metrics
            
        except ImportError:
            logger.warning("quantstats not available, using simple metrics")
            # Fallback: simple metrics
            return {
                'sharpe_ratio': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0.0,
                'total_return': (1 + strategy_returns).prod() - 1,
                'n_trades': len(strategy_returns),
                'win_rate': (strategy_returns > 0).sum() / len(strategy_returns) if len(strategy_returns) > 0 else 0.0
            }
    
    def _create_walkforward_criterion_func(
        self,
        train_start: datetime,
        train_end: datetime,
        test_step: int,
        num_steps: int,
        target_col: str,
        portfolio: Portfolio,
        candles_df: pd.DataFrame,
        verbose: bool,
        **walkforward_kwargs
    ) -> Callable:
        """
        Create criterion function for permutation test.
        
        The criterion function runs walkforward_test on permuted data and returns
        the aggregated metric (Sharpe ratio).
        """
        def criterion_func(data: pd.DataFrame, feature_col: str) -> float:
            """
            Run walkforward test and return aggregated metric.
            
            Args:
                data: DataFrame with features and targets (ALREADY PERMUTED)
                feature_col: Feature column name (or 'portfolio')
            
            Returns:
                Aggregated Sharpe ratio across all folds
            """
            # Extract features and targets from permuted data
            features_df = data.drop(columns=[target_col], errors='ignore').copy()
            targets_df = data[[target_col]].copy()
            
            # Temporarily replace self data with permuted data
            original_features = self.features_df
            original_targets = self.targets_df
            
            try:
                self.features_df = features_df
                self.targets_df = targets_df
                
                # Run walkforward test
                fold_results, summary_df, aggregate_metrics = self.walkforward_test(
                    portfolio=portfolio,
                    candles_df=candles_df,
                    train_start=train_start,
                    train_end=train_end,
                    test_step=test_step,
                    num_steps=num_steps,
                    target_col=target_col,
                    verbose=False,  # Suppress verbose output in permutations
                    **walkforward_kwargs
                )
                
                # Return Sharpe ratio as primary metric
                return aggregate_metrics.get('sharpe_ratio', 0.0)
                
            except Exception as e:
                logger.warning(f"Walkforward test failed on permuted data: {e}")
                return 0.0
                
            finally:
                # Restore original data
                self.features_df = original_features
                self.targets_df = original_targets
        
        return criterion_func
    
    def get_results(self, test_name: str) -> Dict[str, Any]:
        """
        Retrieve stored results for a specific test.
        
        Parameters
        ----------
        test_name : str
            Name of the test ('walkforward_test', 'walkforward_permutation_test')
            
        Returns
        -------
        Dict[str, Any]
            The dictionary stored in self.results[test_name]
        """
        if test_name not in self.results:
            raise ValueError(
                f"Test '{test_name}' not found. "
                f"Available: {list(self.results.keys())}"
            )
        return self.results[test_name]
    
    def __repr__(self) -> str:
        """String representation of the FeatureValidator."""
        return (
            f"FeatureValidator(n_features={self.n_features}, "
            f"n_samples={self.n_samples}, "
            f"date_range={self.date_range[0].date()} to {self.date_range[1].date()})"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [f"FeatureValidator with {self.n_features} features:"]
        
        # Show first few features
        for feature_name in self.feature_names[:5]:
            lines.append(f"  - {feature_name}")
        
        if self.n_features > 5:
            lines.append(f"  ... and {self.n_features - 5} more")
        
        lines.append(f"\nSamples: {self.n_samples}")
        lines.append(f"Date range: {self.date_range[0].date()} to {self.date_range[1].date()}")
        
        if self.has_ticker:
            lines.append(f"Tickers: {self.tickers}")
        
        lines.append(f"Tests run: {list(self.results.keys())}")
        
        return '\n'.join(lines)
