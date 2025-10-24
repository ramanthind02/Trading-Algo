"""
Feature Explorer Class for Systematic Feature Validation and Analysis

This module provides a comprehensive framework for validating trading features through
rigorous statistical tests, visualizations, and performance analysis. It implements
a professional workflow for moving features from "archive" (candidate) to "live" (production-ready).

The FeatureExplorer class includes methods for:
- Decile/quintile plots
- Time series visualization
- Statistical tests (stationarity, permutation, autocorrelation)
- Performance metrics (IC, monotonicity, turnover)
- Walk-forward analysis
- Correlation analysis
- Report generation

Author: Trading Research Team
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Union, Tuple, Dict, Any
import scipy.stats as stats
from pathlib import Path
import json
from datetime import datetime as dt
import warnings
from utils.enums import Ticker, TimeFrame
from feature_extraction.feature_extractor import extract_bias
import utils.helpers as helpers
from eda.decile_plots import plot_decile_analysis, plot_2bin_analysis
from eda.rolling_decile import plot_rolling_decile_whiskers, plot_rolling_decile_heatmap
from feature_selection.walkforward.walkforward_bin_selection import (
    walkforward_bin_selection, plot_walkforward_results
)


class FeatureExplorer:
    """
    Comprehensive feature validation and exploration framework for trading features.
    
    This class provides a systematic approach to feature engineering by offering
    standardized tests and visualizations that help determine if a feature should
    be promoted from "archive" (candidate) to "live" (production-ready) status.
    
    Parameters
    ----------
    feature_name : str
        Name of the feature being explored
    feature_data : pd.Series
        The feature values (must have datetime index)
    target_data : pd.Series
        The target variable (must have datetime index matching feature_data)
    price_data : Optional[pd.DataFrame], default=None
        Price data (OHLCV) for additional analysis. Should have datetime index.
    metadata : Optional[Dict[str, Any]], default=None
        Additional metadata about the feature (e.g., creation_date, version, notes)
        
    Attributes
    ----------
    feature_name : str
        Name of the feature
    df : pd.DataFrame
        Combined dataframe with feature and target (aligned by index)
    price_data : Optional[pd.DataFrame]
        Price data if provided
    metadata : Dict[str, Any]
        Feature metadata
    results : Dict[str, Any]
        Dictionary storing all analysis results
        
    Examples
    --------
    >>> explorer = FeatureExplorer(
    ...     feature_name="return_zscore_20",
    ...     feature_data=feature_series,
    ...     target_data=target_series,
    ...     price_data=ohlcv_df
    ... )
    >>> 
    >>> # Run full analysis suite
    >>> results = explorer.run_full_analysis(output_dir="./reports/")
    >>> 
    >>> # Or run individual analyses
    >>> decile_figs = explorer.plot_feature_deciles(n_bins=10)
    >>> stats_results = explorer.run_statistical_tests()
    >>> perf_metrics = explorer.compute_performance_metrics()
    """
    
    def __init__(
        self,
        feature_name: str,
        feature_data: pd.Series,
        target_data: pd.Series,
        price_data: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict[str, Any]] = None,
        raw_return: Optional[pd.Series] = None,
        log_return: Optional[pd.Series] = None,
        log_return_atr: Optional[pd.Series] = None,
        log_return_ewsd: Optional[pd.Series] = None
    ):
        """Initialize the FeatureExplorer with feature and target data."""
        self.feature_name = feature_name
        self.price_data = price_data
        self.metadata = metadata or {}
        self.results = {}
        
        # Validate inputs
        if not isinstance(feature_data, pd.Series):
            raise TypeError("feature_data must be a pandas Series")
        if not isinstance(target_data, pd.Series):
            raise TypeError("target_data must be a pandas Series")
        
        # Align feature and target data by index
        df_dict = {
            'feature': feature_data,
            'target': target_data
        }
        
        # Add optional return columns if provided
        if raw_return is not None:
            df_dict['raw_return'] = raw_return
        if log_return is not None:
            df_dict['log_return'] = log_return
        if log_return_atr is not None:
            df_dict['log_return_atr'] = log_return_atr
        if log_return_ewsd is not None:
            df_dict['log_return_ewsd'] = log_return_ewsd
        
        self.df = pd.DataFrame(df_dict).dropna()
        
        if self.df.empty:
            raise ValueError("No valid data after aligning feature and target (all NaN)")
        
        # Store basic statistics
        self.n_samples = len(self.df)
        self.date_range = (self.df.index.min(), self.df.index.max())
        
    def plot_feature_deciles(
        self,
        n_bins: int = 10,
        figsize: Tuple[int, int] = (12, 8),
        plot_type: str = "bar",
        save_path: Optional[str] = None,
        target_col: str = 'target'
    ) -> plt.Figure:
        """
        Create decile plot showing how the target variable changes across different 
        buckets of feature values.
        
        This is one of the most important visualizations for feature validation. A good
        predictive feature should show a monotonic relationship between feature deciles
        and mean target values. For mean reversion features, we expect higher feature
        values to predict lower (more negative) returns, and vice versa.
        
        Parameters
        ----------
        n_bins : int, default=10
            Number of bins/buckets to create (10 = deciles, 5 = quintiles, etc.)
        figsize : Tuple[int, int], default=(12, 8)
            Figure size for the plot
        plot_type : str, default="bar"
            Type of plot to create: "bar" or "line"
        save_path : Optional[str], default=None
            If provided, save the figure to this path
        target_col : str, default='target'
            Which target column to use. Options: 'target', 'raw_return', 'log_return',
            'log_return_atr', 'log_return_ewsd'
            
        Returns
        -------
        plt.Figure
            Matplotlib Figure object containing the decile plot
            
        Notes
        -----
        The plot includes:
        - Mean target value for each decile (green bars = positive, red = negative)
        - Spearman rank correlation showing monotonicity of the relationship
        - P-value indicating statistical significance
        - Feature value ranges for each decile (top x-axis)
        
        A strong predictive feature should have:
        - High absolute Spearman correlation (|ρ| > 0.7)
        - Low p-value (p < 0.05)
        - Monotonic pattern in the bars
        
        Examples
        --------
        >>> fig = explorer.plot_feature_deciles(n_bins=10, plot_type="bar")
        >>> plt.show()
        """
        # Validate target_col
        if target_col not in self.df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in dataframe. "
                f"Available columns: {list(self.df.columns)}"
            )
        
        # Use abstracted plotting function from eda module
        fig, decile_table = plot_decile_analysis(
            feature_data=self.df['feature'],
            target_data=self.df[target_col],
            feature_name=self.feature_name,
            n_bins=n_bins,
            figsize=figsize,
            plot_type=plot_type,
            save_path=save_path
        )
        
        # Store results (including decile table for display)
        self.results['decile_analysis'] = {
            'n_bins': n_bins,
            'decile_data': decile_table  # Table for display
        }
        
        return fig
    
    def plot_2bin(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
        target_col: str = 'target'
    ) -> plt.Figure:
        """
        Create a 2-bin plot: one bin for positive feature values, one for negative.
        
        This is useful for features that are expected to have symmetric behavior
        around zero, or to quickly assess if positive vs negative values have
        different target characteristics.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default=(10, 6)
            Figure size for the plot
        save_path : Optional[str], default=None
            If provided, save the figure to this path
        target_col : str, default='target'
            Which target column to use. Options: 'target', 'raw_return', 'log_return',
            'log_return_atr', 'log_return_ewsd'
            
        Returns
        -------
        plt.Figure
            Matplotlib Figure object containing the 2-bin plot
            
        Examples
        --------
        >>> fig = explorer.plot_2bin()
        >>> plt.show()
        """
        # Validate target_col
        if target_col not in self.df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in dataframe. "
                f"Available columns: {list(self.df.columns)}"
            )
        
        # Use abstracted plotting function from eda module
        fig, bin_table = plot_2bin_analysis(
            feature_data=self.df['feature'],
            target_data=self.df[target_col],
            feature_name=self.feature_name,
            figsize=figsize,
            save_path=save_path
        )
        
        # Store results (including bin table for display)
        self.results['2bin_analysis'] = {
            'bin_data': bin_table  # Table for display
        }
        
        return fig
    
    def plot_rolling_decile_whiskers(
        self,
        window_size: int = 252,
        step_size: int = 63,
        n_bins: int = 10,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None,
        target_col: str = 'target'
    ) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        Create a rolling decile whisker plot showing variance over time.
        
        This plot shows how the mean target return for each decile changes
        over rolling windows, with whiskers showing the standard deviation.
        This helps identify if the feature-target relationship is stable over time.
        
        Parameters
        ----------
        window_size : int, default=252
            Number of days in each rolling window (252 = ~1 year)
        step_size : int, default=63
            Number of days to roll forward (~3 months for quarterly analysis)
        n_bins : int, default=10
            Number of decile bins
        figsize : Tuple[int, int], default=(14, 8)
            Figure size
        save_path : Optional[str], default=None
            If provided, save the figure to this path
        target_col : str, default='target'
            Which target column to use. Options: 'target', 'raw_return', 'log_return',
            'log_return_atr', 'log_return_ewsd'
            
        Returns
        -------
        Tuple[plt.Figure, pd.DataFrame]
            Figure object and DataFrame with aggregated rolling window results
            
        Examples
        --------
        >>> fig, agg_df = explorer.plot_rolling_decile_whiskers(window_size=252, step_size=63)
        >>> plt.show()
        """
        # Validate target_col
        if target_col not in self.df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in dataframe. "
                f"Available columns: {list(self.df.columns)}"
            )
        
        # Use abstracted rolling analysis function
        fig, agg_df = plot_rolling_decile_whiskers(
            feature_data=self.df['feature'],
            target_data=self.df[target_col],
            feature_name=self.feature_name,
            window_size=window_size,
            step_size=step_size,
            n_bins=n_bins,
            figsize=figsize,
            save_path=save_path
        )
        
        # Store results
        self.results['rolling_decile_whiskers'] = {
            'aggregate_data': agg_df,
            'window_size': window_size,
            'step_size': step_size
        }
        
        return fig, agg_df
    
    def walkforward_bin_selection(
        self,
        train_start: dt,
        train_end: dt,
        test_step: int = 252,
        num_steps: int = 10,
        n_bins: int = 3,
        selection_metric: str = 'sharpe',
        verbose: bool = True,
        plot_results: bool = True,
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None,
        target_col: str = 'target'
    ) -> Tuple[pd.DataFrame, List[Dict], Optional[Tuple[plt.Figure, plt.Figure]]]:
        """
        Perform walk-forward bin selection analysis.
        
        For each walk-forward step:
        1. Train on historical window to find optimal bin (highest mean return)
        2. Test on next period using that bin selection rule
        3. Track performance and which bin was selected
        
        This addresses the threshold selection problem: instead of hoping a specific
        bin is best, the system learns which bin is best on each training window and
        applies that rule to test data.
        
        Parameters
        ----------
        train_start : datetime
            Start date for initial training window
        train_end : datetime
            End date for initial training window
        test_step : int, default=252
            Number of days for test period (~1 year)
        num_steps : int, default=10
            Number of walk-forward steps
        n_bins : int, default=3
            Number of bins to create (3 or 4 recommended)
        selection_metric : str, default='sharpe'
            Metric to use for bin selection:
            - 'sharpe': mean / std (risk-adjusted, favors stable returns)
            - 'mean': mean return only (ignores variance)
        verbose : bool, default=True
            Print detailed progress
        plot_results : bool, default=True
            Generate visualization of results
        figsize : Tuple[int, int], default=(16, 10)
            Figure size for plots
        save_path : Optional[str], default=None
            If provided, save figure to this path
        target_col : str, default='target'
            Which target column to use. Options: 'target', 'raw_return', 'log_return',
            'log_return_atr', 'log_return_ewsd'
            
        Returns
        -------
        Tuple[pd.DataFrame, List[Dict], Optional[Tuple[plt.Figure, plt.Figure]]]
            - results_df: DataFrame with test period results
            - step_info: List of detailed info per step
            - figs: Tuple of (metrics_fig, equity_fig) if plot_results=True, else None
            
        Examples
        --------
        >>> results_df, step_info, fig = explorer.walkforward_bin_selection(
        ...     train_start=datetime(2000, 1, 1),
        ...     train_end=datetime(2010, 1, 1),
        ...     test_step=252,
        ...     num_steps=15,
        ...     n_bins=3
        ... )
        >>> plt.show()
        >>> 
        >>> # Analyze which bin was selected most often
        >>> print(results_df['best_bin'].value_counts())
        """
        # Validate target_col
        if target_col not in self.df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in dataframe. "
                f"Available columns: {list(self.df.columns)}"
            )
        
        # Run walk-forward bin selection
        # Pass raw_return if available for equity curve calculation
        raw_return_data = self.df['raw_return'] if 'raw_return' in self.df.columns else None
        
        results_df, step_info = walkforward_bin_selection(
            feature_data=self.df['feature'],
            target_data=self.df[target_col],
            train_start=train_start,
            train_end=train_end,
            test_step=test_step,
            num_steps=num_steps,
            n_bins=n_bins,
            selection_metric=selection_metric,
            verbose=verbose,
            raw_return=raw_return_data
        )
        
        # Store results
        self.results['walkforward_bin_selection'] = {
            'results_df': results_df,
            'step_info': step_info,
            'train_start': train_start,
            'train_end': train_end,
            'test_step': test_step,
            'num_steps': num_steps,
            'n_bins': n_bins
        }
        
        # Plot results if requested
        figs = None
        if plot_results:
            fig_metrics, fig_equity = plot_walkforward_results(
                results_df=results_df,
                step_info=step_info,
                feature_name=self.feature_name,
                figsize=figsize
            )
            figs = (fig_metrics, fig_equity)
            
            if save_path:
                # Save both figures
                base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
                ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'png'
                
                metrics_path = f"{base_path}_metrics.{ext}"
                equity_path = f"{base_path}_equity.{ext}"
                
                fig_metrics.savefig(metrics_path, dpi=300, bbox_inches='tight')
                fig_equity.savefig(equity_path, dpi=300, bbox_inches='tight')
                print(f"\nSaved walk-forward results:")
                print(f"  Metrics: {metrics_path}")
                print(f"  Equity:  {equity_path}")
        
        # Print summary for both LONG and SHORT strategies
        print(f"\n{'='*70}")
        print(f"WALK-FORWARD BIN SELECTION SUMMARY (LONG & SHORT)")
        print(f"{'='*70}")
        print(f"Feature: {self.feature_name}")
        print(f"Total steps: {len(results_df)}")
        
        print(f"\n--- LONG STRATEGY ---")
        print(f"  Average test return: {results_df['test_mean_return_long'].mean():.6f}")
        print(f"  Average test Sortino: {results_df['test_sortino_long'].mean():.2f}")
        print(f"  Total trades: {results_df['n_trades_long'].sum()}")
        print(f"  Bin selection frequency: {results_df['best_long_bin'].value_counts().sort_index().to_dict()}")
        
        print(f"\n--- SHORT STRATEGY ---")
        print(f"  Average test return: {results_df['test_mean_return_short'].mean():.6f}")
        print(f"  Average test Sortino: {results_df['test_sortino_short'].mean():.2f}")
        print(f"  Total trades: {results_df['n_trades_short'].sum()}")
        print(f"  Bin selection frequency: {results_df['best_short_bin'].value_counts().sort_index().to_dict()}")
        
        print(f"\n--- COMBINED (LONG + SHORT) ---")
        combined_return = (results_df['test_mean_return_long'].mean() + results_df['test_mean_return_short'].mean()) / 2
        print(f"  Average combined return: {combined_return:.6f}")
        print(f"  Total trades: {results_df['n_trades_long'].sum() + results_df['n_trades_short'].sum()}")
        print(f"{'='*70}")
        
        return results_df, step_info, figs
    
    def feature_permutation_test(
        self,
        objective_metric: 'ObjectiveMetric',
        nreps: int = 100,
        alpha: float = 0.05,
        n_jobs: int = -1,
        random_seed: Optional[int] = None,
        verbose: bool = True,
        target_col: str = 'target'
    ) -> Dict[str, Any]:
        """
        Perform feature permutation test by shuffling feature values.
        
        This test:
        1. Uses the CACHED data in self.df (no re-extraction needed)
        2. Shuffles only the feature values, keeping target fixed
        3. Computes objective metric directly on all returns (no model needed)
        4. Calculates p-value: fraction of permutations with metric >= original
        
        This is FAST because it uses cached data and only shuffles feature values.
        For bar permutation (shuffling price bars), use bar_permutation_test() instead.
        
        Parameters
        ----------
        objective_metric : ObjectiveMetric
            Metric instance (e.g., SortinoRatio, SharpeRatio)
            Must implement compute() method
        nreps : int, default=100
            Number of permutation replications
        alpha : float, default=0.05
            Significance level for hypothesis test
        n_jobs : int, default=-1
            Number of parallel jobs (-1 = all CPUs)
        random_seed : Optional[int], default=None
            Random seed for reproducibility
        verbose : bool, default=True
            Print progress and results
        target_col : str, default='target'
            Which target column to use. Options: 'target', 'raw_return', 'log_return',
            'log_return_atr', 'log_return_ewsd'
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with:
            - 'original_metric': Original objective metric value
            - 'pval': Permutation p-value
            - 'significant': Whether feature is significant (pval <= alpha)
            - 'metric_type': Type of objective metric used
            - 'alpha': Significance level
            - 'nreps': Number of replications
            
        Examples
        --------
        >>> from feature_selection.objective_metric import SortinoRatio
        >>> 
        >>> metric = SortinoRatio(annualization_factor=252)
        >>> 
        >>> result = explorer.feature_permutation_test(
        ...     objective_metric=metric,
        ...     nreps=1000,
        ...     alpha=0.05
        ... )
        >>> if result['significant']:
        ...     print(f"Feature is significant! Metric: {result['original_metric']:.2f}")
        ...     print(f"P-value: {result['pval']:.4f}")
        """
        from feature_selection.permutation_test.permutation_engine import (
            PermutationEngine,
            FeaturePermutationStrategy
        )
        
        if verbose:
            print("\n" + "="*80)
            print("FEATURE PERMUTATION TEST")
            print("="*80)
            print(f"Feature: {self.feature_name}")
            print(f"Objective Metric: {objective_metric.__class__.__name__}")
            print(f"Target Column: {target_col}")
            print(f"Replications: {nreps}")
            print(f"Alpha level: {alpha}")
            print("="*80)
        
        # Validate target_col
        if target_col not in self.df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in dataframe. "
                f"Available columns: {list(self.df.columns)}"
            )
        
        # Define criterion function that computes metric directly on all returns
        # No model needed - just compute metric on the target values
        def criterion_func(data: pd.DataFrame, feature_col: str) -> float:
            """
            Criterion function: compute objective metric directly on all returns.
            
            For feature permutation, we don't need a model - we just compute
            the metric on all the target values to see if the feature has any
            relationship with the target.
            """
            y = data[target_col]
            
            # Compute metric directly on all returns
            return objective_metric.compute(y)
        
        # Compute original criterion
        if verbose:
            print(f"\nComputing original {objective_metric.__class__.__name__}...")
        
        original_metric = criterion_func(self.df, 'feature')
        
        if verbose:
            print(f"  Original Metric: {original_metric:.4f}")
        
        # Set up feature permutation strategy
        strategy = FeaturePermutationStrategy()
        
        # Set up permutation engine
        engine = PermutationEngine(
            strategy=strategy,
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        # Create temporary DataFrame with feature name as column
        # (FeatureExplorer stores data in 'feature' column, but engine expects actual feature name)
        temp_df = self.df.copy()
        temp_df[self.feature_name] = temp_df['feature']
        
        # Run permutation test
        perm_results = engine.run_permutation_test(
            data=temp_df,
            feature_cols=[self.feature_name],
            criterion_func=criterion_func,
            nreps=nreps,
            random_seed=random_seed,
            alpha=alpha,
            target_col=target_col
        )
        
        # Extract results
        pval = perm_results.loc[0, 'pval']
        significant = perm_results.loc[0, 'significant']
        
        # Compile final results
        result = {
            'original_metric': original_metric,
            'pval': pval,
            'significant': significant,
            'metric_type': objective_metric.__class__.__name__,
            'alpha': alpha,
            'nreps': nreps
        }
        
        # Store in results
        self.results['feature_permutation_test'] = result
        
        if verbose:
            print("\n" + "="*80)
            print("FEATURE PERMUTATION TEST RESULTS")
            print("="*80)
            print(f"Feature: {self.feature_name}")
            print(f"Original Metric: {original_metric:.4f}")
            print(f"P-value: {pval:.4f}")
            print(f"Significant (α={alpha}): {'YES' if significant else 'NO'}")
            print("="*80)
        
        return result
    
   
    
    def plot_rolling_decile_heatmap(
        self,
        window_size: int = 252,
        step_size: int = 63,
        n_bins: int = 10,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None,
        target_col: str = 'target'
    ) -> Tuple[plt.Figure, List[pd.DataFrame]]:
        """
        Create a heatmap showing how decile returns evolve over rolling windows.
        
        This provides a time-series view of decile performance, showing if
        certain deciles become more/less predictive over time.
        
        Parameters
        ----------
        window_size : int, default=252
            Number of days in each rolling window
        step_size : int, default=63
            Number of days to roll forward
        n_bins : int, default=10
            Number of decile bins
        figsize : Tuple[int, int], default=(14, 10)
            Figure size
        save_path : Optional[str], default=None
            If provided, save the figure to this path
        target_col : str, default='target'
            Which target column to use. Options: 'target', 'raw_return', 'log_return',
            'log_return_atr', 'log_return_ewsd'
            
        Returns
        -------
        Tuple[plt.Figure, List[pd.DataFrame]]
            Figure object and list of decile stats for each window
            
        Examples
        --------
        >>> fig, all_stats = explorer.plot_rolling_decile_heatmap()
        >>> plt.show()
        """
        # Validate target_col
        if target_col not in self.df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in dataframe. "
                f"Available columns: {list(self.df.columns)}"
            )
        
        # Use abstracted rolling analysis function
        fig, all_stats = plot_rolling_decile_heatmap(
            feature_data=self.df['feature'],
            target_data=self.df[target_col],
            feature_name=self.feature_name,
            window_size=window_size,
            step_size=step_size,
            n_bins=n_bins,
            figsize=figsize,
            save_path=save_path
        )
        
        # Store results
        self.results['rolling_decile_heatmap'] = {
            'all_window_stats': all_stats,
            'window_size': window_size,
            'step_size': step_size
        }
        
        return fig, all_stats
    
    def __repr__(self) -> str:
        """String representation of the FeatureExplorer."""
        return (f"FeatureExplorer(feature='{self.feature_name}', "
                f"n_samples={self.n_samples}, "
                f"date_range={self.date_range[0].date()} to {self.date_range[1].date()})")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()



# Example usage and testing
if __name__ == "__main__":
    # Create synthetic data for testing
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Create a feature with some predictive power
    feature = pd.Series(np.random.randn(500), index=dates, name='test_feature')
    
    # Create target that's negatively correlated with feature (mean reversion)
    target = pd.Series(-0.3 * feature + 0.7 * np.random.randn(500), 
                      index=dates, name='target')
    
    # Initialize explorer
    explorer = FeatureExplorer(
        feature_name="test_mean_reversion_feature",
        feature_data=feature,
        target_data=target,
        metadata={'created_date': '2025-10-18', 'version': 1}
    )
    
    print(explorer)
    print(f"\nFeature statistics:")
    print(f"  Mean: {explorer.df['feature'].mean():.4f}")
    print(f"  Std: {explorer.df['feature'].std():.4f}")
    print(f"  Correlation with target: {explorer.df['feature'].corr(explorer.df['target']):.4f}")
    
    # Test decile plot
    print("\nGenerating decile plot...")
    fig = explorer.plot_feature_deciles(n_bins=10, plot_type='bar')
    
    # Uncomment to show plot
    # plt.show()
