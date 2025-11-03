"""
Feature Selector Class for Model-Based Feature Selection

This module provides a framework for selecting features using walk-forward analysis
and other model-based selection techniques. It separates the selection logic from
the exploration/visualization logic in FeatureExplorer.

The FeatureSelector class includes methods for:
- Walk-forward analysis with any BaseModel
- Rolling decile analysis for temporal stability
- Model-based feature selection strategies

Author: Trading Research Team
Date: 2025-10-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime as dt
from eda.rolling_decile import plot_rolling_decile_whiskers


class FeatureSelector:
    """
    Feature selection framework using model-based approaches.
    
    This class provides methods for selecting features based on their predictive
    power in walk-forward tests and temporal stability analysis. It works with
    any model that implements the BaseModel interface.
    
    Parameters
    ----------
    feature_name : str
        Name of the feature being analyzed
    feature_data : pd.Series
        The feature values (must have datetime index)
    target_data : pd.DataFrame
        DataFrame containing target columns (raw_return, log_return, etc.)
        Must have datetime index matching feature_data
    metadata : Optional[Dict[str, Any]], default=None
        Additional metadata about the feature
        
    Attributes
    ----------
    feature_name : str
        Name of the feature
    df : pd.DataFrame
        Combined dataframe with feature and all target columns
    metadata : Dict[str, Any]
        Feature metadata
    results : Dict[str, Any]
        Dictionary storing all selection results
    """
    
    def __init__(
        self,
        feature_name: str,
        feature_data: pd.Series,
        target_data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize the FeatureSelector with feature and target data."""
        self.feature_name = feature_name
        self.metadata = metadata or {}
        self.results = {}
        
        # Validate inputs
        if not isinstance(feature_data, pd.Series):
            raise TypeError("feature_data must be a pandas Series")
        if not isinstance(target_data, pd.DataFrame):
            raise TypeError("target_data must be a pandas DataFrame")
        
        # Combine feature and target data
        self.df = pd.DataFrame({'feature': feature_data})
        
        # Add all target columns from target_data
        for col in target_data.columns:
            self.df[col] = target_data[col]
        
        # Drop rows with NaN values
        self.df = self.df.dropna()
        
        if self.df.empty:
            raise ValueError("No valid data after aligning feature and targets (all NaN)")
        
        # Store basic statistics
        self.n_samples = len(self.df)
        self.date_range = (self.df.index.min(), self.df.index.max())
    

    def plot_rolling_decile_whiskers(
        self,
        window_size: int = 252,
        step_size: int = 63,
        n_bins: int = 10,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None,
        target_col: str = 'log_return'
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
        target_col : str, default='log_return'
            Which target column to use. Options: 'raw_return', 'log_return',
            'log_return_atr', 'log_return_ewsd'
            
        Returns
        -------
        Tuple[plt.Figure, pd.DataFrame]
            Figure object and DataFrame with aggregated rolling window results
            
        Examples
        --------
        >>> fig, agg_df = selector.plot_rolling_decile_whiskers(window_size=252, step_size=63)
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


    def walkforward_analysis(
        self,
        model: 'BaseModel',
        objective_metric: 'ObjectiveMetric',
        train_start: dt,
        train_end: dt,
        test_step: int = 252,
        num_steps: int = 10,
        strategy: str = 'long',
        verbose: bool = True,
        plot_results: bool = True,
        save_path: Optional[str] = None,
        target_col: str = 'log_return'
    ) -> Tuple[pd.DataFrame, List[Dict], Optional[plt.Figure]]:
        """
        Perform walk-forward analysis using any BaseModel.
        
        This method is now FLEXIBLE and accepts any model that implements the
        BaseModel interface (QuantileBinningModel, DecisionTreeBinningModel, etc.).
        
        For each walk-forward step:
        1. Fit model on training window
        2. Generate predictions on test window
        3. Compute objective metric on test returns
        4. Track performance
        
        Parameters
        ----------
        model : BaseModel
            Model to use for walk-forward analysis (e.g., QuantileBinningModel,
            DecisionTreeBinningModel). The model will be fitted on each training
            window and used to predict on test windows.
        objective_metric : ObjectiveMetric
            Metric to compute on test returns (e.g., SortinoRatio, SharpeRatio)
        train_start : datetime
            Start date for initial training window
        train_end : datetime
            End date for initial training window
        test_step : int, default=252
            Number of days for test period (~1 year)
        num_steps : int, default=10
            Number of walk-forward steps
        strategy : str, default='long'
            Trading strategy: 'long' or 'short'
        verbose : bool, default=True
            Print detailed progress
        plot_results : bool, default=True
            Generate visualization of results
        figsize : Tuple[int, int], default=(16, 10)
            Figure size for plots
        save_path : Optional[str], default=None
            If provided, save figure to this path
        target_col : str, default='log_return'
            Which target column to use. Options: 'raw_return', 'log_return',
            'log_return_atr', 'log_return_ewsd'
            
        Returns
        -------
        Tuple[pd.DataFrame, List[Dict], Optional[plt.Figure]]
            - results_df: DataFrame with test period results
            - step_info: List of detailed info per step
            - fig: Figure if plot_results=True, else None
            
        Examples
        --------
        >>> from plotting.base_models import QuantileBinningModel
        >>> from plotting.objective_metric import SortinoRatio
        >>> 
        >>> # Use quantile binning model
        >>> model = QuantileBinningModel(n_bins=3, selection_metric='sortino')
        >>> metric = SortinoRatio(annualization_factor=252)
        >>> 
        >>> results_df, step_info, fig = explorer.walkforward_analysis(
        ...     model=model,
        ...     objective_metric=metric,
        ...     train_start=datetime(2000, 1, 1),
        ...     train_end=datetime(2010, 1, 1),
        ...     test_step=252,
        ...     num_steps=15,
        ...     strategy='long'
        ... )
        >>> plt.show()
        """
        # Validate target_col
        if target_col not in self.df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in dataframe. "
                f"Available columns: {list(self.df.columns)}"
            )
        
        from plotting.walkforward.walkforward_model import WalkForwardSplitter
        
        # Get data
        feature_data = self.df['feature']
        target_data = self.df[target_col]  # Used for binning/training
        
        # IMPORTANT: For equity curves, ALWAYS use raw_return (actual P&L)
        # This is distinct from target_col which is used for binning
        if 'raw_return' not in self.df.columns:
            raise ValueError(
                "raw_return column not found. This is required for equity curve calculation. "
                f"Available columns: {list(self.df.columns)}"
            )
        raw_return_data = self.df['raw_return']  # Used for equity curves
        
        # Extract normalization data if model requires it
        normalization_data = None
        using_volatility_scaling = False
        
        if hasattr(model, 'normalize_by') and model.normalize_by is not None:
            norm_col = model.normalize_by.lower()  # 'ewsd' or 'atr'
            
            # Search for the normalization column in the dataframe (case-insensitive)
            # Look for columns like 'ewsd_252_D', 'atr_252_D', etc.
            matching_cols = [col for col in self.df.columns if col.lower().startswith(norm_col)]
            
            if len(matching_cols) == 0:
                raise ValueError(
                    f"Model requires normalization by '{model.normalize_by}' but no {norm_col} column found. "
                    f"Available columns: {list(self.df.columns)}. "
                    f"Make sure to include {norm_col} in your feature extraction."
                )
            
            # Use the first matching column (typically there's only one)
            norm_col_name = matching_cols[0]
            normalization_data = self.df[norm_col_name]
            using_volatility_scaling = True
            
            if verbose:
                print(f"\n{'='*70}")
                print("VOLATILITY SCALING CONFIGURATION")
                print(f"{'='*70}")
                print(f"✓ Volatility scaling: ENABLED")
                print(f"  Normalization method: {model.normalize_by.upper()}")
                print(f"  Normalization column: {norm_col_name}")
                print(f"  Scaling applied: AFTER binning (to signals)")
                print(f"{'='*70}")
        else:
            if verbose:
                print(f"\n{'='*70}")
                print("VOLATILITY SCALING CONFIGURATION")
                print(f"{'='*70}")
                print(f"✗ Volatility scaling: DISABLED")
                print(f"  Signals will NOT be scaled by volatility")
                print(f"  Consider using normalize_by='ewsd' or 'atr' for better risk-adjusted returns")
                print(f"{'='*70}")
        
        # Create splitter
        splitter = WalkForwardSplitter(
            train_start=train_start,
            train_end=train_end,
            test_step=test_step,
            num_steps=num_steps
        )
        
        # Get splits
        splits = splitter.split(feature_data.index)
        
        if len(splits) == 0:
            raise ValueError("No valid walk-forward splits found")
        
        # Process each step
        step_results = []
        
        for step, (train_idx, test_idx) in enumerate(splits):
            # Get train/test data
            X_train = feature_data.iloc[train_idx]
            y_train = target_data.iloc[train_idx]
            X_test = feature_data.iloc[test_idx]
            y_test = target_data.iloc[test_idx]
            
            if verbose:
                print(f"\n=== Step {step + 1}/{len(splits)} ===")
                print(f"  Train: {X_train.index.min().date()} to {X_train.index.max().date()} (n={len(X_train)})")
                print(f"  Test:  {X_test.index.min().date()} to {X_test.index.max().date()} (n={len(X_test)})")
            
            try:
                # Get normalization data for this split if needed
                norm_train = None
                norm_test = None
                if normalization_data is not None:
                    norm_train = normalization_data.iloc[train_idx]
                    norm_test = normalization_data.iloc[test_idx]
                
                # Fit model on training data (binning on raw features)
                model.fit(X_train, y_train, normalization_data=norm_train)
                
                # DEBUG: Show binning statistics
                if verbose:
                    print(f"  DEBUG - Training binning:")
                    if hasattr(model, 'thresholds_'):
                        print(f"    Thresholds: {model.thresholds_}")
                    if hasattr(model, 'best_long_bin_'):
                        print(f"    Best long bin: {model.best_long_bin_}")
                    if hasattr(model, 'best_short_bin_'):
                        print(f"    Best short bin: {model.best_short_bin_}")
                    
                    # Show bin counts in training data
                    train_bins = np.digitize(X_train.values, model.thresholds_)
                    unique, counts = np.unique(train_bins, return_counts=True)
                    print(f"    Training bin distribution:")
                    for bin_num, count in zip(unique, counts):
                        pct = 100 * count / len(train_bins)
                        print(f"      Bin {bin_num}: {count} samples ({pct:.1f}%)")
                
                # Generate predictions on test data
                # Use volatility scaling if model has normalize_by set
                scaled = hasattr(model, 'normalize_by') and model.normalize_by is not None
                test_signals = model.predict(
                    X_test,
                    strategy=strategy,
                    normalization_data=norm_test,
                    scaled=scaled
                )
                
                # DEBUG: Show test binning
                if verbose:
                    test_bins = np.digitize(X_test.values, model.thresholds_)
                    unique_test, counts_test = np.unique(test_bins, return_counts=True)
                    print(f"  DEBUG - Test binning:")
                    for bin_num, count in zip(unique_test, counts_test):
                        pct = 100 * count / len(test_bins)
                        in_best = "<-- SELECTED" if bin_num == (model.best_long_bin_ if strategy == 'long' else model.best_short_bin_) else ""
                        print(f"    Bin {bin_num}: {count} samples ({pct:.1f}%) {in_best}")
                    # Count trades where signal > 0 (handles both binary and scaled signals)
                    n_signals = (test_signals > 0).sum()
                    print(f"    Signals: {n_signals} out of {len(test_signals)} ({100*n_signals/len(test_signals):.1f}%)")
                    if scaled:
                        print(f"    Position sizes (scaled): min={test_signals[test_signals>0].min():.2f}, max={test_signals[test_signals>0].max():.2f}, mean={test_signals[test_signals>0].mean():.2f}")
                
                # Get test returns for selected signals (use > 0 for both binary and scaled)
                test_returns = y_test[test_signals > 0]
                
                # Compute metrics
                if len(test_returns) > 0:
                    test_metric = objective_metric.compute(test_returns)
                    n_trades = len(test_returns)
                    mean_return = test_returns.mean()
                else:
                    test_metric = 0.0
                    n_trades = 0
                    mean_return = 0.0
                
                if verbose:
                    model_name = model.__class__.__name__
                    print(f"  Model: {model_name}")
                    if hasattr(model, 'best_long_bin_') and hasattr(model, 'best_short_bin_'):
                        best_bin = model.best_long_bin_ if strategy == 'long' else model.best_short_bin_
                        print(f"  Selected bin: {best_bin}")
                    print(f"  Test performance: metric={test_metric:.2f}, mean_return={mean_return:.6f}, n_trades={n_trades}")
                
                step_results.append({
                    'step': step,
                    'test_start': X_test.index.min(),
                    'test_end': X_test.index.max(),
                    'test_metric': test_metric,
                    'mean_return': mean_return,
                    'n_trades': n_trades,
                    'strategy': strategy
                })
                
            except Exception as e:
                if verbose:
                    print(f"  ERROR: {e}")
                # Still include all required columns for consistency
                step_results.append({
                    'step': step,
                    'test_start': X_test.index.min(),
                    'test_end': X_test.index.max(),
                    'test_metric': np.nan,
                    'mean_return': np.nan,
                    'n_trades': 0,
                    'strategy': strategy,
                    'error': str(e)
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(step_results)
        
        # Store results
        self.results['walkforward_analysis'] = {
            'results_df': results_df,
            'model': model.__class__.__name__,
            'objective_metric': objective_metric.__class__.__name__,
            'strategy': strategy,
            'train_start': train_start,
            'train_end': train_end,
            'test_step': test_step,
            'num_steps': num_steps
        }
        
        # Construct DAILY returns series for QuantStats
        # QuantStats needs daily returns, not aggregated walk-forward results!
        # IMPORTANT: Use raw_return for equity curves (actual P&L), not target_col
        if verbose:
            print(f"\n{'='*70}")
            print("Constructing daily returns series for QuantStats...")
            print(f"  Binning used: {target_col}")
            print(f"  Equity curve uses: raw_return (actual P&L)")
            print(f"{'='*70}")
        
        from plotting.graphing.quantstats_reports import generate_tearsheet
        
        # Create daily returns series for strategy and baseline
        # Initialize with zeros for all dates
        # Use raw_return for equity curves (actual P&L), not the target_col used for binning
        strategy_daily_returns = pd.Series(0.0, index=raw_return_data.index)
        
        # IMPORTANT: Baseline should ONLY include out-of-sample (test) periods
        # Training data should NOT be included in the baseline comparison
        baseline_daily_returns = pd.Series(0.0, index=raw_return_data.index)
        
        # For each walk-forward step, fill in the actual returns where signals were generated
        for _, row in results_df.iterrows():
            test_start = row['test_start']
            test_end = row['test_end']
            
            # Get the test period data
            test_mask = (raw_return_data.index >= test_start) & (raw_return_data.index <= test_end)
            test_idx = raw_return_data.index[test_mask]
            
            if len(test_idx) > 0:
                # Get the feature data for this test period
                X_test = feature_data.loc[test_idx]
                
                # Get RAW returns for equity curve (actual P&L)
                raw_returns_test = raw_return_data.loc[test_idx]
                
                # IMPORTANT: Fill baseline with ALL returns for this test period (always-in)
                # This ensures baseline only includes out-of-sample periods, not training data
                baseline_daily_returns.loc[test_idx] = raw_returns_test
                
                # Get normalization data if needed
                norm_test = None
                if normalization_data is not None:
                    norm_test = normalization_data.loc[test_idx]
                
                # Generate predictions for this test period
                # Note: Predictions are based on model trained with target_col,
                # but we apply them to raw_return for equity curve
                # Use volatility scaling if model has normalize_by set
                scaled = hasattr(model, 'normalize_by') and model.normalize_by is not None
                test_signals = model.predict(
                    X_test,
                    strategy=strategy,
                    normalization_data=norm_test,
                    scaled=scaled
                )
                
                # Fill in STRATEGY returns where signals > 0 (handles both binary and scaled)
                # For scaled signals, this applies position sizing to returns
                # This gives us the actual P&L for the equity curve
                strategy_daily_returns.loc[test_idx] = raw_returns_test.where(test_signals > 0, 0.0)
        
        if verbose:
            strategy_total = strategy_daily_returns.sum()
            baseline_total = baseline_daily_returns.sum()
            print(f"Strategy Total Return: {strategy_total:.6f}")
            print(f"Baseline Total Return: {baseline_total:.6f}")
            print(f"Improvement: {strategy_total - baseline_total:+.6f}")
        
        # Generate QuantStats tearsheet if requested
        fig = None
        if plot_results and len(results_df) > 0:
            # Filter returns to only include out-of-sample periods (where baseline != 0)
            # This ensures the graph starts from the first test period, not training start
            oos_mask = baseline_daily_returns != 0
            
            if oos_mask.sum() == 0:
                if verbose:
                    print("\n⚠️  Warning: No out-of-sample data found. Skipping QuantStats tearsheet.")
            else:
                strategy_oos = strategy_daily_returns[oos_mask]
                baseline_oos = baseline_daily_returns[oos_mask]
                
                # Check if strategy has any variance (non-zero returns)
                if strategy_oos.std() == 0:
                    if verbose:
                        print("\n⚠️  Warning: Strategy generated no trades (all returns are zero). Skipping QuantStats tearsheet.")
                        print("    This feature may not be predictive or the model couldn't find good bins.")
                else:
                    if verbose:
                        print(f"\nGenerating QuantStats tearsheet for OOS period:")
                        print(f"  Start: {strategy_oos.index.min().date()}")
                        print(f"  End: {strategy_oos.index.max().date()}")
                        print(f"  Days: {len(strategy_oos)}")
                    
                    if save_path:
                        # Generate HTML tearsheet and save to file
                        # Ensure .html extension
                        if not save_path.endswith('.html'):
                            save_path = save_path.replace('.png', '.html')
                        
                        generate_tearsheet(
                            strategy_returns=strategy_oos,
                            baseline_returns=baseline_oos,
                            feature_name=self.feature_name,
                            output_file=save_path,
                            mode='html'
                        )
                    else:
                        # Display full tearsheet in notebook
                        generate_tearsheet(
                            strategy_returns=strategy_oos,
                            baseline_returns=baseline_oos,
                            feature_name=self.feature_name,
                            mode='full'
                        )
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"WALK-FORWARD ANALYSIS SUMMARY ({strategy.upper()})")
        print(f"{'='*70}")
        print(f"Feature: {self.feature_name}")
        print(f"Model: {model.__class__.__name__}")
        print(f"Objective Metric: {objective_metric.__class__.__name__}")
        print(f"Volatility Scaling: {'ENABLED (' + model.normalize_by.upper() + ')' if using_volatility_scaling else 'DISABLED'}")
        print(f"Binning Target: {target_col}")
        print(f"Equity Curve: raw_return (actual P&L)")
        print(f"Total steps: {len(results_df)}")
        print(f"Average test metric: {results_df['test_metric'].mean():.2f}")
        print(f"Average test return: {results_df['mean_return'].mean():.6f}")
        print(f"Total trades: {results_df['n_trades'].sum()}")
        print(f"{'='*70}")
        
        return results_df, step_results, fig
    

    def __repr__(self) -> str:
        """String representation of the FeatureSelector."""
        return (
            f"FeatureSelector(feature='{self.feature_name}', "
            f"n_samples={self.n_samples}, "
            f"date_range={self.date_range[0].date()} to {self.date_range[1].date()})"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()
