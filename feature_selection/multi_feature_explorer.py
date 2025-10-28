"""
Multi-Feature Explorer Wrapper

This module provides a wrapper class that manages multiple FeatureExplorer instances,
allowing you to work with all features from a bias node as a single unit.

Author: Trading Research Team
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List, Union
from datetime import datetime as dt
import warnings

from utils.enums import Ticker, TimeFrame
from feature_extraction.feature_extractor import extract_features_from_bias_node
import utils.helpers as helpers


class MultiFeatureExplorer:
    """
    Wrapper class that manages multiple FeatureExplorer instances.
    
    This class uses dynamic method delegation to automatically forward any
    FeatureExplorer method to all managed explorers. This eliminates the need
    for manual wrapper methods and makes the class highly maintainable.
    
    When you call a method on MultiFeatureExplorer that doesn't exist directly,
    it will automatically:
    1. Check if the method exists on FeatureExplorer
    2. Call it on all explorers
    3. Return a dict mapping feature names to results
    
    Parameters
    ----------
    explorers : Dict[str, FeatureExplorer]
        Dictionary mapping feature names to FeatureExplorer instances
        
    Attributes
    ----------
    explorers : Dict[str, FeatureExplorer]
        Dictionary of individual feature explorers
    feature_names : List[str]
        List of all feature names
    n_features : int
        Number of features
        
    Examples
    --------
    >>> # Create from bias node (returns MultiFeatureExplorer)
    >>> explorer = FeatureExplorer.from_bias_node(
    ...     module_name='cmma',
    ...     ticker=Ticker.SPY,
    ...     params={'lookback': 20, 'atr_length': 252}
    ... )
    >>> 
    >>> # Call methods directly - they apply to all features automatically!
    >>> figs = explorer.plot_feature_deciles(n_bins=10)
    >>> # Returns dict: {feature_name: figure}
    >>> 
    >>> # Any FeatureExplorer method works automatically
    >>> results = explorer.binning_permutation_test(n_bins=3, nreps=100)
    >>> # No need to manually wrap each method!
    >>> 
    >>> # Access individual explorers if needed
    >>> for name, individual_explorer in explorer.explorers.items():
    ...     print(f"{name}: {individual_explorer.results}")
    """
    
    def __init__(self, explorers: Dict[str, 'FeatureExplorer']):
        """Initialize MultiFeatureExplorer with a dict of explorers."""
        import re
        
        # Filter out ATR and EWSD features (used for normalization, not for analysis)
        # Use regex to match any feature containing 'atr' or 'ewsd'
        self.explorers = {
            name: explorer for name, explorer in explorers.items()
            if not (re.search(r'atr', name, re.IGNORECASE) or 
                   re.search(r'ewsd', name, re.IGNORECASE))
        }
        self.feature_names = list(self.explorers.keys())
        self.n_features = len(self.explorers)
        
        # Store all explorers (including ATR/EWSD) for reference
        self._all_explorers = explorers
    

    
    @classmethod
    def from_bias_node(
        cls,
        module_name: str,
        ticker: Union[Ticker, List[Ticker]],
        params: dict = None,
        timeframes: list = None,
        target_data: pd.Series = None,
        start: dt = None,
        end: dt = None,
        strategy_key: str = None,
        metadata: dict = None,
        feature_filter: list = None
    ) -> 'MultiFeatureExplorer':
        """
        Create MultiFeatureExplorer by extracting ALL features from a bias node.
        
        This is the main method for feature exploration. It automatically extracts all
        features that a bias node outputs and creates a FeatureExplorer for each one.
        No need to specify feature names - just specify the bias node and parameters!
        
        Supports both single and multiple tickers. For multiple tickers, features are
        extracted from each ticker and combined using millisecond offsets.
        
        Parameters
        ----------
        module_name : str
            Name of the bias node module (e.g., 'rsi', 'atr', 'cmma', 'volatility_regime')
        ticker : Ticker or List[Ticker]
            Ticker symbol(s) to extract features for
        params : dict, optional
            Parameters for the bias node (e.g., {'lookback': 14} for RSI)
        timeframes : list, optional
            List of TimeFrame objects. Defaults to [TimeFrame.D]
        target_data : pd.Series, optional
            Target variable. If None, automatically extracts log returns.
        start : datetime, optional
            Start date for feature extraction. Defaults to datetime(1990, 1, 1)
        end : datetime, optional
            End date for feature extraction. Defaults to datetime.now()
        strategy_key : str, optional
            Custom strategy key. If None, auto-generated from module_name and params
        metadata : dict, optional
            Additional metadata about the features
        feature_filter : list, optional
            If provided, only create explorers for features in this list.
            Useful if you only want specific outputs from a multi-feature node.
            
        Returns
        -------
        MultiFeatureExplorer
            Instance managing all extracted features
            
        Examples
        --------
        >>> # Single ticker
        >>> explorer = MultiFeatureExplorer.from_bias_node(
        ...     module_name='cmma',
        ...     ticker=Ticker.SPY,
        ...     params={'lookback': 20, 'atr_length': 252}
        ... )
        >>> 
        >>> # Multiple tickers
        >>> explorer = MultiFeatureExplorer.from_bias_node(
        ...     module_name='rsi',
        ...     ticker=[Ticker.ES, Ticker.NQ, Ticker.YM],
        ...     params={'lookback': 14}
        ... )
        >>> 
        >>> # Use the explorer
        >>> figs = explorer.plot_feature_deciles(n_bins=10)
        >>> results = explorer.binning_permutation_test(n_bins=3, nreps=100)
        """
        from feature_selection.feature_explorer import FeatureExplorer
        
        # Set defaults
        if params is None:
            params = {}
        if timeframes is None:
            timeframes = [TimeFrame.D]
        if start is None:
            start = dt(1990, 1, 1)
        if end is None:
            end = dt.now()
        
        # Convert ticker to list for consistent handling
        if isinstance(ticker, Ticker):
            tickers = [ticker]
        else:
            tickers = ticker
        
        # Use universal extraction function
        print(f"Extracting features from '{module_name}'...")
        features_df, targets_df = extract_features_from_bias_node(
            ticker=ticker,
            module_name=module_name,
            params=params,
            timeframes=timeframes,
            start=start,
            end=end,
            strategy_key=strategy_key,
            use_millisecond_offset=True
        )
        
        # Filter features if requested
        available_features = [col for col in features_df.columns if col != 'ticker'] # Exclude 'ticker' column when using multiple tickers
        if feature_filter:
            features_to_explore = [f for f in feature_filter if f in available_features]
            if not features_to_explore:
                raise ValueError(
                    f"None of the requested features found in extracted data.\n"
                    f"Requested: {feature_filter}\n"
                    f"Available: {available_features}"
                )
        else:
            features_to_explore = available_features
        
        print(f"Found {len(available_features)} features from '{module_name}':")
        for feat in available_features:
            print(f"  - {feat}")
        
        if feature_filter:
            print(f"\nCreating explorers for {len(features_to_explore)} filtered features")
        else:
            print(f"\nCreating explorers for all {len(features_to_explore)} features")
        
        # Determine which target to use
        if target_data is None:
            # Use log_return as default target for backward compatibility
            print(f"\nUsing extracted log_return as target (mean: {targets_df['log_return'].mean():.6f})")
            default_target = targets_df['log_return']
        else:
            # User provided custom target
            print(f"\nUsing provided custom target (mean: {target_data.mean():.6f})")
            default_target = target_data
        
        # Create FeatureExplorer instance for each feature
        explorers = {}
        for feature_name in features_to_explore:
            feature_data = features_df[feature_name]
            
            # Add extraction info to metadata
            feat_metadata = metadata.copy() if metadata else {}
            feat_metadata.update({
                'module_name': module_name,
                'params': params,
                'timeframes': [str(tf) for tf in timeframes],
                'extracted_date': dt.now().isoformat(),
                'tickers': [str(t) for t in tickers],
                'n_tickers': len(tickers),
                'start_date': start.isoformat(),
                'end_date': end.isoformat(),
                'total_features_from_node': len(available_features),
                'target_auto_extracted': target_data is None
            })
            
            # Create explorer with all target types
            explorer = FeatureExplorer(
                feature_name=feature_name,
                feature_data=feature_data,
                target_data=default_target,
                metadata=feat_metadata,
                raw_return=targets_df['raw_return'],
                log_return=targets_df['log_return'],
                log_return_atr=targets_df['log_return_atr'],
                log_return_ewsd=targets_df['log_return_ewsd']
            )
            
            explorers[feature_name] = explorer
            print(f"  ✓ Created explorer for '{feature_name}'")
        
        print(f"\nSuccessfully created {len(explorers)} FeatureExplorer instances")
        
        # Return MultiFeatureExplorer wrapper
        return cls(explorers)
    
    def __getattr__(self, name: str):
        """
        Dynamically delegate method calls to all explorers.
        
        This magic method is called when an attribute/method is not found on
        MultiFeatureExplorer itself. It checks if the method exists on the
        underlying FeatureExplorer instances and creates a wrapper that calls
        it on all explorers.
        
        Parameters
        ----------
        name : str
            Name of the attribute/method being accessed
            
        Returns
        -------
        callable
            A wrapper function that calls the method on all explorers
            
        Raises
        ------
        AttributeError
            If the attribute doesn't exist on FeatureExplorer either
        """
        # Check if this is a method that exists on FeatureExplorer
        if self.explorers:
            first_explorer = next(iter(self.explorers.values()))
            if hasattr(first_explorer, name):
                attr = getattr(first_explorer, name)
                
                # If it's a callable method, return a wrapper
                if callable(attr):
                    return self._create_multi_method(name)
                else:
                    # If it's a property/attribute, return dict of values
                    return {fname: getattr(exp, name) 
                            for fname, exp in self.explorers.items()}
        
        # If not found, raise AttributeError
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
    
    def _create_multi_method(self, method_name: str):
        """
        Create a wrapper function that calls method_name on all explorers.
        
        Parameters
        ----------
        method_name : str
            Name of the method to call on each explorer
            
        Returns
        -------
        callable
            Wrapper function that applies the method to all explorers
        """
        def multi_method(*args, **kwargs):
            """
            Call the method on all explorers and return dict of results.
            
            Parameters
            ----------
            *args
                Positional arguments to pass to the method
            **kwargs
                Keyword arguments to pass to the method
                
            Returns
            -------
            Dict[str, Any]
                Dictionary mapping feature names to method results
            """
            # Check if verbose output is requested
            verbose = kwargs.get('verbose', True)
            
            if verbose and self.n_features > 1:
                print(f"\nApplying {method_name}() to {self.n_features} features...")
            
            results = {}
            for i, (feature_name, explorer) in enumerate(self.explorers.items(), 1):
                if verbose and self.n_features > 1:
                    print(f"\n[{i}/{self.n_features}] {feature_name}")
                    print("=" * 60)
                
                method = getattr(explorer, method_name)
                result = method(*args, **kwargs)
                results[feature_name] = result
            
            if verbose and self.n_features > 1:
                print(f"\n{'='*60}")
                print(f"Completed {method_name}() for all {self.n_features} features")
                print(f"{'='*60}")
            
            return results
        
        # Set the wrapper's name for better debugging
        multi_method.__name__ = method_name
        multi_method.__doc__ = f"Apply {method_name}() to all features. See FeatureExplorer.{method_name}() for details."
        
        return multi_method
    
    # Note: plot_feature_deciles, plot_2bin, plot_rolling_decile_whiskers, etc.
    # are now automatically delegated via __getattr__. No need for wrapper methods!
    
    # All plotting methods (plot_2bin, plot_rolling_decile_whiskers, 
    # plot_rolling_decile_heatmap) are automatically delegated via __getattr__
    
    def walkforward_bin_selection(
        self,
        train_start,
        train_end,
        test_step: int = 252,
        num_steps: int = 10,
        n_bins: int = 3,
        selection_metric: str = 'sharpe',
        verbose: bool = True,
        plot_results: bool = True,
        figsize: Tuple[int, int] = (16, 10),
        save_dir: Optional[str] = None
    ) -> Dict[str, Tuple]:
        """
        Perform walk-forward bin selection for ALL features.
        
        This method automatically applies walk-forward bin selection analysis
        to all features managed by this MultiFeatureExplorer.
        
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
        selection_metric : str, default='sortino'
            Metric to use for bin selection:
            - 'sortino': mean / downside_std (risk-adjusted, penalizes only downside)
            - 'mean': mean return only (ignores variance)
        verbose : bool, default=True
            Print detailed progress
        plot_results : bool, default=True
            Generate visualization of results
        figsize : Tuple[int, int], default=(16, 10)
            Figure size for plots
        save_dir : Optional[str], default=None
            If provided, save all figures to this directory
            
        Returns
        -------
        Dict[str, Tuple]
            Dictionary mapping feature names to (results_df, step_info, figs) tuples
            where figs is a tuple of (metrics_fig, equity_fig)
            
        Examples
        --------
        >>> from datetime import datetime
        >>> results = explorer.walkforward_bin_selection(
        ...     train_start=datetime(2000, 1, 1),
        ...     train_end=datetime(2010, 1, 1),
        ...     test_step=252,
        ...     num_steps=15,
        ...     n_bins=3
        ... )
        >>> 
        >>> # Access individual feature results
        >>> for feature_name, (results_df, step_info, figs) in results.items():
        ...     print(f"{feature_name}: Avg Sortino = {results_df['test_sortino'].mean():.2f}")
        ...     if figs:
        ...         metrics_fig, equity_fig = figs
        """
        # Use auto-delegation to call walkforward_bin_selection on all explorers
        # But handle save_dir specially to generate per-feature save paths
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD BIN SELECTION FOR {self.n_features} FEATURES")
        print(f"{'='*60}")
        
        results = {}
        for feature_name, explorer in self.explorers.items():
            # Generate save path if directory provided
            save_path = f"{save_dir}/{feature_name}_walkforward.png" if save_dir else None
            
            # Call the method on individual explorer
            result = explorer.walkforward_bin_selection(
                train_start=train_start,
                train_end=train_end,
                test_step=test_step,
                num_steps=num_steps,
                n_bins=n_bins,
                selection_metric=selection_metric,
                verbose=verbose,
                plot_results=plot_results,
                figsize=figsize,
                save_path=save_path
            )
            results[feature_name] = result
        
        print(f"\n{'='*60}")
        print(f"COMPLETED WALK-FORWARD ANALYSIS FOR ALL {self.n_features} FEATURES")
        print(f"{'='*60}")
        
        # Print comparative summary for LONG and SHORT strategies
        print(f"\nComparative Summary (LONG Strategy):")
        print(f"{'Feature':<40} {'Avg Return':>12} {'Avg Sortino':>12} {'Total Trades':>12}")
        print(f"{'-'*80}")
        for feature_name, (results_df, _, _) in results.items():
            avg_return_long = results_df['test_mean_return_long'].mean()
            avg_sortino_long = results_df['test_sortino_long'].mean()
            total_trades_long = results_df['n_trades_long'].sum()
            print(f"{feature_name:<40} {avg_return_long:>12.6f} {avg_sortino_long:>12.2f} {total_trades_long:>12.0f}")
        print(f"{'-'*80}")
        
        print(f"\nComparative Summary (SHORT Strategy):")
        print(f"{'Feature':<40} {'Avg Return':>12} {'Avg Sortino':>12} {'Total Trades':>12}")
        print(f"{'-'*80}")
        for feature_name, (results_df, _, _) in results.items():
            avg_return_short = results_df['test_mean_return_short'].mean()
            avg_sortino_short = results_df['test_sortino_short'].mean()
            total_trades_short = results_df['n_trades_short'].sum()
            print(f"{feature_name:<40} {avg_return_short:>12.6f} {avg_sortino_short:>12.2f} {total_trades_short:>12.0f}")
        print(f"{'-'*80}")
        
        print(f"\nComparative Summary (COMBINED L+S):")
        print(f"{'Feature':<40} {'Avg Return':>12} {'Total Trades':>12}")
        print(f"{'-'*80}")
        for feature_name, (results_df, _, _) in results.items():
            avg_return_combined = (results_df['test_mean_return_long'].mean() + results_df['test_mean_return_short'].mean()) / 2
            total_trades_combined = results_df['n_trades_long'].sum() + results_df['n_trades_short'].sum()
            print(f"{feature_name:<40} {avg_return_combined:>12.6f} {total_trades_combined:>12.0f}")
        print(f"{'-'*80}")
        
        return results
    
    # Note: walkforward_bin_selection_tree, binning_permutation_test, and other
    # FeatureExplorer methods are automatically delegated via __getattr__.
    # No wrapper methods needed!
    
    # binning_permutation_test is automatically delegated via __getattr__
    # It will work automatically without a wrapper method!
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all analysis results across features.
        
        Returns
        -------
        pd.DataFrame
            Summary of results with one row per feature
        """
        summary_data = []
        
        for feature_name, explorer in self.explorers.items():
            row = {'feature_name': feature_name}
            
            # Add basic statistics only (no statistical tests)
            row['n_samples'] = explorer.n_samples
            row['feature_mean'] = explorer.df['feature'].mean()
            row['feature_std'] = explorer.df['feature'].std()
            row['feature_min'] = explorer.df['feature'].min()
            row['feature_max'] = explorer.df['feature'].max()
            row['target_mean'] = explorer.df['target'].mean()
            row['target_std'] = explorer.df['target'].std()
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"MultiFeatureExplorer(n_features={self.n_features}, "
                f"features={self.feature_names})")
    
    def __str__(self) -> str:
        """Human-readable string."""
        return self.__repr__()
    
    def __getitem__(self, feature_name: str) -> 'FeatureExplorer':
        """Allow dict-like access to individual explorers."""
        return self.explorers[feature_name]
    
    def __iter__(self):
        """Allow iteration over feature names."""
        return iter(self.explorers.items())
