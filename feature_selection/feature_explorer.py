"""
FeatureExplorer - Simplified Feature Analysis Class

This module provides a streamlined feature exploration framework that consumes
dataframes from FeatureExtractor. It follows the Single Responsibility Principle:
- FeatureExtractor: handles extraction
- FeatureExplorer: handles analysis and visualization

The new FeatureExplorer:
- Takes dataframes as input (from FeatureExtractor)
- Works with single or multiple features automatically
- Provides visualization and statistical analysis methods
- No extraction logic - purely analysis

Author: Trading Research Team
Date: 2025-10-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime as dt
from eda.decile_plots import plot_decile_analysis, plot_2bin_analysis
from eda.distribution import plot_feature_distribution, plot_feature_timeseries
from feature_selection.permutation_test.permutation_engine import (
    PermutationEngine,
    FeaturePermutationStrategy
)
from feature_selection.base_models.quantile_binning import QuantileBinningModel
from feature_selection.objective_metric.sortino import SortinoRatio


class FeatureExplorer:
    """
    Simplified feature exploration class that consumes FeatureExtractor outputs.
    
    This class provides visualization and statistical analysis for features,
    working directly with the dataframes produced by FeatureExtractor. It
    automatically handles single or multiple features.
    
    Key Design:
    - Input: Dataframes from FeatureExtractor (features_df, targets_df)
    - Output: Plots, statistics, analysis results
    - No extraction logic - purely analysis
    - Works with any number of features
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Features dataframe from FeatureExtractor
        Columns: feature columns + optional 'ticker' column
    targets_df : pd.DataFrame
        Targets dataframe from FeatureExtractor
        Columns: raw_return, log_return, log_return_atr, log_return_ewsd, optional 'ticker'
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
        Dictionary storing analysis results
        
    Examples
    --------
    >>> # Step 1: Extract features using FeatureExtractor
    >>> from feature_extraction.feature_extractor_class import FeatureExtractor
    >>> extractor = FeatureExtractor(ticker=Ticker.SPY)
    >>> result = extractor.extract(modules={'rsi': {'lookback': [14, 21]}})
    >>> 
    >>> # Step 2: Create FeatureExplorer from extracted dataframes
    >>> from feature_selection.feature_explorer import FeatureExplorer
    >>> explorer = FeatureExplorer(
    ...     features_df=result['features'],
    ...     targets_df=result['targets']
    ... )
    >>> 
    >>> # Step 3: Analyze features
    >>> explorer.plot_all_deciles(n_bins=10)
    >>> summary = explorer.get_summary()
    >>> 
    >>> # Note: Normalization columns (ATR/EWSD) are in result['normalization']
    >>> # They don't clutter the feature analysis!
    """
    
    @staticmethod
    def _permutation_criterion(
        df: pd.DataFrame, 
        feature_col: str, 
        target_col: str,
        base_model: Any,
        metric: Any,
        verbose: bool
    ) -> float:
        """
        Compute criterion for a single feature (static method for pickling).
        
        Args:
            df: DataFrame with features and target
            feature_col: Name of feature to evaluate
            target_col: Name of target column
            base_model: Model to use for prediction
            metric: Metric to compute
            verbose: Whether to print warnings
            
        Returns:
            Metric value (higher is better)
        """
        # Extract feature and target
        X = df[feature_col]
        y = df[target_col]
        
        # Remove NaN values
        valid_mask = ~(X.isna() | y.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(X_clean) < 10:
            return 0.0
        
        try:
            # Fit model
            base_model.fit(X_clean, y_clean)
            
            # Get predictions (binary signals)
            signals = base_model.predict(X_clean, strategy='long')
            
            # Get returns for selected signals
            selected_returns = y_clean[signals == 1]
            
            if len(selected_returns) < 5:
                return 0.0
            
            # Compute metric
            metric_value = metric.compute(selected_returns)
            
            return metric_value
            
        except Exception as e:
            if verbose:
                print(f"  [WARNING] Failed to compute criterion for '{feature_col}': {e}")
            return 0.0
            
    def __init__(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize FeatureExplorer with features and targets dataframes."""
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
        
        # Store feature metadata if available
        self.feature_metadata = self.metadata.get('feature_metadata', {})
        
        # Extract feature names (exclude 'ticker', 'atr', and 'ewsd' columns)
        # ATR and EWSD are normalization columns, not features to analyze
        # Use substring search to catch all variations (e.g., "atr_252_D_atr_252_lookback2")
        self.feature_names = [
            col for col in features_df.columns 
            if col != 'ticker' 
            and 'atr' not in col.lower()
            and 'ewsd' not in col.lower()
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
            
        # Initialize parameter mapping
        self._param_mapping = {}
        self._initialize_parameter_mapping()
        
    def _initialize_parameter_mapping(self):
        """Initialize parameter mapping from feature metadata."""
        from collections import defaultdict
        
        # Get feature metadata if available
        self._param_mapping = self.metadata.get('feature_metadata', {})
        self._feature_groups = defaultdict(list)
        
        # If no metadata, fall back to name parsing
        if not self._param_mapping:
            self._initialize_parameter_mapping_from_names()
            return
            
        # Group features by module and parameter combinations
        for feature, meta in self._param_mapping.items():
            # Skip if feature not in our filtered list
            if feature not in self.feature_names:
                continue
                
            # Get module name and parameters
            module = meta.get('module', '')
            params = meta.get('parameters', {})
            
            # Skip features without parameters
            if not params:
                continue
                
            # Create groups for each parameter
            for param_name, param_value in params.items():
                if param_name == 'module_name':
                    continue
                    
                # Create a unique key for this parameter
                group_key = (module, param_name)
                
                # Store parameter value and feature name
                self._feature_groups[group_key].append((str(param_value), feature))
    
    def _initialize_parameter_mapping_from_names(self):
        """Fallback method to extract parameters from feature names when metadata is not available."""
        from collections import defaultdict
        
        self._feature_groups = defaultdict(list)
        
        for feature in self.feature_names:
            # Special handling for CMMA features (e.g., 'cmma_10_D_cmma_10_252')
            if feature.startswith('cmma_') and feature.count('_') >= 4:
                # Format: cmma_<lookback>_D_cmma_<lookback>_<atr_length>
                parts = feature.split('_')
                if len(parts) >= 5 and parts[0] == 'cmma' and parts[3] == 'cmma':
                    lookback = parts[1]  # First number is lookback
                    atr_length = parts[4]  # Last number is atr_length
                    
                    # Store the mapping with proper parameter names
                    self._param_mapping[feature] = {
                        'base_name': 'cmma',
                        'module': 'cmma',
                        'parameters': {
                            'lookback': lookback,
                            'atr_length': atr_length
                        },
                        'full_name': feature
                    }
                    
                    # Add to feature groups for both parameters
                    for param_name in ['lookback', 'atr_length']:
                        group_key = ('cmma', param_name)
                        self._feature_groups[group_key].append((self._param_mapping[feature]['parameters'][param_name], feature))
                    continue
            
            # Default handling for other features
            parts = feature.split('_')
            if len(parts) < 2:
                continue
                
            # The last part is usually the parameter value
            param_value = parts[-1]
            
            # The part before the last underscore is usually the parameter name
            param_name = parts[-2] if len(parts) > 1 else None
            
            # The base name is everything before the parameter name
            base_name = '_'.join(parts[:-2]) if param_name else feature
            
            # Store the mapping
            self._param_mapping[feature] = {
                'base_name': base_name,
                'module': base_name.split('_')[0],  # First part is usually module name
                'parameters': {param_name: param_value} if param_name else {},
                'full_name': feature
            }
            
            # Add to feature groups
            if param_name and param_name.isalpha():
                group_key = (base_name, param_name)
                self._feature_groups[group_key].append((param_value, feature))
    
    def get_parameterized_features(self) -> Dict[Tuple[str, str], List[Tuple[Any, str]]]:
        """
        Get features grouped by their module and parameter.
        
        Returns
        -------
        Dict[Tuple[str, str], List[Tuple[Any, str]]]
            Dictionary mapping (module_name, param_name) to list of (param_value, feature_name) tuples
            Parameters are converted to their appropriate types (int, float, or str)
            
        Examples
        --------
        >>> # Get all parameterized features
        >>> param_groups = explorer.get_parameterized_features()
        >>> 
        >>> # Find all RSI lookback parameters
        >>> rsi_lookbacks = param_groups.get(('rsi', 'lookback'), [])
        >>> for value, feature_name in sorted(rsi_lookbacks):
        ...     print(f"RSI Lookback {value}: {feature_name}")
        """
        result = {}
        
        for (module_param, features) in self._feature_groups.items():
            # Convert parameter values to appropriate types
            converted = []
            for value, feature in features:
                try:
                    # Try to convert to int first, then float, otherwise keep as string
                    try:
                        converted_value = int(value)
                    except (ValueError, TypeError):
                        try:
                            converted_value = float(value)
                        except (ValueError, TypeError):
                            converted_value = value
                    converted.append((converted_value, feature))
                except Exception as e:
                    print(f"Warning: Could not convert parameter value '{value}' for feature '{feature}': {e}")
                    converted.append((value, feature))
            
            # Sort by parameter value
            try:
                result[module_param] = sorted(converted, key=lambda x: x[0] if isinstance(x[0], (int, float)) else str(x[0]))
            except Exception as e:
                print(f"Warning: Could not sort parameter group {module_param}: {e}")
                result[module_param] = converted
        
        return result
    
    def plot_parameter_sensitivity(
        self,
        module_name: str,
        param_name: str,
        target_col: str = 'log_return',
        metric: str = 'sortino',
        n_bins: int = 3,
        show_plot: bool = True,
        **plot_kwargs
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Plot the sensitivity of a parameter across different values.
        
        Parameters
        ----------
        module_name : str
            Name of the module (e.g., 'cmma', 'rsi')
        param_name : str
            Name of the parameter to analyze (e.g., 'lookback')
        target_col : str, default='log_return'
            Target column to use for computing metrics
        metric : str, default='sortino'
            Metric to compute. Options: 'sortino', 'sharpe', 'mean', 'std', 'max_drawdown'
        n_bins : int, default=5
            Number of bins for QuantileBinningModel
        show_plot : bool, default=True
            Whether to show the plot
        **plot_kwargs
            Additional keyword arguments passed to Plotly
            
        Returns
        -------
        Tuple[pd.DataFrame, Any]
            - DataFrame with parameter values and corresponding metrics
            - Plotly figure
        """
        from feature_selection.visualization.parameter_analysis import ParameterAnalyzer
        
        # Get the feature group
        group_key = (module_name, param_name)
        if group_key not in self._feature_groups:
            available_modules = list(set(k[0] for k in self._feature_groups.keys()))
            available_params = list(set(k[1] for k in self._feature_groups.keys() 
                                     if k[0].lower() == module_name.lower()))
            
            error_msg = [
                f"No features found for module='{module_name}' with parameter='{param_name}'."
            ]
            
            if available_modules:
                error_msg.append(f"\nAvailable modules: {available_modules}")
            if available_params:
                error_msg.append(f"\nAvailable parameters for {module_name}: {available_params}")
                
            raise ValueError(''.join(error_msg))
        
        # Initialize analyzer
        analyzer = ParameterAnalyzer(self.features_df, self.targets_df)
        
        # Get features in this group and sort by parameter value
        features = sorted(
            self._feature_groups[group_key],
            key=lambda x: float(x[0]) if str(x[0]).replace('.', '').isdigit() else x[0]
        )
        
        try:
            # Analyze parameter sensitivity
            df = analyzer.analyze_parameter(
                feature_group=features,
                target_col=target_col,
                metric=metric,
                n_bins=n_bins,
                annualization_factor=252
            )
            
            # Generate title with parameter info
            title = f"{module_name.upper()} Parameter Sensitivity: {param_name}"
            
            # Add other parameter info to title if available
            if module_name in [k[0] for k in self._feature_groups.keys()]:
                module_params = {k[1]: v for k, v in self._feature_groups.items() 
                              if k[0] == module_name}
                other_params = {k: next(iter(v))[0] 
                              for k, v in module_params.items() 
                              if k != param_name}
                if other_params:
                    param_str = ", ".join(f"{k}={v}" for k, v in other_params.items())
                    title += f"<br><sup>Other params: {param_str}</sup>"
            
            # Create the plot
            fig = analyzer.plot_parameter_sensitivity(
                df=df,
                param_name=param_name,
                metric=metric,
                title=title,
                show_plot=show_plot
            )
            
            return df, fig
            
        except Exception as e:
            # Print debug info
            print("\nDebug Info:")
            print(f"Feature group: {group_key}")
            print(f"Features found: {[f[1] for f in features]}")
            print(f"Available columns in features_df: {self.features_df.columns.tolist()}")
            print(f"Available columns in targets_df: {self.targets_df.columns.tolist()}")
            
            # Check if any features exist in the dataframe
            missing_features = [f[1] for f in features if f[1] not in self.features_df.columns]
            if missing_features:
                print(f"\nError: The following features are not in features_df: {missing_features}")
            
            raise ValueError(f"Error in parameter sensitivity analysis: {str(e)}")
    
    def plot_2d_parameter_surface(
        self,
        module_name: str,
        param1_name: str,
        param2_name: str,
        target_col: str = 'log_return',
        metric: str = 'sortino',
        n_bins: int = 5,
        show_plot: bool = True,
        plot_type: str = 'surface',  # Add this parameter
        **plot_kwargs
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Create a 3D surface plot of parameter sensitivity for two parameters.
        
        Parameters
        ----------
        module_name : str
            Name of the module (e.g., 'cmma', 'rsi')
        param1_name : str
            Name of the first parameter (x-axis)
        param2_name : str
            Name of the second parameter (y-axis)
        target_col : str, default='log_return'
            Target column to use for computing metrics
        metric : str, default='sortino'
            Metric to compute. Options: 'sortino', 'sharpe', 'mean', 'std', 'max_drawdown'
        n_bins : int, default=5
            Number of bins for QuantileBinningModel
        show_plot : bool, default=True
            Whether to show the plot
        plot_type : str, default='surface'
            Type of plot to generate: 'surface', 'scatter', 'heatmap', 'contour', 'lines'
        **plot_kwargs
            Additional keyword arguments passed to Plotly
            
        Returns
        -------
        Tuple[pd.DataFrame, Any]
            - DataFrame with parameter values and computed metrics
            - Plotly figure
        """
        from feature_selection.visualization.parameter_analysis import ParameterAnalyzer
        
        # Get all features for this module from metadata
        module_features = {}
        print(f"Building parameter grid for module '{module_name}'")
        
        # Create a mapping from feature to parameters
        feature_to_params = {}
        for feature_name, meta in self.feature_metadata.items():
            if meta.get('module', '').lower() == module_name.lower():
                feature_to_params[feature_name] = meta.get('parameters', {})
        
        # Create parameter grid with all unique parameter combinations
        param_combinations = set()
        for params in feature_to_params.values():
            if param1_name in params and param2_name in params:
                param_combinations.add((params[param1_name], params[param2_name]))
        
        # Map each parameter combination to all features that match it
        for (param1_val, param2_val) in param_combinations:
            matching_features = [
                feature for feature, params in feature_to_params.items()
                if params.get(param1_name) == param1_val and params.get(param2_name) == param2_val
            ]
            module_features[(param1_val, param2_val)] = matching_features
        
        if not module_features:
            # Provide more detailed error information
            available_modules = list(set(meta['module'] for meta in self.feature_metadata.values()))
            
            # Find available parameters for this module
            available_params = set()
            for meta in self.feature_metadata.values():
                if meta.get('module', '').lower() == module_name.lower():
                    available_params.update(meta.get('parameters', {}).keys())
            
            error_msg = [
                f"No features found for module='{module_name}' with parameters '{param1_name}' and '{param2_name}'."
            ]
            
            if available_modules:
                error_msg.append(f"\nAvailable modules: {available_modules}")
            if available_params:
                error_msg.append(f"\nAvailable parameters for {module_name}: {list(available_params)}")
            else:
                error_msg.append(f"\nNo parameters found for module '{module_name}'")
                
            # List features that match the module
            module_features_list = [
                feature for feature, meta in self.feature_metadata.items()
                if meta.get('module', '').lower() == module_name.lower()
            ]
            
            if module_features_list:
                error_msg.append(f"\nFeatures found for module '{module_name}': {module_features_list[:5]}")
                if len(module_features_list) > 5:
                    error_msg.append(f" (and {len(module_features_list)-5} more)")
            
            raise ValueError(''.join(error_msg))
        
        # Initialize analyzer
        analyzer = ParameterAnalyzer(self.features_df, self.targets_df)
        
        try:
            # Analyze 2D parameter sensitivity
            df = analyzer.analyze_2d_parameters(
                feature_grid=module_features,
                target_col=target_col,
                metric=metric,
                n_bins=n_bins,
                annualization_factor=252
            )
            
            # Create the 3D surface plot
            fig = analyzer.plot_2d_parameter_surface(
                df=df,
                param1=param1_name,
                param2=param2_name,
                metric=metric,
                title=f"{module_name.upper()} {metric.capitalize()} vs {param1_name} and {param2_name}",
                show_plot=show_plot,
                plot_type=plot_type  # Pass the plot type
            )
            
            return df, fig
            
        except Exception as e:
            # Print debug info
            print("\nDebug Info:")
            print(f"Module: {module_name}")
            print(f"Parameters: {param1_name}, {param2_name}")
            print(f"Feature grid: {list(module_features.items())[:5]}")
            print(f"Features in dataframe: {self.features_df.columns.tolist()}")
            print(f"Targets in dataframe: {self.targets_df.columns.tolist()}")
            
            raise ValueError(f"Error in 2D parameter sensitivity analysis: {str(e)}")
    
    def plot_all_deciles(
        self,
        n_bins: int = 10,
        target_col: str = 'log_return',
        figsize: Tuple[int, int] = (12, 8),
        plot_type: str = "bar",
        save_dir: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, plt.Figure]:
        """
        Plot decile analysis for all features.
        
        Parameters
        ----------
        n_bins : int, default=10
            Number of bins
        target_col : str, default='log_return'
            Target column to use
        figsize : Tuple[int, int], default=(12, 8)
            Figure size
        plot_type : str, default="bar"
            Plot type
        save_dir : Optional[str], default=None
            Directory to save figures (one per feature)
        verbose : bool, default=True
            Print progress
            
        Returns
        -------
        Dict[str, plt.Figure]
            Dictionary mapping feature names to figures
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Plotting decile analysis for {self.n_features} features")
            print(f"Target: {target_col}")
            print(f"{'='*70}")
        
        figures = {}
        for i, feature_name in enumerate(self.feature_names, 1):
            if verbose:
                print(f"\n[{i}/{self.n_features}] {feature_name}")
            
            try:
                # Determine save path
                save_path = None
                if save_dir:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{feature_name}_deciles.png")
                
                # Plot
                fig = self.plot_deciles(
                    feature_name=feature_name,
                    n_bins=n_bins,
                    target_col=target_col,
                    figsize=figsize,
                    plot_type=plot_type,
                    save_path=save_path
                )
                figures[feature_name] = fig
                
                if verbose:
                    print(f"  ✓ Complete")
                    
            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed: {e}")
                continue
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Completed {len(figures)}/{self.n_features} features")
            print(f"{'='*70}")
        
        return figures
    
    def plot_2bin(
        self,
        feature_name: str,
        target_col: str = 'log_return',
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot 2-bin analysis (positive vs negative feature values).
        
        Parameters
        ----------
        feature_name : str
            Name of the feature
        target_col : str, default='log_return'
            Target column to use
        figsize : Tuple[int, int], default=(10, 6)
            Figure size
        save_path : Optional[str], default=None
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        # Validate
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found")
        if target_col not in self.targets_df.columns:
            raise ValueError(f"Target '{target_col}' not found")
        
        # Get data
        feature_data = self.features_df[feature_name]
        target_data = self.targets_df[target_col]
        
        # Check numeric
        if not pd.api.types.is_numeric_dtype(feature_data):
            raise TypeError(f"Feature '{feature_name}' is non-numeric")
        
        # Plot
        fig, bin_table = plot_2bin_analysis(
            feature_data=feature_data,
            target_data=target_data,
            feature_name=feature_name,
            figsize=figsize,
            save_path=save_path
        )
        
        # Store results
        if '2bin_analysis' not in self.results:
            self.results['2bin_analysis'] = {}
        self.results['2bin_analysis'][feature_name] = {
            'target_col': target_col,
            'bin_data': bin_table
        }
        
        return fig
    
    def plot_deciles(
        self,
        feature_name: str,
        n_bins: int = 10,
        target_col: str = 'log_return',
        figsize: Tuple[int, int] = (12, 8),
        plot_type: str = "bar",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot decile analysis for a single feature.
        
        Parameters
        ----------
        feature_name : str
            Name of the feature
        n_bins : int, default=10
            Number of bins
        target_col : str, default='log_return'
            Target column to use
        figsize : Tuple[int, int], default=(12, 8)
            Figure size
        plot_type : str, default="bar"
            Plot type
        save_path : Optional[str], default=None
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        # Validate
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found")
        if target_col not in self.targets_df.columns:
            raise ValueError(f"Target '{target_col}' not found")
        
        # Get data
        feature_data = self.features_df[feature_name]
        target_data = self.targets_df[target_col]
        
        # Check numeric
        if not pd.api.types.is_numeric_dtype(feature_data):
            raise TypeError(f"Feature '{feature_name}' is non-numeric")
        
        # Plot using the imported function
        fig, bin_table = plot_decile_analysis(
            feature_data=feature_data,
            target_data=target_data,
            feature_name=feature_name,
            n_bins=n_bins,
            figsize=figsize,
            plot_type=plot_type,
            save_path=save_path
        )
        
        # Store results
        if 'decile_analysis' not in self.results:
            self.results['decile_analysis'] = {}
        self.results['decile_analysis'][feature_name] = {
            'target_col': target_col,
            'n_bins': n_bins,
            'bin_data': bin_table
        }
        
        return fig
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for all features.
        
        Returns
        -------
        pd.DataFrame
            Summary with columns: feature_name, n_samples, mean, std, min, max, n_nan
        """
        summaries = []
        for feature_name in self.feature_names:
            feature_data = self.features_df[feature_name]
            summaries.append({
                'feature_name': feature_name,
                'n_samples': len(feature_data),
                'n_nan': feature_data.isna().sum(),
                'mean': feature_data.mean(),
                'std': feature_data.std(),
                'min': feature_data.min(),
                'max': feature_data.max(),
                'dtype': str(feature_data.dtype)
            })
        
        return pd.DataFrame(summaries)
    
    def get_correlations(
        self,
        target_col: str = 'log_return',
        method: str = 'spearman'
    ) -> pd.Series:
        """
        Compute correlations between features and target.
        
        Parameters
        ----------
        target_col : str, default='log_return'
            Target column to use
        method : str, default='spearman'
            Correlation method: 'pearson' or 'spearman'
            
        Returns
        -------
        pd.Series
            Correlations for each feature
        """
        if target_col not in self.targets_df.columns:
            raise ValueError(f"Target '{target_col}' not found")
        
        target_data = self.targets_df[target_col]
        correlations = {}
        
        for feature_name in self.feature_names:
            feature_data = self.features_df[feature_name]
            
            # Skip non-numeric
            if not pd.api.types.is_numeric_dtype(feature_data):
                correlations[feature_name] = np.nan
                continue
            
            # Compute correlation
            if method == 'pearson':
                corr = feature_data.corr(target_data)
            elif method == 'spearman':
                corr = feature_data.corr(target_data, method='spearman')
            else:
                raise ValueError(f"Unknown method: {method}")
            
            correlations[feature_name] = corr
        
        return pd.Series(correlations, name=f'{method}_correlation')
    
    def plot_feature_target_correlations(
        self,
        target_col: str = 'log_return',
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'RdBu_r',
        annot: bool = True,
        fmt: str = '.2f',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot heatmap of feature-target correlations using Spearman rank correlation.
        
        Spearman correlation captures monotonic relationships, making it ideal for
        quantile-binned features that have strong predictive power through monotonic
        (but not necessarily linear) relationships with the target.
        
        Parameters
        ----------
        target_col : str, default='log_return'
            Target column to use
        figsize : Tuple[int, int], default=(10, 8)
            Figure size
        cmap : str, default='RdBu_r'
            Colormap for heatmap
        annot : bool, default=True
            Whether to annotate cells with correlation values
        fmt : str, default='.2f'
            Format string for annotations
        save_path : Optional[str], default=None
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
            
        Notes
        -----
        Uses Spearman rank correlation which measures monotonic relationships:
        - Captures non-linear but monotonic patterns
        - Robust to outliers
        - Perfect for quantile-binned features
        - Range: [-1, 1] where ±1 indicates perfect monotonic relationship
        """
        import seaborn as sns
        
        # Compute Spearman correlations for all features
        correlations = self.get_correlations(target_col=target_col, method='spearman')
        
        # Filter out NaN values
        correlations = correlations.dropna()
        
        if correlations.empty:
            raise ValueError("No valid correlations computed (all features are non-numeric)")
        
        # Create a DataFrame for the heatmap (single column)
        corr_df = pd.DataFrame({
            target_col: correlations
        })
        
        # Sort by absolute correlation value
        corr_df = corr_df.reindex(correlations.abs().sort_values(ascending=False).index)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            corr_df,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Spearman Correlation'},
            ax=ax
        )
        
        ax.set_title(
            f'Feature-Target Correlations (Spearman)\n'
            f'Target: {target_col}\n'
            f'Captures monotonic relationships',
            fontsize=12,
            pad=20
        )
        ax.set_xlabel('')
        ax.set_ylabel('Features (sorted by |correlation|)')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Store results
        self.results['feature_target_correlations'] = {
            'target_col': target_col,
            'method': 'spearman',
            'correlations': correlations
        }
        
        return fig
    
    def plot_feature_correlations(
        self,
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = 'coolwarm',
        annot: bool = False,
        fmt: str = '.2f',
        save_path: Optional[str] = None,
        mask_diagonal: bool = True
    ) -> plt.Figure:
        """
        Plot heatmap of intra-feature correlations using Pearson correlation.
        
        This shows linear relationships between features, useful for identifying
        redundant or highly correlated features.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default=(12, 10)
            Figure size
        cmap : str, default='coolwarm'
            Colormap for heatmap
        annot : bool, default=False
            Whether to annotate cells with correlation values (can be cluttered for many features)
        fmt : str, default='.2f'
            Format string for annotations
        save_path : Optional[str], default=None
            Path to save figure
        mask_diagonal : bool, default=True
            Whether to mask the diagonal (self-correlations)
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
            
        Notes
        -----
        Uses Pearson correlation which measures linear relationships:
        - Standard correlation metric
        - Range: [-1, 1] where ±1 indicates perfect linear relationship
        - Useful for identifying redundant features
        - Features with |correlation| > 0.8 are often considered highly correlated
        """
        import seaborn as sns
        
        # Get only numeric features
        numeric_features = [
            col for col in self.feature_names
            if pd.api.types.is_numeric_dtype(self.features_df[col])
        ]
        
        if len(numeric_features) < 2:
            raise ValueError("Need at least 2 numeric features for correlation matrix")
        
        # Compute correlation matrix
        corr_matrix = self.features_df[numeric_features].corr(method='pearson')
        
        # Create mask for upper triangle if desired
        mask = None
        if mask_diagonal:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Pearson Correlation'},
            ax=ax
        )
        
        ax.set_title(
            f'Intra-Feature Correlations (Pearson)\n'
            f'{len(numeric_features)} features\n'
            f'Measures linear relationships',
            fontsize=12,
            pad=20
        )
        
        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Store results
        self.results['feature_correlations'] = {
            'method': 'pearson',
            'correlation_matrix': corr_matrix
        }
        
        return fig
    
    def plot_distributions(
        self,
        figsize: Tuple[int, int] = (12, 6),
        bins: int = 50,
        show_stats: bool = True,
        save_dir: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, plt.Figure]:
        """
        Plot distribution analysis for all features.
        
        Creates histogram + KDE + Q-Q plot for each feature to assess
        distribution shape, normality, and statistical properties.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default=(12, 6)
            Figure size for each plot
        bins : int, default=50
            Number of histogram bins
        show_stats : bool, default=True
            Whether to show statistical summary box
        save_dir : Optional[str], default=None
            Directory to save figures (one per feature)
        verbose : bool, default=True
            Print progress
            
        Returns
        -------
        Dict[str, plt.Figure]
            Dictionary mapping feature names to figures
            
        Examples
        --------
        >>> # Plot distributions for all features in the explorer
        >>> figures = explorer.plot_distributions(bins=50)
        >>> plt.show()  # Show all figures
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Plotting distributions for {self.n_features} features")
            print(f"{'='*70}")
        
        figures = {}
        for i, feature_name in enumerate(self.feature_names, 1):
            if verbose:
                print(f"\n[{i}/{self.n_features}] {feature_name}")
            
            try:
                # Determine save path
                save_path = None
                if save_dir:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{feature_name}_distribution.png")
                
                # Get feature data
                feature_data = self.features_df[feature_name]
                
                # Plot using abstracted function
                fig = plot_feature_distribution(
                    feature_data=feature_data,
                    feature_name=feature_name,
                    figsize=figsize,
                    bins=bins,
                    show_stats=show_stats,
                    save_path=save_path
                )
                
                # Store results
                if 'distributions' not in self.results:
                    self.results['distributions'] = {}
                self.results['distributions'][feature_name] = {
                    'n_samples': len(feature_data.dropna()),
                    'mean': feature_data.mean(),
                    'std': feature_data.std(),
                    'skew': feature_data.skew(),
                    'kurtosis': feature_data.kurtosis()
                }
                figures[feature_name] = fig
                
                if verbose:
                    print(f"  ✓ Complete")
                    
            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed: {e}")
                continue
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Completed {len(figures)}/{self.n_features} features")
            print(f"{'='*70}")
        
        return figures
    
    def plot_timeseries(
        self,
        figsize: Tuple[int, int] = (14, 6),
        show_rolling_mean: bool = True,
        rolling_window: int = 20,
        show_rolling_std: bool = True,
        save_dir: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, plt.Figure]:
        """
        Plot time series analysis for all features.
        
        Creates time series plots with rolling statistics for each feature
        to assess temporal patterns, trends, and volatility.
        
        Parameters
        ----------
        figsize : Tuple[int, int], default=(14, 6)
            Figure size for each plot
        show_rolling_mean : bool, default=True
            Whether to show rolling mean overlay
        rolling_window : int, default=20
            Window size for rolling statistics (in days for daily data)
        show_rolling_std : bool, default=True
            Whether to show ±2σ rolling standard deviation bands
        save_dir : Optional[str], default=None
            Directory to save figures (one per feature)
        verbose : bool, default=True
            Print progress
            
        Returns
        -------
        Dict[str, plt.Figure]
            Dictionary mapping feature names to figures
            
        Examples
        --------
        >>> # Plot time series for all features in the explorer
        >>> figures = explorer.plot_timeseries(rolling_window=50)
        >>> plt.show()  # Show all figures
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Plotting time series for {self.n_features} features")
            print(f"{'='*70}")
        
        figures = {}
        for i, feature_name in enumerate(self.feature_names, 1):
            if verbose:
                print(f"\n[{i}/{self.n_features}] {feature_name}")
            
            try:
                # Determine save path
                save_path = None
                if save_dir:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{feature_name}_timeseries.png")
                
                # Get feature data
                feature_data = self.features_df[feature_name]
                
                # Plot using abstracted function
                fig = plot_feature_timeseries(
                    feature_data=feature_data,
                    feature_name=feature_name,
                    figsize=figsize,
                    show_rolling_mean=show_rolling_mean,
                    rolling_window=rolling_window,
                    show_rolling_std=show_rolling_std,
                    save_path=save_path
                )
                
                # Store results
                if 'timeseries' not in self.results:
                    self.results['timeseries'] = {}
                self.results['timeseries'][feature_name] = {
                    'date_range': (feature_data.index.min(), feature_data.index.max()),
                    'n_samples': len(feature_data.dropna()),
                    'rolling_window': rolling_window
                }
                figures[feature_name] = fig
                
                if verbose:
                    print(f"  ✓ Complete")
                    
            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed: {e}")
                continue
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Completed {len(figures)}/{self.n_features} features")
            print(f"{'='*70}")
        
        return figures
    
    def run_permutation_test(
        self,
        target_col: str = 'log_return',
        base_model: Optional[Any] = None,
        metric: Optional[Any] = None,
        nreps: int = 100,
        n_jobs: int = -1,
        random_seed: Optional[int] = 42,
        alpha: float = 0.1,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run in-sample permutation test to assess feature significance.
        
        This method tests whether each feature has predictive power by:
        1. Fitting a base model (default: quantile binning) on each feature
        2. Computing a performance metric (default: Sortino ratio)
        3. Shuffling feature values and repeating nreps times
        4. Computing p-values: fraction of permutations with metric >= original
        
        Features with low p-values (< alpha) are considered significant.
        
        Parameters
        ----------
        target_col : str, default='log_return'
            Target column to use for prediction
        base_model : Optional[Any], default=None
            Base model to use (must have sklearn API: fit, predict)
            If None, uses QuantileBinningModel(n_bins=3, selection_metric='sortino')
        metric : Optional[Any], default=None
            Objective metric to compute (must be callable: metric(returns) -> float)
            If None, uses SortinoRatio(annualization_factor=252)
        nreps : int, default=100
            Number of permutation replications
        n_jobs : int, default=-1
            Number of parallel jobs (-1 = all CPUs)
        random_seed : Optional[int], default=42
            Random seed for reproducibility
        alpha : float, default=0.05
            Significance level
        verbose : bool, default=True
            Print progress
            
        Returns
        -------
        pd.DataFrame
            Results with columns:
            - feature: Feature name
            - original_criterion: Original metric value
            - pval: Permutation p-value
            - significant: Whether feature is significant (pval <= alpha)
            
        Examples
        --------
        >>> # Run permutation test with default settings
        >>> results = explorer.run_permutation_test(
        ...     target_col='log_return_ewsd',
        ...     nreps=100
        ... )
        >>> 
        >>> # View significant features
        >>> print(results[results['significant']])
        >>> 
        >>> # Use custom model and metric
        >>> from feature_selection.base_models import QuantileBinningModel
        >>> from feature_selection.objective_metric import SharpeRatio
        >>> 
        >>> model = QuantileBinningModel(n_bins=5, selection_metric='mean')
        >>> metric = SharpeRatio(annualization_factor=252)
        >>> 
        >>> results = explorer.run_permutation_test(
        ...     target_col='log_return',
        ...     base_model=model,
        ...     metric=metric,
        ...     nreps=200
        ... )
        
        Notes
        -----
        This is an IN-SAMPLE test, which means:
        - The model is fit and evaluated on the SAME data
        - Results may be optimistic due to overfitting
        - Use for initial feature screening
        - For more realistic estimates, use cross-validation or walk-forward tests
        
        The permutation test works by:
        1. Computing the original metric for each feature
        2. Shuffling each feature independently (breaks feature-target relationship)
        3. Re-computing the metric on shuffled data
        4. Counting how many permutations achieve metric >= original
        5. p-value = (count + 1) / (nreps + 1)
        
        Low p-values indicate the feature has genuine predictive power.
        """
        # Validate target column
        if target_col not in self.targets_df.columns:
            raise ValueError(
                f"Target '{target_col}' not found. "
                f"Available targets: {list(self.targets_df.columns)}"
            )
        
        # Set defaults
        if base_model is None:
            base_model = QuantileBinningModel(n_bins=3, selection_metric='sortino')
        
        if metric is None:
            metric = SortinoRatio(annualization_factor=252)
        
        # Combine features and target into single DataFrame
        data = self.features_df.copy()
        data[target_col] = self.targets_df[target_col]
        
        # Create a partial function with the required parameters
        from functools import partial
        criterion_func = partial(
            self._permutation_criterion,
            target_col=target_col,
            base_model=base_model,
            metric=metric,
            verbose=verbose
        )
        
        # Create permutation strategy and engine
        strategy = FeaturePermutationStrategy()
        engine = PermutationEngine(strategy, n_jobs=n_jobs, verbose=verbose)
        
        # Run permutation test
        results = engine.run_permutation_test(
            data=data,
            feature_cols=self.feature_names,
            criterion_func=criterion_func,
            nreps=nreps,
            random_seed=random_seed,
            alpha=alpha,
            target_col=target_col  # Pass to strategy for validation
        )
        
        # Store results
        self.results['permutation_test'] = {
            'target_col': target_col,
            'model': str(base_model),
            'metric': str(metric),
            'nreps': nreps,
            'alpha': alpha,
            'results': results
        }
        
        return results
    
    def get_feature(self, feature_name: str) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Get a single feature and all targets.
        
        Parameters
        ----------
        feature_name : str
            Name of the feature
            
        Returns
        -------
        Tuple[pd.Series, pd.DataFrame]
            (feature_series, targets_df)
        """
        if feature_name not in self.feature_names:
            raise ValueError(
                f"Feature '{feature_name}' not found. "
                f"Available: {self.feature_names}"
            )
        
        return self.features_df[feature_name], self.targets_df
    
    def sensitivity_analysis(
        self,
        target_col: str = 'log_return',
        base_model: Optional[Any] = None,
        metric: Optional[Any] = None,
        nreps: int = 100,
        n_jobs: int = -1,
        random_seed: Optional[int] = 42,
        alpha: float = 0.1,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis to assess feature robustness.
        
        This method tests whether each feature has robust predictive power by:
        1. Fitting a base model (default: quantile binning) on each feature
        2. Computing a performance metric (default: Sortino ratio)
        3. Shuffling feature values and repeating nreps times
        4. Computing p-values: fraction of permutations with metric >= original
        
        Features with low p-values (< alpha) are considered robust.
        
        Parameters
        ----------
        target_col : str, default='log_return'
            Target column to use for prediction
        base_model : Optional[Any], default=None
            Base model to use (must have sklearn API: fit, predict)
            If None, uses QuantileBinningModel(n_bins=3, selection_metric='sortino')
        metric : Optional[Any], default=None
            Objective metric to compute (must be callable: metric(returns) -> float)
            If None, uses SortinoRatio(annualization_factor=252)
        nreps : int, default=100
            Number of permutation replications
        n_jobs : int, default=-1
            Number of parallel jobs (-1 = all CPUs)
        random_seed : Optional[int], default=42
            Random seed for reproducibility
        alpha : float, default=0.05
            Significance level
        verbose : bool, default=True
            Print progress
            
        Returns
        -------
        pd.DataFrame
            Results with columns:
            - feature: Feature name
            - original_criterion: Original metric value
            - pval: Permutation p-value
            - robust: Whether feature is robust (pval <= alpha)
            
        Examples
        --------
        >>> # Run sensitivity analysis with default settings
        >>> results = explorer.sensitivity_analysis(
        ...     target_col='log_return_ewsd',
        ...     nreps=100
        ... )
        >>> 
        >>> # View robust features
        >>> print(results[results['robust']])
        >>> 
        >>> # Use custom model and metric
        >>> from feature_selection.base_models import QuantileBinningModel
        >>> from feature_selection.objective_metric import SharpeRatio
        >>> 
        >>> model = QuantileBinningModel(n_bins=5, selection_metric='mean')
        >>> metric = SharpeRatio(annualization_factor=252)
        >>> 
        >>> results = explorer.sensitivity_analysis(
        ...     target_col='log_return',
        ...     base_model=model,
        ...     metric=metric,
        ...     nreps=200
        ... )
        
        Notes
        -----
        This is an IN-SAMPLE test, which means:
        - The model is fit and evaluated on the SAME data
        - Results may be optimistic due to overfitting
        - Use for initial feature screening
        - For more realistic estimates, use cross-validation or walk-forward tests
        
        The sensitivity analysis works by:
        1. Computing the original metric for each feature
        2. Shuffling each feature independently (breaks feature-target relationship)
        3. Re-computing the metric on shuffled data
        4. Counting how many permutations achieve metric >= original
        5. p-value = (count + 1) / (nreps + 1)
        
        Low p-values indicate the feature has genuine predictive power.
        """
        # Validate target column
        if target_col not in self.targets_df.columns:
            raise ValueError(
                f"Target '{target_col}' not found. "
                f"Available targets: {list(self.targets_df.columns)}"
            )
        
        # Set defaults
        if base_model is None:
            base_model = QuantileBinningModel(n_bins=3, selection_metric='sortino')
        
        if metric is None:
            metric = SortinoRatio(annualization_factor=252)
        
        # Combine features and target into single DataFrame
        data = self.features_df.copy()
        data[target_col] = self.targets_df[target_col]
        
        # Create a partial function with the required parameters
        from functools import partial
        criterion_func = partial(
            self._permutation_criterion,
            target_col=target_col,
            base_model=base_model,
            metric=metric,
            verbose=verbose
        )
        
        # Create permutation strategy and engine
        strategy = FeaturePermutationStrategy()
        engine = PermutationEngine(strategy, n_jobs=n_jobs, verbose=verbose)
        
        # Run sensitivity analysis
        results = engine.run_permutation_test(
            data=data,
            feature_cols=self.feature_names,
            criterion_func=criterion_func,
            nreps=nreps,
            random_seed=random_seed,
            alpha=alpha,
            target_col=target_col  # Pass to strategy for validation
        )
        
        # Store results
        self.results['sensitivity_analysis'] = {
            'target_col': target_col,
            'model': str(base_model),
            'metric': str(metric),
            'nreps': nreps,
            'alpha': alpha,
            'results': results
        }
        
        return results
    
    def __str__(self) -> str:
        """Human-readable string."""
        return (
            f"FeatureExplorer(n_features={self.n_features}, "
            f"n_samples={self.n_samples}, "
            f"date_range={self.date_range[0].date()} to {self.date_range[1].date()}, "
            f"n_parameter_groups={len(self._feature_groups)})"
        )
    
    def __str__(self) -> str:
        """Human-readable string."""
        lines = [f"FeatureExplorer with {self.n_features} features:"]
        
        # Show first few features
        for feature_name in self.feature_names[:5]:
            lines.append(f"  - {feature_name}")
        
        if self.n_features > 5:
            lines.append(f"  ... and {self.n_features - 5} more")
        
        lines.append(f"\nSamples: {self.n_samples}")
        lines.append(f"Date range: {self.date_range[0].date()} to {self.date_range[1].date()}")
        
        if self.has_ticker:
            lines.append(f"Tickers: {self.tickers}")
        
        return '\n'.join(lines)
    
    def __getitem__(self, feature_name: str) -> Tuple[pd.Series, pd.DataFrame]:
        """Allow dict-like access to features."""
        return self.get_feature(feature_name)
    
    def __iter__(self):
        """Iterate over feature names."""
        return iter(self.feature_names)
    
    def __len__(self):
        """Number of features."""
        return self.n_features
