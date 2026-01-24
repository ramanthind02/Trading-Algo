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
import utils.helpers as helpers
from datetime import datetime as dt
from metrics.plotting.decile_plots import plot_decile_analysis, plot_2bin_analysis, plot_uniform_binning
from metrics.plotting.distribution import plot_feature_distribution, plot_feature_timeseries
from metrics.plotting.feature_explorer_plots import (
    plot_all_feature_deciles,
    plot_feature_2bin,
    plot_all_feature_uniform_bins,
    plot_feature_target_correlations as plot_feature_target_correlations_pure,
    plot_feature_correlation_matrix,
    plot_all_feature_distributions,
    plot_all_feature_timeseries,
    plot_feature_signal_cumsum,
    combine_decile_plots,
    combine_signal_cumsum_plots,
    combine_distribution_plots,
    combine_timeseries_plots,
)
from metrics.plotting.parameter_plots import (
    plot_parameter_sensitivity as plot_parameter_sensitivity_pure,
    plot_2d_parameter_surface as plot_2d_parameter_surface_pure,
)
from utils.permutation_test.permutation_engine import (
    PermutationEngine,
    FeaturePermutationStrategy
)



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
    >>> result = extractor.extract(bias_node_specs=[...])
    >>> 
    >>> # Step 2: Create FeatureExplorer from extracted dataframes
    >>> # Pass metadata for optimal performance (uses standardized parsing)
    >>> from eda.feature_explorer import FeatureExplorer
    >>> explorer = FeatureExplorer(
    ...     features_df=result['features'],
    ...     targets_df=result['targets'],
    ...     metadata={'feature_metadata': result['feature_metadata']}
    ... )
    >>> 
    >>> # Step 3: Analyze features
    >>> explorer.plot_all_deciles(n_bins=10)
    >>> summary = explorer.get_summary()
    >>> 
    >>> # Note: Normalization columns (ATR/EWSD) are in result['normalization']
    >>> # They don't clutter the feature analysis!
    >>> # Note: If metadata is not passed, FeatureExplorer will parse names automatically
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
        """Initialize parameter mapping from feature metadata.
        
        Trusts metadata from FeatureExtractor. If metadata is missing,
        falls back to parsing feature names using the standardized parser.
        """
        from collections import defaultdict
        
        # Get feature metadata if available (from FeatureExtractor)
        self._param_mapping = self.metadata.get('feature_metadata', {})
        self._feature_groups = defaultdict(list)
        
        # If no metadata provided, parse feature names using standardized parser
        if not self._param_mapping:
            for feature in self.feature_names:
                try:
                    parsed = helpers.parse_feature_column_name(feature)
                    if parsed and parsed.get('module') and parsed.get('params') is not None:
                        self._param_mapping[feature] = {
                            'module': parsed['module'],
                            'parameters': parsed['params'],
                            'base_name': parsed['module'],
                            'full_name': feature
                        }
                except Exception:
                    # Skip features that can't be parsed
                    continue
            
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
                # Metadata already has correct types from parse_feature_column_name
                self._feature_groups[group_key].append((param_value, feature))
    
    def get_parameterized_features(self) -> Dict[Tuple[str, str], List[Tuple[Any, str]]]:
        """
        Get features grouped by their module and parameter.
        
        Returns
        -------
        Dict[Tuple[str, str], List[Tuple[Any, str]]]
            Dictionary mapping (module_name, param_name) to list of (param_value, feature_name) tuples
            Parameter values are already correctly typed from metadata (int, float, or str)
            
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
            # Parameter values are already correctly typed from metadata
            # Sort by parameter value
            try:
                result[module_param] = sorted(
                    features, 
                    key=lambda x: x[0] if isinstance(x[0], (int, float)) else str(x[0])
                )
            except Exception as e:
                print(f"Warning: Could not sort parameter group {module_param}: {e}")
                result[module_param] = features
        
        return result
    
    def _has_parameterized_features(self) -> bool:
        """
        Check if any features have parameters (for conditional analysis).
        
        Returns
        -------
        bool
            True if any features have parameters, False otherwise
        """
        return len(self._feature_groups) > 0
    
    def _count_parameters_per_module(self) -> Dict[str, int]:
        """
        Count the number of unique parameters per module.
        
        Returns
        -------
        Dict[str, int]
            Dictionary mapping module_name to number of unique parameters
        """
        module_param_counts = {}
        for (module_name, param_name), features in self._feature_groups.items():
            if module_name not in module_param_counts:
                module_param_counts[module_name] = set()
            module_param_counts[module_name].add(param_name)
        
        return {module: len(params) for module, params in module_param_counts.items()}

    @staticmethod
    def _canonicalize_param_name(name: str) -> str:
        """Normalize parameter name for user input matching.
        
        Used to normalize user-provided parameter names (e.g., from plot_parameter_sensitivity)
        to match the standardized naming convention. This is only for user input normalization,
        not for parsing feature names (which are already standardized).
        
        - If already camelCase (contains uppercase and no separators), return as-is.
        - Otherwise, convert snake/kebab to camelCase.
        """
        if not isinstance(name, str):
            return str(name)
        token = name.strip()
        # preserve existing camelCase
        if ('_' not in token and '-' not in token) and any(ch.isupper() for ch in token[1:]):
            return token
        token = token.replace('-', '_')
        parts = [p for p in token.split('_') if p]
        if not parts:
            return ''
        head = parts[0].lower()
        tail = ''.join(p.capitalize() for p in parts[1:])
        return head + tail
    
    def plot_parameter_sensitivity(
        self,
        module_name: str,
        param_name: str,
        target_col: str = 'log_return',
        metric: Optional[Any] = None,
        n_bins: int = 3,
        show_plot: bool = True,
        feature_name: Optional[str] = None,
        base_model: Optional[Any] = None,
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
        metric : Optional[Any], default=None
            Metric object from metrics.performance (e.g., SortinoRatio, SharpeRatio).
            Must have a .compute() method. If None, defaults to SortinoRatio.
        n_bins : int, default=3
            Number of bins (model-specific)
        show_plot : bool, default=True
            Whether to show the plot
        feature_name : Optional[str], default=None
            Optional filter for specific feature type (e.g., 'signal')
        base_model : Optional[Any], default=None
            Model instance with fit/predict methods. If None, uses QuantileBinningModel.
        **plot_kwargs
            Additional keyword arguments passed to Plotly
            
        Examples
        --------
        >>> from metrics.performance import SortinoRatio
        >>> from feature_selection.base_models.quantile_binning import QuantileBinningModel
        >>> 
        >>> metric = SortinoRatio(annualization_factor=252)
        >>> model = QuantileBinningModel(n_bins=3, selection_metric='sortino')
        >>> 
        >>> df, fig = explorer.plot_parameter_sensitivity(
        ...     module_name='rsi',
        ...     param_name='lookback',
        ...     feature_name='signal',
        ...     target_col='log_return',
        ...     base_model=model,
        ...     metric=metric
        ... )
            
        Returns
        -------
        Tuple[pd.DataFrame, Any]
            - DataFrame with parameter values and corresponding metrics
            - Plotly figure
        """
        from eda.parameter_analysis import ParameterAnalyzer
        
        # Parameter names in metadata are now in camelCase
        # Convert user input to camelCase to match
        param_name_camel = self._canonicalize_param_name(param_name)
        group_key = (module_name, param_name_camel)
        
        if group_key not in self._feature_groups:
            available_modules = list(set(k[0] for k in self._feature_groups.keys()))
            available_params = sorted(list(set(k[1] for k in self._feature_groups.keys() 
                                 if k[0].lower() == module_name.lower())))
            
            error_msg = [
                f"No features found for module='{module_name}' with parameter='{param_name}'."
            ]
            if available_modules:
                error_msg.append(f"\nAvailable modules: {available_modules}")
            if available_params:
                error_msg.append(f"\nAvailable parameters for {module_name}: {available_params}")
            raise ValueError(''.join(error_msg))
        
        # Use the camelCase version
        param_name = param_name_camel
        
        analyzer = ParameterAnalyzer(self.features_df, self.targets_df)
        features = list(self._feature_groups[group_key])
        
        # Optional: filter by feature_name token (e.g., 'signal' vs 'signalBool')
        if feature_name is not None:
            try:
                filtered = []
                for value, feat in features:
                    meta = helpers.parse_feature_column_name(feat)
                    if str(meta.get('feature')).lower() == str(feature_name).lower():
                        filtered.append((value, feat))
                if filtered:
                    features = filtered
            except Exception:
                pass
        else:
            # Heuristic: prefer non-bool features by default
            non_bool = []
            try:
                for value, feat in features:
                    meta = helpers.parse_feature_column_name(feat)
                    feat_tok = str(meta.get('feature', ''))
                    if 'bool' not in feat_tok.lower():
                        non_bool.append((value, feat))
                if non_bool:
                    features = non_bool
            except Exception:
                pass

        # Sort by parameter value
        features = sorted(
            features,
            key=lambda x: float(x[0]) if str(x[0]).replace('.', '').isdigit() else x[0]
        )
        
        try:
            # Use default metric if not provided
            if metric is None:
                from metrics.performance import SortinoRatio
                metric = SortinoRatio(annualization_factor=252)
            
            # Infer metric name from metric object for display/plotting
            from eda.parameter_analysis import _get_metric_name_from_object
            metric_name = _get_metric_name_from_object(metric)
            
            # Analyze parameter with metric object
            df = analyzer.analyze_parameter(
                feature_group=features,
                target_col=target_col,
                metric=metric,
                n_bins=n_bins,
                base_model=base_model
            )
            title = f"{module_name.upper()} Parameter Sensitivity: {param_name}"
            if module_name in [k[0] for k in self._feature_groups.keys()]:
                module_params = {k[1]: v for k, v in self._feature_groups.items() 
                              if k[0] == module_name}
                other_params = {k: next(iter(v))[0] 
                              for k, v in module_params.items() 
                              if k != param_name}
                if other_params:
                    param_str = ", ".join(f"{k}={v}" for k, v in other_params.items())
                    title += f"<br><sup>Other params: {param_str}</sup>"
            fig = plot_parameter_sensitivity_pure(
                df=df,
                param_name=param_name,
                metric=metric_name,
                title=title,
                show_plot=show_plot
            )
            return df, fig
        except Exception as e:
            print("\nDebug Info:")
            print(f"Feature group: {group_key}")
            print(f"Features found: {[f[1] for f in features]}")
            print(f"Available columns in features_df: {self.features_df.columns.tolist()}")
            print(f"Available columns in targets_df: {self.targets_df.columns.tolist()}")
            try:
                print("Result df (first 10 rows):")
                print(df.head(10))
                print("Result df columns:", list(df.columns))
            except Exception:
                pass
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
        metric: Optional[Any] = None,
        n_bins: int = 5,
        show_plot: bool = True,
        plot_type: str = 'surface',
        feature_name: Optional[str] = None,
        base_model: Optional[Any] = None,
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
        metric : Optional[Any], default=None
            Metric object from metrics.performance (e.g., SortinoRatio, SharpeRatio).
            Must have a .compute() method. If None, defaults to SortinoRatio.
        n_bins : int, default=5
            Number of bins (model-specific)
        show_plot : bool, default=True
            Whether to show the plot
        plot_type : str, default='surface'
            Type of plot to generate: 'surface', 'scatter', 'heatmap', 'contour', 'lines'
        feature_name : Optional[str], default=None
            Optional filter for specific feature type (e.g., 'signal')
        base_model : Optional[Any], default=None
            Model instance with fit/predict methods. If None, uses QuantileBinningModel.
        **plot_kwargs
            Additional keyword arguments passed to Plotly
            
        Returns
        -------
        Tuple[pd.DataFrame, Any]
            - DataFrame with parameter values and computed metrics
            - Plotly figure
        """
        from eda.parameter_analysis import ParameterAnalyzer
        
        param1_name = self._canonicalize_param_name(param1_name)
        param2_name = self._canonicalize_param_name(param2_name)
        module_features = {}
        print(f"Building parameter grid for module '{module_name}'")
        
        feature_to_params = {}
        for feature_name, meta in self.feature_metadata.items():
            if meta.get('module', '').lower() == module_name.lower():
                feature_to_params[feature_name] = meta.get('parameters', {})
        
        def _norm_val(v):
            try:
                if isinstance(v, str) and v.replace('.', '', 1).isdigit():
                    return int(v) if v.isdigit() else float(v)
            except Exception:
                pass
            return v

        param_combinations = set()
        for params in feature_to_params.values():
            if param1_name in params and param2_name in params:
                v1 = _norm_val(params[param1_name])
                v2 = _norm_val(params[param2_name])
                param_combinations.add((v1, v2))
        
        for (param1_val, param2_val) in param_combinations:
            matching_features = []
            for feature, params in feature_to_params.items():
                p1 = _norm_val(params.get(param1_name))
                p2 = _norm_val(params.get(param2_name))
                if p1 == param1_val and p2 == param2_val:
                    matching_features.append(feature)
            print(f"  [DEBUG] Param combo ({param1_val}, {param2_val}) initial matches: {len(matching_features)}")
            before_filter = list(matching_features)
            if feature_name is not None:
                try:
                    matching_features = [
                        f for f in matching_features
                        if (
                            str(helpers.parse_feature_column_name(f).get('feature')).lower() == str(feature_name).lower()
                            or f"_{str(feature_name).lower()}_" in f.lower()
                        )
                    ]
                except Exception:
                    pass
            else:
                try:
                    non_bool = [
                        f for f in matching_features
                        if 'bool' not in str(helpers.parse_feature_column_name(f).get('feature', '')).lower()
                    ]
                    if non_bool:
                        matching_features = non_bool
                except Exception:
                    pass
            if feature_name is not None and not matching_features:
                matching_features = before_filter
            print(f"  [DEBUG] Param combo ({param1_val}, {param2_val}) after filter: {len(matching_features)}")
            module_features[(param1_val, param2_val)] = matching_features
        
        if not module_features:
            available_modules = list(set(meta['module'] for meta in self.feature_metadata.values()))
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
            module_features_list = [
                feature for feature, meta in self.feature_metadata.items()
                if meta.get('module', '').lower() == module_name.lower()
            ]
            if module_features_list:
                error_msg.append(f"\nFeatures found for module '{module_name}': {module_features_list[:5]}")
                if len(module_features_list) > 5:
                    error_msg.append(f" (and {len(module_features_list)-5} more)")
            raise ValueError(''.join(error_msg))
        
        analyzer = ParameterAnalyzer(self.features_df, self.targets_df)
        try:
            # Use default metric if not provided
            if metric is None:
                from metrics.performance import SortinoRatio
                metric = SortinoRatio(annualization_factor=252)
            
            # Infer metric name from metric object for display/plotting
            from eda.parameter_analysis import _get_metric_name_from_object
            metric_name = _get_metric_name_from_object(metric)
            
            df = analyzer.analyze_2d_parameters(
                feature_grid=module_features,
                target_col=target_col,
                metric=metric,
                n_bins=n_bins,
                base_model=base_model
            )
            fig = plot_2d_parameter_surface_pure(
                df=df,
                param1=param1_name,
                param2=param2_name,
                metric=metric_name,
                title=f"{module_name.upper()} {metric_name.capitalize()} vs {param1_name} and {param2_name}",
                show_plot=show_plot,
                plot_type=plot_type
            )
            return df, fig
        except Exception as e:
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
        
        Delegates to pure function in metrics.plotting.
        """
        return plot_all_feature_deciles(
            features_df=self.features_df,
            targets_df=self.targets_df,
            feature_names=self.feature_names,
            target_col=target_col,
            n_bins=n_bins,
            figsize=figsize,
            plot_type=plot_type,
            save_dir=save_dir,
            verbose=verbose
        )
    
    def plot_2bin(
        self,
        feature_name: str,
        target_col: str = 'log_return',
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot 2-bin analysis (positive vs negative feature values).
        
        Delegates to pure function in metrics.plotting.
        """
        # Validate
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found")
        if target_col not in self.targets_df.columns:
            raise ValueError(f"Target '{target_col}' not found")
        
        feature_data = self.features_df[feature_name]
        target_data = self.targets_df[target_col]
        
        fig = plot_feature_2bin(
            feature_data=feature_data,
            target_data=target_data,
            feature_name=feature_name,
            figsize=figsize,
            save_path=save_path
        )
        
        # Store results (analysis logic stays in class)
        if '2bin_analysis' not in self.results:
            self.results['2bin_analysis'] = {}
        # Note: bin_table not available from pure function, but we can compute if needed
        self.results['2bin_analysis'][feature_name] = {
            'target_col': target_col,
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
        (Unchanged)
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found")
        if target_col not in self.targets_df.columns:
            raise ValueError(f"Target '{target_col}' not found")
        
        feature_data = self.features_df[feature_name]
        target_data = self.targets_df[target_col]
        if not pd.api.types.is_numeric_dtype(feature_data):
            raise TypeError(f"Feature '{feature_name}' is non-numeric")
        
        fig, bin_table = plot_decile_analysis(
            feature_data=feature_data,
            target_data=target_data,
            feature_name=feature_name,
            n_bins=n_bins,
            figsize=figsize,
            plot_type=plot_type,
            save_path=save_path
        )
        if 'decile_analysis' not in self.results:
            self.results['decile_analysis'] = {}
        self.results['decile_analysis'][feature_name] = {
            'target_col': target_col,
            'n_bins': n_bins,
            'bin_data': bin_table
        }
        return fig
    
    def plot_uniform_bins(
        self,
        feature_name: str,
        n_bins: int = 10,
        target_col: str = 'log_return',
        figsize: Tuple[int, int] = (12, 8),
        plot_type: str = "bar",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot uniform binning analysis for a single feature.
        
        Uniform binning uses equal-width bins (unlike deciles which use equal-frequency bins).
        This is useful for understanding behavior at specific feature value ranges.
        
        Parameters
        ----------
        feature_name : str
            Name of the feature to plot
        n_bins : int, default=10
            Number of bins to create
        target_col : str, default='log_return'
            Target column to use
        figsize : Tuple[int, int], default=(12, 8)
            Figure size
        plot_type : str, default="bar"
            Type of plot: "bar" or "line"
        save_path : Optional[str], default=None
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found")
        if target_col not in self.targets_df.columns:
            raise ValueError(f"Target '{target_col}' not found")
        
        feature_data = self.features_df[feature_name]
        target_data = self.targets_df[target_col]
        if not pd.api.types.is_numeric_dtype(feature_data):
            raise TypeError(f"Feature '{feature_name}' is non-numeric")
        
        fig, bin_table = plot_uniform_binning(
            feature_data=feature_data,
            target_data=target_data,
            feature_name=feature_name,
            n_bins=n_bins,
            figsize=figsize,
            plot_type=plot_type,
            save_path=save_path
        )
        if 'uniform_binning' not in self.results:
            self.results['uniform_binning'] = {}
        self.results['uniform_binning'][feature_name] = {
            'target_col': target_col,
            'n_bins': n_bins,
            'bin_data': bin_table
        }
        return fig
    
    def plot_all_uniform_bins(
        self,
        n_bins: int = 10,
        target_col: str = 'log_return',
        figsize: Tuple[int, int] = (12, 8),
        plot_type: str = "bar",
        save_dir: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, plt.Figure]:
        """
        Plot uniform binning analysis for all features.
        
        Delegates to pure function in metrics.plotting.
        """
        return plot_all_feature_uniform_bins(
            features_df=self.features_df,
            targets_df=self.targets_df,
            feature_names=self.feature_names,
            target_col=target_col,
            n_bins=n_bins,
            figsize=figsize,
            plot_type=plot_type,
            save_dir=save_dir,
            verbose=verbose
        )
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for all features.
        (Unchanged)
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
        (Unchanged)
        """
        if target_col not in self.targets_df.columns:
            raise ValueError(f"Target '{target_col}' not found")
        target_data = self.targets_df[target_col]
        correlations = {}
        for feature_name in self.feature_names:
            feature_data = self.features_df[feature_name]
            if not pd.api.types.is_numeric_dtype(feature_data):
                correlations[feature_name] = np.nan
                continue
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
        
        Delegates to pure function in metrics.plotting.
        """
        fig = plot_feature_target_correlations_pure(
            features_df=self.features_df,
            targets_df=self.targets_df,
            feature_names=self.feature_names,
            target_col=target_col,
            figsize=figsize,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            save_path=save_path
        )
        
        # Store results (analysis logic stays in class)
        correlations = self.get_correlations(target_col=target_col, method='spearman')
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
        
        Delegates to pure function in metrics.plotting.
        """
        fig = plot_feature_correlation_matrix(
            features_df=self.features_df,
            feature_names=self.feature_names,
            figsize=figsize,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            save_path=save_path,
            mask_diagonal=mask_diagonal
        )
        
        # Store results (analysis logic stays in class)
        numeric_features = [
            col for col in self.feature_names
            if pd.api.types.is_numeric_dtype(self.features_df[col])
        ]
        corr_matrix = self.features_df[numeric_features].corr(method='pearson')
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
        
        Delegates to pure function in metrics.plotting.
        """
        figures = plot_all_feature_distributions(
            features_df=self.features_df,
            feature_names=self.feature_names,
            figsize=figsize,
            bins=bins,
            show_stats=show_stats,
            save_dir=save_dir,
            verbose=verbose
        )
        
        # Store results (analysis logic stays in class)
        if 'distributions' not in self.results:
            self.results['distributions'] = {}
        for feature_name in figures.keys():
            feature_data = self.features_df[feature_name]
            self.results['distributions'][feature_name] = {
                'n_samples': len(feature_data.dropna()),
                'mean': feature_data.mean(),
                'std': feature_data.std(),
                'skew': feature_data.skew(),
                'kurtosis': feature_data.kurtosis()
            }
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
        
        Delegates to pure function in metrics.plotting.
        """
        figures = plot_all_feature_timeseries(
            features_df=self.features_df,
            feature_names=self.feature_names,
            figsize=figsize,
            show_rolling_mean=show_rolling_mean,
            rolling_window=rolling_window,
            show_rolling_std=show_rolling_std,
            save_dir=save_dir,
            verbose=verbose
        )
        
        # Store results (analysis logic stays in class)
        if 'timeseries' not in self.results:
            self.results['timeseries'] = {}
        for feature_name in figures.keys():
            feature_data = self.features_df[feature_name]
            self.results['timeseries'][feature_name] = {
                'date_range': (feature_data.index.min(), feature_data.index.max()),
                'n_samples': len(feature_data.dropna()),
                'rolling_window': rolling_window
            }
        return figures
    
    def plot_signal_cumsum(
        self,
        target_col: str = 'log_return',
        features: Optional[List[str]] = None,
        binning_model: Optional[Any] = None,
        metric: Optional[Any] = None,
        strategy: str = 'long',
        figsize: Tuple[int, int] = (12, 6),
        save_dir: Optional[str] = None,
        show_plot: bool = True,
        verbose: bool = True
    ) -> Tuple[Dict[str, plt.Figure], pd.DataFrame, Dict[str, pd.Series]]:
        """
        Plot cumulative sum of target returns gated by model signals for each feature.
        
        For each feature, this will:
        - If binning_model is provided: Fit the binning model on (feature, target) and generate binary signals
        - If binning_model is None: Use the feature series directly as signals (for features that are already binary)
        - Compute product: target * signal
        - Plot cumulative sum over time
        
        Parameters
        ----------
        target_col : str, default='log_return'
            Target column to multiply with signals
        features : Optional[List[str]], default=None
            Subset of features to analyze. If None, uses all features
        binning_model : Optional[Any], default=None
            Binning model instance (e.g., QuantileBinningModel) implementing fit(X, y) and predict(X, strategy) -> {0,1}.
            If None, the feature series itself is used as signals (useful for binary features).
        metric : Optional[Any], default=None
            Metric object with compute(returns) -> float for title/summary.
            If None, metric computation is skipped.
        strategy : str, default='long'
            Strategy flag: 'long', 'short', or 'long-short'.
            - 'long': Only take long positions when long bin is selected
            - 'short': Only take short positions when short bin is selected
            - 'long-short': Take long positions when long bin is selected, short positions when short bin is selected
            Only used if binning_model is provided
        figsize : Tuple[int, int], default=(12, 6)
            Figure size
        save_dir : Optional[str], default=None
            Directory to save per-feature plots
        show_plot : bool, default=True
            Whether to display the plots
        verbose : bool, default=True
            Print progress
        
        Returns
        -------
        Tuple[Dict[str, plt.Figure], pd.DataFrame, Dict[str, pd.Series]]
            - Mapping from feature name to matplotlib Figure
            - Summary DataFrame with final cumulative sum and metric
            - Dictionary mapping feature name to gated returns series (for combining plots)
        """
        # Validate target
        if target_col not in self.targets_df.columns:
            raise ValueError(
                f"Target '{target_col}' not found. Available targets: {list(self.targets_df.columns)}"
            )
        
        if features is None:
            features = list(self.feature_names)
        
        # Ensure save directory exists if provided
        if save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)
        
        figures: Dict[str, plt.Figure] = {}
        summary_rows: List[Dict[str, Any]] = []
        gated_returns_dict: Dict[str, pd.Series] = {}  # Store gated returns for combining plots
        
        target_series = self.targets_df[target_col]
        
        # Validate strategy
        if strategy not in ['long', 'short', 'long-short']:
            raise ValueError(f"strategy must be 'long', 'short', or 'long-short', got '{strategy}'")
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Plotting signal-gated cumulative returns for {len(features)} features")
            print(f"Target: {target_col} | Strategy: {strategy}")
            print(f"{'='*70}")
        
        for i, feature_name in enumerate(features, 1):
            if feature_name not in self.features_df.columns:
                if verbose:
                    print(f"\n[{i}/{len(features)}] {feature_name} -> skipped (not in features_df)")
                continue
            if verbose:
                print(f"\n[{i}/{len(features)}] {feature_name}")
            
            X = self.features_df[feature_name]
            y = target_series
            
            # Align X and y to same index before filtering
            common_index = X.index.intersection(y.index)
            X_aligned = X.reindex(common_index)
            y_aligned = y.reindex(common_index)
            
            valid_mask = ~(X_aligned.isna() | y_aligned.isna())
            X_clean = X_aligned[valid_mask]
            y_clean = y_aligned[valid_mask]
            
            if len(X_clean) < 5:
                if verbose:
                    print("  ✗ Insufficient non-NaN samples (<5)")
                continue
            
            try:
                # Generate signals: use binning model if provided, otherwise use feature series directly
                if binning_model is not None:
                    # Fit binning model on (feature, target) for this feature
                    binning_model.fit(X_clean, y_clean)
                    
                    if strategy == 'long-short':
                        # Get signals for both long and short bins
                        long_signals = binning_model.predict(X_clean, strategy='long')
                        short_signals = binning_model.predict(X_clean, strategy='short')
                        
                        if isinstance(long_signals, (pd.Series, pd.DataFrame)):
                            long_signals = long_signals.squeeze()
                        else:
                            long_signals = pd.Series(long_signals, index=X_clean.index)
                        
                        if isinstance(short_signals, (pd.Series, pd.DataFrame)):
                            short_signals = short_signals.squeeze()
                        else:
                            short_signals = pd.Series(short_signals, index=X_clean.index)
                        
                        # Long positions: use returns as-is
                        # Short positions: negate returns (shorting profits from negative returns)
                        gated_returns = (y_clean * long_signals) + (-y_clean * short_signals)
                        signals_series = long_signals + short_signals  # For summary stats
                    else:
                        signals = binning_model.predict(X_clean, strategy=strategy)
                        if isinstance(signals, (pd.Series, pd.DataFrame)):
                            signals_series = signals.squeeze()
                        else:
                            signals_series = pd.Series(signals, index=X_clean.index)
                        
                        # For short strategy, negate returns (shorting profits from negative returns)
                        if strategy == 'short':
                            gated_returns = -y_clean * signals_series
                        else:
                            gated_returns = y_clean * signals_series
                else:
                    # Use feature series directly as signals (for binary features)
                    signals_series = X_clean
                    if strategy == 'short':
                        gated_returns = -y_clean * signals_series
                    elif strategy == 'long-short':
                        # For binary features, long-short doesn't make sense without a model
                        # Treat as long-only
                        gated_returns = y_clean * signals_series
                    else:
                        gated_returns = y_clean * signals_series
                
                # CRITICAL: Sort by datetime index before calculating cumulative sum
                # This ensures chronological order, especially important for multi-ticker data
                # where datetime index might not be sorted or have duplicates
                if isinstance(gated_returns.index, pd.DatetimeIndex):
                    gated_returns = gated_returns.sort_index()
                elif hasattr(gated_returns.index, 'sort_values'):
                    # If index has sort_values method (e.g., MultiIndex), try to sort
                    try:
                        gated_returns = gated_returns.sort_index()
                    except Exception:
                        # If sorting fails, at least ensure we have a consistent order
                        pass
                
                # Compute cumulative returns for summary (now in chronological order)
                cum_returns = gated_returns.cumsum()
                
                # Compute metric if provided
                metric_value = float('nan')
                if metric is not None:
                    try:
                        selected_returns = gated_returns[gated_returns != 0]
                        metric_value = float(metric.compute(selected_returns)) if len(selected_returns) > 0 else float('nan')
                    except Exception:
                        metric_value = float('nan')
                
                # Use pure plotting function (with sorted data)
                save_path = None
                if save_dir is not None:
                    save_path = os.path.join(save_dir, f"{feature_name}_signal_cumsum.png")
                
                fig = plot_feature_signal_cumsum(
                    gated_returns=gated_returns,
                    feature_name=feature_name,
                    strategy=strategy,
                    metric_value=metric_value,
                    figsize=figsize,
                    save_path=save_path,
                    show_plot=show_plot
                )
                figures[feature_name] = fig
                gated_returns_dict[feature_name] = gated_returns  # Store for combining plots
                summary_rows.append({
                    'feature': feature_name,
                    'n_samples': int(len(X_clean)),
                    'n_signals': int(signals_series.sum()) if pd.api.types.is_numeric_dtype(signals_series) else int((signals_series != 0).sum()),
                    'final_cumsum': float(cum_returns.iloc[-1]),
                    'metric': metric_value
                })
                if verbose:
                    print("  ✓ Complete")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed: {e}")
                continue
        
        summary_df = pd.DataFrame(summary_rows)
        self.results['signal_cumsum'] = {
            'target_col': target_col,
            'strategy': strategy,
            'model': str(binning_model) if binning_model is not None else 'feature_direct',
            'metric': str(metric) if metric is not None else 'none',
            'summary': summary_df
        }
        return figures, summary_df, gated_returns_dict
    
    def generate_summary_report(
        self,
        binning_model: Any,
        target_col: str = 'log_return',
        strategy: str = 'long',
        metric: Optional[Any] = None,
        save_dir: Optional[str] = None,
        show_plots: bool = True,
        verbose: bool = True,
        permutation_test_nreps: int = 1000,
        permutation_test_alpha: float = 0.1,
        permutation_test_n_jobs: int = -1,
        export_report: bool = False,
        export_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report with all relevant analysis methods.
        
        This method runs all relevant analysis methods and returns a dictionary
        with all results and figures. Methods are conditionally included based on
        available features (e.g., parameter sensitivity only if parameterized features exist).
        
        Parameters
        ----------
        binning_model : Any
            Binning model instance (subclass of BinningModelBase).
            Must have fit() and predict() methods. User should configure
            all hyperparameters (n_bins, selection_metric, etc.) before passing.
        target_col : str, default='log_return'
            Target column to use for analysis
        strategy : str, default='long'
            Strategy for signal generation: 'long', 'short', or 'long-short'
        metric : Optional[Any], default=None
            Metric object from metrics.performance (e.g., SortinoRatio, SharpeRatio).
            Must have a .compute() method. If None, defaults to SortinoRatio.
        save_dir : Optional[str], default=None
            Directory to save plots. If None, plots are not saved.
        show_plots : bool, default=True
            Whether to display plots
        verbose : bool, default=True
            Print progress
        permutation_test_nreps : int, default=100
            Number of permutation replications for permutation test
        permutation_test_alpha : float, default=0.1
            Significance level for permutation test
        permutation_test_n_jobs : int, default=-1
            Number of parallel jobs for permutation test (-1 = all CPUs)
        export_report : bool, default=False
            Whether to export the summary report to files
        export_path : Optional[str], default=None
            Path to export directory. If None and export_report=True, uses save_dir.
            Exports DataFrames to CSV and creates a summary text file.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'summary_stats': DataFrame with basic feature statistics
            - 'correlations': Series with feature-target correlations
            - 'feature_correlations': DataFrame with intra-feature correlation matrix
            - 'decile_figures': Dict of decile analysis figures
            - 'signal_cumsum_figures': Dict of signal cumulative sum figures
            - 'signal_cumsum_summary': DataFrame with signal performance summary
            - 'parameter_sensitivity': Dict with parameter sensitivity results (if applicable)
            - 'parameter_2d_surface': Dict with 2D parameter surface results (if applicable)
            - 'distribution_figures': Dict of distribution figures
            - 'timeseries_figures': Dict of time series figures
            - 'permutation_test': DataFrame with permutation test results
        """
        if verbose:
            print(f"\n{'='*70}")
            print("Generating Comprehensive Feature Analysis Report")
            print(f"{'='*70}")
        
        # Disable interactive mode and close all existing figures if not showing plots
        if not show_plots:
            import matplotlib.pyplot as plt
            plt.ioff()  # Turn off interactive mode
            plt.close('all')  # Close all existing figures
        
        # Validate binning model
        if binning_model is None:
            raise ValueError("binning_model is required. Please provide a binning model instance (e.g., QuantileBinningModel or DecisionTreeBinningModel) with parameters already configured.")
        
        # Create metric if needed
        if metric is None:
            from metrics.performance import SortinoRatio
            metric = SortinoRatio(annualization_factor=252)
        
        # When exporting, set save_dir to export_dir so all plots get saved during generation
        if export_report:
            import os
            export_dir = export_path if export_path is not None else save_dir
            if export_dir is None:
                import tempfile
                export_dir = tempfile.mkdtemp(prefix='feature_explorer_report_')
                if verbose:
                    print(f"\n[Export] No export path specified, using temporary directory: {export_dir}")
            # Create export directory if it doesn't exist
            os.makedirs(export_dir, exist_ok=True)
            # Override save_dir to ensure plots are saved during generation
            save_dir = export_dir
        
        results: Dict[str, Any] = {}
        
        # 1. Basic summary statistics
        if verbose:
            print("\n[1/10] Computing basic summary statistics...")
        results['summary_stats'] = self.get_summary()
        if verbose:
            print(f"  ✓ Computed statistics for {len(results['summary_stats'])} features")
        
        # 2. Feature-target correlations
        if verbose:
            print("\n[2/10] Computing feature-target correlations...")
        results['correlations'] = self.get_correlations(target_col=target_col, method='spearman')
        if verbose:
            print(f"  ✓ Computed correlations for {len(results['correlations'])} features")
        
        # 3. Intra-feature correlations
        if verbose:
            print("\n[3/10] Computing intra-feature correlations...")
        import os
        import matplotlib.pyplot as plt
        corr_save_path = None
        if save_dir is not None:
            corr_save_path = os.path.join(save_dir, 'feature_correlations.png')
        results['feature_correlations_figure'] = self.plot_feature_correlations(
            save_path=corr_save_path
        )
        if show_plots:
            plt.show()
        elif not export_report:
            # Only close if not exporting (export will handle it)
            plt.close(results['feature_correlations_figure'])
        # Store correlation matrix in results
        numeric_features = [
            col for col in self.feature_names
            if pd.api.types.is_numeric_dtype(self.features_df[col])
        ]
        results['feature_correlations'] = self.features_df[numeric_features].corr(method='pearson')
        if verbose:
            print(f"  ✓ Computed correlation matrix for {len(numeric_features)} features")
        
        # 4. Decile analysis
        if verbose:
            print("\n[4/10] Plotting decile analysis...")
        decile_save_dir = None
        # Only save individual files if not exporting (export will create combined file)
        if save_dir is not None and not export_report:
            decile_save_dir = os.path.join(save_dir, 'deciles')
        results['decile_figures'] = self.plot_all_deciles(
            n_bins=10,
            target_col=target_col,
            save_dir=decile_save_dir,
            verbose=False
        )
        # Close figures if not showing plots and not exporting (export will handle closing)
        if not show_plots and not export_report:
            import matplotlib.pyplot as plt
            for fig in results['decile_figures'].values():
                plt.close(fig)
        if verbose:
            print(f"  ✓ Generated {len(results['decile_figures'])} decile plots")
        
        # 5. Signal cumulative sum
        if verbose:
            print("\n[5/10] Plotting signal-gated cumulative returns...")
        signal_save_dir = None
        # Only save individual files if not exporting (export will create combined file)
        if save_dir is not None and not export_report:
            signal_save_dir = os.path.join(save_dir, 'signal_cumsum')
        signal_figures, signal_summary, signal_gated_returns = self.plot_signal_cumsum(
            target_col=target_col,
            binning_model=binning_model,
            metric=metric,
            strategy=strategy,
            save_dir=signal_save_dir,
            show_plot=show_plots,
            verbose=False
        )
        results['signal_cumsum_figures'] = signal_figures
        results['signal_cumsum_summary'] = signal_summary
        results['signal_cumsum_gated_returns'] = signal_gated_returns  # Store for combining plots
        # Close figures if not showing plots and not exporting (export will handle closing)
        if not show_plots and not export_report:
            import matplotlib.pyplot as plt
            for fig in signal_figures.values():
                plt.close(fig)
        if verbose:
            print(f"  ✓ Generated {len(signal_figures)} signal plots")
        
        # 6. Parameter sensitivity (if parameterized features exist)
        if self._has_parameterized_features():
            if verbose:
                print("\n[6/10] Analyzing parameter sensitivity...")
            param_counts = self._count_parameters_per_module()
            results['parameter_sensitivity'] = {}
            
            for module_name, n_params in param_counts.items():
                if n_params >= 1:
                    # Get first parameter for sensitivity analysis
                    param_groups = self.get_parameterized_features()
                    module_params = [
                        (module, param) for (module, param) in param_groups.keys()
                        if module == module_name
                    ]
                    
                    if module_params:
                        first_param = module_params[0][1]
                        try:
                            # Get n_bins from binning_model if available
                            n_bins = getattr(binning_model, 'n_bins', 10)
                            param_df, param_fig = self.plot_parameter_sensitivity(
                                module_name=module_name,
                                param_name=first_param,
                                target_col=target_col,
                                metric=metric,
                                n_bins=n_bins,
                                base_model=binning_model,  # Note: parameter name is 'base_model' for compatibility
                                show_plot=show_plots
                            )
                            results['parameter_sensitivity'][f"{module_name}_{first_param}"] = {
                                'dataframe': param_df,
                                'figure': param_fig
                            }
                            # For Plotly figures, prevent auto-display in notebooks when show_plots=False
                            if not show_plots and hasattr(param_fig, 'update_layout'):
                                # Update layout to prevent auto-display (Plotly might still show in notebooks)
                                param_fig.update_layout(template=None)
                            if verbose:
                                print(f"  ✓ Analyzed {module_name}.{first_param} sensitivity")
                        except Exception as e:
                            if verbose:
                                print(f"  ✗ Failed to analyze {module_name}.{first_param}: {e}")
        else:
            if verbose:
                print("\n[6/10] Skipping parameter sensitivity (no parameterized features)")
            results['parameter_sensitivity'] = None
        
        # 7. 2D parameter surface (if 2+ parameters exist)
        if self._has_parameterized_features():
            param_counts = self._count_parameters_per_module()
            has_2d_params = any(n_params >= 2 for n_params in param_counts.values())
            
            if has_2d_params:
                if verbose:
                    print("\n[7/10] Analyzing 2D parameter surface...")
                results['parameter_2d_surface'] = {}
                
                for module_name, n_params in param_counts.items():
                    if n_params >= 2:
                        # Get first two parameters for 2D analysis
                        param_groups = self.get_parameterized_features()
                        module_params = [
                            (module, param) for (module, param) in param_groups.keys()
                            if module == module_name
                        ]
                        
                        if len(module_params) >= 2:
                            param1 = module_params[0][1]
                            param2 = module_params[1][1]
                            try:
                                # Get n_bins from binning_model if available
                                n_bins = getattr(binning_model, 'n_bins', 5)
                                surface_df, surface_fig = self.plot_2d_parameter_surface(
                                    module_name=module_name,
                                    param1_name=param1,
                                    param2_name=param2,
                                    target_col=target_col,
                                    metric=metric,
                                    n_bins=n_bins,
                                    base_model=binning_model,  # Note: parameter name is 'base_model' for compatibility
                                    show_plot=show_plots
                                )
                                results['parameter_2d_surface'][f"{module_name}_{param1}_{param2}"] = {
                                    'dataframe': surface_df,
                                    'figure': surface_fig
                                }
                                # For Plotly figures, prevent auto-display in notebooks when show_plots=False
                                if not show_plots and hasattr(surface_fig, 'update_layout'):
                                    # Update layout to prevent auto-display (Plotly might still show in notebooks)
                                    surface_fig.update_layout(template=None)
                                if verbose:
                                    print(f"  ✓ Analyzed {module_name}.{param1} vs {param2} surface")
                            except Exception as e:
                                if verbose:
                                    print(f"  ✗ Failed to analyze {module_name} 2D surface: {e}")
            else:
                if verbose:
                    print("\n[7/10] Skipping 2D parameter surface (insufficient parameters)")
                results['parameter_2d_surface'] = None
        else:
            if verbose:
                print("\n[7/10] Skipping 2D parameter surface (no parameterized features)")
            results['parameter_2d_surface'] = None
        
        # 8. Feature distributions
        if verbose:
            print("\n[8/10] Plotting feature distributions...")
        dist_save_dir = None
        # Only save individual files if not exporting (export will create combined file)
        if save_dir is not None and not export_report:
            dist_save_dir = os.path.join(save_dir, 'distributions')
        results['distribution_figures'] = self.plot_distributions(
            save_dir=dist_save_dir,
            verbose=False
        )
        # Close figures if not showing plots and not exporting (export will handle closing)
        if not show_plots and not export_report:
            import matplotlib.pyplot as plt
            for fig in results['distribution_figures'].values():
                plt.close(fig)
        if verbose:
            print(f"  ✓ Generated {len(results['distribution_figures'])} distribution plots")
        
        # 9. Time series plots
        if verbose:
            print("\n[9/10] Plotting time series...")
        ts_save_dir = None
        # Only save individual files if not exporting (export will create combined file)
        if save_dir is not None and not export_report:
            ts_save_dir = os.path.join(save_dir, 'timeseries')
        results['timeseries_figures'] = self.plot_timeseries(
            save_dir=ts_save_dir,
            verbose=False
        )
        # Close figures if not showing plots and not exporting (export will handle closing)
        if not show_plots and not export_report:
            import matplotlib.pyplot as plt
            for fig in results['timeseries_figures'].values():
                plt.close(fig)
        if verbose:
            print(f"  ✓ Generated {len(results['timeseries_figures'])} time series plots")
        
        # 10. Permutation test
        if verbose:
            print("\n[10/10] Running permutation test...")
        try:
            results['permutation_test'] = self.run_permutation_test(
                target_col=target_col,
                base_model=binning_model,
                metric=metric,
                nreps=permutation_test_nreps,
                n_jobs=permutation_test_n_jobs,
                alpha=permutation_test_alpha,
                verbose=False
            )
            if verbose:
                n_significant = results['permutation_test']['significant'].sum() if 'significant' in results['permutation_test'].columns else 0
                print(f"  ✓ Completed permutation test for {len(results['permutation_test'])} features")
                print(f"  ✓ Found {n_significant} significant features (p <= {permutation_test_alpha})")
                print(f"\n  Permutation test p-values:")
                for _, row in results['permutation_test'].iterrows():
                    pval = row.get('pval', 'N/A')
                    sig = '✓' if row.get('significant', False) else '✗'
                    print(f"    {sig} {row.get('feature', 'unknown')}: p = {pval:.4f}")
        except Exception as e:
            if verbose:
                print(f"  ✗ Permutation test failed: {e}")
            results['permutation_test'] = None
        
        if verbose:
            print(f"\n{'='*70}")
            print("Summary Report Complete!")
            print(f"{'='*70}")
            print(f"\nResults Summary:")
            print(f"  - Summary statistics: {len(results['summary_stats'])} features")
            print(f"  - Feature-target correlations: {len(results['correlations'])} features")
            print(f"  - Intra-feature correlations: {len(results.get('feature_correlations', pd.DataFrame()).columns)} features")
            print(f"  - Decile plots: {len(results['decile_figures'])} features")
            print(f"  - Signal cumsum plots: {len(results['signal_cumsum_figures'])} features")
            print(f"  - Distribution plots: {len(results['distribution_figures'])} features")
            print(f"  - Time series plots: {len(results['timeseries_figures'])} features")
            if results.get('parameter_sensitivity'):
                print(f"  - Parameter sensitivity: {len(results['parameter_sensitivity'])} analyses")
            if results.get('parameter_2d_surface'):
                print(f"  - 2D parameter surfaces: {len(results['parameter_2d_surface'])} analyses")
            if results.get('permutation_test') is not None:
                n_significant = results['permutation_test']['significant'].sum() if 'significant' in results['permutation_test'].columns else 0
                print(f"  - Permutation test: {len(results['permutation_test'])} features tested, {n_significant} significant")
                print(f"\n  Permutation test p-values:")
                for _, row in results['permutation_test'].iterrows():
                    pval = row.get('pval', 'N/A')
                    sig = '✓' if row.get('significant', False) else '✗'
                    print(f"    {sig} {row.get('feature', 'unknown')}: p = {pval:.4f}")
        
        # Export report if requested (plots should already be saved since save_dir was set above)
        if export_report:
            self._export_summary_report(results, save_dir, target_col=target_col, verbose=verbose)
        
        # Close all remaining figures if show_plots is False and not exporting
        # (export function will close figures after saving them)
        if not show_plots and not export_report:
            import matplotlib.pyplot as plt
            plt.close('all')
        
        # Re-enable interactive mode if we disabled it
        if not show_plots:
            import matplotlib.pyplot as plt
            plt.ion()  # Turn interactive mode back on for future use
        
        return results
    
    def _export_summary_report(
        self,
        results: Dict[str, Any],
        export_dir: str,
        target_col: str = 'log_return',
        verbose: bool = True
    ) -> None:
        """
        Export summary report to files.
        
        Saves plots/figures to image files and creates a summary text file.
        Only exports raw data (CSV) for permutation test results.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Results dictionary from generate_summary_report
        export_dir : str
            Directory to save exported files
        verbose : bool, default=True
            Print progress
        """
        import os
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        # Create export directory if it doesn't exist
        os.makedirs(export_dir, exist_ok=True)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Exporting Summary Report to: {export_dir}")
            print(f"{'='*70}")
        
        # 1. Export feature correlations figure (already saved during generation, but ensure it's there)
        if 'feature_correlations_figure' in results:
            corr_fig_path = os.path.join(export_dir, 'feature_correlations.png')
            if not os.path.exists(corr_fig_path):
                results['feature_correlations_figure'].savefig(corr_fig_path, dpi=150, bbox_inches='tight')
            if verbose:
                print(f"  ✓ Exported feature correlations plot: {corr_fig_path}")
        
        # 2. Export decile plots (combined into single file)
        if 'decile_figures' in results and results['decile_figures']:
            decile_path = os.path.join(export_dir, 'all_deciles_combined.png')
            try:
                feature_names = list(results['decile_figures'].keys())
                combined_fig = combine_decile_plots(
                    features_df=self.features_df,
                    targets_df=self.targets_df,
                    feature_names=feature_names,
                    target_col=target_col,
                    n_bins=10,
                    n_cols=4,
                    figsize_per_plot=(6, 4),
                    plot_type="bar",
                    save_path=decile_path
                )
                plt.close(combined_fig)
                # Close individual figures
                for fig in results['decile_figures'].values():
                    plt.close(fig)
                if verbose:
                    print(f"  ✓ Exported combined decile plots: {decile_path} ({len(feature_names)} features)")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed to combine decile plots: {e}")
        
        # 3. Export signal cumsum plots (combined into single file)
        if 'signal_cumsum_figures' in results and results['signal_cumsum_figures']:
            signal_path = os.path.join(export_dir, 'all_signal_cumsum_combined.png')
            try:
                if 'signal_cumsum_gated_returns' in results:
                    combined_fig = combine_signal_cumsum_plots(
                        gated_returns_dict=results['signal_cumsum_gated_returns'],
                        n_cols=4,
                        figsize_per_plot=(6, 3),
                        save_path=signal_path
                    )
                    plt.close(combined_fig)
                    # Close individual figures
                    for fig in results['signal_cumsum_figures'].values():
                        plt.close(fig)
                    if verbose:
                        print(f"  ✓ Exported combined signal cumsum plots: {signal_path} ({len(results['signal_cumsum_figures'])} features)")
                else:
                    if verbose:
                        print(f"  ✗ Cannot combine signal cumsum plots: gated returns data not available")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed to combine signal cumsum plots: {e}")
        
        # 4. Export distribution plots (combined into single file)
        if 'distribution_figures' in results and results['distribution_figures']:
            dist_path = os.path.join(export_dir, 'all_distributions_combined.png')
            try:
                feature_names = list(results['distribution_figures'].keys())
                combined_fig = combine_distribution_plots(
                    features_df=self.features_df,
                    feature_names=feature_names,
                    n_cols=4,
                    figsize_per_plot=(5, 3),
                    bins=50,
                    save_path=dist_path
                )
                plt.close(combined_fig)
                # Close individual figures
                for fig in results['distribution_figures'].values():
                    plt.close(fig)
                if verbose:
                    print(f"  ✓ Exported combined distribution plots: {dist_path} ({len(feature_names)} features)")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed to combine distribution plots: {e}")
        
        # 5. Export time series plots (combined into single file)
        if 'timeseries_figures' in results and results['timeseries_figures']:
            ts_path = os.path.join(export_dir, 'all_timeseries_combined.png')
            try:
                feature_names = list(results['timeseries_figures'].keys())
                combined_fig = combine_timeseries_plots(
                    features_df=self.features_df,
                    feature_names=feature_names,
                    n_cols=4,
                    figsize_per_plot=(6, 3),
                    rolling_window=20,
                    save_path=ts_path
                )
                plt.close(combined_fig)
                # Close individual figures
                for fig in results['timeseries_figures'].values():
                    plt.close(fig)
                if verbose:
                    print(f"  ✓ Exported combined time series plots: {ts_path} ({len(feature_names)} features)")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Failed to combine time series plots: {e}")
        
        # 6. Export parameter sensitivity plots (if available)
        if results.get('parameter_sensitivity'):
            param_dir = os.path.join(export_dir, 'parameter_sensitivity')
            os.makedirs(param_dir, exist_ok=True)
            for key, data in results['parameter_sensitivity'].items():
                if isinstance(data, dict) and 'figure' in data:
                    # Save plotly figure if it's a plotly figure
                    fig = data['figure']
                    param_path = os.path.join(param_dir, f'{key}.png')
                    try:
                        # Try plotly save as PNG (requires kaleido package)
                        if hasattr(fig, 'write_image'):
                            fig.write_image(param_path, width=1200, height=800, scale=2)
                        else:
                            # Fallback to HTML if write_image not available
                            param_path = os.path.join(param_dir, f'{key}.html')
                            fig.write_html(param_path)
                    except (AttributeError, Exception) as e:
                        # If it's a matplotlib figure or plotly save failed, try matplotlib save
                        try:
                            param_path = os.path.join(param_dir, f'{key}.png')
                            fig.savefig(param_path, dpi=150, bbox_inches='tight')
                            plt.close(fig)
                        except Exception:
                            # Last resort: save as HTML
                            param_path = os.path.join(param_dir, f'{key}.html')
                            fig.write_html(param_path)
            if verbose:
                print(f"  ✓ Exported parameter sensitivity plots: {param_dir}")
        
        # 7. Export 2D parameter surface plots (if available)
        if results.get('parameter_2d_surface'):
            surface_dir = os.path.join(export_dir, 'parameter_2d_surface')
            os.makedirs(surface_dir, exist_ok=True)
            for key, data in results['parameter_2d_surface'].items():
                if isinstance(data, dict) and 'figure' in data:
                    fig = data['figure']
                    surface_path = os.path.join(surface_dir, f'{key}.png')
                    try:
                        # Try plotly save as PNG (requires kaleido package)
                        if hasattr(fig, 'write_image'):
                            fig.write_image(surface_path, width=1200, height=800, scale=2)
                        else:
                            # Fallback to HTML if write_image not available
                            surface_path = os.path.join(surface_dir, f'{key}.html')
                            fig.write_html(surface_path)
                    except (AttributeError, Exception) as e:
                        # If it's a matplotlib figure or plotly save failed, try matplotlib save
                        try:
                            surface_path = os.path.join(surface_dir, f'{key}.png')
                            fig.savefig(surface_path, dpi=150, bbox_inches='tight')
                            plt.close(fig)
                        except Exception:
                            # Last resort: save as HTML
                            surface_path = os.path.join(surface_dir, f'{key}.html')
                            fig.write_html(surface_path)
            if verbose:
                print(f"  ✓ Exported 2D parameter surface plots: {surface_dir}")
        
        # 8. Export permutation test results (ONLY raw data export)
        if results.get('permutation_test') is not None and isinstance(results['permutation_test'], pd.DataFrame):
            perm_path = os.path.join(export_dir, 'permutation_test_results.csv')
            results['permutation_test'].to_csv(perm_path, index=False)
            if verbose:
                print(f"  ✓ Exported permutation test results (CSV): {perm_path}")
        
        # 8. Create summary text file
        summary_text_path = os.path.join(export_dir, 'summary_report.txt')
        with open(summary_text_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("Feature Analysis Summary Report\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Date Range: {self.date_range[0].date()} to {self.date_range[1].date()}\n")
            f.write(f"Number of Features: {self.n_features}\n")
            f.write(f"Number of Samples: {self.n_samples}\n")
            if self.has_ticker:
                f.write(f"Tickers: {self.tickers}\n")
            f.write("\n" + "="*70 + "\n")
            f.write("Results Summary\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Summary Statistics: {len(results['summary_stats'])} features\n")
            f.write(f"Feature-Target Correlations: {len(results['correlations'])} features\n")
            f.write(f"Intra-Feature Correlations: {len(results.get('feature_correlations', pd.DataFrame()).columns)} features\n")
            f.write(f"Decile Plots: {len(results['decile_figures'])} features\n")
            f.write(f"Signal Cumsum Plots: {len(results['signal_cumsum_figures'])} features\n")
            f.write(f"Distribution Plots: {len(results['distribution_figures'])} features\n")
            f.write(f"Time Series Plots: {len(results['timeseries_figures'])} features\n")
            
            if results.get('parameter_sensitivity'):
                f.write(f"Parameter Sensitivity: {len(results['parameter_sensitivity'])} analyses\n")
            
            if results.get('parameter_2d_surface'):
                f.write(f"2D Parameter Surfaces: {len(results['parameter_2d_surface'])} analyses\n")
            
            if results.get('permutation_test') is not None:
                n_significant = results['permutation_test']['significant'].sum() if 'significant' in results['permutation_test'].columns else 0
                f.write(f"\nPermutation Test: {len(results['permutation_test'])} features tested, {n_significant} significant\n")
                f.write("\nPermutation Test p-values:\n")
                for _, row in results['permutation_test'].iterrows():
                    pval = row.get('pval', 'N/A')
                    sig = '✓' if row.get('significant', False) else '✗'
                    feature_name = row.get('feature', 'unknown')
                    original_criterion = row.get('original_criterion', 'N/A')
                    f.write(f"  {sig} {feature_name}: p = {pval:.4f}, metric = {original_criterion:.4f}\n")
            
            # Add top correlations
            if 'correlations' in results:
                f.write("\n" + "="*70 + "\n")
                f.write("Top Feature-Target Correlations (Spearman)\n")
                f.write("="*70 + "\n")
                top_corr = results['correlations'].sort_values(ascending=False).head(10)
                for feature, corr in top_corr.items():
                    f.write(f"  {feature}: {corr:.4f}\n")
        
        if verbose:
            print(f"  ✓ Exported summary text report: {summary_text_path}")
            print(f"\n{'='*70}")
            print(f"Export Complete! Files saved to: {export_dir}")
            print(f"{'='*70}")
    
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
        1. Fitting a base model on each feature
        2. Computing a performance metric
        3. Shuffling feature values and repeating nreps times
        4. Computing p-values: fraction of permutations with metric >= original
        
        Features with low p-values (< alpha) are considered significant.
        
        Parameters
        ----------
        target_col : str, default='log_return'
            Target column to use for prediction
        base_model : Optional[Any], default=None
            Base model to use (must have sklearn API: fit, predict)
        metric : Optional[Any], default=None
            Objective metric to compute (must be callable: metric(returns) -> float)
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
        """
        # Validate target column
        if target_col not in self.targets_df.columns:
            raise ValueError(
                f"Target '{target_col}' not found. "
                f"Available targets: {list(self.targets_df.columns)}"
            )
        if base_model is None:
            raise ValueError("You must provide a base_model instance (fit/predict methods) for permutation test.")
        if metric is None:
            raise ValueError("You must provide a metric object (with .compute()) for permutation test.")
        
        # Combine features and target into single DataFrame
        data = self.features_df.copy()
        data[target_col] = self.targets_df[target_col]
        from functools import partial
        criterion_func = partial(
            self._permutation_criterion,
            target_col=target_col,
            base_model=base_model,
            metric=metric,
            verbose=verbose
        )
        strategy = FeaturePermutationStrategy()
        engine = PermutationEngine(strategy, n_jobs=n_jobs, verbose=verbose)
        results = engine.run_permutation_test(
            data=data,
            feature_cols=self.feature_names,
            criterion_func=criterion_func,
            nreps=nreps,
            random_seed=random_seed,
            alpha=alpha,
            target_col=target_col
        )
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
        (Unchanged)
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
        1. Fitting a base model on each feature
        2. Computing a performance metric
        3. Shuffling feature values and repeating nreps times
        4. Computing p-values: fraction of permutations with metric >= original
        
        Features with low p-values (< alpha) are considered robust.
        
        Parameters
        ----------
        target_col : str, default='log_return'
            Target column to use for prediction
        base_model : Optional[Any], default=None
            Base model to use (must have sklearn API: fit, predict)
        metric : Optional[Any], default=None
            Objective metric to compute (must be callable: metric(returns) -> float)
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
        """
        # Validate target column
        if target_col not in self.targets_df.columns:
            raise ValueError(
                f"Target '{target_col}' not found. "
                f"Available targets: {list(self.targets_df.columns)}"
            )
        if base_model is None:
            raise ValueError("You must provide a base_model instance (fit/predict methods) for sensitivity analysis.")
        if metric is None:
            raise ValueError("You must provide a metric object (with .compute()) for sensitivity analysis.")
        
        # Combine features and target into single DataFrame
        data = self.features_df.copy()
        data[target_col] = self.targets_df[target_col]
        from functools import partial
        criterion_func = partial(
            self._permutation_criterion,
            target_col=target_col,
            base_model=base_model,
            metric=metric,
            verbose=verbose
        )
        strategy = FeaturePermutationStrategy()
        engine = PermutationEngine(strategy, n_jobs=n_jobs, verbose=verbose)
        results = engine.run_permutation_test(
            data=data,
            feature_cols=self.feature_names,
            criterion_func=criterion_func,
            nreps=nreps,
            random_seed=random_seed,
            alpha=alpha,
            target_col=target_col
        )
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
