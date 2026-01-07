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
        base_model: Optional[Any] = None,
        metric: Optional[Any] = None,
        strategy: str = 'long',
        figsize: Tuple[int, int] = (12, 6),
        save_dir: Optional[str] = None,
        show_plot: bool = True,
        verbose: bool = True
    ) -> Tuple[Dict[str, plt.Figure], pd.DataFrame]:
        """
        Plot cumulative sum of target returns gated by model signals for each feature.
        
        For each feature, this will:
        - If base_model is provided: Fit the base model on (feature, target) and generate binary signals
        - If base_model is None: Use the feature series directly as signals (for features that are already binary)
        - Compute product: target * signal
        - Plot cumulative sum over time
        
        Parameters
        ----------
        target_col : str, default='log_return'
            Target column to multiply with signals
        features : Optional[List[str]], default=None
            Subset of features to analyze. If None, uses all features
        base_model : Optional[Any], default=None
            Model implementing fit(X, y) and predict(X, strategy) -> {0,1}.
            If None, the feature series itself is used as signals (useful for binary features).
        metric : Optional[Any], default=None
            Metric object with compute(returns) -> float for title/summary.
            If None, metric computation is skipped.
        strategy : str, default='long'
            Strategy flag: 'long', 'short', or 'long-short'.
            - 'long': Only take long positions when long bin is selected
            - 'short': Only take short positions when short bin is selected
            - 'long-short': Take long positions when long bin is selected, short positions when short bin is selected
            Only used if base_model is provided
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
        Tuple[Dict[str, plt.Figure], pd.DataFrame]
            - Mapping from feature name to matplotlib Figure
            - Summary DataFrame with final cumulative sum and metric
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
            
            valid_mask = ~(X.isna() | y.isna())
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]
            
            if len(X_clean) < 5:
                if verbose:
                    print("  ✗ Insufficient non-NaN samples (<5)")
                continue
            
            try:
                # Generate signals: use model if provided, otherwise use feature series directly
                if base_model is not None:
                    base_model.fit(X_clean, y_clean)
                    
                    if strategy == 'long-short':
                        # Get signals for both long and short bins
                        long_signals = base_model.predict(X_clean, strategy='long')
                        short_signals = base_model.predict(X_clean, strategy='short')
                        
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
                        signals = base_model.predict(X_clean, strategy=strategy)
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
                
                # Compute cumulative returns for summary
                cum_returns = gated_returns.cumsum()
                
                # Compute metric if provided
                metric_value = float('nan')
                if metric is not None:
                    try:
                        selected_returns = gated_returns[gated_returns != 0]
                        metric_value = float(metric.compute(selected_returns)) if len(selected_returns) > 0 else float('nan')
                    except Exception:
                        metric_value = float('nan')
                
                # Use pure plotting function
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
            'model': str(base_model) if base_model is not None else 'feature_direct',
            'metric': str(metric) if metric is not None else 'none',
            'summary': summary_df
        }
        return figures, summary_df
    
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
