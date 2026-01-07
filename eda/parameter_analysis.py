"""
Parameter Sensitivity Analysis and Visualization

This module provides tools for analyzing and visualizing parameter sensitivity
for feature engineering modules, including 1D and 2D parameter sweeps.
"""
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd
from feature_selection.base_models.quantile_binning import QuantileBinningModel
from metrics.performance import SortinoRatio, SharpeRatio
from metrics.plotting.parameter_plots import (
    plot_parameter_sensitivity as plot_parameter_sensitivity_pure,
    plot_2d_parameter_surface as plot_2d_parameter_surface_pure,
)

def _get_metric_name_from_object(metric_obj: Any) -> str:
    """
    Extract metric name from a metric object for display/plotting purposes.
    
    Converts class names like 'SortinoRatio' -> 'sortino', 'SharpeRatio' -> 'sharpe'
    
    Parameters
    ----------
    metric_obj : Any
        Metric object with .compute() method
        
    Returns
    -------
    str
        Metric name (lowercase, without 'Ratio' suffix)
    """
    if metric_obj is None:
        return 'sortino'  # default
    
    # Get class name
    class_name = metric_obj.__class__.__name__
    
    # Remove 'Ratio' suffix if present and convert to lowercase
    if class_name.endswith('Ratio'):
        return class_name[:-5].lower()  # 'SortinoRatio' -> 'sortino'
    
    # Fallback: just lowercase the class name
    return class_name.lower()


class ParameterAnalyzer:
    """
    Analyze and visualize parameter sensitivity for feature engineering modules.
    
    Supports both 1D and 2D parameter sweeps with interactive visualizations.
    """
    
    def __init__(self, features_df: pd.DataFrame, targets_df: pd.DataFrame):
        """
        Initialize the parameter analyzer.
        
        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame containing feature values (columns) for each sample (rows)
        targets_df : pd.DataFrame
            DataFrame containing target values with matching index to features_df
        """
        self.features_df = features_df
        self.targets_df = targets_df
        
    def _compute_metrics(
        self,
        feature_data: pd.Series,
        target_data: pd.Series,
        model: Any,
        metric_funcs: Dict[str, callable]
    ) -> Dict[str, float]:
        """Compute performance metrics for a given feature and target."""
        try:
            # Ensure data is properly aligned and 1D
            X = feature_data.values.ravel()
            y = target_data.values.ravel()
            
            if len(X) != len(y):
                raise ValueError(f"Feature and target length mismatch: {len(X)} vs {len(y)}")
            
            # Create clean DataFrame with aligned data
            df = pd.DataFrame({'feature': X, 'target': y}).dropna()
            
            if len(df) < 10:  # Minimum samples needed
                raise ValueError(f"Not enough valid samples ({len(df)})")
                
            # Fit model and get predictions
            model.fit(df['feature'], df['target'])
            signals = model.predict(df['feature'], strategy='long')
            signals = np.asarray(signals).astype(bool).flatten()
            selected_returns = df['target'].values[signals]
            
            if len(selected_returns) < 5:
                # Fallback: use full target when too few selected samples
                # This enables plotting while still providing a comparable metric
                selected_returns = y.values if hasattr(y, 'values') else np.asarray(y)
            
            # Compute metrics
            metrics = {}
            for name, func in metric_funcs.items():
                if name == 'drawdown':
                    try:
                        returns_series = pd.Series(selected_returns, name='returns')
                        cum_returns = (1 + returns_series).cumprod()
                        running_max = cum_returns.cummax()
                        drawdowns = (cum_returns - running_max) / running_max
                        metrics['max_drawdown'] = drawdowns.min()
                    except Exception as e:
                        metrics['max_drawdown'] = np.nan
                        raise
                else:
                    metrics[name] = func(selected_returns)
            
            metrics['n_samples'] = len(selected_returns)
            return metrics
            
        except Exception as e:
            raise ValueError(f"Error computing metrics: {str(e)}")
    
    def analyze_parameter(
        self,
        feature_group: List[Tuple[Union[float, str], str]],
        target_col: str = 'log_return',
        metric: Optional[Any] = None,
        n_bins: int = 5,
        base_model: Optional[Any] = None,
        **metric_kwargs
    ) -> pd.DataFrame:
        """
        Analyze sensitivity of a single parameter.
        
        Parameters
        ----------
        feature_group : List[Tuple[param_value, feature_name]]
            List of (parameter_value, feature_name) tuples
        target_col : str, default='log_return'
            Target column to use for computing metrics
        metric : Optional[Any], default=None
            Metric object from metrics.performance (e.g., SortinoRatio, SharpeRatio).
            Must have a .compute() method. If None, defaults to SortinoRatio.
        n_bins : int, default=5
            Number of bins for QuantileBinningModel
        base_model : Optional[Any], default=None
            Model instance with fit/predict methods. If None, uses QuantileBinningModel.
        **metric_kwargs
            DEPRECATED: Additional keyword arguments for metric functions.
            Pass metric configuration directly to the metric object constructor.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with parameter values and computed metrics
        """
        # Initialize model
        model = base_model if base_model is not None else QuantileBinningModel(n_bins=n_bins)
        
        # Use provided metric or default to SortinoRatio
        if metric is None:
            metric = SortinoRatio(annualization_factor=252)
        
        # Infer metric name for display/plotting
        metric_name = _get_metric_name_from_object(metric)
        
        # Build metric functions
        metric_funcs = {
            'mean': lambda x: np.mean(x),
            'std': lambda x: np.std(x),
            'drawdown': None  # Handled specially in _compute_metrics
        }
        # Use provided metric
        metric_funcs[metric_name] = metric.compute
        
        results = []
        for param_value, feature_name in feature_group:
            try:
                # Get feature and target data
                feature_series = self.features_df[feature_name]
                target_series = self.targets_df[target_col]
                # Align and drop NaNs jointly
                aligned = pd.concat([feature_series.rename('feature'), target_series.rename('target')], axis=1).dropna()
                feature_data = aligned['feature']
                target_data = aligned['target']
                # Debug: basic counts
                print(f"[DEBUG] analyze_parameter: '{feature_name}' samples={len(aligned)} (feature NaNs={feature_series.isna().sum()}, target NaNs={target_series.isna().sum()})")
                
                if len(feature_data) == 0 or len(target_data) == 0:
                    print(f"[DEBUG] Skipping '{feature_name}' due to empty aligned data")
                    continue
                
                metrics = self._compute_metrics(feature_data, target_data, model, metric_funcs)
                metrics.update({
                    'param_value': float(param_value) if str(param_value).replace('.', '').isdigit() else param_value,
                    'feature': feature_name,
                    'param1_value': param_value
                })
                results.append(metrics)
                
            except Exception as e:
                print(f"  [WARNING] Error processing {feature_name}: {str(e)}")
                try:
                    # More diagnostics
                    unique_non_nan = feature_series.dropna().nunique()
                    print(f"  [DEBUG] Unique non-NaN values in feature: {unique_non_nan}")
                    print(f"  [DEBUG] Head aligned:\n{aligned.head(5)}")
                except Exception:
                    pass
                continue
        
        if not results:
            raise ValueError("No valid data points to analyze.")
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Group by parameter values and aggregate metrics (1D: only param1)
        possible_cols = ['sortino', 'sharpe', 'mean', 'std', 'max_drawdown', 'n_samples']
        # Also include the metric name if it's not in the standard list
        if metric_name not in possible_cols:
            possible_cols.append(metric_name)
        present = [c for c in possible_cols if c in results_df.columns]
        agg_map = {c: ('sum' if c == 'n_samples' else 'mean') for c in present}
        grouped_df = results_df.groupby(['param1_value']).agg(agg_map).reset_index()
        
        return grouped_df
    
    def analyze_2d_parameters(
        self,
        feature_grid: Dict[Tuple, List[str]],
        target_col: str = 'log_return',
        metric: Optional[Any] = None,
        n_bins: int = 5,
        base_model: Optional[Any] = None,
        **metric_kwargs
    ) -> pd.DataFrame:
        """
        Analyze sensitivity of two parameters simultaneously.
        
        Parameters
        ----------
        feature_grid : Dict[Tuple[param1, param2], List[feature_name]]
            Dictionary mapping (param1, param2) tuples to feature names
        target_col : str, default='log_return'
            Target column to use for computing metrics
        metric : Optional[Any], default=None
            Metric object from metrics.performance (e.g., SortinoRatio, SharpeRatio).
            Must have a .compute() method. If None, defaults to SortinoRatio.
        n_bins : int, default=5
            Number of bins for QuantileBinningModel
        base_model : Optional[Any], default=None
            Model instance with fit/predict methods. If None, uses QuantileBinningModel.
        **metric_kwargs
            DEPRECATED: Additional keyword arguments for metric functions.
            Pass metric configuration directly to the metric object constructor.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with parameter values and computed metrics
        """
        # Initialize model
        model = base_model if base_model is not None else QuantileBinningModel(n_bins=n_bins)
        
        # Use provided metric or default to SortinoRatio
        if metric is None:
            metric = SortinoRatio(annualization_factor=252)
        
        # Infer metric name for display/plotting
        metric_name = _get_metric_name_from_object(metric)
        
        metric_funcs = {
            'mean': lambda x: np.mean(x),
            'std': lambda x: np.std(x),
            'drawdown': None
        }
        # Use provided metric
        metric_funcs[metric_name] = metric.compute
        
        results = []
        print(f"Analyzing 2D parameters with {len(feature_grid)} parameter combinations")
        for (param1, param2), feature_names in feature_grid.items():
            print(f"  Processing parameter combination: ({param1}, {param2}) with {len(feature_names)} features")
            
            feature_metrics = []
            for feature_name in feature_names:
                print(f"    Processing feature: {feature_name}")
                try:
                    # Get feature and target data (aligned)
                    feature_series = self.features_df[feature_name]
                    target_series = self.targets_df[target_col]
                    aligned = pd.concat([feature_series.rename('feature'), target_series.rename('target')], axis=1).dropna()
                    feature_data = aligned['feature']
                    target_data = aligned['target']
                    print(f"      [DEBUG] aligned samples={len(aligned)} (feature NaNs={feature_series.isna().sum()}, target NaNs={target_series.isna().sum()})")
                    
                    if len(feature_data) == 0 or len(target_data) == 0:
                        print(f"      [WARNING] Empty data for {feature_name}")
                        continue
                    
                    metrics = self._compute_metrics(feature_data, target_data, model, metric_funcs)
                    metrics.update({
                        'param1_value': param1,
                        'param2_value': param2,
                        'feature': feature_name
                    })
                    feature_metrics.append(metrics)
                    print(f"      Completed")
                    
                except Exception as e:
                    print(f"      Error processing {feature_name}: {str(e)}")
                    try:
                        unique_non_nan = feature_series.dropna().nunique()
                        print(f"      [DEBUG] Unique non-NaN values in feature: {unique_non_nan}")
                        print(f"      [DEBUG] Head aligned:\n{aligned.head(5)}")
                    except Exception:
                        pass
                    continue
            
            if feature_metrics:
                results.extend(feature_metrics)
        
        if not results:
            raise ValueError("No valid data points to analyze.")
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Group by parameter values and aggregate metrics
        possible_cols = ['sortino', 'sharpe', 'mean', 'std', 'max_drawdown', 'n_samples']
        # Also include the metric name if it's not in the standard list
        if metric_name not in possible_cols:
            possible_cols.append(metric_name)
        present = [c for c in possible_cols if c in results_df.columns]
        agg_map = {c: ('sum' if c == 'n_samples' else 'mean') for c in present}
        grouped_df = results_df.groupby(['param1_value', 'param2_value']).agg(agg_map).reset_index()
        
        return grouped_df
    
    def plot_parameter_sensitivity(
        self,
        df: pd.DataFrame,
        param_name: str,
        metric: str = 'sortino',
        title: Optional[str] = None,
        show_plot: bool = True
    ):
        """
        Plot parameter sensitivity for 1D parameter sweep.
        
        Delegates to pure function in metrics.plotting.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame from analyze_parameter()
        param_name : str
            Name of the parameter being analyzed
        metric : str, default='sortino'
            Metric to plot on primary y-axis
        title : Optional[str], default=None
            Plot title. If None, will be generated automatically.
        show_plot : bool, default=True
            Whether to show the plot
            
        Returns
        -------
        Plotly figure object
        """
        return plot_parameter_sensitivity_pure(
            df=df,
            param_name=param_name,
            metric=metric,
            title=title,
            show_plot=show_plot
        )

    def plot_2d_parameter_surface(
        self,
        df: pd.DataFrame,
        param1: str,
        param2: str,
        metric: str = 'sortino',
        title: Optional[str] = None,
        show_plot: bool = True,
        plot_type: str = 'surface'
    ):
        """
        Create a 3D surface plot of parameter sensitivity.
        
        Delegates to pure function in metrics.plotting.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame from analyze_2d_parameters()
        param1 : str
            Name of the first parameter (x-axis)
        param2 : str
            Name of the second parameter (y-axis)
        metric : str, default='sortino'
            Metric to plot on z-axis
        title : Optional[str], default=None
            Plot title. If None, will be generated automatically.
        show_plot : bool, default=True
            Whether to show the plot
        plot_type : str, default='surface'
            Type of plot to generate: 'surface', 'scatter', 'heatmap', 'contour', 'lines'
            
        Returns
        -------
        Plotly figure object
        """
        return plot_2d_parameter_surface_pure(
            df=df,
            param1=param1,
            param2=param2,
            metric=metric,
            title=title,
            show_plot=show_plot,
            plot_type=plot_type
        )
