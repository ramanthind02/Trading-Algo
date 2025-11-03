"""
Parameter Sensitivity Analysis and Visualization

This module provides tools for analyzing and visualizing parameter sensitivity
for feature engineering modules, including 1D and 2D parameter sweeps.
"""
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from feature_selection.base_models.quantile_binning import QuantileBinningModel
from utils.objective_metric import SortinoRatio, SharpeRatio

def _get_metric_name_from_object(metric_obj: Any) -> str:
    """
    Extract metric name from a metric object.
    
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
        metric: str = 'sortino',
        n_bins: int = 5,
        base_model: Optional[Any] = None,
        custom_metric: Optional[Any] = None,
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
        metric : str, default='sortino'
            Primary metric to compute
        n_bins : int, default=5
            Number of bins for QuantileBinningModel
        **metric_kwargs
            Additional keyword arguments for metric functions
            
        Returns
        -------
        pd.DataFrame
            DataFrame with parameter values and computed metrics
        """
        # Initialize model and metrics
        model = base_model if base_model is not None else QuantileBinningModel(n_bins=n_bins)
        
        # Infer metric name from custom_metric if provided
        if custom_metric is not None:
            metric_name = _get_metric_name_from_object(custom_metric)
        elif metric is None:
            metric_name = 'sortino'  # default
        else:
            metric_name = metric
        
        # Build metric functions with override support
        metric_funcs = {
            'mean': lambda x: np.mean(x),
            'std': lambda x: np.std(x),
            'drawdown': None  # Handled specially in _compute_metrics
        }
        if custom_metric is not None:
            # Use provided custom metric under the inferred metric name
            metric_funcs[metric_name] = custom_metric.compute
        else:
            # Default known metrics
            metric_funcs['sortino'] = SortinoRatio(**metric_kwargs).compute
            metric_funcs['sharpe'] = SharpeRatio(**metric_kwargs).compute
        
        # Update metric variable for later use
        metric = metric_name
        
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
        if metric not in possible_cols:
            possible_cols.append(metric)
        present = [c for c in possible_cols if c in results_df.columns]
        agg_map = {c: ('sum' if c == 'n_samples' else 'mean') for c in present}
        grouped_df = results_df.groupby(['param1_value']).agg(agg_map).reset_index()
        
        return grouped_df
    
    def analyze_2d_parameters(
        self,
        feature_grid: Dict[Tuple, List[str]],
        target_col: str = 'log_return',
        metric: str = 'sortino',
        n_bins: int = 5,
        base_model: Optional[Any] = None,
        custom_metric: Optional[Any] = None,
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
        metric : str, default='sortino'
            Primary metric to compute
        n_bins : int, default=5
            Number of bins for QuantileBinningModel
        **metric_kwargs
            Additional keyword arguments for metric functions
            
        Returns
        -------
        pd.DataFrame
            DataFrame with parameter values and computed metrics
        """
        # Initialize model and metrics
        model = base_model if base_model is not None else QuantileBinningModel(n_bins=n_bins)
        
        # Infer metric name from custom_metric if provided
        if custom_metric is not None:
            metric_name = _get_metric_name_from_object(custom_metric)
        elif metric is None:
            metric_name = 'sortino'  # default
        else:
            metric_name = metric
        
        metric_funcs = {
            'mean': lambda x: np.mean(x),
            'std': lambda x: np.std(x),
            'drawdown': None
        }
        if custom_metric is not None:
            # Use provided custom metric under the inferred metric name
            metric_funcs[metric_name] = custom_metric.compute
        else:
            metric_funcs['sortino'] = SortinoRatio(**metric_kwargs).compute
            metric_funcs['sharpe'] = SharpeRatio(**metric_kwargs).compute
        
        # Update metric variable for later use
        metric = metric_name
        
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
        if metric not in possible_cols:
            possible_cols.append(metric)
        present = [c for c in possible_cols if c in results_df.columns]
        agg_map = {c: ('sum' if c == 'n_samples' else 'mean') for c in present}
        grouped_df = results_df.groupby(['param1_value', 'param2_value']).agg(agg_map).reset_index()
        
        return grouped_df
    
    def plot_parameter_sensitivity(
        self,
        df: pd.DataFrame,
        param_name: str,
        metric: str = 'sortino',
        title: str = None,
        show_plot: bool = True
    ) -> go.Figure:
        """
        Plot parameter sensitivity for 1D parameter sweep.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame from analyze_parameter()
        param_name : str
            Name of the parameter being analyzed
        metric : str, default='sortino'
            Metric to plot on primary y-axis
        title : str, optional
            Plot title. If None, will be generated automatically.
        show_plot : bool, default=True
            Whether to show the plot
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Choose x-axis series: prefer 'param_value' else fallback to 'param1_value'
        x_series = 'param_value' if 'param_value' in df.columns else 'param1_value'
        # Prepare plotting frame: keep only needed cols, coerce to numeric, drop NaN
        cols_needed = [x_series, metric, 'n_samples']
        cols_present = [c for c in cols_needed if c in df.columns]
        df_plot = df[cols_present].copy()
        # Coerce x and y to numeric when possible
        if x_series in df_plot.columns:
            df_plot[x_series] = pd.to_numeric(df_plot[x_series], errors='coerce')
        if metric in df_plot.columns:
            df_plot[metric] = pd.to_numeric(df_plot[metric], errors='coerce')
        if 'n_samples' in df_plot.columns:
            df_plot['n_samples'] = pd.to_numeric(df_plot['n_samples'], errors='coerce')
        df_plot = df_plot.dropna(subset=[x_series, metric])
        # Sort by x for nicer lines
        if not df_plot.empty:
            df_plot = df_plot.sort_values(by=x_series)
        else:
            # Debug prints when no valid points
            try:
                print("[DEBUG] plot_parameter_sensitivity: empty df_plot")
                print(f"[DEBUG] Columns: {list(df.columns)}")
                print(f"[DEBUG] dtypes: {df.dtypes.to_dict()}")
                print(f"[DEBUG] head:\n{df.head(10)}")
                if x_series in df.columns:
                    print(f"[DEBUG] {x_series} NaNs: {df[x_series].isna().sum()} unique: {df[x_series].nunique(dropna=True)}")
                if metric in df.columns:
                    print(f"[DEBUG] {metric} NaNs: {df[metric].isna().sum()} unique: {df[metric].nunique(dropna=True)}")
            except Exception:
                pass
            raise ValueError("No valid points to plot (all metric/x values are NaN or missing)")

        # Add main metric trace
        fig.add_trace(
            go.Scatter(
                x=df_plot[x_series],
                y=df_plot[metric],
                mode='lines+markers',
                name=metric.capitalize(),
                line=dict(color='#1f77b4'),
                marker=dict(size=8)
            ),
            secondary_y=False,
        )
        
        # Add sample size as bar chart on secondary y-axis
        fig.add_trace(
            go.Bar(
                x=df_plot[x_series],
                y=df_plot['n_samples'] if 'n_samples' in df_plot.columns else None,
                name='Sample Size',
                opacity=0.2,
                marker_color='gray',
                showlegend=True
            ),
            secondary_y=True,
        )
        
        # Update layout
        if title is None:
            title = f"Parameter Sensitivity: {param_name}"
            
        fig.update_layout(
            title=title,
            xaxis_title=param_name,
            yaxis_title=metric.capitalize(),
            yaxis2_title="Sample Size",
            hovermode='x unified',
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Customize hover template
        fig.update_traces(
            hovertemplate=f"<b>{param_name}</b>: %{{x}}<br>" +
                         f"<b>{metric.capitalize()}</b>: %{{y:.4f}}<br>" +
                         "<extra></extra>"
        )
        
        if show_plot:
            fig.show()
            
        return fig

    def plot_2d_parameter_surface(
        self,
        df: pd.DataFrame,
        param1: str,
        param2: str,
        metric: str = 'sortino',
        title: str = None,
        show_plot: bool = True,
        plot_type: str = 'surface'
    ) -> go.Figure:
        """
        Create a 3D surface plot of parameter sensitivity.
        
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
        title : str, optional
            Plot title. If None, will be generated automatically.
        show_plot : bool, default=True
            Whether to show the plot
        plot_type : str, default='surface'
            Type of plot to generate: 'surface', 'scatter', 'heatmap', 'contour', 'lines'
        """
        # Create pivot table with sorted parameters
        param1_vals = sorted(df['param1_value'].unique())
        param2_vals = sorted(df['param2_value'].unique())
        
        pivot_df = df.pivot_table(index='param1_value', columns='param2_value', values=metric)
        
        # Fill NaN with zeros for plotting
        pivot_df = pivot_df.fillna(0)
        
        # Sort index and columns
        pivot_df = pivot_df.sort_index(axis=0)
        pivot_df = pivot_df.sort_index(axis=1)
        
        # Print pivot table for debugging
        print("Pivot table for surface plot:")
        print(pivot_df)
        
        # Check if we have enough data for surface plot
        if len(pivot_df) < 2 or len(pivot_df.columns) < 2:
            print(f"Warning: Insufficient data for surface plot ({pivot_df.shape[0]}x{pivot_df.shape[1]}). "
                  f"Need at least 2x2 grid.")
        
        # Check if data is too sparse for surface plot
        if len(pivot_df) < 3 or len(pivot_df.columns) < 3:
            print("Warning: Data is too sparse for a smooth surface plot. "
                  "Consider adding more parameter combinations.")
        
        # Create 3D surface plot
        fig = go.Figure()
        
        if len(pivot_df) > 1 and len(pivot_df.columns) > 1:
            # Normal case with multiple points
            if plot_type == 'surface':
                fig.add_trace(go.Surface(
                    z=pivot_df.values,
                    x=pivot_df.columns.values,
                    y=pivot_df.index.values,
                    colorscale='Viridis',
                    colorbar=dict(title=metric.capitalize()),
                    hovertemplate=(
                        f"<b>{param1}</b>: %{{y:.2f}}<br>" +
                        f"<b>{param2}</b>: %{{x:.2f}}<br>" +
                        f"<b>{metric}</b>: %{{z:.4f}}<extra></extra>"
                    )
                ))
            elif plot_type == 'scatter':
                fig.add_trace(go.Scatter3d(
                    x=df['param2_value'],
                    y=df['param1_value'],
                    z=df[metric],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=df[metric],
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    text=df['feature']
                ))
            elif plot_type == 'heatmap':
                fig.add_trace(go.Heatmap(
                    z=pivot_df.values,
                    x=pivot_df.columns.values,
                    y=pivot_df.index.values,
                    colorscale='Viridis',
                    colorbar=dict(title=metric.capitalize()),
                    hovertemplate=(
                        f"<b>{param1}</b>: %{{y:.2f}}<br>" +
                        f"<b>{param2}</b>: %{{x:.2f}}<br>" +
                        f"<b>{metric}</b>: %{{z:.4f}}<extra></extra>"
                    )
                ))
            elif plot_type == 'contour':
                fig.add_trace(go.Contour(
                    z=pivot_df.values,
                    x=pivot_df.columns.values,
                    y=pivot_df.index.values,
                    colorscale='Viridis',
                    colorbar=dict(title=metric.capitalize()),
                    hovertemplate=(
                        f"<b>{param1}</b>: %{{y:.2f}}<br>" +
                        f"<b>{param2}</b>: %{{x:.2f}}<br>" +
                        f"<b>{metric}</b>: %{{z:.4f}}<extra></extra>"
                    )
                ))
            elif plot_type == 'lines':
                fig.add_trace(go.Scatter3d(
                    x=df['param2_value'],
                    y=df['param1_value'],
                    z=df[metric],
                    mode='lines',
                    line=dict(
                        color=df[metric],
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    text=df['feature']
                ))
            else:
                raise ValueError(f"Invalid plot type: {plot_type}")
        else:
            # Fallback for insufficient data - show as 3D scatter plot
            print("Warning: Using 3D scatter plot due to insufficient data for surface")
            fig.add_trace(go.Scatter3d(
                x=df['param2_value'],
                y=df['param1_value'],
                z=df[metric],
                mode='markers',
                marker=dict(
                    size=5,
                    color=df[metric],
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=df['feature']
            ))
        
        # Update layout
        if title is None:
            title = f"{metric.capitalize()} vs {param1} and {param2}"
            
        fig.update_layout(
            scene=dict(
                xaxis_title=param2,
                yaxis_title=param1,
                zaxis_title=metric.capitalize(),
                xaxis=dict(showspikes=False, title_font=dict(size=12)),
                yaxis=dict(showspikes=False, title_font=dict(size=12)),
                zaxis=dict(showspikes=False, title_font=dict(size=12)),
                aspectmode='auto',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=0.8)
                )
            ),
            margin=dict(l=60, r=60, t=80, b=60),
            template='plotly_white',
            title=title if title else f"{metric.capitalize()} vs {param1} and {param2}",
            title_x=0.5,
            showlegend=False
        )
        
        if show_plot:
            fig.show()
            
        return fig
