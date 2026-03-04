"""
Parameter Sensitivity Analysis and Visualization

This module provides tools for analyzing and visualizing parameter sensitivity
for feature engineering modules, including 1D and 2D parameter sweeps,
grid-aware neighbor smoothing, stable region identification, and report generation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from metrics.performance import SortinoRatio, SharpeRatio
from metrics.plotting.parameter_plots import (
    plot_parameter_sensitivity as plot_parameter_sensitivity_pure,
    plot_2d_parameter_surface as plot_2d_parameter_surface_pure,
)
from utils.compute.grid_smoothing import add_smoothed_objective

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


# ---------------------------------------------------------------------------
# T009 — Grid-Aware Neighbor Smoothing (wrapper around utils.compute.grid_smoothing)
# ---------------------------------------------------------------------------

def identify_neighbors(
    param_values: Tuple,
    grid_structure: Dict[str, List],
) -> List[Tuple]:
    """
    Return all 1-step axis-aligned neighbor tuples for a given parameter combination.

    Parameters
    ----------
    param_values : Tuple
        Current parameter combination (p1, p2, ..., pN).
    grid_structure : Dict[str, List]
        Ordered mapping of param names to their sorted unique values.
        Example: {"lookback": [2, 3, 4, 5], "threshold": [0.3, 0.5, 0.7]}

    Returns
    -------
    List[Tuple]
        Neighbor tuples that differ in exactly one dimension by one grid step.
    """
    param_names = list(grid_structure.keys())
    neighbors: List[Tuple] = []

    for dim_idx, name in enumerate(param_names):
        sorted_vals = grid_structure[name]
        current_val = param_values[dim_idx]

        try:
            pos = sorted_vals.index(current_val)
        except ValueError:
            continue

        for offset in (-1, 1):
            neighbor_pos = pos + offset
            if 0 <= neighbor_pos < len(sorted_vals):
                neighbor_list = list(param_values)
                neighbor_list[dim_idx] = sorted_vals[neighbor_pos]
                neighbors.append(tuple(neighbor_list))

    return neighbors


def compute_neighbor_smoothing(
    results_df: pd.DataFrame,
    param_names: List[str],
    metric_col: str,
    self_weight: float = 2.0,
) -> pd.DataFrame:
    """
    Add smoothed objective, stability ratio, and neighbor count columns.

    Wraps ``utils.compute.grid_smoothing.add_smoothed_objective()`` and enriches the
    result with:
    - ``smoothed_{metric_col}``: neighbor-averaged metric
    - ``stability_ratio``: smoothed / raw (NaN when raw == 0)
    - ``n_neighbors``: number of existing grid neighbors per row

    Parameters
    ----------
    results_df : pd.DataFrame
        Grid search results with ``param1_value`` … ``paramN_value`` columns
        and the metric column.
    param_names : List[str]
        Human-readable parameter names (same order as paramK_value columns).
    metric_col : str
        Column name of the raw objective metric (e.g. ``'sortino'``).
    self_weight : float, default 2.0
        Weight for the param's own value vs. neighbors. Higher values reduce
        smoothing (param counts more).

    Returns
    -------
    pd.DataFrame
        Copy of *results_df* with three new columns added.
    """
    n_params = len(param_names)
    param_cols = [f"param{k}_value" for k in range(1, n_params + 1)]

    smoothed_col = f"smoothed_{metric_col}"

    # Delegate to the core smoothing algorithm (self_weight > 1 = less aggressive)
    smoothed_df = add_smoothed_objective(
        df=results_df,
        param_columns=param_cols,
        objective_column=metric_col,
        output_column=smoothed_col,
        self_weight=self_weight,
    )

    # Stability ratio: smoothed / raw  (NaN when raw == 0)
    raw = smoothed_df[metric_col].values.astype(float)
    smoothed = smoothed_df[smoothed_col].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(raw != 0.0, smoothed / raw, np.nan)
    smoothed_df["stability_ratio"] = ratio

    # Compute n_neighbors per row using identify_neighbors()
    grid_structure: Dict[str, List] = {}
    for col, name in zip(param_cols, param_names):
        unique_vals = sorted(smoothed_df[col].dropna().unique())
        grid_structure[name] = unique_vals

    # Build a set of all existing grid points for O(1) membership checks
    existing_points = set(
        tuple(row) for row in smoothed_df[param_cols].values
    )

    n_neighbors_arr = np.empty(len(smoothed_df), dtype=int)
    param_values_arr = smoothed_df[param_cols].values

    for i, row_params in enumerate(param_values_arr):
        all_neighbors = identify_neighbors(tuple(row_params), grid_structure)
        n_neighbors_arr[i] = sum(1 for nb in all_neighbors if nb in existing_points)

    smoothed_df["n_neighbors"] = n_neighbors_arr

    return smoothed_df


# ---------------------------------------------------------------------------
# T010 — Stable Region Identification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StableRegion:
    """A contiguous set of parameter combinations with high stability ratio."""

    param_ranges: Dict[str, Tuple[Any, Any]]
    param_combinations: List[Tuple]
    mean_stability_ratio: float
    mean_objective: float
    min_objective: float
    max_objective: float
    n_combinations: int
    is_boundary_region: bool


def identify_stable_regions(
    smoothed_df: pd.DataFrame,
    metric_col: str,
    stability_threshold: float = 0.8,
) -> List[StableRegion]:
    """
    Find contiguous regions of the parameter grid where stability_ratio > threshold.

    Uses BFS on an axis-aligned adjacency graph to discover connected components
    among grid points that exceed the stability threshold.

    Parameters
    ----------
    smoothed_df : pd.DataFrame
        DataFrame from ``compute_neighbor_smoothing()`` with ``paramK_value``,
        ``smoothed_{metric_col}``, and ``stability_ratio`` columns.
    metric_col : str
        Name of the *smoothed* metric column (e.g. ``'smoothed_sortino'``).
        The corresponding raw column is inferred by stripping the ``smoothed_``
        prefix if present.
    stability_threshold : float, default 0.8
        Minimum stability ratio to include a point in a stable region.

    Returns
    -------
    List[StableRegion]
        Stable regions with ≥ 2 parameter combinations, sorted by
        ``mean_objective`` descending.
    """
    param_cols = sorted(
        [c for c in smoothed_df.columns if c.startswith("param") and c.endswith("_value")]
    )
    n_params = len(param_cols)

    # Infer human-readable param names from column names (param1_value -> param1)
    param_names = [c.replace("_value", "") for c in param_cols]

    # Build grid structure for neighbor lookups
    grid_structure: Dict[str, List] = {}
    for col, name in zip(param_cols, param_names):
        grid_structure[name] = sorted(smoothed_df[col].dropna().unique())

    # Grid boundary values for boundary detection
    grid_min_max: Dict[str, Tuple] = {}
    for name, vals in grid_structure.items():
        grid_min_max[name] = (vals[0], vals[-1]) if vals else (None, None)

    # Filter to stable points
    stable_mask = smoothed_df["stability_ratio"] > stability_threshold
    stable_df = smoothed_df[stable_mask].copy()

    if stable_df.empty:
        return []

    # Build point -> index mapping for BFS
    stable_points: Dict[Tuple, int] = {}
    for idx, row in enumerate(stable_df[param_cols].values):
        stable_points[tuple(row)] = idx

    # BFS to find connected components
    visited: set = set()
    components: List[List[Tuple]] = []

    for point in stable_points:
        if point in visited:
            continue

        # BFS from this point
        component: List[Tuple] = []
        queue = deque([point])
        visited.add(point)

        while queue:
            current = queue.popleft()
            component.append(current)

            neighbors = identify_neighbors(current, grid_structure)
            for nb in neighbors:
                if nb in stable_points and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        components.append(component)

    # Build StableRegion objects (minimum size = 2)
    smoothed_metric_col = metric_col if metric_col.startswith("smoothed_") else f"smoothed_{metric_col}"
    raw_metric_col = metric_col.replace("smoothed_", "") if metric_col.startswith("smoothed_") else metric_col

    regions: List[StableRegion] = []
    for component in components:
        if len(component) < 2:
            continue

        # Gather metrics for points in this component
        component_set = set(component)
        mask = stable_df[param_cols].apply(lambda row: tuple(row) in component_set, axis=1)
        region_df = stable_df[mask]

        # Param ranges
        param_range_dict: Dict[str, Tuple[Any, Any]] = {}
        for col, name in zip(param_cols, param_names):
            vals = region_df[col].values
            param_range_dict[name] = (vals.min(), vals.max())

        # Boundary detection: check if any param value touches grid min/max
        is_boundary = False
        for col, name in zip(param_cols, param_names):
            gmin, gmax = grid_min_max[name]
            if gmin in region_df[col].values or gmax in region_df[col].values:
                is_boundary = True
                break

        # Use the smoothed metric column for objective values
        obj_col = smoothed_metric_col if smoothed_metric_col in region_df.columns else raw_metric_col
        obj_values = region_df[obj_col].values

        regions.append(StableRegion(
            param_ranges=param_range_dict,
            param_combinations=component,
            mean_stability_ratio=float(region_df["stability_ratio"].mean()),
            mean_objective=float(np.nanmean(obj_values)),
            min_objective=float(np.nanmin(obj_values)),
            max_objective=float(np.nanmax(obj_values)),
            n_combinations=len(component),
            is_boundary_region=is_boundary,
        ))

    # Sort by mean_objective descending
    regions.sort(key=lambda r: r.mean_objective, reverse=True)
    return regions


# ---------------------------------------------------------------------------
# T012 — ParameterSensitivityReport dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParameterSensitivityReport:
    """Structured output of Phase 3 parameter sensitivity analysis."""

    # Input metadata
    param_names: List[str]
    metric_name: str
    stability_threshold: float
    metric_floor: float | None

    # Grid analysis results
    grid_results: pd.DataFrame
    stable_regions: List[StableRegion]

    # Summary statistics
    mean_stability_ratio: float
    median_stability_ratio: float
    pct_stable_combinations: float

    # Recommendations
    recommended_combinations: List[Tuple]
    top_k_combinations: List[Tuple]

    # Visualizations
    plot_1d: Optional[go.Figure] = None
    plot_2d: Optional[go.Figure] = None
    plot_3d: Optional[go.Figure] = None

    # Diagnostic info
    n_parameter_combinations: int = 0
    n_stable_regions: int = 0
    timestamp: str = ""

    class Config:
        arbitrary_types_allowed = True


def generate_parameter_sensitivity_report(
    results_df: pd.DataFrame,
    param_names: List[str],
    metric_col: str,
    stability_threshold: float = 0.8,
    top_k: int = 3,
    plot_3d_mode: str = "heatmap_slices",
    smoothing_self_weight: float = 2.0,
    metric_floor: float | None = 2.0,
) -> ParameterSensitivityReport:
    """
    Orchestrate smoothing and parameter sensitivity plots/recommendations.

    Selection recommendations are produced from the smoothed objective surface.

    Parameters
    ----------
    results_df : pd.DataFrame
        Grid search results with ``paramK_value`` columns and *metric_col*.
    param_names : List[str]
        Human-readable parameter names (same order as paramK_value columns).
    metric_col : str
        Column name of the raw objective metric (e.g. ``'sortino'``).
    stability_threshold : float, default 0.8
        Plot shading threshold when stable_regions are provided (unused when []).
    top_k : int, default 3
        Number of top recommended parameter combinations.
    plot_3d_mode : str, default "heatmap_slices"
        3D plot style when ``len(param_names) >= 3``.
    smoothing_self_weight : float, default 2.0
        Weight for param's own value vs. neighbors in neighbor smoothing.

    Returns
    -------
    ParameterSensitivityReport
    """
    from metrics.plotting.parameter_plots import (
        plot_parameter_sensitivity_with_stability,
        plot_2d_stability_heatmap,
        plot_3d_slices,
    )

    n_dims = len(param_names)
    smoothed_metric_col = f"smoothed_{metric_col}"

    smoothed_df = compute_neighbor_smoothing(
        results_df, param_names, metric_col, self_weight=smoothing_self_weight
    )
    sorted_smoothed = smoothed_df.sort_values(smoothed_metric_col, ascending=False)
    recommended = [
        tuple(row[f"param{k}_value"] for k in range(1, n_dims + 1))
        for _, row in sorted_smoothed.iterrows()
    ]
    top_k_combos = recommended[:top_k]
    stable_regions: List[StableRegion] = []
    pct_stable = len(top_k_combos) / max(len(results_df), 1)

    # Summary statistics (from smoothed grid)
    ratios = smoothed_df["stability_ratio"].dropna()
    mean_ratio = float(ratios.mean()) if len(ratios) > 0 else 0.0
    median_ratio = float(ratios.median()) if len(ratios) > 0 else 0.0

    # Plots
    plot_1d: Optional[go.Figure] = None
    plot_2d: Optional[go.Figure] = None
    plot_3d: Optional[go.Figure] = None
    plot_threshold = stability_threshold
    if n_dims == 1:
        plot_1d = plot_parameter_sensitivity_with_stability(
            df=smoothed_df,
            param_name=param_names[0],
            metric=metric_col,
            stable_regions=stable_regions,
            stability_threshold=plot_threshold,
            metric_floor=metric_floor,
            show_plot=False,
        )
    elif n_dims == 2:
        plot_2d = plot_2d_stability_heatmap(
            df=smoothed_df,
            param1=param_names[0],
            param2=param_names[1],
            metric=metric_col,
            stable_regions=stable_regions,
            metric_floor=metric_floor,
            show_plot=False,
        )
    else:
        if plot_3d_mode == "heatmap_slices":
            plot_type = "heatmap"
        elif plot_3d_mode == "surface_slices":
            plot_type = "surface"
        else:
            raise ValueError(
                f"Invalid plot_3d_mode: {plot_3d_mode}. "
                "Expected one of: ['heatmap_slices', 'surface_slices']."
            )
        plot_3d = plot_3d_slices(
            df=smoothed_df,
            param_names=param_names,
            metric=metric_col,
            plot_type=plot_type,
            metric_floor=metric_floor,
            show_plot=False,
        )

    return ParameterSensitivityReport(
        param_names=param_names,
        metric_name=metric_col,
        stability_threshold=stability_threshold,
        metric_floor=metric_floor,
        grid_results=smoothed_df,
        stable_regions=stable_regions,
        mean_stability_ratio=mean_ratio,
        median_stability_ratio=median_ratio,
        pct_stable_combinations=float(pct_stable),
        recommended_combinations=recommended,
        top_k_combinations=top_k_combos,
        plot_1d=plot_1d,
        plot_2d=plot_2d,
        plot_3d=plot_3d,
        n_parameter_combinations=len(smoothed_df),
        n_stable_regions=len(stable_regions),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


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
            Number of bins for ContinuousBinningModel
        base_model : Optional[Any], default=None
            Model instance with fit/predict methods. If None, uses ContinuousBinningModel.
        **metric_kwargs
            DEPRECATED: Additional keyword arguments for metric functions.
            Pass metric configuration directly to the metric object constructor.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with parameter values and computed metrics
        """
        # Initialize model
        model = base_model if base_model is not None else ContinuousBinningModel(n_bins=n_bins)
        
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
                
                
                if len(feature_data) == 0 or len(target_data) == 0:

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
            Number of bins for ContinuousBinningModel
        base_model : Optional[Any], default=None
            Model instance with fit/predict methods. If None, uses ContinuousBinningModel.
        **metric_kwargs
            DEPRECATED: Additional keyword arguments for metric functions.
            Pass metric configuration directly to the metric object constructor.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with parameter values and computed metrics
        """
        # Initialize model
        model = base_model if base_model is not None else ContinuousBinningModel(n_bins=n_bins)
        
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

    def analyze_nd_parameters(
        self,
        feature_grid: Dict[Tuple, List[str]],
        param_names: List[str],
        target_col: str = 'log_return',
        metric: Optional[Any] = None,
        n_bins: int = 5,
        base_model: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Analyze sensitivity of N parameters simultaneously.

        Generalizes analyze_2d_parameters to support 1-4 parameters.

        Parameters
        ----------
        feature_grid : Dict[Tuple, List[str]]
            Dictionary mapping parameter value tuples to feature column names.
            Keys are tuples of (val1, val2, ..., valN) matching param_names order.
        param_names : List[str]
            Names of the parameters (1-4).
        target_col : str, default='log_return'
            Target column to use for computing metrics.
        metric : Optional[Any], default=None
            Metric object with .compute() method. Defaults to SortinoRatio.
        n_bins : int, default=5
            Number of bins for ContinuousBinningModel.
        base_model : Optional[Any], default=None
            Model instance with fit/predict methods. Defaults to ContinuousBinningModel.

        Returns
        -------
        pd.DataFrame
            DataFrame with paramK_value columns (K=1..N) and computed metrics.
        """
        model = base_model if base_model is not None else ContinuousBinningModel(n_bins=n_bins)

        if metric is None:
            metric = SortinoRatio(annualization_factor=252)

        metric_name = _get_metric_name_from_object(metric)

        metric_funcs = {
            'mean': lambda x: np.mean(x),
            'std': lambda x: np.std(x),
            'drawdown': None
        }
        metric_funcs[metric_name] = metric.compute

        results = []
        n_params = len(param_names)
        print(f"Analyzing {n_params}D parameters with {len(feature_grid)} parameter combinations")

        for param_vals, feature_names in feature_grid.items():
            # Ensure param_vals is a tuple
            if not isinstance(param_vals, tuple):
                param_vals = (param_vals,)

            for feature_name in feature_names:
                try:
                    feature_series = self.features_df[feature_name]
                    target_series = self.targets_df[target_col]
                    aligned = pd.concat(
                        [feature_series.rename('feature'), target_series.rename('target')], axis=1
                    ).dropna()
                    feature_data = aligned['feature']
                    target_data = aligned['target']

                    if len(feature_data) == 0 or len(target_data) == 0:
                        continue

                    metrics = self._compute_metrics(feature_data, target_data, model, metric_funcs)
                    # Add paramK_value columns
                    for k, val in enumerate(param_vals, start=1):
                        metrics[f'param{k}_value'] = val
                    metrics['feature'] = feature_name
                    results.append(metrics)

                except Exception as e:
                    print(f"  [WARNING] Error processing {feature_name}: {str(e)}")
                    continue

        if not results:
            raise ValueError("No valid data points to analyze.")

        results_df = pd.DataFrame(results)

        # Group by all paramK_value columns and aggregate
        group_cols = [f'param{k}_value' for k in range(1, n_params + 1)]
        possible_cols = ['sortino', 'sharpe', 'mean', 'std', 'max_drawdown', 'n_samples']
        if metric_name not in possible_cols:
            possible_cols.append(metric_name)
        present = [c for c in possible_cols if c in results_df.columns]
        agg_map = {c: ('sum' if c == 'n_samples' else 'mean') for c in present}
        grouped_df = results_df.groupby(group_cols).agg(agg_map).reset_index()

        return grouped_df

    def compute_robustness_metrics(
        self,
        results_df: pd.DataFrame,
        metric_col: str,
        metric_threshold: float = 1.0
    ) -> Dict[str, Any]:
        """
        Compute robustness metrics for parameter sensitivity analysis.

        Scores how robust the strategy is across the parameter space.

        Parameters
        ----------
        results_df : pd.DataFrame
            Results DataFrame from analyze_parameter, analyze_2d_parameters,
            or analyze_nd_parameters.
        metric_col : str
            Column name of the metric to evaluate (e.g., 'sortino').
        metric_threshold : float, default=1.0
            Threshold for "acceptable" performance.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys: variance_score, consistency_score, risk_score,
            overall_score, rating, statistics, parameter_sensitivity.
        """
        metric_values = results_df[metric_col].dropna()

        if len(metric_values) == 0:
            return {
                'variance_score': 0.0,
                'consistency_score': 0.0,
                'risk_score': 0.0,
                'overall_score': 0.0,
                'rating': 'HIGHLY FRAGILE',
                'statistics': {},
                'parameter_sensitivity': {}
            }

        mean_val = metric_values.mean()
        std_val = metric_values.std()

        # Variance score: penalize high coefficient of variation
        cv = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else float('inf')
        variance_score = max(0.0, 10.0 - cv * 20.0)

        # Consistency score: percentage above threshold
        pct_above = (metric_values >= metric_threshold).mean()
        consistency_score = pct_above * 10.0

        # Risk score: penalize large drawdowns
        if 'max_drawdown' in results_df.columns:
            worst_dd = results_df['max_drawdown'].dropna().min()
            risk_score = max(0.0, 10.0 - abs(worst_dd) * 50.0)
        else:
            risk_score = 5.0  # neutral if no drawdown data

        # Overall score: average of components
        overall_score = (variance_score + consistency_score + risk_score) / 3.0

        # Rating
        if overall_score >= 8.0:
            rating = 'HIGHLY ROBUST'
        elif overall_score >= 6.0:
            rating = 'ROBUST'
        elif overall_score >= 4.0:
            rating = 'MODERATELY ROBUST'
        elif overall_score >= 2.0:
            rating = 'FRAGILE'
        else:
            rating = 'HIGHLY FRAGILE'

        # Statistics
        statistics = {
            'mean': float(mean_val),
            'median': float(metric_values.median()),
            'std': float(std_val),
            'min': float(metric_values.min()),
            'max': float(metric_values.max()),
            'cv': float(cv) if cv != float('inf') else None,
            'n_configurations': len(metric_values),
            'pct_above_threshold': float(pct_above),
        }

        # Parameter sensitivity: variance contribution per param
        param_cols = [c for c in results_df.columns if c.startswith('param') and c.endswith('_value')]
        parameter_sensitivity = {}

        if param_cols and len(metric_values) > 1:
            total_var = 0.0
            param_vars = {}
            for pc in param_cols:
                if pc in results_df.columns:
                    group_means = results_df.groupby(pc)[metric_col].mean()
                    inter_group_var = group_means.var()
                    if np.isnan(inter_group_var):
                        inter_group_var = 0.0
                    param_vars[pc] = inter_group_var
                    total_var += inter_group_var

            # Normalize to sum to 1
            if total_var > 0:
                for pc in param_vars:
                    # Use param name without _value suffix for cleaner output
                    param_key = pc.replace('_value', '')
                    parameter_sensitivity[param_key] = param_vars[pc] / total_var
            else:
                for pc in param_cols:
                    param_key = pc.replace('_value', '')
                    parameter_sensitivity[param_key] = 1.0 / len(param_cols)

        return {
            'variance_score': float(variance_score),
            'consistency_score': float(consistency_score),
            'risk_score': float(risk_score),
            'overall_score': float(overall_score),
            'rating': rating,
            'statistics': statistics,
            'parameter_sensitivity': parameter_sensitivity,
        }

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

    def compute_neighbor_smoothed_grid(
        self,
        param_results: pd.DataFrame,
        param_cols: List[str],
        objective_col: str = 'objective',
    ) -> pd.DataFrame:
        """Compute axis-aligned 1-step neighbor-smoothed objective column.

        smoothed(P) = mean([obj(P)] + [obj(N) for N in 1-step_axis_neighbors(P)])

        Args:
            param_results: DataFrame with parameter columns and an objective column.
            param_cols: Names of parameter dimension columns.
            objective_col: Name of the objective metric column.

        Returns:
            Copy of param_results with added 'smoothed_objective' column.
        """
        result_df = param_results.copy()
        param_value_sets = {
            col: sorted(param_results[col].unique())
            for col in param_cols
        }

        smoothed_values: List[float] = []
        for _, row in param_results.iterrows():
            values = [float(row[objective_col])]

            for col in param_cols:
                sorted_vals = param_value_sets[col]
                current_val = row[col]
                try:
                    val_idx = sorted_vals.index(current_val)
                except ValueError:
                    continue

                for neighbor_idx in (val_idx - 1, val_idx + 1):
                    if 0 <= neighbor_idx < len(sorted_vals):
                        mask = pd.Series(True, index=param_results.index)
                        for c in param_cols:
                            if c == col:
                                mask &= param_results[c] == sorted_vals[neighbor_idx]
                            else:
                                mask &= param_results[c] == row[c]
                        matching = param_results[mask]
                        if not matching.empty:
                            values.append(float(matching.iloc[0][objective_col]))

            smoothed_values.append(float(np.mean(values)))

        result_df['smoothed_objective'] = smoothed_values
        return result_df

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
