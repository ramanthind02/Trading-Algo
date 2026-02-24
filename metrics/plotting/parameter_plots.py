"""
Pure plotting functions for parameter sensitivity analysis.

This module provides pure functions for visualizing parameter sensitivity results.
All functions are stateless and follow functional programming principles.

Author: Trading Research Team
Date: 2025-01-XX
"""

from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING, Union
from itertools import combinations
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from eda.parameter_analysis import StableRegion


def plot_parameter_sensitivity(
    df: pd.DataFrame,
    param_name: str,
    metric: str = 'sortino',
    title: Optional[str] = None,
    show_plot: bool = True
) -> go.Figure:
    """
    Plot parameter sensitivity for 1D parameter sweep.
    
    Pure function version of ParameterAnalyzer.plot_parameter_sensitivity().
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from parameter analysis with columns: param_value (or param1_value), metric, n_samples
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
    go.Figure
        Plotly figure object
    """
    # Create figure with config to prevent auto-display when show_plot=False
    config = {'displayModeBar': False} if not show_plot else {}
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
    if 'n_samples' in df_plot.columns:
        fig.add_trace(
            go.Bar(
                x=df_plot[x_series],
                y=df_plot['n_samples'],
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
        fig.show(config=config)
    # When show_plot=False, we don't call .show() to prevent display
    # Note: Plotly figures in Jupyter may still auto-display - this is a Jupyter limitation
        
    return fig


def plot_2d_parameter_surface(
    df: pd.DataFrame,
    param1: str,
    param2: str,
    metric: str = 'sortino',
    title: Optional[str] = None,
    show_plot: bool = True,
    plot_type: str = 'surface'
) -> go.Figure:
    """
    Create a 3D surface plot of parameter sensitivity.
    
    Pure function version of ParameterAnalyzer.plot_2d_parameter_surface().
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from 2D parameter analysis with columns: param1_value, param2_value, metric
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
    go.Figure
        Plotly figure object
    """
    # Create pivot table with sorted parameters
    param1_vals = sorted(df['param1_value'].unique())
    param2_vals = sorted(df['param2_value'].unique())
    
    pivot_df = df.pivot_table(index='param1_value', columns='param2_value', values=metric)
    
    # NaN values remain as gaps (not filled with false zeros)
    
    # Sort index and columns
    pivot_df = pivot_df.sort_index(axis=0)
    pivot_df = pivot_df.sort_index(axis=1)
    
    # Check if we have enough data for surface plot
    if len(pivot_df) < 2 or len(pivot_df.columns) < 2:
        raise ValueError(
            f"Insufficient data for surface plot ({pivot_df.shape[0]}x{pivot_df.shape[1]}). "
            f"Need at least 2x2 grid."
        )
    
    # Create 3D surface plot with config to prevent auto-display when show_plot=False
    config = {'displayModeBar': False} if not show_plot else {}
    fig = go.Figure()
    
    if len(pivot_df) > 1 and len(pivot_df.columns) > 1:
        # Normal case with multiple points
        if plot_type == 'surface':
            fig.add_trace(go.Surface(
                z=pivot_df.values,
                x=pivot_df.columns.values,
                y=pivot_df.index.values,
                colorscale='Viridis',
                connectgaps=True,
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
                text=df.get('feature', '')
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
                text=df.get('feature', '')
            ))
        else:
            raise ValueError(f"Invalid plot type: {plot_type}")
    else:
        # Fallback for insufficient data - show as 3D scatter plot
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
            text=df.get('feature', '')
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
        title=title,
        title_x=0.5,
        showlegend=False
    )

    if show_plot:
        fig.show(config=config)
    # When show_plot=False, we don't call .show() to prevent display
    # Note: Plotly figures in Jupyter may still auto-display - this is a Jupyter limitation

    return fig


def plot_3d_parameter_interactive(
    df: pd.DataFrame,
    param_names: List[str],
    metric: str = 'sortino',
    title: Optional[str] = None,
    show_plot: bool = True,
    plot_type: str = 'surface'
) -> go.Figure:
    """
    Create an interactive 3D parameter sensitivity visualization.

    Uses a dropdown to select which parameter to fix and a slider to step
    through values of the fixed parameter, showing a 2D surface of the
    remaining two parameters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from analyze_nd_parameters with param1_value, param2_value,
        param3_value columns and the metric column.
    param_names : List[str]
        List of 3 parameter names matching param1_value, param2_value, param3_value.
    metric : str, default='sortino'
        Metric column to visualize.
    title : Optional[str], default=None
        Plot title. Auto-generated if None.
    show_plot : bool, default=True
        Whether to show the plot.
    plot_type : str, default='surface'
        Type of plot: 'surface', 'heatmap', 'contour'.

    Returns
    -------
    go.Figure
        Plotly figure with dropdown + slider controls.
    """
    if len(param_names) != 3:
        raise ValueError(f"Expected 3 parameter names, got {len(param_names)}")

    param_cols = ['param1_value', 'param2_value', 'param3_value']

    # Build traces: for each fixed-param choice, for each value of that param
    traces = []
    dropdown_buttons = []
    slider_groups = {}  # fixed_idx -> group info

    trace_idx = 0
    for fixed_idx in range(3):
        free_indices = [i for i in range(3) if i != fixed_idx]
        free_cols = [param_cols[i] for i in free_indices]
        fixed_col = param_cols[fixed_idx]

        fixed_values = sorted(df[fixed_col].unique())
        group_start = trace_idx

        for fixed_val in fixed_values:
            subset = df[df[fixed_col] == fixed_val]
            if subset.empty:
                traces.append(go.Surface(z=[[0]], x=[0], y=[0], visible=False, showscale=False))
                trace_idx += 1
                continue

            pivot = subset.pivot_table(
                index=free_cols[0], columns=free_cols[1], values=metric
            )
            pivot = pivot.sort_index(axis=0).sort_index(axis=1)

            if plot_type == 'heatmap':
                trace = go.Heatmap(
                    z=pivot.values,
                    x=[str(v) for v in pivot.columns.values],
                    y=[str(v) for v in pivot.index.values],
                    colorscale='Viridis',
                    visible=False,
                    showscale=True,
                    colorbar=dict(title=metric.capitalize()),
                )
            elif plot_type == 'contour':
                trace = go.Contour(
                    z=pivot.values,
                    x=[str(v) for v in pivot.columns.values],
                    y=[str(v) for v in pivot.index.values],
                    colorscale='Viridis',
                    visible=False,
                    showscale=True,
                    colorbar=dict(title=metric.capitalize()),
                )
            else:
                trace = go.Surface(
                    z=pivot.values,
                    x=pivot.columns.values,
                    y=pivot.index.values,
                    colorscale='Viridis',
                    connectgaps=True,
                    visible=False,
                    showscale=True,
                    colorbar=dict(title=metric.capitalize()),
                )

            traces.append(trace)
            trace_idx += 1

        group_end = trace_idx
        slider_groups[fixed_idx] = {
            'start': group_start,
            'end': group_end,
            'values': fixed_values,
            'free_names': [param_names[i] for i in free_indices],
        }

    fig = go.Figure(data=traces)
    total_traces = len(traces)

    # Build dropdown buttons for selecting which param to fix
    for fixed_idx in range(3):
        group = slider_groups[fixed_idx]
        visibility = [False] * total_traces
        if group['start'] < group['end']:
            visibility[group['start']] = True

        free_names = group['free_names']
        if plot_type in ('heatmap', 'contour'):
            layout_update = {
                'xaxis_title': free_names[1],
                'yaxis_title': free_names[0],
            }
        else:
            layout_update = {
                'scene.xaxis.title': free_names[1],
                'scene.yaxis.title': free_names[0],
                'scene.zaxis.title': metric.capitalize(),
            }

        dropdown_buttons.append(dict(
            label=f"Fix {param_names[fixed_idx]}",
            method='update',
            args=[
                {'visible': visibility},
                layout_update
            ]
        ))

    # Build slider for default (fixed_idx=0)
    default_group = slider_groups[0]
    slider_steps = []
    for i, val in enumerate(default_group['values']):
        vis = [False] * total_traces
        vis[default_group['start'] + i] = True
        slider_steps.append(dict(
            method='update',
            args=[{'visible': vis}],
            label=str(val),
        ))

    # Make first trace of first group visible by default
    if default_group['start'] < default_group['end']:
        fig.data[default_group['start']].visible = True

    if title is None:
        title = f"{metric.capitalize()} — 3D Parameter Analysis: {', '.join(param_names)}"

    scene_layout = dict(
        xaxis_title=default_group['free_names'][1],
        yaxis_title=default_group['free_names'][0],
        zaxis_title=metric.capitalize(),
        aspectmode='auto',
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=0.8)
        ),
    )

    layout_kwargs = dict(
        title=title,
        title_x=0.5,
        template='plotly_white',
        margin=dict(l=60, r=60, t=120, b=100),
        updatemenus=[dict(
            type='dropdown',
            direction='down',
            x=0.05,
            xanchor='left',
            y=1.12,
            yanchor='top',
            buttons=dropdown_buttons,
            active=0,
        )],
    )

    if plot_type not in ('heatmap', 'contour'):
        layout_kwargs['scene'] = scene_layout

    if slider_steps:
        layout_kwargs['sliders'] = [dict(
            active=0,
            currentvalue=dict(prefix=f"{param_names[0]}="),
            pad=dict(t=60),
            steps=slider_steps,
        )]

    fig.update_layout(**layout_kwargs)

    if show_plot:
        fig.show()

    return fig


def plot_4d_parameter_interactive(
    df: pd.DataFrame,
    param_names: List[str],
    metric: str = 'sortino',
    title: Optional[str] = None,
    show_plot: bool = True,
    plot_type: str = 'surface'
) -> go.Figure:
    """
    Create an interactive 4D parameter sensitivity visualization.

    Uses a dropdown to select which pair of parameters to fix (C(4,2)=6 options)
    and a slider to step through all value combinations of the two fixed params.
    Shows a 2D surface of the remaining two parameters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from analyze_nd_parameters with param1_value through param4_value
        columns and the metric column.
    param_names : List[str]
        List of 4 parameter names matching param1_value through param4_value.
    metric : str, default='sortino'
        Metric column to visualize.
    title : Optional[str], default=None
        Plot title. Auto-generated if None.
    show_plot : bool, default=True
        Whether to show the plot.
    plot_type : str, default='surface'
        Type of plot: 'surface', 'heatmap', 'contour'.

    Returns
    -------
    go.Figure
        Plotly figure with dropdown + slider controls.
    """
    if len(param_names) != 4:
        raise ValueError(f"Expected 4 parameter names, got {len(param_names)}")

    param_cols = ['param1_value', 'param2_value', 'param3_value', 'param4_value']

    # Build all C(4,2)=6 pair combinations for fixing
    fixed_pairs = list(combinations(range(4), 2))

    traces = []
    dropdown_buttons = []
    pair_groups = {}

    trace_idx = 0
    for pair_idx, (fi, fj) in enumerate(fixed_pairs):
        free_indices = [i for i in range(4) if i not in (fi, fj)]
        free_cols = [param_cols[i] for i in free_indices]
        fixed_cols = [param_cols[fi], param_cols[fj]]

        # All unique combinations of fixed param values
        fixed_combos = sorted(
            df.groupby(fixed_cols).size().index.tolist()
        )

        group_start = trace_idx

        for combo in fixed_combos:
            v_i, v_j = combo
            mask = (df[fixed_cols[0]] == v_i) & (df[fixed_cols[1]] == v_j)
            subset = df[mask]

            if subset.empty:
                traces.append(go.Surface(z=[[0]], x=[0], y=[0], visible=False, showscale=False))
                trace_idx += 1
                continue

            pivot = subset.pivot_table(
                index=free_cols[0], columns=free_cols[1], values=metric
            )
            pivot = pivot.sort_index(axis=0).sort_index(axis=1)

            if plot_type == 'heatmap':
                trace = go.Heatmap(
                    z=pivot.values,
                    x=[str(v) for v in pivot.columns.values],
                    y=[str(v) for v in pivot.index.values],
                    colorscale='Viridis',
                    visible=False,
                    showscale=True,
                    colorbar=dict(title=metric.capitalize()),
                )
            elif plot_type == 'contour':
                trace = go.Contour(
                    z=pivot.values,
                    x=[str(v) for v in pivot.columns.values],
                    y=[str(v) for v in pivot.index.values],
                    colorscale='Viridis',
                    visible=False,
                    showscale=True,
                    colorbar=dict(title=metric.capitalize()),
                )
            else:
                trace = go.Surface(
                    z=pivot.values,
                    x=pivot.columns.values,
                    y=pivot.index.values,
                    colorscale='Viridis',
                    connectgaps=True,
                    visible=False,
                    showscale=True,
                    colorbar=dict(title=metric.capitalize()),
                )

            traces.append(trace)
            trace_idx += 1

        group_end = trace_idx
        pair_groups[pair_idx] = {
            'start': group_start,
            'end': group_end,
            'combos': fixed_combos,
            'fixed_names': [param_names[fi], param_names[fj]],
            'free_names': [param_names[i] for i in free_indices],
        }

    fig = go.Figure(data=traces)
    total_traces = len(traces)

    # Build dropdown buttons for selecting which pair to fix
    for pair_idx, (fi, fj) in enumerate(fixed_pairs):
        group = pair_groups[pair_idx]
        visibility = [False] * total_traces
        if group['start'] < group['end']:
            visibility[group['start']] = True

        free_names = group['free_names']
        if plot_type in ('heatmap', 'contour'):
            layout_update = {
                'xaxis_title': free_names[1],
                'yaxis_title': free_names[0],
            }
        else:
            layout_update = {
                'scene.xaxis.title': free_names[1],
                'scene.yaxis.title': free_names[0],
                'scene.zaxis.title': metric.capitalize(),
            }

        fixed_names = group['fixed_names']
        dropdown_buttons.append(dict(
            label=f"Fix {fixed_names[0]} & {fixed_names[1]}",
            method='update',
            args=[
                {'visible': visibility},
                layout_update,
            ]
        ))

    # Default: first pair group
    default_group = pair_groups[0]
    slider_steps = []
    for i, combo in enumerate(default_group['combos']):
        vis = [False] * total_traces
        vis[default_group['start'] + i] = True
        label = f"{default_group['fixed_names'][0]}={combo[0]}, {default_group['fixed_names'][1]}={combo[1]}"
        slider_steps.append(dict(
            method='update',
            args=[{'visible': vis}],
            label=label,
        ))

    # Make first trace of first group visible by default
    if default_group['start'] < default_group['end']:
        fig.data[default_group['start']].visible = True

    if title is None:
        title = f"{metric.capitalize()} — 4D Parameter Analysis: {', '.join(param_names)}"

    scene_layout = dict(
        xaxis_title=default_group['free_names'][1],
        yaxis_title=default_group['free_names'][0],
        zaxis_title=metric.capitalize(),
        aspectmode='auto',
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=0.8)
        ),
    )

    layout_kwargs = dict(
        title=title,
        title_x=0.5,
        template='plotly_white',
        margin=dict(l=60, r=60, t=120, b=100),
        updatemenus=[dict(
            type='dropdown',
            direction='down',
            x=0.05,
            xanchor='left',
            y=1.12,
            yanchor='top',
            buttons=dropdown_buttons,
            active=0,
        )],
    )

    if plot_type not in ('heatmap', 'contour'):
        layout_kwargs['scene'] = scene_layout

    if slider_steps:
        fixed_label = f"{default_group['fixed_names'][0]} & {default_group['fixed_names'][1]}"
        layout_kwargs['sliders'] = [dict(
            active=0,
            currentvalue=dict(prefix=f"{fixed_label}: "),
            pad=dict(t=60),
            steps=slider_steps,
        )]

    fig.update_layout(**layout_kwargs)

    if show_plot:
        fig.show()

    return fig


# ---------------------------------------------------------------------------
# T011 — Stability-Aware Visualizations
# ---------------------------------------------------------------------------

def plot_parameter_sensitivity_with_stability(
    df: pd.DataFrame,
    param_name: str,
    metric: str = "sortino",
    stable_regions: Optional[List["StableRegion"]] = None,
    stability_threshold: float = 0.8,
    title: Optional[str] = None,
    show_plot: bool = True,
    phase3_selected_mask: Optional[Union[pd.Series, np.ndarray]] = None,
) -> go.Figure:
    """
    1D parameter sensitivity with raw + smoothed lines and shaded stable regions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``param1_value``, *metric*, ``smoothed_{metric}``,
        ``stability_ratio``, and optionally ``n_neighbors``.
    param_name : str
        Human-readable parameter name for axis labels.
    metric : str
        Raw metric column name.
    stable_regions : list of StableRegion, optional
        Regions to shade. When *None*, shading is derived from
        ``stability_ratio > stability_threshold``.
    stability_threshold : float
        Threshold for inline stability shading when *stable_regions* is None.
    title : str, optional
        Plot title.
    show_plot : bool
        Whether to call ``fig.show()``.
    phase3_selected_mask : pd.Series or np.ndarray, optional
        Boolean mask aligned with *df* (same index/length). When provided,
        points where True are drawn as "Phase 3/4 selected" markers.

    Returns
    -------
    go.Figure
    """
    smoothed_col = f"smoothed_{metric}"
    x_col = "param1_value"

    plot_df = df[[c for c in [x_col, metric, smoothed_col, "stability_ratio", "n_neighbors"]
                  if c in df.columns]].copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col, metric]).sort_values(x_col)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Raw metric line
    fig.add_trace(
        go.Scatter(
            x=plot_df[x_col], y=plot_df[metric],
            mode="lines+markers",
            name=f"{metric.capitalize()} (raw)",
            line=dict(color="#1f77b4"),
            marker=dict(size=8),
            hovertemplate=(
                f"<b>{param_name}</b>: %{{x}}<br>"
                f"<b>{metric.capitalize()} (raw)</b>: %{{y:.4f}}<br>"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    # Smoothed metric line
    if smoothed_col in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df[x_col], y=plot_df[smoothed_col],
                mode="lines+markers",
                name=f"{metric.capitalize()} (smoothed)",
                line=dict(color="#ff7f0e", dash="dash"),
                marker=dict(size=6, symbol="diamond"),
                hovertemplate=(
                    f"<b>{param_name}</b>: %{{x}}<br>"
                    f"<b>{metric.capitalize()} (smoothed)</b>: %{{y:.4f}}<br>"
                    "<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    # Stability ratio on secondary y-axis
    if "stability_ratio" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df[x_col], y=plot_df["stability_ratio"],
                mode="lines",
                name="Stability Ratio",
                line=dict(color="gray", dash="dot", width=1),
                opacity=0.5,
                hovertemplate=(
                    f"<b>{param_name}</b>: %{{x}}<br>"
                    "<b>Stability Ratio</b>: %{y:.2f}<br>"
                    "<extra></extra>"
                ),
            ),
            secondary_y=True,
        )

    # Shaded stable regions
    if stable_regions:
        for region in stable_regions:
            x_vals = [combo[0] for combo in region.param_combinations]
            x_lo, x_hi = min(x_vals), max(x_vals)
            fig.add_vrect(
                x0=x_lo, x1=x_hi,
                fillcolor="rgba(0,200,0,0.12)",
                layer="below",
                line_width=0,
                annotation_text="stable",
                annotation_position="top left",
                annotation_font_size=9,
                annotation_font_color="green",
            )
    else:
        # Derive contiguous stable runs from stability_ratio directly
        _add_inline_stable_shading(fig, plot_df, x_col, stability_threshold)

    # Phase 3/4 selected overlay
    if phase3_selected_mask is not None:
        if isinstance(phase3_selected_mask, np.ndarray):
            mask_series = pd.Series(phase3_selected_mask, index=df.index)
        else:
            mask_series = phase3_selected_mask
        sel = mask_series.reindex(plot_df.index).fillna(False)
        if sel.any():
            sub = plot_df.loc[sel]
            fig.add_trace(
                go.Scatter(
                    x=sub[x_col],
                    y=sub[smoothed_col] if smoothed_col in sub.columns else sub[metric],
                    mode="markers",
                    name="Phase 3/4 selected",
                    marker=dict(symbol="star", size=14, color="gold", line=dict(width=1, color="darkorange")),
                    hovertemplate=f"<b>{param_name}</b>: %{{x}}<br>Phase 3/4 selected<extra></extra>",
                ),
                secondary_y=False,
            )

    if title is None:
        title = f"Parameter Sensitivity with Stability: {param_name}"

    fig.update_layout(
        title=title,
        xaxis_title=param_name,
        yaxis_title=metric.capitalize(),
        yaxis2_title="Stability Ratio",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=100, b=50),
    )

    if show_plot:
        fig.show()
    return fig


def _add_inline_stable_shading(
    fig: go.Figure,
    plot_df: pd.DataFrame,
    x_col: str,
    threshold: float,
) -> None:
    """Add vrects for contiguous runs where stability_ratio > threshold."""
    if "stability_ratio" not in plot_df.columns:
        return

    x_vals = plot_df[x_col].values
    ratios = plot_df["stability_ratio"].values
    stable = ratios > threshold

    i = 0
    while i < len(stable):
        if stable[i]:
            start = i
            while i < len(stable) and stable[i]:
                i += 1
            end = i - 1
            if end > start:  # at least 2 contiguous points
                fig.add_vrect(
                    x0=float(x_vals[start]), x1=float(x_vals[end]),
                    fillcolor="rgba(0,200,0,0.12)",
                    layer="below",
                    line_width=0,
                )
        else:
            i += 1


def plot_2d_stability_heatmap(
    df: pd.DataFrame,
    param1: str,
    param2: str,
    metric: str = "sortino",
    stable_regions: Optional[List["StableRegion"]] = None,
    base_layers: Optional[List[str]] = None,
    default_layer: str = "smoothed",
    show_stability_contours: bool = False,
    show_region_markers: bool = True,
    enable_controls: bool = True,
    title: Optional[str] = None,
    show_plot: bool = True,
    phase3_selected_mask: Optional[Union[pd.Series, np.ndarray]] = None,
) -> go.Figure:
    """
    2D heatmap explorer with selectable base layers and optional overlays.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``param1_value``, ``param2_value``,
        ``smoothed_{metric}``, ``stability_ratio``.
    param1 : str
        Name for the y-axis parameter.
    param2 : str
        Name for the x-axis parameter.
    metric : str
        Raw metric column name.
    stable_regions : list of StableRegion, optional
        Regions whose boundaries are highlighted.
    base_layers : list of str, optional
        Base layer keys to include in the metric dropdown. Supported aliases:
        ``raw``, ``smoothed``, ``stability_ratio``, ``n_neighbors``, ``delta``.
        Any item that matches a DataFrame column is also supported.
    default_layer : str, default ``"smoothed"``
        Layer key shown on initial render.
    show_stability_contours : bool, default ``False``
        Initial visibility for stability contour overlay.
    show_region_markers : bool, default ``True``
        Initial visibility for stable-region marker overlay.
    enable_controls : bool, default ``True``
        Whether to add interactive dropdown/buttons for layer and overlays.
    title : str, optional
    show_plot : bool

    Returns
    -------
    go.Figure
    """
    smoothed_col = f"smoothed_{metric}"
    layer_alias_to_col = {
        "raw": metric,
        "smoothed": smoothed_col if smoothed_col in df.columns else metric,
        "stability_ratio": "stability_ratio",
        "n_neighbors": "n_neighbors",
    }

    if smoothed_col in df.columns and metric in df.columns:
        delta_col = f"delta_{metric}"
        working_df = df.copy()
        working_df[delta_col] = working_df[metric] - working_df[smoothed_col]
        layer_alias_to_col["delta"] = delta_col
    else:
        working_df = df

    if base_layers is None:
        base_layers = ["smoothed", "raw", "stability_ratio", "n_neighbors", "delta"]

    resolved_layers: List[tuple[str, str]] = []
    for layer_key in base_layers:
        if layer_key in layer_alias_to_col and layer_alias_to_col[layer_key] in working_df.columns:
            resolved_layers.append((layer_key, layer_alias_to_col[layer_key]))
        elif layer_key in working_df.columns:
            resolved_layers.append((layer_key, layer_key))

    if not resolved_layers:
        raise ValueError("No valid base layers available for 2D heatmap plotting.")

    if default_layer not in [k for k, _ in resolved_layers]:
        default_layer = resolved_layers[0][0]

    fig = go.Figure()
    base_trace_indices: dict[str, int] = {}

    def _layer_colorscale(layer_key: str) -> str:
        if layer_key == "stability_ratio":
            return "RdYlGn"
        if layer_key == "delta":
            return "RdBu"
        if layer_key == "n_neighbors":
            return "Cividis"
        return "Viridis"

    def _layer_title(layer_key: str) -> str:
        if layer_key == "raw":
            return f"{metric.capitalize()}<br>(raw)"
        if layer_key == "smoothed":
            return f"{metric.capitalize()}<br>(smoothed)"
        if layer_key == "stability_ratio":
            return "Stability<br>Ratio"
        if layer_key == "n_neighbors":
            return "Neighbor<br>Count"
        if layer_key == "delta":
            return f"{metric.capitalize()}<br>(raw-smoothed)"
        return layer_key

    # Base heatmap layers (one visible at a time)
    for layer_key, layer_col in resolved_layers:
        pivot = working_df.pivot_table(
            index="param1_value", columns="param2_value", values=layer_col,
        ).sort_index(axis=0).sort_index(axis=1)

        trace = go.Heatmap(
            z=pivot.values,
            x=[str(v) for v in pivot.columns],
            y=[str(v) for v in pivot.index],
            colorscale=_layer_colorscale(layer_key),
            colorbar=dict(title=_layer_title(layer_key), x=1.0),
            name=f"Layer: {layer_key}",
            visible=(layer_key == default_layer),
            hovertemplate=(
                f"<b>{param1}</b>: %{{y}}<br>"
                f"<b>{param2}</b>: %{{x}}<br>"
                f"<b>{layer_key}</b>: %{{z:.4f}}<br>"
                "<extra></extra>"
            ),
        )
        fig.add_trace(trace)
        base_trace_indices[layer_key] = len(fig.data) - 1

    contour_idx: Optional[int] = None
    region_indices: List[int] = []

    # Stability ratio contour overlay (optional visibility)
    if "stability_ratio" in working_df.columns:
        pivot_sr = working_df.pivot_table(
            index="param1_value", columns="param2_value", values="stability_ratio",
        ).sort_index(axis=0).sort_index(axis=1)

        fig.add_trace(go.Contour(
            z=pivot_sr.values,
            x=[str(v) for v in pivot_sr.columns],
            y=[str(v) for v in pivot_sr.index],
            colorscale="RdYlGn",
            opacity=0.30,
            showscale=False,
            contours=dict(showlabels=False),
            name="Overlay: stability contours",
            visible=show_stability_contours,
            hovertemplate=(
                f"<b>{param1}</b>: %{{y}}<br>"
                f"<b>{param2}</b>: %{{x}}<br>"
                "<b>Stability Ratio</b>: %{z:.2f}<br>"
                "<extra></extra>"
            ),
        ))
        contour_idx = len(fig.data) - 1

    # Stable region markers overlay
    if stable_regions:
        for region in stable_regions:
            xs = [str(combo[1]) for combo in region.param_combinations]
            ys = [str(combo[0]) for combo in region.param_combinations]
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="markers",
                marker=dict(
                    size=12,
                    color="rgba(0,0,0,0)",
                    line=dict(color="lime", width=2),
                    symbol="square",
                ),
                name=f"Overlay: stable region (n={region.n_combinations})",
                visible=show_region_markers,
                hovertemplate=(
                    f"<b>Stable Region</b><br>"
                    f"<b>Mean Obj</b>: {region.mean_objective:.4f}<br>"
                    f"<b>Stability</b>: {region.mean_stability_ratio:.2f}<br>"
                    "<extra></extra>"
                ),
            ))
            region_indices.append(len(fig.data) - 1)

    # Phase 3/4 selected overlay
    phase3_idx: Optional[int] = None
    if phase3_selected_mask is not None:
        if isinstance(phase3_selected_mask, np.ndarray):
            mask_series = pd.Series(phase3_selected_mask, index=df.index)
        else:
            mask_series = phase3_selected_mask
        sel = mask_series.reindex(working_df.index).fillna(False)
        if sel.any():
            sub = working_df.loc[sel]
            fig.add_trace(go.Scatter(
                x=[str(v) for v in sub["param2_value"]],
                y=[str(v) for v in sub["param1_value"]],
                mode="markers",
                name="Phase 3/4 selected",
                marker=dict(symbol="star", size=14, color="gold", line=dict(width=1.5, color="darkorange")),
                hovertemplate=f"<b>{param1}</b>: %{{y}}<br><b>{param2}</b>: %{{x}}<br>Phase 3/4 selected<extra></extra>",
            ))
            phase3_idx = len(fig.data) - 1

    if enable_controls:
        total = len(fig.data)

        def _visibility(selected_layer: str, contour_on: bool, regions_on: bool) -> List[bool]:
            vis = [False] * total
            vis[base_trace_indices[selected_layer]] = True
            if contour_idx is not None:
                vis[contour_idx] = contour_on
            if region_indices:
                for idx in region_indices:
                    vis[idx] = regions_on
            if phase3_idx is not None:
                vis[phase3_idx] = True
            return vis

        layer_buttons = []
        for layer_key, _ in resolved_layers:
            layer_buttons.append(dict(
                label=layer_key,
                method="update",
                args=[{"visible": _visibility(layer_key, show_stability_contours, show_region_markers)}],
            ))

        overlay_modes = [
            ("none", False, False),
            ("regions", False, True),
            ("contours", True, False),
            ("both", True, True),
        ]
        overlay_buttons = [
            dict(
                label=label,
                method="update",
                args=[{"visible": _visibility(default_layer, contour_on, regions_on)}],
            )
            for label, contour_on, regions_on in overlay_modes
        ]

        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    x=0.0,
                    xanchor="left",
                    y=1.18,
                    yanchor="top",
                    active=[k for k, _ in resolved_layers].index(default_layer),
                    buttons=layer_buttons,
                ),
                dict(
                    type="buttons",
                    direction="right",
                    x=0.36,
                    xanchor="left",
                    y=1.18,
                    yanchor="top",
                    active=3 if (show_stability_contours and show_region_markers)
                    else 2 if show_stability_contours
                    else 1 if show_region_markers
                    else 0,
                    buttons=overlay_buttons,
                ),
            ]
        )

    if title is None:
        title = f"2D Stability Heatmap: {param1} vs {param2} ({metric.capitalize()})"

    fig.update_layout(
        title=title,
        xaxis_title=param2,
        yaxis_title=param1,
        template="plotly_white",
        margin=dict(l=60, r=120, t=80, b=60),
    )

    if show_plot:
        fig.show()
    return fig


def plot_3d_slices(
    df: pd.DataFrame,
    param_names: List[str],
    metric: str = "sortino",
    plot_type: str = "heatmap",
    title: Optional[str] = None,
    show_plot: bool = True,
) -> go.Figure:
    """
    3D+ parameter sensitivity via 2D slices with dropdown + slider.

    For each choice of a fixed parameter, show a 2D slice (heatmap/contour)
    or a true 3D surface of the smoothed metric across the remaining two
    parameters. A dropdown selects which
    parameter to fix; a slider steps through its values.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``param1_value`` … ``paramN_value`` (N ≥ 3) and
        ``smoothed_{metric}`` (or falls back to *metric*).
    param_names : list of str
        Human-readable names for each dimension.
    metric : str
        Raw metric column name.
    plot_type : str
        Slice visualization type: ``"heatmap"`` (default), ``"surface"``,
        or ``"contour"``.
    title : str, optional
    show_plot : bool

    Returns
    -------
    go.Figure
    """
    n_params = len(param_names)
    if n_params < 3:
        raise ValueError(f"plot_3d_slices requires >= 3 params, got {n_params}")
    if plot_type not in {"heatmap", "surface", "contour"}:
        raise ValueError(f"Invalid plot_type: {plot_type}")

    param_cols = [f"param{k}_value" for k in range(1, n_params + 1)]
    smoothed_col = f"smoothed_{metric}"
    obj_col = smoothed_col if smoothed_col in df.columns else metric

    traces: list = []
    dropdown_buttons: list = []
    slider_groups: dict = {}

    trace_idx = 0
    for fixed_idx in range(n_params):
        fixed_col = param_cols[fixed_idx]
        free_indices = [i for i in range(n_params) if i != fixed_idx]
        # Pick first two free dims for the 2D slice
        free_cols = [param_cols[free_indices[0]], param_cols[free_indices[1]]]
        free_names = [param_names[free_indices[0]], param_names[free_indices[1]]]

        fixed_values = sorted(df[fixed_col].unique())
        group_start = trace_idx

        for fixed_val in fixed_values:
            subset = df[df[fixed_col] == fixed_val]

            # If there are more free dims (>2 remaining), aggregate over them
            if len(free_indices) > 2:
                subset = subset.groupby(free_cols, as_index=False)[obj_col].mean()

            if subset.empty or subset[free_cols[0]].nunique() < 2 or subset[free_cols[1]].nunique() < 2:
                if plot_type == "surface":
                    traces.append(go.Surface(z=[[0]], x=[0], y=[0], visible=False, showscale=False))
                elif plot_type == "contour":
                    traces.append(go.Contour(z=[[0]], x=["0"], y=["0"], visible=False, showscale=False))
                else:
                    traces.append(go.Heatmap(z=[[0]], x=["0"], y=["0"], visible=False, showscale=False))
                trace_idx += 1
                continue

            pivot = subset.pivot_table(
                index=free_cols[0], columns=free_cols[1], values=obj_col,
            ).sort_index(axis=0).sort_index(axis=1)

            if plot_type == "surface":
                trace = go.Surface(
                    z=pivot.values,
                    x=pivot.columns.values,
                    y=pivot.index.values,
                    colorscale="Viridis",
                    visible=False,
                    showscale=True,
                    colorbar=dict(title=metric.capitalize()),
                    hovertemplate=(
                        f"<b>{free_names[0]}</b>: %{{y}}<br>"
                        f"<b>{free_names[1]}</b>: %{{x}}<br>"
                        f"<b>{metric}</b>: %{{z:.4f}}<br>"
                        f"<b>{param_names[fixed_idx]}</b>={fixed_val}<br>"
                        "<extra></extra>"
                    ),
                )
            elif plot_type == "contour":
                trace = go.Contour(
                    z=pivot.values,
                    x=[str(v) for v in pivot.columns],
                    y=[str(v) for v in pivot.index],
                    colorscale="Viridis",
                    visible=False,
                    showscale=True,
                    colorbar=dict(title=metric.capitalize()),
                    hovertemplate=(
                        f"<b>{free_names[0]}</b>: %{{y}}<br>"
                        f"<b>{free_names[1]}</b>: %{{x}}<br>"
                        f"<b>{metric}</b>: %{{z:.4f}}<br>"
                        f"<b>{param_names[fixed_idx]}</b>={fixed_val}<br>"
                        "<extra></extra>"
                    ),
                )
            else:
                trace = go.Heatmap(
                    z=pivot.values,
                    x=[str(v) for v in pivot.columns],
                    y=[str(v) for v in pivot.index],
                    colorscale="Viridis",
                    visible=False,
                    showscale=True,
                    colorbar=dict(title=metric.capitalize()),
                    hovertemplate=(
                        f"<b>{free_names[0]}</b>: %{{y}}<br>"
                        f"<b>{free_names[1]}</b>: %{{x}}<br>"
                        f"<b>{metric}</b>: %{{z:.4f}}<br>"
                        f"<b>{param_names[fixed_idx]}</b>={fixed_val}<br>"
                        "<extra></extra>"
                    ),
                )
            traces.append(trace)
            trace_idx += 1

        group_end = trace_idx
        slider_groups[fixed_idx] = {
            "start": group_start,
            "end": group_end,
            "values": fixed_values,
            "free_names": free_names,
        }

    fig = go.Figure(data=traces)
    total_traces = len(traces)

    # Dropdown buttons
    def _build_slider_for_group(fixed_idx: int) -> List[dict]:
        group = slider_groups[fixed_idx]
        steps: List[dict] = []
        for i, val in enumerate(group["values"]):
            vis = [False] * total_traces
            vis[group["start"] + i] = True
            steps.append(dict(
                method="update",
                args=[{"visible": vis}],
                label=str(val),
            ))
        return [dict(
            active=0,
            currentvalue=dict(prefix=f"{param_names[fixed_idx]}="),
            pad=dict(t=60),
            steps=steps,
        )]

    for fixed_idx in range(n_params):
        group = slider_groups[fixed_idx]
        vis = [False] * total_traces
        if group["start"] < group["end"]:
            vis[group["start"]] = True

        if plot_type == "surface":
            axis_update = {
                "scene.xaxis.title": group["free_names"][1],
                "scene.yaxis.title": group["free_names"][0],
                "scene.zaxis.title": metric.capitalize(),
            }
        else:
            axis_update = {
                "xaxis_title": group["free_names"][1],
                "yaxis_title": group["free_names"][0],
            }

        dropdown_buttons.append(dict(
            label=f"Fix {param_names[fixed_idx]}",
            method="update",
            args=[
                {"visible": vis},
                {**axis_update, "sliders": _build_slider_for_group(fixed_idx)},
            ],
        ))

    # Slider for default group (fixed_idx=0)
    default_group = slider_groups[0]
    slider_steps = _build_slider_for_group(0)[0]["steps"]

    if default_group["start"] < default_group["end"]:
        fig.data[default_group["start"]].visible = True

    if title is None:
        title = f"{metric.capitalize()} — Multi-Parameter Slices: {', '.join(param_names)}"

    layout_kwargs: dict = dict(
        title=title,
        title_x=0.5,
        template="plotly_white",
        margin=dict(l=60, r=60, t=120, b=100),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.05, xanchor="left",
            y=1.12, yanchor="top",
            buttons=dropdown_buttons,
            active=0,
        )],
    )
    if plot_type == "surface":
        layout_kwargs["scene"] = dict(
            xaxis_title=default_group["free_names"][1],
            yaxis_title=default_group["free_names"][0],
            zaxis_title=metric.capitalize(),
            aspectmode="auto",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=0.8),
            ),
        )
    else:
        layout_kwargs["xaxis_title"] = default_group["free_names"][1]
        layout_kwargs["yaxis_title"] = default_group["free_names"][0]

    if slider_steps:
        layout_kwargs["sliders"] = [dict(
            active=0,
            currentvalue=dict(prefix=f"{param_names[0]}="),
            pad=dict(t=60),
            steps=slider_steps,
        )]

    fig.update_layout(**layout_kwargs)

    if show_plot:
        fig.show()
    return fig
