"""
Pure plotting functions for parameter sensitivity analysis.

This module provides pure functions for visualizing parameter sensitivity results.
All functions are stateless and follow functional programming principles.

Author: Trading Research Team
Date: 2025-01-XX
"""

from typing import Optional, List
from itertools import combinations
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

