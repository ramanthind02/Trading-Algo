import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Union, Tuple, Dict
import scipy.stats as stats


def plot_feature_correlations(
    df: pd.DataFrame, 
    target_col: str = "target", 
    features: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    max_plots: int = 100,
    alpha: float = 0.3,
    add_trend_line: bool = True,
    add_correlation: bool = True
) -> Dict[str, plt.Figure]:
    """
    Compute and plot correlations between features and a target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features and target
    target_col : str, default="target"
        Name of the target column
    features : Optional[List[str]], default=None
        List of feature columns to plot. If None, all numeric columns except target_col are used
    figsize : Tuple[int, int], default=(10, 6)
        Figure size for the plot
    max_plots : int, default=100
        Maximum number of subplots to show
    alpha : float, default=0.3
        Transparency of scatter points
    add_trend_line : bool, default=True
        Whether to add a trend line to each scatter plot
    add_correlation : bool, default=True
        Whether to add correlation coefficient to plot titles

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary of feature name to matplotlib Figure objects
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    # Select features (exclude non-numeric and string columns)
    if features is None:
        features = [col for col in df.columns
                    if col != target_col
                    and pd.api.types.is_numeric_dtype(df[col])
                    and not pd.api.types.is_bool_dtype(df[col])
                    and not pd.api.types.is_string_dtype(df[col])]

    # Filter out any specified features that don't meet criteria
    valid_features = []
    for col in features:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            valid_features.append(col)
        else:
            print(f"Skipping column '{col}': not numeric or is binary/string type")

    # Limit number of plots
    if len(valid_features) > max_plots:
        print(f"Limiting to {max_plots} features with highest absolute correlation")
        corrs = df[valid_features + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
        valid_features = corrs.index[:max_plots].tolist()
        if target_col in valid_features:
            valid_features.remove(target_col)

    if not valid_features:
        raise ValueError("No valid numeric features found for correlation plotting")

    plot_df = df

    figures = {}

    for feature in valid_features:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # Calculate correlations - use both Pearson and Spearman
        # Only use rows where both feature and target_col are not NaN
        sub_df = plot_df[[feature, target_col]].dropna()
        if sub_df.empty:
            print(f"Skipping '{feature}': no valid data after dropping NaNs")
            continue

        spearman_corr, spearman_p = stats.spearmanr(sub_df[feature], sub_df[target_col])
        pearson_corr, pearson_p = stats.pearsonr(sub_df[feature], sub_df[target_col])

        corr = spearman_corr
        p_value = spearman_p

        sns.scatterplot(x=feature, y=target_col, data=sub_df, alpha=alpha, ax=ax)

        if add_trend_line:
            sns.regplot(x=feature, y=target_col, data=sub_df,
                        scatter=False, ci=None, line_kws={'color': 'red'}, ax=ax)

        if add_correlation:
            title = f"{feature} vs {target_col}\nSpearman: {corr:.3f}, Pearson: {pearson_corr:.3f}"
            if p_value < 0.05:
                title += f" (p={p_value:.3e})*"
            else:
                title += f" (p={p_value:.3e})"
            ax.set_title(title)
        else:
            ax.set_title(f"{feature} vs {target_col}")

        figures[feature] = fig
        fig.tight_layout()

    return figures


def plot_feature_deciles(
    df: pd.DataFrame,
    target_col: str = "target",
    features: Optional[List[str]] = None,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (12, 8),
    plot_type: str = "bar",
    include_stats: bool = True
) -> Dict[str, plt.Figure]:
    """
    Create decile plots showing how the target variable changes across different buckets of feature values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features and target
    target_col : str, default="target"
        Name of the target column
    features : Optional[List[str]], default=None
        List of feature columns to plot. If None, all numeric columns except target_col are used
    n_bins : int, default=10
        Number of bins/buckets to create (10 = deciles, 5 = quintiles, etc.)
    figsize : Tuple[int, int], default=(12, 8)
        Figure size for the plot
    plot_type : str, default="bar"
        Type of plot to create: "bar" or "line"
    include_stats : bool, default=True
        Whether to include statistics in the plot title
        
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary of feature name to matplotlib Figure objects
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Select features (exclude non-numeric and string columns)
    if features is None:
        features = [col for col in df.columns
                    if col != target_col
                    and pd.api.types.is_numeric_dtype(df[col])
                    and not pd.api.types.is_bool_dtype(df[col])
                    and not pd.api.types.is_string_dtype(df[col])]
    
    # Filter out any specified features that don't meet criteria
    valid_features = []
    for col in features:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            valid_features.append(col)
        else:
            print(f"Skipping column '{col}': not numeric or is binary/string type")
    
    if not valid_features:
        raise ValueError("No valid numeric features found for decile plotting")
    
    figures = {}
    
    for feature in valid_features:
        # Create a clean subset with no NaNs
        sub_df = df[[feature, target_col]].dropna().copy()
        if sub_df.empty:
            print(f"Skipping '{feature}': no valid data after dropping NaNs")
            continue
        
        # Create bins/buckets based on feature quantiles
        sub_df['bin'] = pd.qcut(sub_df[feature], n_bins, labels=False, duplicates='drop')
        
        # If we couldn't create the requested number of bins due to duplicates
        actual_bins = sub_df['bin'].nunique()
        if actual_bins < n_bins:
            print(f"Warning: Could only create {actual_bins} bins for '{feature}' due to duplicate values")
        
        # Calculate statistics for each bin
        bin_stats = sub_df.groupby('bin')[target_col].agg(['mean', 'std', 'count'])
        bin_stats['bin_label'] = [f"{i+1}" for i in range(len(bin_stats))]
        
        # Calculate feature values at bin edges for reference
        bin_edges = sub_df.groupby('bin')[feature].agg(['min', 'max'])
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # Plot the mean target value for each bin with color based on value
        if plot_type == "bar":
            # Determine bar colors based on value (green for positive, red for negative)
            bar_colors = ['#2ecc71' if val >= 0 else '#e74c3c' for val in bin_stats['mean']]
            
            # Plot bars without error bars (whiskers)
            ax.bar(bin_stats.index + 1, bin_stats['mean'], color=bar_colors)
        else:  # line plot
            # For line plot, use points with different colors based on value
            for i, val in enumerate(bin_stats['mean']):
                color = '#2ecc71' if val >= 0 else '#e74c3c'
                ax.plot(i+1, val, 'o', color=color, markersize=8)
            
            # Add connecting line
            ax.plot(bin_stats.index + 1, bin_stats['mean'], '-', color='#555555', alpha=0.5)
        
        # Calculate trend statistics
        trend_corr, trend_p = stats.spearmanr(bin_stats.index, bin_stats['mean'])
        monotonicity = np.abs(trend_corr)
        
        # Set labels and title
        ax.set_xlabel(f"{feature}_decile")
        ax.set_ylabel(f"mean_{target_col}")
        
        if include_stats:
            title = f"{feature} Decile Plot\n"
            title += f"Trend Corr: {trend_corr:.3f} (p={trend_p:.3e})"
            if trend_p < 0.05:
                title += "*"
            ax.set_title(title)
        else:
            ax.set_title(f"{feature} Decile Plot")
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Store the figure
        figures[feature] = fig
        fig.tight_layout()
        
        # Add a second axis with feature value ranges
        ax2 = ax.twiny()
        
        # Get the actual tick positions from the first axis
        tick_positions = ax.get_xticks()
        
        # Ensure we only use valid tick positions (within the data range)
        valid_positions = [pos for pos in tick_positions if 1 <= pos <= len(bin_stats)]
        ax2.set_xticks(valid_positions)
        
        # Create labels showing the range of feature values in each bin
        if len(bin_edges) > 0:  # Only if we have bin edges
            # Create a mapping from tick position to bin index
            tick_to_bin = {int(pos): int(pos-1) for pos in valid_positions if 1 <= pos <= len(bin_edges)}
            
            # Generate labels for each valid tick position
            range_labels = []
            for pos in valid_positions:
                if int(pos) in tick_to_bin and tick_to_bin[int(pos)] < len(bin_edges):
                    bin_idx = tick_to_bin[int(pos)]
                    min_val = bin_edges['min'].iloc[bin_idx]
                    max_val = bin_edges['max'].iloc[bin_idx]
                    
                    # Format the values based on their magnitude
                    if abs(min_val) < 0.01 or abs(min_val) > 1000:
                        min_str = f"{min_val:.2e}"
                    else:
                        min_str = f"{min_val:.3f}"
                        
                    if abs(max_val) < 0.01 or abs(max_val) > 1000:
                        max_str = f"{max_val:.2e}"
                    else:
                        max_str = f"{max_val:.3f}"
                        
                    range_labels.append(f"[{min_str}, {max_str}]")
                else:
                    range_labels.append("")
            
            # Set the labels
            ax2.set_xticklabels(range_labels)
        
        ax2.set_xlabel(f"{feature} value ranges")
        fig.tight_layout()
    
    return figures
