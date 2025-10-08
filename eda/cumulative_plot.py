import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple


def plot_cumulative_product(
    df: pd.DataFrame,
    target_col: str = "target",
    features: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    normalize: bool = False,
    sort_by_date: bool = True,
    top_n: int = 50
) -> Dict[str, plt.Figure]:
    """
    Compute the product between the target and each feature, then plot the cumulative sum.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features and target
    target_col : str, default="target"
        Name of the target column
    features : Optional[List[str]], default=None
        List of feature columns to plot. If None, all numeric columns except target_col are used
    figsize : Tuple[int, int], default=(12, 8)
        Figure size for the plot
    normalize : bool, default=True
        Whether to normalize the cumulative sum to start at 0 and end at 1 (or -1)
    sort_by_date : bool, default=True
        Whether to sort the dataframe by index (assumed to be datetime) before computing cumulative sum
    top_n : int, default=10
        Number of top features to plot. If 0, plot all features
        
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary of figure types to matplotlib Figure objects
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
        raise ValueError("No valid numeric features found for plotting")
    
    # Create a clean subset with no NaNs
    sub_df = df[[target_col] + valid_features].dropna().copy()
    if sub_df.empty:
        raise ValueError("No valid data after dropping NaNs")
    
    # Sort by date if requested
    if sort_by_date and isinstance(sub_df.index, pd.DatetimeIndex):
        sub_df = sub_df.sort_index()
    
    # Compute product and cumulative sum for each feature
    cum_products = {}
    final_values = {}
    
    for feature in valid_features:
        # Compute product between target and feature
        product = sub_df[target_col] * sub_df[feature]
        
        # Compute cumulative sum
        cum_sum = product.cumsum()
        
        # Normalize if requested
        if normalize and len(cum_sum) > 0:
            max_abs = abs(cum_sum.iloc[-1])
            if max_abs > 0:
                cum_sum = cum_sum / max_abs
        
        cum_products[feature] = cum_sum
        final_values[feature] = cum_sum.iloc[-1] if len(cum_sum) > 0 else 0
    
    # Sort features by final cumulative sum value (absolute)
    sorted_features = sorted(valid_features, key=lambda x: abs(final_values[x]), reverse=True)
    
    # Limit to top N features if specified
    if top_n > 0 and len(sorted_features) > top_n:
        plot_features = sorted_features[:top_n]
        print(f"Limiting to top {top_n} features by absolute cumulative product")
    else:
        plot_features = sorted_features
    
    figures = {}
    
    # Create individual plots for each feature
    for feature in plot_features:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        cum_sum = cum_products[feature]
        final_value = final_values[feature]
        
        # Determine color based on final value
        color = '#2ecc71' if final_value >= 0 else '#e74c3c'  # Green if positive, red if negative
        
        # Plot cumulative sum
        ax.plot(cum_sum.index, cum_sum, color=color, linewidth=2)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative Product')
        ax.set_title(f"{feature} vs {target_col} - Cumulative Product\nFinal Value: {final_value:.3f}")
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Store the figure
        figures[feature] = fig
        fig.tight_layout()
    
    # Create a combined plot with all features
    if len(plot_features) > 1:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        for feature in plot_features:
            cum_sum = cum_products[feature]
            final_value = final_values[feature]
            
            # Determine color based on final value
            color = '#2ecc71' if final_value >= 0 else '#e74c3c'  # Green if positive, red if negative
            
            # Plot cumulative sum
            ax.plot(cum_sum.index, cum_sum, label=f"{feature} ({final_value:.3f})")
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative Product')
        ax.set_title(f"Feature vs {target_col} - Cumulative Products")
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        # Store the figure
        figures['combined'] = fig
        fig.tight_layout()
    
    return figures
