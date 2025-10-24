"""PCA Analysis Utility

This module provides functions for performing Principal Component Analysis (PCA)
on feature DataFrames and creating comprehensive visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, List


def perform_pca_analysis(
    df: pd.DataFrame,
    n_components: Optional[int] = None,
    exclude_columns: Optional[List[str]] = None,
    standardize: bool = True,
    plot: bool = True,
    figsize: Tuple[int, int] = (16, 12)
) -> Tuple[PCA, np.ndarray, pd.DataFrame]:
    """
    Perform PCA on a DataFrame and optionally create comprehensive visualizations.
    
    This function:
    1. Standardizes the data (optional)
    2. Performs PCA
    3. Creates 4 plots: scree plot, cumulative variance, 2D scatter, loadings heatmap
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with features as columns
    n_components : int, optional
        Number of principal components to compute. If None, uses min(n_samples, n_features)
    exclude_columns : list of str, optional
        Column names to exclude from PCA (e.g., ['ticker', 'target', 'datetime'])
    standardize : bool, default=True
        Whether to standardize features before PCA (recommended)
    plot : bool, default=True
        Whether to create visualization plots
    figsize : tuple, default=(16, 12)
        Figure size for the plot grid
        
    Returns
    -------
    pca : PCA
        Fitted PCA object
    transformed : np.ndarray
        Transformed data in principal component space
    loadings_df : pd.DataFrame
        DataFrame showing feature loadings on each principal component
        
    Examples
    --------
    >>> # Basic usage
    >>> pca, transformed, loadings = perform_pca_analysis(features_df)
    >>> 
    >>> # Exclude non-feature columns
    >>> pca, transformed, loadings = perform_pca_analysis(
    ...     features_df,
    ...     exclude_columns=['ticker', 'datetime'],
    ...     n_components=10
    ... )
    >>> 
    >>> # Access results
    >>> print(f"Explained variance: {pca.explained_variance_ratio_[:5]}")
    >>> print(f"Top loadings on PC1:\\n{loadings['PC1'].abs().sort_values(ascending=False).head()}")
    """
    # Prepare data
    df_clean = df.copy()
    
    # Exclude specified columns
    if exclude_columns:
        feature_cols = [col for col in df_clean.columns if col not in exclude_columns]
    else:
        feature_cols = df_clean.columns.tolist()
    
    # Extract feature data and handle missing values
    X = df_clean[feature_cols].fillna(0)
    
    print(f"PCA Analysis")
    print(f"{'='*60}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    
    # Standardize features
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"Features standardized: Yes")
    else:
        X_scaled = X.values
        print(f"Features standardized: No")
    
    # Determine number of components
    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(X_scaled)
    
    print(f"Number of components: {n_components}")
    print(f"Total variance explained by {n_components} components: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"\nTop 5 components explain: {pca.explained_variance_ratio_[:5].sum():.2%}")
    
    # Create loadings DataFrame
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_cols
    )
    
    # Create visualizations
    if plot:
        _plot_pca_results(pca, transformed, loadings_df, df_clean, figsize)
    
    return pca, transformed, loadings_df


def _plot_pca_results(
    pca: PCA,
    transformed: np.ndarray,
    loadings_df: pd.DataFrame,
    original_df: pd.DataFrame,
    figsize: Tuple[int, int]
):
    """Create comprehensive PCA visualization plots."""
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Scree Plot
    ax1 = fig.add_subplot(gs[0, 0])
    n_components = len(pca.explained_variance_ratio_)
    components = np.arange(1, min(21, n_components + 1))  # Show first 20 components
    
    ax1.bar(components, pca.explained_variance_ratio_[:len(components)], alpha=0.7, color='steelblue')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Scree Plot - Variance Explained by Each Component', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for i, v in enumerate(pca.explained_variance_ratio_[:len(components)]):
        ax1.text(i + 1, v, f'{v:.1%}', ha='center', va='bottom', fontsize=8)
    
    # 2. Cumulative Variance Plot
    ax2 = fig.add_subplot(gs[0, 1])
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    ax2.plot(components, cumsum[:len(components)], 'o-', linewidth=2, markersize=6, color='darkgreen')
    ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% variance')
    ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% variance')
    ax2.axhline(y=0.95, color='purple', linestyle='--', alpha=0.7, label='95% variance')
    
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Find components needed for 80%, 90%, 95%
    n_80 = np.argmax(cumsum >= 0.8) + 1
    n_90 = np.argmax(cumsum >= 0.9) + 1
    n_95 = np.argmax(cumsum >= 0.95) + 1
    
    ax2.text(0.02, 0.98, 
             f'Components for:\n80%: {n_80}\n90%: {n_90}\n95%: {n_95}',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. 2D Scatter Plot (PC1 vs PC2)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Color by ticker if available
    if 'ticker' in original_df.columns:
        tickers = original_df['ticker'].values
        unique_tickers = np.unique(tickers)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_tickers)))
        
        for i, ticker in enumerate(unique_tickers):
            mask = tickers == ticker
            ax3.scatter(transformed[mask, 0], transformed[mask, 1], 
                       alpha=0.6, s=20, label=ticker, color=colors[i])
        ax3.legend(title='Ticker', loc='best', fontsize=9)
    else:
        ax3.scatter(transformed[:, 0], transformed[:, 1], alpha=0.6, s=20, color='steelblue')
    
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax3.set_title('First Two Principal Components', fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # 4. Loadings Heatmap (top features on first 5 PCs)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Select top features by absolute loading on PC1
    n_top_features = min(15, len(loadings_df))
    n_pcs_to_show = min(5, loadings_df.shape[1])
    
    top_features = loadings_df['PC1'].abs().nlargest(n_top_features).index
    loadings_subset = loadings_df.loc[top_features, [f'PC{i+1}' for i in range(n_pcs_to_show)]]
    
    sns.heatmap(loadings_subset, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Loading'}, ax=ax4, linewidths=0.5)
    ax4.set_title(f'Feature Loadings on First {n_pcs_to_show} PCs\n(Top {n_top_features} features by PC1)', 
                  fontsize=13, fontweight='bold')
    ax4.set_xlabel('Principal Component', fontsize=12)
    ax4.set_ylabel('Feature', fontsize=12)
    
    plt.suptitle('PCA Analysis Results', fontsize=16, fontweight='bold', y=0.995)
    
    return fig


def get_top_features_per_component(
    loadings_df: pd.DataFrame,
    n_components: int = 5,
    n_features: int = 10
) -> pd.DataFrame:
    """
    Get the top features contributing to each principal component.
    
    Parameters
    ----------
    loadings_df : pd.DataFrame
        DataFrame of feature loadings from perform_pca_analysis
    n_components : int, default=5
        Number of principal components to analyze
    n_features : int, default=10
        Number of top features to return per component
        
    Returns
    -------
    pd.DataFrame
        DataFrame with top features and their loadings for each component
    """
    results = {}
    
    for i in range(min(n_components, loadings_df.shape[1])):
        pc_name = f'PC{i+1}'
        top_features = loadings_df[pc_name].abs().nlargest(n_features)
        
        results[pc_name] = pd.DataFrame({
            'feature': top_features.index,
            'loading': loadings_df.loc[top_features.index, pc_name].values,
            'abs_loading': top_features.values
        })
    
    return results


def reconstruct_from_pca(
    pca: PCA,
    transformed: np.ndarray,
    n_components: int,
    feature_names: List[str],
    scaler: Optional[StandardScaler] = None
) -> pd.DataFrame:
    """
    Reconstruct original features using only the first n principal components.
    
    Useful for dimensionality reduction and noise filtering.
    
    Parameters
    ----------
    pca : PCA
        Fitted PCA object
    transformed : np.ndarray
        Transformed data in PC space
    n_components : int
        Number of components to use for reconstruction
    feature_names : list of str
        Original feature names
    scaler : StandardScaler, optional
        Scaler used before PCA (for inverse transform)
        
    Returns
    -------
    pd.DataFrame
        Reconstructed features
    """
    # Use only first n components
    transformed_reduced = transformed[:, :n_components]
    
    # Reconstruct in original space
    reconstructed = transformed_reduced @ pca.components_[:n_components, :]
    
    # Inverse standardization if scaler was used
    if scaler is not None:
        reconstructed = scaler.inverse_transform(reconstructed)
    
    return pd.DataFrame(reconstructed, columns=feature_names)
