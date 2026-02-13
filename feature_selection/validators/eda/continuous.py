"""Continuous feature EDA methods."""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Literal

from feature_selection.validators.reports.base import MonotonicityTestResult


def compute_decile_stats(
    feature: pd.Series,
    target: pd.Series,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Compute per-decile target statistics.

    Args:
        feature: Feature series
        target: Target series
        n_deciles: Number of deciles (default 10)

    Returns:
        DataFrame with decile statistics (mean, std, Sharpe, t-stat per decile)
    """
    # Align and drop NaN
    aligned = pd.DataFrame({'feature': feature, 'target': target}).dropna()

    if len(aligned) < n_deciles * 5:
        raise ValueError(f"Insufficient data for decile analysis: {len(aligned)}")

    # Assign deciles
    aligned['decile'] = pd.qcut(aligned['feature'], q=n_deciles, labels=False, duplicates='drop')

    # Compute per-decile stats
    decile_stats = []

    for decile in sorted(aligned['decile'].unique()):
        decile_data = aligned[aligned['decile'] == decile]['target']

        mean_target = decile_data.mean()
        std_target = decile_data.std()
        n_samples = len(decile_data)

        # Sharpe ratio
        sharpe = mean_target / std_target if std_target > 0 else 0.0

        # t-statistic
        t_stat = mean_target / (std_target / np.sqrt(n_samples)) if std_target > 0 else 0.0

        decile_stats.append({
            'decile': decile,
            'mean_target': mean_target,
            'std_target': std_target,
            'sharpe': sharpe,
            't_stat': t_stat,
            'n_samples': n_samples,
        })

    return pd.DataFrame(decile_stats)


def check_monotonicity(decile_means: pd.Series) -> MonotonicityTestResult:
    """
    Check if decile means show monotonic trend.

    Args:
        decile_means: Series of mean target values per decile

    Returns:
        MonotonicityTestResult
    """
    # Kendall's tau correlation with decile index
    decile_index = np.arange(len(decile_means))
    tau, p_value = stats.kendalltau(decile_index, decile_means.values)

    # Determine monotonicity and direction
    is_monotonic = bool(abs(tau) > 0.3 and p_value < 0.05)

    if is_monotonic:
        direction = 'increasing' if tau > 0 else 'decreasing'
    else:
        direction = 'none'

    return MonotonicityTestResult(
        kendall_tau=float(tau),
        p_value=float(p_value),
        is_monotonic=is_monotonic,
        direction=direction,
    )


def detect_outliers(
    data: pd.Series,
    method: Literal['iqr', 'zscore'] = 'iqr',
    threshold: float = 3.0,
) -> tuple[pd.Series, float]:
    """
    Detect outliers using IQR or Z-score method.

    Args:
        data: Numeric series
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection (IQR multiplier or Z-score)

    Returns:
        Tuple of (outlier_mask, outlier_fraction)
    """
    if method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        outlier_mask = (data < lower) | (data > upper)
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        outlier_mask = pd.Series(z_scores > threshold, index=data.index)
    else:
        raise ValueError(f"Unknown method: {method}")

    outlier_fraction = outlier_mask.sum() / len(data)

    return outlier_mask, outlier_fraction


def fit_polynomial_regression(
    feature: pd.Series,
    target: pd.Series,
    max_degree: int = 3,
) -> dict[int, float]:
    """
    Fit polynomial regressions of varying degrees.

    Args:
        feature: Feature series
        target: Target series
        max_degree: Maximum polynomial degree

    Returns:
        Dictionary mapping degree → R²
    """
    # Align and drop NaN
    aligned = pd.DataFrame({'feature': feature, 'target': target}).dropna()

    X = aligned['feature'].values.reshape(-1, 1)
    y = aligned['target'].values

    r2_scores = {}

    for degree in range(1, max_degree + 1):
        # Create polynomial features
        X_poly = np.column_stack([X**d for d in range(1, degree + 1)])

        # Fit linear regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_poly, y)

        # Compute R²
        r2 = model.score(X_poly, y)
        r2_scores[degree] = r2

    return r2_scores
