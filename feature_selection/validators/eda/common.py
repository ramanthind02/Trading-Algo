"""Common EDA methods for both feature types."""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss

from feature_selection.validators.reports.base import (
    DescriptiveStats,
    ADFTestResult,
    KPSSTestResult,
)


def compute_distribution_stats(data: pd.Series) -> DescriptiveStats:
    """
    Compute descriptive statistics for a numeric series.

    Args:
        data: Numeric series

    Returns:
        DescriptiveStats dataclass
    """
    return DescriptiveStats.from_series(data)


def compute_correlations(
    feature: pd.Series,
    target: pd.Series,
) -> dict[str, float]:
    """
    Compute feature-target correlations.

    Args:
        feature: Feature series
        target: Target series

    Returns:
        Dictionary with pearson, spearman, kendall correlations
    """
    # Align and drop NaN
    aligned = pd.DataFrame({'feature': feature, 'target': target}).dropna()

    if len(aligned) < 10:
        return {'pearson': 0.0, 'spearman': 0.0, 'kendall': 0.0}

    pearson_corr = aligned['feature'].corr(aligned['target'], method='pearson')
    spearman_corr = aligned['feature'].corr(aligned['target'], method='spearman')
    kendall_corr = aligned['feature'].corr(aligned['target'], method='kendall')

    return {
        'pearson': float(pearson_corr),
        'spearman': float(spearman_corr),
        'kendall': float(kendall_corr),
    }


def run_stationarity_tests(
    data: pd.Series,
) -> tuple[ADFTestResult, KPSSTestResult]:
    """
    Run ADF and KPSS stationarity tests.

    Args:
        data: Time series data

    Returns:
        Tuple of (ADFTestResult, KPSSTestResult)
    """
    # Drop NaN
    clean_data = data.dropna()

    if len(clean_data) < 20:
        raise ValueError(f"Insufficient data for stationarity tests: {len(clean_data)} < 20")

    # Run ADF test
    adf_output = adfuller(clean_data, autolag='AIC')
    adf_result = ADFTestResult.from_adf_output(adf_output)

    # Run KPSS test
    kpss_output = kpss(clean_data, regression='c', nlags='auto')
    kpss_result = KPSSTestResult.from_kpss_output(kpss_output)

    return adf_result, kpss_result


def compute_lagged_correlations(
    feature: pd.Series,
    target: pd.Series,
    max_lag: int = 10,
) -> pd.Series:
    """
    Compute lagged correlations (feature_t vs. target_t+k).

    Args:
        feature: Feature series
        target: Target series
        max_lag: Maximum lag to compute

    Returns:
        Series with correlations at each lag
    """
    correlations = []

    for lag in range(max_lag + 1):
        if lag == 0:
            corr = feature.corr(target)
        else:
            # Shift target forward by lag
            target_lagged = target.shift(-lag)
            corr = feature.corr(target_lagged)

        correlations.append(corr)

    return pd.Series(correlations, index=range(max_lag + 1))


def compute_rolling_correlation(
    feature: pd.Series,
    target: pd.Series,
    window: int = 252,  # ~1 year for daily data
) -> pd.Series:
    """
    Compute rolling correlation over time.

    Args:
        feature: Feature series
        target: Target series
        window: Rolling window size

    Returns:
        Series with rolling correlation
    """
    # Align series
    aligned = pd.DataFrame({'feature': feature, 'target': target}).dropna()

    if len(aligned) < window:
        raise ValueError(f"Insufficient data for rolling correlation: {len(aligned)} < {window}")

    rolling_corr = aligned['feature'].rolling(window=window).corr(aligned['target'])

    return rolling_corr
