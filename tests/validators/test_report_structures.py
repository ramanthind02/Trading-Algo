import pytest
import numpy as np
from feature_selection.validators.reports.base import DescriptiveStats


def test_descriptive_stats_creation():
    """Test DescriptiveStats dataclass creation."""
    stats = DescriptiveStats(
        mean=0.5,
        std=1.2,
        skew=-0.3,
        kurtosis=2.1,
        min_val=-3.0,
        max_val=4.5,
        q25=0.1,
        median=0.6,
        q75=1.1,
        n_samples=1000,
    )

    assert stats.mean == 0.5
    assert stats.std == 1.2
    assert stats.n_samples == 1000


def test_descriptive_stats_from_series():
    """Test DescriptiveStats.from_series() factory method."""
    import pandas as pd

    data = pd.Series(np.random.randn(100))
    stats = DescriptiveStats.from_series(data)

    assert stats.n_samples == 100
    assert stats.mean == pytest.approx(data.mean(), rel=1e-6)
    assert stats.std == pytest.approx(data.std(), rel=1e-6)
    assert stats.median == pytest.approx(data.median(), rel=1e-6)


def test_adf_test_result_from_output():
    """Test ADFTestResult.from_adf_output() factory method."""
    from feature_selection.validators.reports.base import ADFTestResult

    # Mock statsmodels adfuller output format
    mock_output = (
        -3.5,  # statistic
        0.008,  # p_value
        5,  # n_lags
        100,  # n_obs
        {'1%': -3.43, '5%': -2.86, '10%': -2.57},  # critical_values
        None  # icbest
    )

    result = ADFTestResult.from_adf_output(mock_output)

    assert result.statistic == -3.5
    assert result.p_value == 0.008
    assert result.is_stationary is True  # p < 0.05
    assert result.critical_values['5%'] == -2.86


def test_kpss_test_result_from_output():
    """Test KPSSTestResult.from_kpss_output() factory method."""
    from feature_selection.validators.reports.base import KPSSTestResult

    # Mock statsmodels kpss output format
    mock_output = (
        0.3,  # statistic
        0.1,  # p_value
        5,  # n_lags
        {'1%': 0.739, '5%': 0.463, '10%': 0.347}  # critical_values
    )

    result = KPSSTestResult.from_kpss_output(mock_output)

    assert result.statistic == 0.3
    assert result.p_value == 0.1
    assert result.is_stationary is True  # p > 0.05 (null = stationary)


def test_eda_report_creation():
    """Test EDAReport dataclass creation."""
    from feature_selection.validators.reports.eda import EDAReport
    from feature_selection.validators.reports.base import (
        DescriptiveStats,
        ADFTestResult,
        KPSSTestResult,
    )
    import pandas as pd

    feature_stats = DescriptiveStats.from_series(pd.Series(np.random.randn(100)))
    target_stats = DescriptiveStats.from_series(pd.Series(np.random.randn(100)))

    mock_adf = (-3.5, 0.008, 5, 100, {'5%': -2.86}, None)
    mock_kpss = (0.3, 0.1, 5, {'5%': 0.463})

    report = EDAReport(
        feature_stats=feature_stats,
        target_stats=target_stats,
        correlations={'pearson': 0.15, 'spearman': 0.18},
        lagged_correlations=pd.Series([0.15, 0.10, 0.05]),
        adf_test=ADFTestResult.from_adf_output(mock_adf),
        kpss_test=KPSSTestResult.from_kpss_output(mock_kpss),
        rolling_correlation=pd.Series([0.12, 0.15, 0.18]),
        regime_stats={},
        distribution_plot=None,
        correlation_plot=None,
        time_series_plot=None,
        stationarity_plot=None,
    )

    assert report.feature_stats == feature_stats
    assert report.correlations['pearson'] == 0.15
    assert report.adf_test.is_stationary is True


def test_permutation_report_creation():
    """Test PermutationReport dataclass creation."""
    from feature_selection.validators.reports.permutation import PermutationReport
    import numpy as np

    report = PermutationReport(
        stage='stage1_vector_shuffle',
        observed_sharpe=0.8,
        observed_t_stat=2.5,
        observed_returns_mean=0.05,
        permuted_sharpes=np.random.randn(1000),
        permuted_t_stats=np.random.randn(1000),
        p_value=0.01,
        confidence_level=0.95,
        critical_value=0.5,
        passed=True,
        margin=0.3,
        permutation_histogram=None,
        qq_plot=None,
        n_permutations=1000,
        random_seed=42,
        execution_time=5.2,
    )

    assert report.stage == 'stage1_vector_shuffle'
    assert report.passed is True
    assert report.p_value == 0.01
