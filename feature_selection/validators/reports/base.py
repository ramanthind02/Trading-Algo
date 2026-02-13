"""Base report data structures."""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class DescriptiveStats:
    """Descriptive statistics for a numeric series."""

    mean: float
    std: float
    skew: float
    kurtosis: float
    min_val: float
    max_val: float
    q25: float
    median: float
    q75: float
    n_samples: int

    @classmethod
    def from_series(cls, data: pd.Series) -> "DescriptiveStats":
        """Create DescriptiveStats from a pandas Series."""
        return cls(
            mean=float(data.mean()),
            std=float(data.std()),
            skew=float(data.skew()),
            kurtosis=float(data.kurtosis()),
            min_val=float(data.min()),
            max_val=float(data.max()),
            q25=float(data.quantile(0.25)),
            median=float(data.median()),
            q75=float(data.quantile(0.75)),
            n_samples=len(data),
        )


@dataclass(frozen=True)
class ADFTestResult:
    """Augmented Dickey-Fuller test result."""

    statistic: float
    p_value: float
    n_lags: int
    critical_values: dict[str, float]  # {'1%': -3.43, '5%': -2.86, '10%': -2.57}
    is_stationary: bool  # True if p_value < 0.05

    @classmethod
    def from_adf_output(cls, adf_output: tuple) -> "ADFTestResult":
        """Create from statsmodels adfuller output."""
        statistic, p_value, n_lags, _, critical_values, _ = adf_output
        return cls(
            statistic=float(statistic),
            p_value=float(p_value),
            n_lags=int(n_lags),
            critical_values={k: float(v) for k, v in critical_values.items()},
            is_stationary=p_value < 0.05,
        )


@dataclass(frozen=True)
class KPSSTestResult:
    """KPSS stationarity test result."""

    statistic: float
    p_value: float
    n_lags: int
    critical_values: dict[str, float]
    is_stationary: bool  # True if p_value > 0.05 (KPSS null = stationary)

    @classmethod
    def from_kpss_output(cls, kpss_output: tuple) -> "KPSSTestResult":
        """Create from statsmodels kpss output."""
        statistic, p_value, n_lags, critical_values = kpss_output
        return cls(
            statistic=float(statistic),
            p_value=float(p_value),
            n_lags=int(n_lags),
            critical_values={k: float(v) for k, v in critical_values.items()},
            is_stationary=p_value > 0.05,
        )


@dataclass(frozen=True)
class MonotonicityTestResult:
    """Test for monotonic relationship (e.g., across deciles)."""

    kendall_tau: float  # Kendall's tau correlation
    p_value: float
    is_monotonic: bool  # True if |tau| > 0.3 and p < 0.05
    direction: Literal['increasing', 'decreasing', 'none']
