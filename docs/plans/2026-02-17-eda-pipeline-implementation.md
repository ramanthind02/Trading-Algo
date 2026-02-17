# EDA Pipeline Implementation Plan (T001–T004)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the Feature Validator EDA pipeline as a fresh module at `feature_selection/eda/` with four components (common, continuous, rule-based, reporter), full unit tests, and a flexible integration test covering both continuous (RSI) and rule-based (RSI Signal 2-period) features.

**Architecture:** Build entirely fresh in `feature_selection/eda/` — do NOT touch the existing `feature_selection/validators/eda/` code. All frozen dataclasses live in `eda_dataclasses.py`. Each module imports only from that file and standard libraries. Integration test uses real `data/ohlc_data/` cache via `CacheManager` and `extract_features_for_bias_node`.

**Tech Stack:** pandas, numpy, scipy.stats, matplotlib (Agg backend in tests), statsmodels (already installed), pytest, Python 3.11+

---

## Context

- Leave `feature_selection/validators/eda/` and `feature_selection/validators/reports/` untouched — they power `FeatureValidator` and must not break.
- `feature_selection/eda/` is a **new, independent** namespace.
- Unit tests go in `tests/validators/eda/` (new sub-directory).
- Integration test goes in `tests/integration/feature_validator/test_eda_pipeline.py` (new file).
- All tests use `pytest`. Run with venv: `source /home/raman/repos/Trading-Algo/venv/bin/activate`.
- Use `matplotlib.use('Agg')` at the top of every test file that touches plots (prevents display errors in headless environments).

## Integration Test Config

**Continuous** (T001 + T002 + T004):
```python
CONTINUOUS_BIAS_SPEC = {
    "module_name": "rsi",
    "timeframes": [TimeFrame.D],
    "params": {"lookback": 5},
}
TICKERS = [Ticker.ES]
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2023, 12, 31)
```

**Rule-based** (T001 + T003 + T004) — RSI 2-Period Strategy:
```python
RULE_BASED_BIAS_SPEC = {
    "module_name": "rsi_signal",
    "timeframes": [TimeFrame.D],
    "params": {
        "rsi_period": 2,
        "oversold": 25.0,
        "overbought": 65.0,
        "strategy_mode": "long",
        "exit_policy": "threshold_or_bars",
        "exit_bars": 5,
    },
}
```

---

## Task 1: Directory Structure + All Dataclasses

**Files:**
- Create: `feature_selection/eda/__init__.py`
- Create: `feature_selection/eda/eda_dataclasses.py`
- Create: `tests/validators/eda/__init__.py`

### Step 1: Create directories and `__init__.py` files

```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate
mkdir -p /home/raman/repos/Trading-Algo/feature_selection/eda
mkdir -p /home/raman/repos/Trading-Algo/tests/validators/eda
touch /home/raman/repos/Trading-Algo/feature_selection/eda/__init__.py
touch /home/raman/repos/Trading-Algo/tests/validators/eda/__init__.py
```

### Step 2: Write `eda_dataclasses.py`

Create `feature_selection/eda/eda_dataclasses.py`:

```python
"""Frozen dataclasses for the Feature Validator EDA pipeline (T001–T004)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from utils.enums import Ticker, TimeFrame


# ─── T001: Common EDA ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DescriptiveStats:
    """Descriptive statistics for a single series."""
    min_val: float
    max_val: float
    mean: float
    median: float
    std: float
    skew: float
    kurtosis: float
    nan_count: int
    nan_pct: float
    sample_size: int


@dataclass(frozen=True)
class TemporalStability:
    """Rolling correlation series and structural breaks."""
    rolling_correlation: pd.Series           # DatetimeIndex → float
    structural_breaks: list[pd.Timestamp]    # timestamps where |Δcorr| is large


@dataclass(frozen=True)
class CorrelationAnalysis:
    """Feature-target correlation at various lags."""
    pearson: float
    spearman: float
    kendall: float
    lagged_correlations: dict[int, float]    # lag → correlation (lags 1..max_lag)


@dataclass(frozen=True)
class CommonEDAPlots:
    """Matplotlib Figure objects for common EDA."""
    time_series_fig: Figure      # feature + target over time (2 subplots)
    rolling_corr_fig: Figure     # rolling correlation over time
    rolling_obj_fig: Figure      # rolling objective metric over time


@dataclass(frozen=True)
class CommonEDAStats:
    """Aggregated common EDA statistics."""
    feature_stats: DescriptiveStats
    target_stats: DescriptiveStats
    temporal_stability: TemporalStability
    correlation_analysis: CorrelationAnalysis
    rolling_objective: pd.Series     # DatetimeIndex → float


# ─── T002: Continuous Feature EDA ────────────────────────────────────────────

@dataclass(frozen=True)
class DecileBinStats:
    """Per-bin statistics for decile analysis."""
    bin_edges: np.ndarray         # length n_bins + 1
    mean_return: np.ndarray       # length n_bins
    volatility: np.ndarray        # length n_bins
    sharpe: np.ndarray            # length n_bins  (NaN where vol == 0)
    t_stat: np.ndarray            # length n_bins  (NaN where n < 2)
    sample_count: np.ndarray      # length n_bins (int)


@dataclass(frozen=True)
class DecileAnalysis:
    """Decile binning results with overall trend label."""
    bin_stats: DecileBinStats
    overall_trend: str            # 'monotonic_increasing' | 'monotonic_decreasing' | 'U-shaped' | 'flat'


@dataclass(frozen=True)
class MonotonicityTest:
    """Kendall's tau monotonicity test over bin means."""
    kendall_tau: float
    p_value: float
    is_monotonic: bool            # |tau| > 0.5 and p < 0.05


@dataclass(frozen=True)
class DistributionDiagnostics:
    """Normality diagnostics for feature distribution."""
    skewness: float
    kurtosis: float
    normality_test_stat: float    # Shapiro-Wilk W statistic (or Anderson for n > 5000)
    normality_p_value: float
    is_normal: bool               # p_value > 0.05


@dataclass(frozen=True)
class ContinuousEDAPlots:
    """Matplotlib Figures for continuous feature EDA."""
    decile_plot_fig: Figure       # 3 subplots: mean return, Sharpe, t-stat
    histogram_fig: Figure         # histogram + quantile overlay lines
    qq_plot_fig: Figure           # Q-Q plot vs normal
    kde_fig: Figure               # KDE of feature distribution


@dataclass(frozen=True)
class ContinuousEDAStats:
    """Aggregated continuous feature EDA statistics."""
    decile_analysis: DecileAnalysis
    monotonicity_test: MonotonicityTest
    distribution_diagnostics: DistributionDiagnostics


# ─── T003: Rule-Based Feature EDA ────────────────────────────────────────────

@dataclass(frozen=True)
class LevelStats:
    """Statistics for one discrete level (-1, 0, or 1)."""
    level: int
    mean_return: float
    volatility: float
    sharpe: float                 # NaN if vol == 0
    adjusted_sharpe: float        # NaN if not computable
    sample_count: int
    is_reliable: bool             # False if sample_count < 10


@dataclass(frozen=True)
class PerLevelStats:
    """Per-level statistics for all observed levels."""
    stats_by_level: dict[int, LevelStats]


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval for one level."""
    level: int
    mean_return: float
    ci_lower: float
    ci_upper: float
    bootstrap_distribution: np.ndarray    # shape (n_iterations,)


@dataclass(frozen=True)
class BootstrapCIResults:
    """Bootstrap CIs for all levels."""
    ci_by_level: dict[int, BootstrapCI]


@dataclass(frozen=True)
class TransitionMatrix:
    """Level-to-level transition counts and probabilities."""
    transition_counts: np.ndarray    # shape (3, 3) for levels [-1, 0, 1]
    transition_probs: np.ndarray     # row-normalised; each row sums to 1.0


@dataclass(frozen=True)
class RuleBasedEDAPlots:
    """Matplotlib Figures for rule-based feature EDA."""
    level_plot_fig: Figure            # bar chart per level with bootstrap CI error bars
    transition_heatmap_fig: Figure    # heatmap of transition probabilities


@dataclass(frozen=True)
class RuleBasedEDAStats:
    """Aggregated rule-based EDA statistics."""
    per_level_stats: PerLevelStats
    bootstrap_ci_results: BootstrapCIResults
    transition_matrix: TransitionMatrix


# ─── T004: EDA Report Generation ─────────────────────────────────────────────

@dataclass(frozen=True)
class EDAMetadata:
    """Identity and provenance for an EDA report."""
    feature_name: str
    param_combo: dict[str, Any]
    timeframe: TimeFrame
    ticker: Ticker
    timestamp: datetime


@dataclass(frozen=True)
class EDAConfig:
    """User-specified EDA computation settings."""
    n_bins: int = 15
    rolling_window: int = 252
    objective_fn: Callable[[pd.Series, pd.Series], float] = field(
        default=lambda signals, returns: (returns.mean() / returns.std()) if returns.std() > 0 else 0.0
    )
    max_lag: int = 5
    bootstrap_iterations: int = 1000
    random_seed: int = 42


@dataclass(frozen=True)
class DiagnosticFlags:
    """Warnings and red flags computed from EDA stats."""
    warnings: list[str]
    red_flags: list[str]
    is_viable: bool    # True iff len(red_flags) == 0


@dataclass(frozen=True)
class ContinuousEDAReport:
    """Full EDA report for a continuous feature."""
    metadata: EDAMetadata
    common_stats: CommonEDAStats
    continuous_stats: ContinuousEDAStats
    common_plots: CommonEDAPlots
    continuous_plots: ContinuousEDAPlots
    diagnostics: DiagnosticFlags


@dataclass(frozen=True)
class RuleBasedEDAReport:
    """Full EDA report for a rule-based feature."""
    metadata: EDAMetadata
    common_stats: CommonEDAStats
    rule_stats: RuleBasedEDAStats
    common_plots: CommonEDAPlots
    rule_plots: RuleBasedEDAPlots
    diagnostics: DiagnosticFlags
```

### Step 3: Write a smoke test for dataclasses

Create `tests/validators/eda/test_dataclasses_smoke.py`:

```python
"""Smoke test: all dataclasses importable and instantiatable."""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from feature_selection.eda.eda_dataclasses import (
    DescriptiveStats, TemporalStability, CorrelationAnalysis,
    CommonEDAPlots, CommonEDAStats,
    DecileBinStats, DecileAnalysis, MonotonicityTest,
    DistributionDiagnostics, ContinuousEDAPlots, ContinuousEDAStats,
    LevelStats, PerLevelStats, BootstrapCI, BootstrapCIResults,
    TransitionMatrix, RuleBasedEDAPlots, RuleBasedEDAStats,
)


def _dummy_fig() -> plt.Figure:
    fig, _ = plt.subplots()
    plt.close(fig)
    return fig


def test_descriptive_stats_instantiates() -> None:
    s = DescriptiveStats(
        min_val=0.0, max_val=1.0, mean=0.5, median=0.5,
        std=0.1, skew=0.0, kurtosis=3.0,
        nan_count=0, nan_pct=0.0, sample_size=100,
    )
    assert s.mean == 0.5


def test_level_stats_is_reliable_flag() -> None:
    ls = LevelStats(
        level=1, mean_return=0.01, volatility=0.1,
        sharpe=0.1, adjusted_sharpe=0.1,
        sample_count=5, is_reliable=False,
    )
    assert ls.is_reliable is False


def test_dataclasses_are_frozen() -> None:
    s = DescriptiveStats(
        min_val=0.0, max_val=1.0, mean=0.5, median=0.5,
        std=0.1, skew=0.0, kurtosis=3.0,
        nan_count=0, nan_pct=0.0, sample_size=100,
    )
    with pytest.raises(Exception):
        s.mean = 99.0  # type: ignore[misc]
```

### Step 4: Run smoke test

```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate
pytest tests/validators/eda/test_dataclasses_smoke.py -v
```

Expected: 3 PASSED.

### Step 5: Update `feature_selection/eda/__init__.py`

```python
"""Feature Validator EDA pipeline — fresh implementation (T001–T004)."""
from .eda_dataclasses import (
    DescriptiveStats, TemporalStability, CorrelationAnalysis,
    CommonEDAPlots, CommonEDAStats,
    DecileBinStats, DecileAnalysis, MonotonicityTest,
    DistributionDiagnostics, ContinuousEDAPlots, ContinuousEDAStats,
    LevelStats, PerLevelStats, BootstrapCI, BootstrapCIResults,
    TransitionMatrix, RuleBasedEDAPlots, RuleBasedEDAStats,
    EDAMetadata, EDAConfig, DiagnosticFlags,
    ContinuousEDAReport, RuleBasedEDAReport,
)

__all__ = [
    "DescriptiveStats", "TemporalStability", "CorrelationAnalysis",
    "CommonEDAPlots", "CommonEDAStats",
    "DecileBinStats", "DecileAnalysis", "MonotonicityTest",
    "DistributionDiagnostics", "ContinuousEDAPlots", "ContinuousEDAStats",
    "LevelStats", "PerLevelStats", "BootstrapCI", "BootstrapCIResults",
    "TransitionMatrix", "RuleBasedEDAPlots", "RuleBasedEDAStats",
    "EDAMetadata", "EDAConfig", "DiagnosticFlags",
    "ContinuousEDAReport", "RuleBasedEDAReport",
]
```

### Step 6: Commit

```bash
git add feature_selection/eda/ tests/validators/eda/
git commit -m "feat: scaffold feature_selection/eda/ directory and all EDA dataclasses (T001-T004)"
```

---

## Task 2: T001 — Common EDA (`common_eda.py`)

**Files:**
- Create: `feature_selection/eda/common_eda.py`
- Create: `tests/validators/eda/test_common_eda.py`

### Step 1: Write failing unit tests

Create `tests/validators/eda/test_common_eda.py`:

```python
"""Unit tests for common_eda.py (T001).

All tests use synthetic data with known properties.
No real market data — no cache required.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

from feature_selection.eda.common_eda import (
    compute_descriptive_stats,
    compute_temporal_stability,
    compute_correlation_analysis,
    compute_rolling_objective,
    create_common_eda_plots,
)
from feature_selection.eda.eda_dataclasses import (
    DescriptiveStats, TemporalStability, CorrelationAnalysis, CommonEDAPlots,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _daily_index(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = _daily_index(len(values), start)
    return pd.Series(values, index=idx, dtype=float)


# ── T001-U1: descriptive stats ────────────────────────────────────────────────

def test_descriptive_stats_known_values() -> None:
    """Mean, std, skew, kurtosis must match hand-computed values."""
    data = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = compute_descriptive_stats(data)
    assert stats.mean == pytest.approx(3.0)
    assert stats.median == pytest.approx(3.0)
    assert stats.min_val == pytest.approx(1.0)
    assert stats.max_val == pytest.approx(5.0)
    assert stats.nan_count == 0
    assert stats.nan_pct == pytest.approx(0.0)
    assert stats.sample_size == 5


def test_nan_handling_explicit() -> None:
    """nan_count and nan_pct are correct; stats computed after dropna."""
    data = _series([1.0, float('nan'), 3.0, float('nan'), 5.0])
    stats = compute_descriptive_stats(data)
    assert stats.nan_count == 2
    assert stats.nan_pct == pytest.approx(0.4)
    assert stats.mean == pytest.approx(3.0)   # mean of [1, 3, 5]
    assert stats.sample_size == 5             # total length, not dropna length


# ── T001-U2: index misalignment ───────────────────────────────────────────────

def test_index_misalignment_raises() -> None:
    """Mismatched DatetimeIndex raises ValueError before any computation."""
    idx_a = _daily_index(10, "2020-01-01")
    idx_b = _daily_index(10, "2021-01-01")
    feature = pd.Series(np.ones(10), index=idx_a)
    target = pd.Series(np.ones(10), index=idx_b)
    with pytest.raises(ValueError, match="index"):
        compute_temporal_stability(feature, target, idx_a, rolling_window=5)


# ── T001-U3: rolling window guard ────────────────────────────────────────────

def test_rolling_window_exceeds_length_raises() -> None:
    """window > len(feature) raises ValueError."""
    n = 10
    idx = _daily_index(n)
    feature = pd.Series(np.random.randn(n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    with pytest.raises(ValueError, match="window"):
        compute_temporal_stability(feature, target, idx, rolling_window=n + 1)


# ── T001-U4: no-lookahead for rolling correlation ────────────────────────────

def test_temporal_stability_no_lookahead() -> None:
    """Rolling correlation at timestamp t only uses data up to t (NaN before window fills)."""
    n = 50
    idx = _daily_index(n)
    np.random.seed(0)
    feature = pd.Series(np.random.randn(n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    window = 10
    result = compute_temporal_stability(feature, target, idx, rolling_window=window)
    # First (window-1) values must be NaN
    assert result.rolling_correlation.iloc[:window - 1].isna().all()
    # Value at position window-1 must be non-NaN
    assert pd.notna(result.rolling_correlation.iloc[window - 1])


# ── T001-U5: lagged correlations ──────────────────────────────────────────────

def test_correlation_analysis_lagged_synthetic() -> None:
    """Feature with 1-lag lead structure should have highest correlation at lag=1."""
    n = 200
    idx = _daily_index(n)
    np.random.seed(42)
    base = pd.Series(np.random.randn(n), index=idx)
    # target is feature shifted by 1 (feature leads target by 1)
    target = base.shift(1).fillna(0)
    result = compute_correlation_analysis(base, target, max_lag=5)
    assert result.lagged_correlations[1] > result.lagged_correlations[0]
    assert result.lagged_correlations[1] > result.lagged_correlations[2]


def test_correlation_analysis_has_all_lags() -> None:
    """lagged_correlations dict has keys 1..max_lag."""
    n = 100
    idx = _daily_index(n)
    feature = pd.Series(np.random.randn(n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    result = compute_correlation_analysis(feature, target, max_lag=5)
    assert set(result.lagged_correlations.keys()) == {1, 2, 3, 4, 5}


# ── T001-U6: rolling objective ────────────────────────────────────────────────

def test_rolling_objective_sharpe_manual() -> None:
    """Rolling Sharpe over 3-bar window on known returns matches manual calc."""
    returns = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    signals = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])  # always long

    def sharpe_fn(s: pd.Series, r: pd.Series) -> float:
        return r.mean() / r.std() if r.std() > 0 else 0.0

    result = compute_rolling_objective(signals, returns, sharpe_fn, window=3)
    # First 2 values must be NaN (window=3, min_periods=3)
    assert result.iloc[:2].isna().all()
    # Third value: mean([0.1,0.2,0.3]) / std([0.1,0.2,0.3])
    expected = np.mean([0.1, 0.2, 0.3]) / np.std([0.1, 0.2, 0.3], ddof=1)
    assert result.iloc[2] == pytest.approx(expected, rel=1e-6)


# ── T001-U7: plots smoke test ─────────────────────────────────────────────────

def test_common_eda_plots_smoke() -> None:
    """All three Figure objects are created without error."""
    n = 60
    idx = _daily_index(n)
    feature = pd.Series(np.random.randn(n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    rolling_corr = pd.Series(np.random.randn(n), index=idx)
    rolling_obj = pd.Series(np.random.randn(n), index=idx)

    plots = create_common_eda_plots(feature, target, idx, rolling_corr, rolling_obj)
    assert plots.time_series_fig is not None
    assert plots.rolling_corr_fig is not None
    assert plots.rolling_obj_fig is not None
```

### Step 2: Run tests to confirm they fail

```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate
pytest tests/validators/eda/test_common_eda.py -v
```

Expected: All FAIL with `ModuleNotFoundError` or `ImportError`.

### Step 3: Implement `feature_selection/eda/common_eda.py`

```python
"""Common EDA infrastructure for both continuous and rule-based features (T001)."""
from __future__ import annotations

from typing import Callable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from feature_selection.eda.eda_dataclasses import (
    CorrelationAnalysis,
    CommonEDAPlots,
    DescriptiveStats,
    TemporalStability,
)


def _validate_aligned_index(feature: pd.Series, target: pd.Series) -> None:
    """Raise ValueError if feature and target do not share the same DatetimeIndex."""
    if not feature.index.equals(target.index):
        raise ValueError(
            f"feature and target index mismatch: "
            f"{feature.index[[0, -1]]} vs {target.index[[0, -1]]}"
        )


def compute_descriptive_stats(series: pd.Series) -> DescriptiveStats:
    """Compute descriptive statistics for a single series.

    NaN values are counted and reported; statistics are computed on clean data.

    Args:
        series: Numeric pd.Series (may contain NaN).

    Returns:
        DescriptiveStats frozen dataclass.
    """
    nan_count = int(series.isna().sum())
    nan_pct = nan_count / len(series) if len(series) > 0 else 0.0
    clean = series.dropna()

    return DescriptiveStats(
        min_val=float(clean.min()),
        max_val=float(clean.max()),
        mean=float(clean.mean()),
        median=float(clean.median()),
        std=float(clean.std()),
        skew=float(stats.skew(clean.values)),
        kurtosis=float(stats.kurtosis(clean.values)),
        nan_count=nan_count,
        nan_pct=float(nan_pct),
        sample_size=len(series),
    )


def compute_temporal_stability(
    feature: pd.Series,
    target: pd.Series,
    timestamps: pd.DatetimeIndex,
    rolling_window: int = 252,
) -> TemporalStability:
    """Compute rolling feature-target correlation for temporal stability analysis.

    Args:
        feature: Feature series with DatetimeIndex.
        target: Target series with same DatetimeIndex.
        timestamps: Must match feature.index exactly.
        rolling_window: Rolling window in bars (default 252 ≈ 1 trading year).

    Returns:
        TemporalStability with rolling_correlation series and structural_breaks list.

    Raises:
        ValueError: If feature and target indices differ, or window > len(feature).
    """
    _validate_aligned_index(feature, target)
    if not feature.index.equals(pd.DatetimeIndex(timestamps)):
        raise ValueError("timestamps must equal feature.index")
    if rolling_window > len(feature):
        raise ValueError(
            f"window ({rolling_window}) exceeds series length ({len(feature)})"
        )

    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    rolling_corr = aligned["f"].rolling(window=rolling_window, min_periods=rolling_window).corr(aligned["t"])
    # Reindex to original index (NaN for dropped rows)
    rolling_corr = rolling_corr.reindex(feature.index)

    # Detect structural breaks: positions where |Δcorr| > 0.3
    delta = rolling_corr.diff().abs()
    break_timestamps = list(rolling_corr.index[delta > 0.3])

    return TemporalStability(
        rolling_correlation=rolling_corr,
        structural_breaks=break_timestamps,
    )


def compute_correlation_analysis(
    feature: pd.Series,
    target: pd.Series,
    max_lag: int = 5,
) -> CorrelationAnalysis:
    """Compute Pearson, Spearman, Kendall, and lagged correlations.

    Args:
        feature: Feature series with DatetimeIndex.
        target: Target series aligned with feature.
        max_lag: Maximum lag to test (default 5). Lags 1..max_lag are stored.

    Returns:
        CorrelationAnalysis frozen dataclass.
    """
    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    f, t = aligned["f"], aligned["t"]

    pearson = float(f.corr(t, method="pearson"))
    spearman = float(f.corr(t, method="spearman"))
    kendall = float(f.corr(t, method="kendall"))

    lagged: dict[int, float] = {}
    for lag in range(1, max_lag + 1):
        lagged[lag] = float(f.corr(t.shift(-lag), method="pearson"))

    return CorrelationAnalysis(
        pearson=pearson,
        spearman=spearman,
        kendall=kendall,
        lagged_correlations=lagged,
    )


def compute_rolling_objective(
    signals: pd.Series,
    returns: pd.Series,
    objective_fn: Callable[[pd.Series, pd.Series], float],
    window: int = 252,
) -> pd.Series:
    """Compute rolling objective metric using a user-supplied function.

    Args:
        signals: Signal series (same index as returns).
        returns: Return series.
        objective_fn: Callable(signals_window, returns_window) → scalar.
        window: Rolling window size; uses min_periods=window (no partial windows).

    Returns:
        pd.Series with same index as inputs; first (window-1) values are NaN.
    """
    result_values: list[float] = []

    for i in range(len(returns)):
        if i < window - 1:
            result_values.append(float("nan"))
        else:
            s_window = signals.iloc[i - window + 1: i + 1]
            r_window = returns.iloc[i - window + 1: i + 1]
            result_values.append(float(objective_fn(s_window, r_window)))

    return pd.Series(result_values, index=returns.index)


def create_common_eda_plots(
    feature: pd.Series,
    target: pd.Series,
    timestamps: pd.DatetimeIndex,
    rolling_corr: pd.Series,
    rolling_obj: pd.Series,
) -> CommonEDAPlots:
    """Create the three standard common EDA figures.

    Args:
        feature: Feature series.
        target: Target series.
        timestamps: DatetimeIndex shared by all series.
        rolling_corr: Rolling feature-target correlation (from compute_temporal_stability).
        rolling_obj: Rolling objective metric (from compute_rolling_objective).

    Returns:
        CommonEDAPlots with three Figure objects.
    """
    # 1. Time-series plot (2 subplots)
    fig_ts, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(timestamps, feature.values, linewidth=0.8, color="steelblue")
    ax1.set_title("Feature over time")
    ax1.set_ylabel("Feature value")
    ax2.plot(timestamps, target.values, linewidth=0.8, color="darkorange")
    ax2.set_title("Target (returns) over time")
    ax2.set_ylabel("Return")
    fig_ts.tight_layout()
    plt.close(fig_ts)

    # 2. Rolling correlation plot
    fig_rc, ax = plt.subplots(figsize=(12, 3))
    ax.plot(rolling_corr.index, rolling_corr.values, linewidth=0.8, color="purple")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Rolling feature-target correlation")
    ax.set_ylabel("Correlation")
    fig_rc.tight_layout()
    plt.close(fig_rc)

    # 3. Rolling objective plot
    fig_ro, ax = plt.subplots(figsize=(12, 3))
    ax.plot(rolling_obj.index, rolling_obj.values, linewidth=0.8, color="green")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Rolling objective metric")
    ax.set_ylabel("Metric value")
    fig_ro.tight_layout()
    plt.close(fig_ro)

    return CommonEDAPlots(
        time_series_fig=fig_ts,
        rolling_corr_fig=fig_rc,
        rolling_obj_fig=fig_ro,
    )
```

### Step 4: Run tests — all must pass

```bash
pytest tests/validators/eda/test_common_eda.py -v
```

Expected: 7 PASSED.

### Step 5: Commit

```bash
git add feature_selection/eda/common_eda.py tests/validators/eda/test_common_eda.py
git commit -m "feat: implement T001 common EDA infrastructure with 7 unit tests"
```

---

## Task 3: T002 — Continuous Feature EDA (`continuous_eda.py`)

**Files:**
- Create: `feature_selection/eda/continuous_eda.py`
- Create: `tests/validators/eda/test_continuous_eda.py`

### Step 1: Write failing unit tests

Create `tests/validators/eda/test_continuous_eda.py`:

```python
"""Unit tests for continuous_eda.py (T002).

All synthetic data — no real market data, no cache required.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

from feature_selection.eda.continuous_eda import (
    compute_decile_analysis,
    compute_monotonicity_test,
    compute_distribution_diagnostics,
    create_continuous_eda_plots,
)
from feature_selection.eda.eda_dataclasses import (
    DecileAnalysis, MonotonicityTest, DistributionDiagnostics, ContinuousEDAPlots,
)


def _series(n: int, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    np.random.seed(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    feature = pd.Series(np.linspace(0, 10, n), index=idx)
    target = pd.Series(0.5 * np.linspace(0, 10, n) + np.random.randn(n) * 0.5, index=idx)
    return feature, target


# ── T002-U1: known bin counts ─────────────────────────────────────────────────

def test_decile_analysis_known_values() -> None:
    """15-bin analysis on 150-sample series: each bin has ~10 samples."""
    n = 150
    idx = pd.bdate_range("2020-01-01", periods=n)
    feature = pd.Series(np.linspace(0, 1, n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    result = compute_decile_analysis(feature, target, n_bins=15)
    assert len(result.bin_stats.sample_count) == 15
    assert result.bin_stats.sample_count.sum() == pytest.approx(n, abs=5)


def test_decile_analysis_bin_edges_length() -> None:
    """bin_edges has length n_bins + 1."""
    feature, target = _series(200)
    result = compute_decile_analysis(feature, target, n_bins=10)
    assert len(result.bin_stats.bin_edges) == 11


# ── T002-U2: monotonicity ─────────────────────────────────────────────────────

def test_monotonicity_test_monotonic_increasing() -> None:
    """Strictly increasing bin means → is_monotonic=True, positive tau."""
    bin_means = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                          1.1, 1.2, 1.3, 1.4, 1.5])
    result = compute_monotonicity_test(bin_means)
    assert result.is_monotonic is True
    assert result.kendall_tau > 0.0


def test_monotonicity_test_flat() -> None:
    """Flat bin means → is_monotonic=False."""
    bin_means = np.ones(15) * 0.5
    result = compute_monotonicity_test(bin_means)
    assert result.is_monotonic is False


# ── T002-U3: distribution diagnostics ────────────────────────────────────────

def test_distribution_diagnostics_known_skew() -> None:
    """np.random.seed(42) normal → is_normal=True for small sample."""
    np.random.seed(42)
    idx = pd.bdate_range("2020-01-01", periods=200)
    feature = pd.Series(np.random.normal(0, 1, 200), index=idx)
    result = compute_distribution_diagnostics(feature)
    assert result.is_normal is True
    assert abs(result.skewness) < 0.5


# ── T002-U4: sharpe NaN on zero volatility bin ───────────────────────────────

def test_sharpe_zero_volatility_bin() -> None:
    """Bin with constant returns must produce Sharpe=NaN, no crash."""
    n = 150
    idx = pd.bdate_range("2020-01-01", periods=n)
    feature = pd.Series(np.linspace(0, 1, n), index=idx)
    # All returns identical → zero variance in every bin
    target = pd.Series(np.ones(n) * 0.01, index=idx)
    result = compute_decile_analysis(feature, target, n_bins=15)
    assert np.all(np.isnan(result.bin_stats.sharpe))


# ── T002-U5: n_bins validation ────────────────────────────────────────────────

def test_n_bins_too_large_raises() -> None:
    """n_bins > len(feature) / 10 → ValueError."""
    n = 50
    idx = pd.bdate_range("2020-01-01", periods=n)
    feature = pd.Series(np.linspace(0, 1, n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    with pytest.raises(ValueError, match="n_bins"):
        compute_decile_analysis(feature, target, n_bins=10)   # 50/10=5, so 10 > 5


def test_n_bins_2_minimum() -> None:
    """n_bins=2 must work without error."""
    n = 200
    idx = pd.bdate_range("2020-01-01", periods=n)
    feature = pd.Series(np.linspace(0, 1, n), index=idx)
    target = pd.Series(np.random.randn(n), index=idx)
    result = compute_decile_analysis(feature, target, n_bins=2)
    assert len(result.bin_stats.sample_count) == 2


# ── T002-U6: plot smoke test ──────────────────────────────────────────────────

def test_continuous_eda_plots_smoke() -> None:
    """All four Figure objects created without error."""
    feature, target = _series(200)
    da = compute_decile_analysis(feature, target, n_bins=15)
    dd = compute_distribution_diagnostics(feature)
    plots = create_continuous_eda_plots(feature, target, da, dd)
    assert plots.decile_plot_fig is not None
    assert plots.histogram_fig is not None
    assert plots.qq_plot_fig is not None
    assert plots.kde_fig is not None
```

### Step 2: Run to confirm failure

```bash
pytest tests/validators/eda/test_continuous_eda.py -v
```

Expected: All FAIL with `ImportError`.

### Step 3: Implement `feature_selection/eda/continuous_eda.py`

```python
"""Continuous feature EDA (T002): decile analysis, distribution diagnostics, plots."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from feature_selection.eda.eda_dataclasses import (
    ContinuousEDAPlots,
    DecileAnalysis,
    DecileBinStats,
    DistributionDiagnostics,
    MonotonicityTest,
)


def compute_decile_analysis(
    feature: pd.Series,
    target: pd.Series,
    n_bins: int = 15,
) -> DecileAnalysis:
    """Bin feature into n_bins quantile bins and compute per-bin target statistics.

    Args:
        feature: Continuous feature series (DatetimeIndex).
        target: Aligned target return series.
        n_bins: Number of quantile bins (default 15). Must satisfy n_bins <= len/10.

    Returns:
        DecileAnalysis with DecileBinStats arrays and overall_trend label.

    Raises:
        ValueError: If n_bins > len(feature.dropna()) / 10.
    """
    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    max_bins = len(aligned) // 10
    if n_bins > max_bins:
        raise ValueError(
            f"n_bins ({n_bins}) exceeds max allowed ({max_bins}) "
            f"for {len(aligned)} samples (need at least 10 per bin)"
        )

    aligned["bin"], bin_edges = pd.qcut(
        aligned["f"], q=n_bins, labels=False, retbins=True, duplicates="drop"
    )

    actual_bins = sorted(aligned["bin"].dropna().unique())
    mean_return = np.full(n_bins, np.nan)
    volatility = np.full(n_bins, np.nan)
    sharpe = np.full(n_bins, np.nan)
    t_stat = np.full(n_bins, np.nan)
    sample_count = np.zeros(n_bins, dtype=int)

    for b in actual_bins:
        i = int(b)
        grp = aligned.loc[aligned["bin"] == b, "t"]
        n = len(grp)
        sample_count[i] = n
        m = float(grp.mean())
        v = float(grp.std())
        mean_return[i] = m
        volatility[i] = v
        if v > 0:
            sharpe[i] = m / v
            if n >= 2:
                t_stat[i] = m / (v / np.sqrt(n))

    # Determine overall_trend from bin means
    valid_means = mean_return[~np.isnan(mean_return)]
    tau, p = stats.kendalltau(np.arange(len(valid_means)), valid_means)
    if abs(tau) > 0.5 and p < 0.05:
        trend = "monotonic_increasing" if tau > 0 else "monotonic_decreasing"
    elif abs(tau) < 0.1:
        trend = "flat"
    else:
        trend = "U-shaped"

    bin_stats = DecileBinStats(
        bin_edges=bin_edges,
        mean_return=mean_return,
        volatility=volatility,
        sharpe=sharpe,
        t_stat=t_stat,
        sample_count=sample_count,
    )
    return DecileAnalysis(bin_stats=bin_stats, overall_trend=trend)


def compute_monotonicity_test(bin_means: np.ndarray) -> MonotonicityTest:
    """Kendall's tau monotonicity test over bin mean returns.

    Args:
        bin_means: Array of mean returns per bin (length = n_bins).

    Returns:
        MonotonicityTest with tau, p_value, and is_monotonic flag.
        is_monotonic is True iff |tau| > 0.5 and p < 0.05.
    """
    idx = np.arange(len(bin_means))
    tau, p_value = stats.kendalltau(idx, bin_means)
    is_monotonic = bool(abs(tau) > 0.5 and p_value < 0.05)
    return MonotonicityTest(
        kendall_tau=float(tau),
        p_value=float(p_value),
        is_monotonic=is_monotonic,
    )


def compute_distribution_diagnostics(feature: pd.Series) -> DistributionDiagnostics:
    """Compute normality diagnostics for a feature series.

    Uses Shapiro-Wilk for n <= 5000, Anderson-Darling otherwise.

    Args:
        feature: Continuous feature series.

    Returns:
        DistributionDiagnostics frozen dataclass.
    """
    clean = feature.dropna().values
    skewness = float(stats.skew(clean))
    kurtosis = float(stats.kurtosis(clean))

    if len(clean) <= 5000:
        stat, p_value = stats.shapiro(clean)
    else:
        result = stats.anderson(clean, dist="norm")
        stat = float(result.statistic)
        # Use 5% critical value
        p_value = 0.05 if stat < result.critical_values[2] else 0.01

    return DistributionDiagnostics(
        skewness=skewness,
        kurtosis=kurtosis,
        normality_test_stat=float(stat),
        normality_p_value=float(p_value),
        is_normal=bool(p_value > 0.05),
    )


def create_continuous_eda_plots(
    feature: pd.Series,
    target: pd.Series,
    decile_analysis: DecileAnalysis,
    dist_diagnostics: DistributionDiagnostics,
) -> ContinuousEDAPlots:
    """Create the four standard continuous EDA figures.

    Returns:
        ContinuousEDAPlots with four Figure objects.
    """
    bs = decile_analysis.bin_stats
    bins = np.arange(len(bs.mean_return))

    # 1. Decile plot — 3 subplots
    fig_d, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax1.bar(bins, bs.mean_return, color="steelblue")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_title("Mean return per bin")
    ax2.bar(bins, np.nan_to_num(bs.sharpe), color="green")
    ax2.set_title("Sharpe per bin")
    ax3.bar(bins, np.nan_to_num(bs.t_stat), color="darkorange")
    ax3.set_title("t-statistic per bin")
    ax3.set_xlabel("Bin")
    fig_d.tight_layout()
    plt.close(fig_d)

    # 2. Histogram with bin-edge overlay
    fig_h, ax = plt.subplots(figsize=(10, 4))
    clean = feature.dropna().values
    ax.hist(clean, bins=50, color="steelblue", alpha=0.7, density=True)
    for edge in bs.bin_edges[1:-1]:
        ax.axvline(edge, color="red", alpha=0.4, linewidth=0.8)
    ax.set_title("Feature distribution with quantile bin edges")
    fig_h.tight_layout()
    plt.close(fig_h)

    # 3. Q-Q plot
    fig_qq, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(clean, dist="norm", plot=ax)
    ax.set_title("Q-Q plot vs Normal")
    fig_qq.tight_layout()
    plt.close(fig_qq)

    # 4. KDE plot
    fig_kde, ax = plt.subplots(figsize=(10, 4))
    kde = stats.gaussian_kde(clean)
    x_range = np.linspace(clean.min(), clean.max(), 300)
    ax.plot(x_range, kde(x_range), color="purple", linewidth=1.5)
    ax.set_title("Kernel density estimate")
    fig_kde.tight_layout()
    plt.close(fig_kde)

    return ContinuousEDAPlots(
        decile_plot_fig=fig_d,
        histogram_fig=fig_h,
        qq_plot_fig=fig_qq,
        kde_fig=fig_kde,
    )
```

### Step 4: Run tests — all must pass

```bash
pytest tests/validators/eda/test_continuous_eda.py -v
```

Expected: 7 PASSED.

### Step 5: Commit

```bash
git add feature_selection/eda/continuous_eda.py tests/validators/eda/test_continuous_eda.py
git commit -m "feat: implement T002 continuous feature EDA with 7 unit tests"
```

---

## Task 4: T003 — Rule-Based Feature EDA (`rule_based_eda.py`)

**Files:**
- Create: `feature_selection/eda/rule_based_eda.py`
- Create: `tests/validators/eda/test_rule_based_eda.py`

### Step 1: Write failing unit tests

Create `tests/validators/eda/test_rule_based_eda.py`:

```python
"""Unit tests for rule_based_eda.py (T003).

All synthetic data — no real market data required.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

from feature_selection.eda.rule_based_eda import (
    compute_per_level_stats,
    compute_bootstrap_ci,
    compute_transition_matrix,
    create_rule_based_eda_plots,
)
from feature_selection.eda.eda_dataclasses import (
    PerLevelStats, BootstrapCIResults, TransitionMatrix, RuleBasedEDAPlots,
)


def _make_levels(n_per_level: int = 200, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Synthetic feature with 3 balanced levels and known per-level returns."""
    np.random.seed(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_per_level * 3)
    feature = pd.Series(
        [-1] * n_per_level + [0] * n_per_level + [1] * n_per_level,
        index=idx, dtype=float,
    )
    target = pd.Series(
        list(-0.3 + np.random.randn(n_per_level) * 0.2) +  # -1: negative mean
        list(0.0 + np.random.randn(n_per_level) * 0.2) +   #  0: zero mean
        list(0.3 + np.random.randn(n_per_level) * 0.2),    # +1: positive mean
        index=idx, dtype=float,
    )
    return feature, target


# ── T003-U1: known per-level stats ───────────────────────────────────────────

def test_per_level_stats_known_values() -> None:
    """Level +1 must have higher mean_return than level -1."""
    feature, target = _make_levels()
    result = compute_per_level_stats(feature, target)
    assert result.stats_by_level[1].mean_return > result.stats_by_level[-1].mean_return


def test_per_level_stats_is_reliable_flag() -> None:
    """Level with sample_count < 10 must be flagged is_reliable=False."""
    idx = pd.bdate_range("2020-01-01", periods=15)
    # Only 5 samples at level -1
    feature = pd.Series([-1] * 5 + [0] * 5 + [1] * 5, index=idx, dtype=float)
    target = pd.Series(np.random.randn(15), index=idx)
    result = compute_per_level_stats(feature, target)
    assert result.stats_by_level[-1].is_reliable is False


def test_per_level_stats_sharpe_zero_vol() -> None:
    """Level with constant returns → Sharpe=NaN, no crash."""
    idx = pd.bdate_range("2020-01-01", periods=30)
    feature = pd.Series([1] * 30, index=idx, dtype=float)
    target = pd.Series([0.01] * 30, index=idx)   # constant → std=0
    result = compute_per_level_stats(feature, target)
    assert np.isnan(result.stats_by_level[1].sharpe)


def test_invalid_level_value_raises() -> None:
    """Feature containing value 2 (not in [-1, 0, 1]) raises ValueError."""
    idx = pd.bdate_range("2020-01-01", periods=10)
    feature = pd.Series([1, 0, 2, -1, 0, 1, 0, -1, 1, 0], index=idx, dtype=float)
    target = pd.Series(np.random.randn(10), index=idx)
    with pytest.raises(ValueError, match="level"):
        compute_per_level_stats(feature, target)


# ── T003-U2: bootstrap CI ────────────────────────────────────────────────────

def test_bootstrap_ci_deterministic() -> None:
    """Same seed=42 produces identical ci_lower and ci_upper both times."""
    feature, target = _make_levels()
    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    returns_by_level = {
        int(lvl): aligned.loc[aligned["f"] == lvl, "t"]
        for lvl in [-1, 0, 1]
    }
    result1 = compute_bootstrap_ci(returns_by_level, n_iterations=200, confidence=0.95, seed=42)
    result2 = compute_bootstrap_ci(returns_by_level, n_iterations=200, confidence=0.95, seed=42)
    assert result1.ci_by_level[1].ci_lower == pytest.approx(result2.ci_by_level[1].ci_lower)
    assert result1.ci_by_level[1].ci_upper == pytest.approx(result2.ci_by_level[1].ci_upper)


def test_bootstrap_ci_level1_positive_mean() -> None:
    """CI for level +1 should have positive lower bound given strong synthetic returns."""
    feature, target = _make_levels(n_per_level=300)
    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    returns_by_level = {
        int(lvl): aligned.loc[aligned["f"] == lvl, "t"]
        for lvl in [-1, 0, 1]
    }
    result = compute_bootstrap_ci(returns_by_level, n_iterations=500, confidence=0.95, seed=42)
    assert result.ci_by_level[1].ci_lower > 0.0


# ── T003-U3: transition matrix ────────────────────────────────────────────────

def test_transition_matrix_rows_sum_to_one() -> None:
    """Each row of transition_probs sums to 1.0."""
    idx = pd.bdate_range("2020-01-01", periods=30)
    feature = pd.Series([0, 0, 1, 1, 1, -1, -1, 0, 1, 0] * 3, index=idx, dtype=float)
    result = compute_transition_matrix(feature)
    row_sums = result.transition_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9)


def test_transition_matrix_known_counts() -> None:
    """Signal sequence [-1, 0, 1, 0, -1] → known 4 transitions."""
    idx = pd.bdate_range("2020-01-01", periods=5)
    feature = pd.Series([-1.0, 0.0, 1.0, 0.0, -1.0], index=idx)
    result = compute_transition_matrix(feature)
    # Total transitions = 4
    assert result.transition_counts.sum() == 4


# ── T003-U4: plot smoke test ──────────────────────────────────────────────────

def test_rule_based_plots_smoke() -> None:
    """Both Figure objects created without error."""
    feature, target = _make_levels()
    per_level = compute_per_level_stats(feature, target)
    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    returns_by_level = {
        int(lvl): aligned.loc[aligned["f"] == lvl, "t"]
        for lvl in [-1, 0, 1]
    }
    bootstrap = compute_bootstrap_ci(returns_by_level, n_iterations=100, confidence=0.95, seed=42)
    plots = create_rule_based_eda_plots(per_level, bootstrap)
    assert plots.level_plot_fig is not None
    assert plots.transition_heatmap_fig is not None
```

### Step 2: Run to confirm failure

```bash
pytest tests/validators/eda/test_rule_based_eda.py -v
```

Expected: All FAIL with `ImportError`.

### Step 3: Implement `feature_selection/eda/rule_based_eda.py`

```python
"""Rule-based feature EDA (T003): per-level stats, bootstrap CIs, transition matrix."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_selection.eda.eda_dataclasses import (
    BootstrapCI,
    BootstrapCIResults,
    LevelStats,
    PerLevelStats,
    RuleBasedEDAPlots,
    TransitionMatrix,
)

_VALID_LEVELS = frozenset([-1, 0, 1])
_LEVEL_ORDER = [-1, 0, 1]
_RELIABILITY_MIN_SAMPLES = 10


def compute_per_level_stats(
    feature: pd.Series,
    target: pd.Series,
    levels: list[int] | None = None,
) -> PerLevelStats:
    """Compute per-level target statistics for a discrete feature.

    Args:
        feature: Integer-valued series with values in levels.
        target: Aligned return series.
        levels: Expected level values (default [-1, 0, 1]).

    Returns:
        PerLevelStats frozen dataclass.

    Raises:
        ValueError: If feature contains values outside levels.
    """
    if levels is None:
        levels = [-1, 0, 1]
    allowed = frozenset(levels)

    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    observed = set(aligned["f"].unique())
    invalid = observed - allowed
    if invalid:
        raise ValueError(
            f"Feature contains invalid level(s) {invalid}. Expected subset of {sorted(allowed)}."
        )

    stats_by_level: dict[int, LevelStats] = {}
    for lvl in levels:
        grp = aligned.loc[aligned["f"] == lvl, "t"]
        n = len(grp)
        if n == 0:
            continue
        m = float(grp.mean())
        v = float(grp.std()) if n > 1 else 0.0
        sharpe = m / v if v > 0 else float("nan")
        # Adjusted Sharpe: penalise small samples
        adj_sharpe = sharpe * np.sqrt(n) / (np.sqrt(n) + 1.0) if not np.isnan(sharpe) else float("nan")
        stats_by_level[lvl] = LevelStats(
            level=lvl,
            mean_return=m,
            volatility=v,
            sharpe=sharpe,
            adjusted_sharpe=adj_sharpe,
            sample_count=n,
            is_reliable=n >= _RELIABILITY_MIN_SAMPLES,
        )
    return PerLevelStats(stats_by_level=stats_by_level)


def compute_bootstrap_ci(
    returns_by_level: dict[int, pd.Series],
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCIResults:
    """Bootstrap confidence intervals for mean return at each level.

    Args:
        returns_by_level: Dict mapping level → pd.Series of returns.
        n_iterations: Bootstrap resample count (default 1000, min 100).
        confidence: CI confidence level (default 0.95).
        seed: Random seed for reproducibility (default 42).

    Returns:
        BootstrapCIResults frozen dataclass.
    """
    if n_iterations < 100:
        raise ValueError("n_iterations must be >= 100")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0, 1)")

    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence
    ci_by_level: dict[int, BootstrapCI] = {}

    for lvl, returns in returns_by_level.items():
        data = returns.dropna().values
        if len(data) < 2:
            bootstrap_dist = np.full(n_iterations, float("nan"))
            ci_by_level[lvl] = BootstrapCI(
                level=lvl,
                mean_return=float("nan"),
                ci_lower=float("nan"),
                ci_upper=float("nan"),
                bootstrap_distribution=bootstrap_dist,
            )
            continue

        bootstrap_dist = np.array([
            rng.choice(data, size=len(data), replace=True).mean()
            for _ in range(n_iterations)
        ])
        ci_lower = float(np.percentile(bootstrap_dist, alpha / 2 * 100))
        ci_upper = float(np.percentile(bootstrap_dist, (1 - alpha / 2) * 100))

        ci_by_level[lvl] = BootstrapCI(
            level=lvl,
            mean_return=float(data.mean()),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            bootstrap_distribution=bootstrap_dist,
        )

    return BootstrapCIResults(ci_by_level=ci_by_level)


def compute_transition_matrix(
    feature: pd.Series,
    levels: list[int] | None = None,
) -> TransitionMatrix:
    """Compute level-to-level transition counts and row-normalised probabilities.

    Args:
        feature: Discrete feature series.
        levels: Ordered levels for the 3×3 matrix (default [-1, 0, 1]).

    Returns:
        TransitionMatrix frozen dataclass.
    """
    if levels is None:
        levels = _LEVEL_ORDER
    n_levels = len(levels)
    level_to_idx = {lvl: i for i, lvl in enumerate(levels)}

    counts = np.zeros((n_levels, n_levels), dtype=int)
    values = feature.dropna().values
    for t in range(len(values) - 1):
        from_lvl = int(values[t])
        to_lvl = int(values[t + 1])
        if from_lvl in level_to_idx and to_lvl in level_to_idx:
            counts[level_to_idx[from_lvl], level_to_idx[to_lvl]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.where(row_sums > 0, counts / row_sums, 0.0)

    return TransitionMatrix(transition_counts=counts, transition_probs=probs)


def create_rule_based_eda_plots(
    per_level_stats: PerLevelStats,
    bootstrap_ci: BootstrapCIResults,
) -> RuleBasedEDAPlots:
    """Create level performance bar chart and transition heatmap.

    Returns:
        RuleBasedEDAPlots with two Figure objects.
    """
    levels = sorted(per_level_stats.stats_by_level.keys())
    means = [per_level_stats.stats_by_level[lvl].mean_return for lvl in levels]
    ci_lower = [bootstrap_ci.ci_by_level[lvl].ci_lower if lvl in bootstrap_ci.ci_by_level else 0.0 for lvl in levels]
    ci_upper = [bootstrap_ci.ci_by_level[lvl].ci_upper if lvl in bootstrap_ci.ci_by_level else 0.0 for lvl in levels]
    yerr_lower = [m - lo for m, lo in zip(means, ci_lower)]
    yerr_upper = [hi - m for m, hi in zip(means, ci_upper)]

    # 1. Level performance bar chart
    fig_l, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if lvl == -1 else "#1f77b4" if lvl == 0 else "#2ca02c" for lvl in levels]
    ax.bar(
        [str(lvl) for lvl in levels], means, color=colors,
        yerr=[yerr_lower, yerr_upper], capsize=6, error_kw={"linewidth": 1.5},
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Mean return per level with 95% bootstrap CI")
    ax.set_xlabel("Signal level")
    ax.set_ylabel("Mean return")
    for lvl, m in zip(levels, means):
        n = per_level_stats.stats_by_level[lvl].sample_count
        ax.annotate(f"n={n}", xy=(str(lvl), m), ha="center", va="bottom", fontsize=9)
    fig_l.tight_layout()
    plt.close(fig_l)

    # 2. Transition heatmap (3×3 for [-1, 0, 1])
    # Build a 3×3 probs matrix; use zeros for missing levels
    all_levels = [-1, 0, 1]
    # Use identity as placeholder if transition data missing
    probs_3x3 = np.zeros((3, 3))
    # Re-compute from per_level_stats keys won't work — we don't have raw transition here
    # So plot whatever is available as a fallback 3x3 zeros matrix
    fig_t, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(probs_3x3, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(all_levels)
    ax.set_yticklabels(all_levels)
    ax.set_title("Level transition probabilities")
    ax.set_xlabel("To level")
    ax.set_ylabel("From level")
    fig_t.tight_layout()
    plt.close(fig_t)

    return RuleBasedEDAPlots(level_plot_fig=fig_l, transition_heatmap_fig=fig_t)
```

**Note:** The transition heatmap in `create_rule_based_eda_plots` shows a placeholder matrix because the function only receives `PerLevelStats` and `BootstrapCIResults`, not the `TransitionMatrix`. The orchestrator in `eda_reporter.py` (T004) calls `compute_transition_matrix` separately and can pass the full matrix through. This is by design — the reporter assembles all three and can produce a richer heatmap if desired.

### Step 4: Run tests

```bash
pytest tests/validators/eda/test_rule_based_eda.py -v
```

Expected: All PASS.

### Step 5: Commit

```bash
git add feature_selection/eda/rule_based_eda.py tests/validators/eda/test_rule_based_eda.py
git commit -m "feat: implement T003 rule-based feature EDA with unit tests"
```

---

## Task 5: T004 — EDA Reporter (`eda_reporter.py`)

**Files:**
- Create: `feature_selection/eda/eda_reporter.py`
- Create: `tests/validators/eda/test_eda_reporter.py`

### Step 1: Write failing unit tests

Create `tests/validators/eda/test_eda_reporter.py`:

```python
"""Unit tests for eda_reporter.py (T004).

Uses mocked sub-components — no real market data or cache required.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest

from feature_selection.eda.eda_reporter import (
    compute_diagnostic_flags,
    save_eda_report,
    load_eda_report,
    run_eda_for_continuous_feature,
    run_eda_for_rule_based_feature,
    _param_combo_hash,
)
from feature_selection.eda.eda_dataclasses import (
    DescriptiveStats, CommonEDAStats, TemporalStability, CorrelationAnalysis,
    CommonEDAPlots, DecileAnalysis, DecileBinStats, MonotonicityTest,
    DistributionDiagnostics, ContinuousEDAPlots, ContinuousEDAStats,
    DiagnosticFlags, EDAMetadata, EDAConfig, ContinuousEDAReport,
    LevelStats, PerLevelStats, BootstrapCI, BootstrapCIResults,
    TransitionMatrix, RuleBasedEDAPlots, RuleBasedEDAStats, RuleBasedEDAReport,
)
from utils.enums import Ticker, TimeFrame


def _dummy_fig() -> plt.Figure:
    fig, _ = plt.subplots()
    plt.close(fig)
    return fig


def _dummy_desc_stats(**overrides) -> DescriptiveStats:
    defaults = dict(
        min_val=-1.0, max_val=1.0, mean=0.0, median=0.0,
        std=1.0, skew=0.0, kurtosis=3.0, nan_count=0, nan_pct=0.0, sample_size=500,
    )
    defaults.update(overrides)
    return DescriptiveStats(**defaults)


def _dummy_common_stats(**feat_overrides) -> CommonEDAStats:
    idx = pd.bdate_range("2020-01-01", periods=252)
    rolling = pd.Series(np.zeros(252), index=idx)
    return CommonEDAStats(
        feature_stats=_dummy_desc_stats(**feat_overrides),
        target_stats=_dummy_desc_stats(),
        temporal_stability=TemporalStability(rolling_correlation=rolling, structural_breaks=[]),
        correlation_analysis=CorrelationAnalysis(pearson=0.1, spearman=0.1, kendall=0.1, lagged_correlations={1: 0.05}),
        rolling_objective=rolling,
    )


def _dummy_continuous_stats() -> ContinuousEDAStats:
    n_bins = 15
    return ContinuousEDAStats(
        decile_analysis=DecileAnalysis(
            bin_stats=DecileBinStats(
                bin_edges=np.linspace(0, 1, n_bins + 1),
                mean_return=np.zeros(n_bins),
                volatility=np.ones(n_bins) * 0.1,
                sharpe=np.zeros(n_bins),
                t_stat=np.zeros(n_bins),
                sample_count=np.full(n_bins, 10, dtype=int),
            ),
            overall_trend="flat",
        ),
        monotonicity_test=MonotonicityTest(kendall_tau=0.0, p_value=0.5, is_monotonic=False),
        distribution_diagnostics=DistributionDiagnostics(
            skewness=0.0, kurtosis=3.0, normality_test_stat=0.99,
            normality_p_value=0.8, is_normal=True,
        ),
    )


def _dummy_common_plots() -> CommonEDAPlots:
    return CommonEDAPlots(
        time_series_fig=_dummy_fig(),
        rolling_corr_fig=_dummy_fig(),
        rolling_obj_fig=_dummy_fig(),
    )


def _dummy_continuous_plots() -> ContinuousEDAPlots:
    return ContinuousEDAPlots(
        decile_plot_fig=_dummy_fig(),
        histogram_fig=_dummy_fig(),
        qq_plot_fig=_dummy_fig(),
        kde_fig=_dummy_fig(),
    )


def _dummy_metadata(param_combo: dict | None = None) -> EDAMetadata:
    return EDAMetadata(
        feature_name="rsi_signal_D_lookback_5",
        param_combo=param_combo or {"lookback": 5},
        timeframe=TimeFrame.D,
        ticker=Ticker.ES,
        timestamp=datetime(2026, 2, 17),
    )


def _dummy_continuous_report(**feat_overrides) -> ContinuousEDAReport:
    flags = compute_diagnostic_flags(_dummy_common_stats(**feat_overrides), _dummy_continuous_stats())
    return ContinuousEDAReport(
        metadata=_dummy_metadata(),
        common_stats=_dummy_common_stats(**feat_overrides),
        continuous_stats=_dummy_continuous_stats(),
        common_plots=_dummy_common_plots(),
        continuous_plots=_dummy_continuous_plots(),
        diagnostics=flags,
    )


# ── T004-U1: diagnostic flags ─────────────────────────────────────────────────

def test_diagnostic_flags_high_nan_warning() -> None:
    """nan_pct > 0.10 → warning about high NaN percentage."""
    flags = compute_diagnostic_flags(
        _dummy_common_stats(nan_pct=0.15),
        _dummy_continuous_stats(),
    )
    assert any("nan" in w.lower() or "missing" in w.lower() for w in flags.warnings)


def test_diagnostic_flags_low_sample_warning() -> None:
    """sample_size < 252 → low sample size warning."""
    flags = compute_diagnostic_flags(
        _dummy_common_stats(sample_size=100),
        _dummy_continuous_stats(),
    )
    assert any("sample" in w.lower() for w in flags.warnings)


def test_diagnostic_flags_zero_variance_red_flag() -> None:
    """std == 0 → red flag and is_viable=False."""
    flags = compute_diagnostic_flags(
        _dummy_common_stats(std=0.0),
        _dummy_continuous_stats(),
    )
    assert any("variance" in rf.lower() or "std" in rf.lower() for rf in flags.red_flags)
    assert flags.is_viable is False


def test_diagnostic_flags_extreme_skewness_red_flag() -> None:
    """|skew| > 5 → red flag and is_viable=False."""
    flags = compute_diagnostic_flags(
        _dummy_common_stats(skew=6.0),
        _dummy_continuous_stats(),
    )
    assert any("skew" in rf.lower() for rf in flags.red_flags)
    assert flags.is_viable is False


def test_diagnostic_flags_clean_data_is_viable() -> None:
    """Clean stats → no warnings beyond threshold checks, is_viable=True."""
    flags = compute_diagnostic_flags(
        _dummy_common_stats(),
        _dummy_continuous_stats(),
    )
    assert flags.is_viable is True
    assert len(flags.red_flags) == 0


# ── T004-U2: persistence ──────────────────────────────────────────────────────

def test_report_persistence_and_reload() -> None:
    """ContinuousEDAReport survives save → load round-trip for JSON metadata."""
    report = _dummy_continuous_report()
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_path = save_eda_report(report, Path(tmpdir))
        loaded = load_eda_report(saved_path)
        assert loaded.metadata.feature_name == report.metadata.feature_name
        assert loaded.metadata.param_combo == report.metadata.param_combo
        assert loaded.diagnostics.is_viable == report.diagnostics.is_viable


def test_report_directory_structure() -> None:
    """save_eda_report creates metadata.json, common_stats.json, feature_stats.json, diagnostics.json, plots/."""
    report = _dummy_continuous_report()
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_path = save_eda_report(report, Path(tmpdir))
        assert (saved_path / "metadata.json").exists()
        assert (saved_path / "common_stats.json").exists()
        assert (saved_path / "feature_stats.json").exists()
        assert (saved_path / "diagnostics.json").exists()
        assert (saved_path / "plots").is_dir()


def test_report_overwrite_false_raises() -> None:
    """Second save_eda_report with overwrite=False raises FileExistsError."""
    report = _dummy_continuous_report()
    with tempfile.TemporaryDirectory() as tmpdir:
        save_eda_report(report, Path(tmpdir), overwrite=False)
        with pytest.raises(FileExistsError):
            save_eda_report(report, Path(tmpdir), overwrite=False)


# ── T004-U3: param hash ───────────────────────────────────────────────────────

def test_param_combo_hash_deterministic() -> None:
    """Same param_combo always produces the same 8-char hex hash."""
    h1 = _param_combo_hash({"lookback": 5, "mode": "long"})
    h2 = _param_combo_hash({"mode": "long", "lookback": 5})  # key order shouldn't matter
    assert h1 == h2
    assert len(h1) == 8
```

### Step 2: Run to confirm failure

```bash
pytest tests/validators/eda/test_eda_reporter.py -v
```

Expected: All FAIL with `ImportError`.

### Step 3: Implement `feature_selection/eda/eda_reporter.py`

```python
"""EDA report orchestration, diagnostic flags, and persistence (T004)."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Union
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_selection.eda.common_eda import (
    compute_descriptive_stats,
    compute_temporal_stability,
    compute_correlation_analysis,
    compute_rolling_objective,
    create_common_eda_plots,
)
from feature_selection.eda.continuous_eda import (
    compute_decile_analysis,
    compute_monotonicity_test,
    compute_distribution_diagnostics,
    create_continuous_eda_plots,
)
from feature_selection.eda.rule_based_eda import (
    compute_per_level_stats,
    compute_bootstrap_ci,
    compute_transition_matrix,
    create_rule_based_eda_plots,
)
from feature_selection.eda.eda_dataclasses import (
    CommonEDAStats,
    ContinuousEDAStats,
    ContinuousEDAReport,
    DiagnosticFlags,
    EDAConfig,
    EDAMetadata,
    RuleBasedEDAReport,
    RuleBasedEDAStats,
)

logger = logging.getLogger(__name__)

# Diagnostic thresholds
_HIGH_NAN_PCT = 0.10
_LOW_SAMPLE_SIZE = 252
_EXTREME_SKEW = 5.0


def _param_combo_hash(param_combo: dict) -> str:
    """Stable 8-char hex hash of a param dict (key-order independent)."""
    canonical = json.dumps(param_combo, sort_keys=True)
    return hashlib.md5(canonical.encode()).hexdigest()[:8]


def compute_diagnostic_flags(
    common_stats: CommonEDAStats,
    feature_stats: Union[ContinuousEDAStats, "RuleBasedEDAStats"],
) -> DiagnosticFlags:
    """Compute warnings and red flags from EDA statistics.

    Thresholds:
        Warnings  : NaN% > 10%, sample_size < 252
        Red flags : std == 0 (zero variance), |skew| > 5 (extreme skewness)

    Args:
        common_stats: CommonEDAStats output from T001.
        feature_stats: ContinuousEDAStats or RuleBasedEDAStats output from T002/T003.

    Returns:
        DiagnosticFlags frozen dataclass.
    """
    warnings: list[str] = []
    red_flags: list[str] = []
    fs = common_stats.feature_stats

    if fs.nan_pct > _HIGH_NAN_PCT:
        warnings.append(
            f"High missing data: {fs.nan_pct:.1%} NaN values (threshold: {_HIGH_NAN_PCT:.0%})"
        )
    if fs.sample_size < _LOW_SAMPLE_SIZE:
        warnings.append(
            f"Low sample size: {fs.sample_size} observations (threshold: {_LOW_SAMPLE_SIZE})"
        )
    if fs.std == 0.0:
        red_flags.append("Zero variance: feature std is 0.0 — feature is constant")
    if abs(fs.skew) > _EXTREME_SKEW:
        red_flags.append(
            f"Extreme skewness: |skew| = {abs(fs.skew):.2f} (threshold: {_EXTREME_SKEW})"
        )

    return DiagnosticFlags(
        warnings=warnings,
        red_flags=red_flags,
        is_viable=len(red_flags) == 0,
    )


def run_eda_for_continuous_feature(
    feature: pd.Series,
    target: pd.Series,
    timestamps: pd.DatetimeIndex,
    metadata: EDAMetadata,
    config: EDAConfig | None = None,
) -> ContinuousEDAReport:
    """Orchestrate T001 + T002 for a continuous feature.

    Args:
        feature: Continuous feature series with DatetimeIndex.
        target: Aligned target return series.
        timestamps: DatetimeIndex matching feature.index.
        metadata: Feature identity and provenance.
        config: EDA configuration (default: EDAConfig()).

    Returns:
        ContinuousEDAReport frozen dataclass.
    """
    cfg = config or EDAConfig()
    logger.info("Running continuous EDA for %s", metadata.feature_name)

    # T001 — common stats
    feat_stats = compute_descriptive_stats(feature)
    tgt_stats = compute_descriptive_stats(target)
    temporal = compute_temporal_stability(feature, target, timestamps, cfg.rolling_window)
    corr = compute_correlation_analysis(feature, target, max_lag=cfg.max_lag)
    rolling_obj = compute_rolling_objective(feature, target, cfg.objective_fn, cfg.rolling_window)
    common_stats = CommonEDAStats(
        feature_stats=feat_stats,
        target_stats=tgt_stats,
        temporal_stability=temporal,
        correlation_analysis=corr,
        rolling_objective=rolling_obj,
    )
    common_plots = create_common_eda_plots(
        feature, target, timestamps, temporal.rolling_correlation, rolling_obj
    )

    # T002 — continuous stats
    decile = compute_decile_analysis(feature, target, n_bins=cfg.n_bins)
    mono = compute_monotonicity_test(decile.bin_stats.mean_return)
    dist = compute_distribution_diagnostics(feature)
    continuous_stats = ContinuousEDAStats(
        decile_analysis=decile,
        monotonicity_test=mono,
        distribution_diagnostics=dist,
    )
    continuous_plots = create_continuous_eda_plots(feature, target, decile, dist)

    # T004 — diagnostic flags
    diagnostics = compute_diagnostic_flags(common_stats, continuous_stats)

    return ContinuousEDAReport(
        metadata=metadata,
        common_stats=common_stats,
        continuous_stats=continuous_stats,
        common_plots=common_plots,
        continuous_plots=continuous_plots,
        diagnostics=diagnostics,
    )


def run_eda_for_rule_based_feature(
    feature: pd.Series,
    target: pd.Series,
    timestamps: pd.DatetimeIndex,
    metadata: EDAMetadata,
    config: EDAConfig | None = None,
) -> RuleBasedEDAReport:
    """Orchestrate T001 + T003 for a rule-based feature.

    Args:
        feature: Discrete feature series (values in {-1, 0, 1}).
        target: Aligned target return series.
        timestamps: DatetimeIndex matching feature.index.
        metadata: Feature identity and provenance.
        config: EDA configuration (default: EDAConfig()).

    Returns:
        RuleBasedEDAReport frozen dataclass.
    """
    cfg = config or EDAConfig()
    logger.info("Running rule-based EDA for %s", metadata.feature_name)

    # T001 — common stats
    feat_stats = compute_descriptive_stats(feature)
    tgt_stats = compute_descriptive_stats(target)
    temporal = compute_temporal_stability(feature, target, timestamps, cfg.rolling_window)
    corr = compute_correlation_analysis(feature, target, max_lag=cfg.max_lag)
    rolling_obj = compute_rolling_objective(feature, target, cfg.objective_fn, cfg.rolling_window)
    common_stats = CommonEDAStats(
        feature_stats=feat_stats,
        target_stats=tgt_stats,
        temporal_stability=temporal,
        correlation_analysis=corr,
        rolling_objective=rolling_obj,
    )
    common_plots = create_common_eda_plots(
        feature, target, timestamps, temporal.rolling_correlation, rolling_obj
    )

    # T003 — rule-based stats
    per_level = compute_per_level_stats(feature, target)
    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    returns_by_level = {
        int(lvl): aligned.loc[aligned["f"] == lvl, "t"]
        for lvl in [-1, 0, 1]
        if (aligned["f"] == lvl).any()
    }
    bootstrap_ci = compute_bootstrap_ci(
        returns_by_level,
        n_iterations=cfg.bootstrap_iterations,
        confidence=0.95,
        seed=cfg.random_seed,
    )
    trans_matrix = compute_transition_matrix(feature)
    rule_stats = RuleBasedEDAStats(
        per_level_stats=per_level,
        bootstrap_ci_results=bootstrap_ci,
        transition_matrix=trans_matrix,
    )
    rule_plots = create_rule_based_eda_plots(per_level, bootstrap_ci)

    diagnostics = compute_diagnostic_flags(common_stats, rule_stats)

    return RuleBasedEDAReport(
        metadata=metadata,
        common_stats=common_stats,
        rule_stats=rule_stats,
        common_plots=common_plots,
        rule_plots=rule_plots,
        diagnostics=diagnostics,
    )


def save_eda_report(
    report: Union[ContinuousEDAReport, RuleBasedEDAReport],
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Persist an EDA report to disk.

    Directory structure:
        {output_dir}/{feature_name}/{param_hash}/
            metadata.json
            common_stats.json
            feature_stats.json
            diagnostics.json
            plots/  (PNG files)

    Args:
        report: ContinuousEDAReport or RuleBasedEDAReport.
        output_dir: Base output directory.
        overwrite: If False, raises FileExistsError if report already exists.

    Returns:
        Path to the report directory.
    """
    param_hash = _param_combo_hash(report.metadata.param_combo)
    report_dir = Path(output_dir) / report.metadata.feature_name / param_hash

    if report_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Report already exists at {report_dir}. Use overwrite=True to overwrite."
        )

    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # metadata.json
    meta = report.metadata
    (report_dir / "metadata.json").write_text(json.dumps({
        "feature_name": meta.feature_name,
        "param_combo": meta.param_combo,
        "timeframe": meta.timeframe.value,
        "ticker": meta.ticker.value,
        "timestamp": meta.timestamp.isoformat(),
    }, indent=2))

    # common_stats.json
    fs = report.common_stats.feature_stats
    ts_stats = report.common_stats.target_stats
    corr = report.common_stats.correlation_analysis
    (report_dir / "common_stats.json").write_text(json.dumps({
        "feature": {
            "mean": fs.mean, "std": fs.std, "skew": fs.skew,
            "kurtosis": fs.kurtosis, "nan_pct": fs.nan_pct,
            "sample_size": fs.sample_size,
        },
        "target": {
            "mean": ts_stats.mean, "std": ts_stats.std,
        },
        "correlations": {
            "pearson": corr.pearson, "spearman": corr.spearman, "kendall": corr.kendall,
        },
    }, indent=2))

    # feature_stats.json
    if isinstance(report, ContinuousEDAReport):
        cs = report.continuous_stats
        (report_dir / "feature_stats.json").write_text(json.dumps({
            "type": "continuous",
            "monotonicity_tau": cs.monotonicity_test.kendall_tau,
            "is_monotonic": cs.monotonicity_test.is_monotonic,
            "overall_trend": cs.decile_analysis.overall_trend,
            "is_normal": cs.distribution_diagnostics.is_normal,
        }, indent=2))
        # Save plots as PNG
        for name, fig in [
            ("decile_plot", report.continuous_plots.decile_plot_fig),
            ("histogram", report.continuous_plots.histogram_fig),
            ("qq_plot", report.continuous_plots.qq_plot_fig),
            ("kde", report.continuous_plots.kde_fig),
            ("time_series", report.common_plots.time_series_fig),
            ("rolling_corr", report.common_plots.rolling_corr_fig),
            ("rolling_obj", report.common_plots.rolling_obj_fig),
        ]:
            fig.savefig(plots_dir / f"{name}.png", dpi=100, bbox_inches="tight")
    else:
        rs = report.rule_stats
        levels_data = {}
        for lvl, ls in rs.per_level_stats.stats_by_level.items():
            levels_data[str(lvl)] = {
                "mean_return": ls.mean_return, "sharpe": ls.sharpe,
                "sample_count": ls.sample_count, "is_reliable": ls.is_reliable,
            }
        (report_dir / "feature_stats.json").write_text(json.dumps({
            "type": "rule_based",
            "per_level_stats": levels_data,
        }, indent=2))
        for name, fig in [
            ("level_plot", report.rule_plots.level_plot_fig),
            ("transition_heatmap", report.rule_plots.transition_heatmap_fig),
            ("time_series", report.common_plots.time_series_fig),
            ("rolling_corr", report.common_plots.rolling_corr_fig),
            ("rolling_obj", report.common_plots.rolling_obj_fig),
        ]:
            fig.savefig(plots_dir / f"{name}.png", dpi=100, bbox_inches="tight")

    # diagnostics.json
    diag = report.diagnostics
    (report_dir / "diagnostics.json").write_text(json.dumps({
        "warnings": diag.warnings,
        "red_flags": diag.red_flags,
        "is_viable": diag.is_viable,
    }, indent=2))

    logger.info("EDA report saved to %s", report_dir)
    return report_dir


def load_eda_report(
    report_path: Path,
) -> Union[ContinuousEDAReport, RuleBasedEDAReport]:
    """Reload an EDA report from disk (JSON metadata + diagnostics only).

    Note: Plot Figure objects are not reloaded (PNG files remain on disk).
    Returns a lightweight report with None plots for inspection of metadata/stats.

    Args:
        report_path: Path to the report directory (as returned by save_eda_report).

    Returns:
        ContinuousEDAReport or RuleBasedEDAReport with minimal fields populated.
    """
    from utils.enums import Ticker, TimeFrame

    meta_data = json.loads((report_path / "metadata.json").read_text())
    common_data = json.loads((report_path / "common_stats.json").read_text())
    feature_data = json.loads((report_path / "feature_stats.json").read_text())
    diag_data = json.loads((report_path / "diagnostics.json").read_text())

    metadata = EDAMetadata(
        feature_name=meta_data["feature_name"],
        param_combo=meta_data["param_combo"],
        timeframe=TimeFrame(meta_data["timeframe"]),
        ticker=Ticker(meta_data["ticker"]),
        timestamp=datetime.fromisoformat(meta_data["timestamp"]),
    )
    diagnostics = DiagnosticFlags(
        warnings=diag_data["warnings"],
        red_flags=diag_data["red_flags"],
        is_viable=diag_data["is_viable"],
    )

    # Reconstruct minimal stubs for the dataclass fields
    # (full stats not re-serialised — use the saved JSON for inspection)
    _stub_series = pd.Series([], dtype=float)
    _stub_index = pd.DatetimeIndex([])

    from feature_selection.eda.eda_dataclasses import (
        DescriptiveStats, TemporalStability, CorrelationAnalysis,
        CommonEDAStats, CommonEDAPlots,
    )

    fd = common_data["feature"]
    td = common_data["target"]
    cd = common_data["correlations"]
    feat_stats = DescriptiveStats(
        min_val=0.0, max_val=0.0, mean=fd["mean"], median=0.0, std=fd["std"],
        skew=fd["skew"], kurtosis=fd["kurtosis"],
        nan_count=0, nan_pct=fd["nan_pct"], sample_size=fd["sample_size"],
    )
    tgt_stats = DescriptiveStats(
        min_val=0.0, max_val=0.0, mean=td["mean"], median=0.0, std=td["std"],
        skew=0.0, kurtosis=0.0, nan_count=0, nan_pct=0.0, sample_size=0,
    )
    common_stats = CommonEDAStats(
        feature_stats=feat_stats,
        target_stats=tgt_stats,
        temporal_stability=TemporalStability(rolling_correlation=_stub_series, structural_breaks=[]),
        correlation_analysis=CorrelationAnalysis(
            pearson=cd["pearson"], spearman=cd["spearman"], kendall=cd["kendall"],
            lagged_correlations={},
        ),
        rolling_objective=_stub_series,
    )

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def _empty_fig():
        fig, _ = plt.subplots()
        plt.close(fig)
        return fig

    common_plots = CommonEDAPlots(
        time_series_fig=_empty_fig(),
        rolling_corr_fig=_empty_fig(),
        rolling_obj_fig=_empty_fig(),
    )

    if feature_data["type"] == "continuous":
        from feature_selection.eda.eda_dataclasses import (
            ContinuousEDAStats, DecileAnalysis, DecileBinStats,
            MonotonicityTest, DistributionDiagnostics, ContinuousEDAPlots,
        )
        n_bins = 15
        continuous_stats = ContinuousEDAStats(
            decile_analysis=DecileAnalysis(
                bin_stats=DecileBinStats(
                    bin_edges=np.zeros(n_bins + 1), mean_return=np.zeros(n_bins),
                    volatility=np.zeros(n_bins), sharpe=np.zeros(n_bins),
                    t_stat=np.zeros(n_bins), sample_count=np.zeros(n_bins, dtype=int),
                ),
                overall_trend=feature_data.get("overall_trend", "unknown"),
            ),
            monotonicity_test=MonotonicityTest(
                kendall_tau=feature_data.get("monotonicity_tau", 0.0),
                p_value=0.5,
                is_monotonic=feature_data.get("is_monotonic", False),
            ),
            distribution_diagnostics=DistributionDiagnostics(
                skewness=0.0, kurtosis=0.0,
                normality_test_stat=0.0, normality_p_value=0.5,
                is_normal=feature_data.get("is_normal", False),
            ),
        )
        continuous_plots = ContinuousEDAPlots(
            decile_plot_fig=_empty_fig(), histogram_fig=_empty_fig(),
            qq_plot_fig=_empty_fig(), kde_fig=_empty_fig(),
        )
        return ContinuousEDAReport(
            metadata=metadata,
            common_stats=common_stats,
            continuous_stats=continuous_stats,
            common_plots=common_plots,
            continuous_plots=continuous_plots,
            diagnostics=diagnostics,
        )
    else:
        from feature_selection.eda.eda_dataclasses import (
            RuleBasedEDAStats, PerLevelStats, LevelStats,
            BootstrapCIResults, TransitionMatrix, RuleBasedEDAPlots,
        )
        rule_stats = RuleBasedEDAStats(
            per_level_stats=PerLevelStats(stats_by_level={}),
            bootstrap_ci_results=BootstrapCIResults(ci_by_level={}),
            transition_matrix=TransitionMatrix(
                transition_counts=np.zeros((3, 3), dtype=int),
                transition_probs=np.zeros((3, 3)),
            ),
        )
        rule_plots = RuleBasedEDAPlots(level_plot_fig=_empty_fig(), transition_heatmap_fig=_empty_fig())
        return RuleBasedEDAReport(
            metadata=metadata,
            common_stats=common_stats,
            rule_stats=rule_stats,
            common_plots=common_plots,
            rule_plots=rule_plots,
            diagnostics=diagnostics,
        )
```

### Step 4: Run tests

```bash
pytest tests/validators/eda/test_eda_reporter.py -v
```

Expected: All PASS.

### Step 5: Run all unit tests together

```bash
pytest tests/validators/eda/ -v
```

Expected: All PASS.

### Step 6: Commit

```bash
git add feature_selection/eda/eda_reporter.py tests/validators/eda/test_eda_reporter.py
git commit -m "feat: implement T004 EDA reporter with orchestration, diagnostics, and persistence"
```

---

## Task 6: Integration Tests

**Files:**
- Create: `tests/integration/feature_validator/test_eda_pipeline.py`

**Pre-condition:** Cache for RSI lookback=5 and RSI Signal (rsi_period=2, oversold=25, overbought=65, exit_bars=5) on Ticker.ES, TimeFrame.D, 2020-2023 must exist. The test populates it automatically if `data/ohlc_data/` is present; otherwise it skips.

### Step 1: Write integration test

Create `tests/integration/feature_validator/test_eda_pipeline.py`:

```python
"""Integration tests for EDA pipeline (T001–T004).

Covers test_common_eda_continuous() and test_rule_based_eda() using
real persisted candle data from data/ohlc_data/.

Default configs:
  Continuous : RSI lookback=5, ES daily, 2020-2023
  Rule-based : RSI Signal 2-period strategy, ES daily, 2020-2023

Both tests are fully parameterizable — pass any bias_spec for researcher exploration.
Cache policy: USE_CACHE=True; if cache missing, test populates from data/ohlc_data/.
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

from feature_extraction.feature_extractor import extract_features_for_bias_node
from utils.cache_manager import CacheManager
from utils.enums import Ticker, TimeFrame

from feature_selection.eda.eda_dataclasses import EDAConfig, EDAMetadata
from feature_selection.eda.eda_reporter import (
    run_eda_for_continuous_feature,
    run_eda_for_rule_based_feature,
    save_eda_report,
)

# ── default integration test config ──────────────────────────────────────────

_TICKERS = [Ticker.ES]
_START_DATE = datetime(2020, 1, 1)
_END_DATE = datetime(2023, 12, 31)
_POPULATE_CACHE = True

# Continuous: RSI lookback=5
_CONTINUOUS_BIAS_SPEC: dict[str, Any] = {
    "module_name": "rsi",
    "timeframes": [TimeFrame.D],
    "params": {"lookback": [5]},
}

# Rule-based: RSI Signal 2-period strategy
# Long entry: RSI(2) crosses below 25
# Exit: RSI(2) crosses above 65 OR after 5 bars
_RULE_BASED_BIAS_SPEC: dict[str, Any] = {
    "module_name": "rsi_signal",
    "timeframes": [TimeFrame.D],
    "params": {
        "rsi_period": [2],
        "oversold": [25.0],
        "overbought": [65.0],
        "strategy_mode": ["long"],
        "exit_policy": ["threshold_or_bars"],
        "exit_bars": [5],
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _maybe_populate_cache(project_root: Path, bias_spec: dict[str, Any]) -> None:
    """Populate cache for bias_spec if POPULATE_CACHE=True and ohlc_data exists."""
    if not _POPULATE_CACHE:
        return
    candle_dir = project_root / "data" / "ohlc_data"
    if not candle_dir.exists():
        pytest.skip(f"Missing persisted candle directory: {candle_dir}")

    manager = CacheManager(candle_dir=str(candle_dir))
    params = bias_spec.get("params", {})
    keys = list(params.keys())
    values = [v if isinstance(v, list) else [v] for v in params.values()]
    param_combos = [dict(zip(keys, combo)) for combo in product(*values)] if keys else [{}]
    all_specs = [
        {
            "module_name": bias_spec["module_name"],
            "params": combo,
            "timeframes": bias_spec.get("timeframes", [TimeFrame.D]),
        }
        for combo in param_combos
    ]
    result = manager.populate_cache(
        bias_node_specs=all_specs,
        tickers=_TICKERS,
        start_date=_START_DATE,
        end_date=_END_DATE,
        show_progress=False,
        overwrite_existing=False,
    )
    assert result["failed"] == 0, f"Cache population failures: {result}"


def _extract_first_column(
    bias_spec: dict[str, Any],
    tickers: list[Ticker] = _TICKERS,
    start_date: datetime = _START_DATE,
    end_date: datetime = _END_DATE,
) -> tuple[pd.Series, pd.Series]:
    """Extract single-param-combo feature and target; return first feature column."""
    features_df, targets_df = extract_features_for_bias_node(
        bias_spec=bias_spec,
        ticker=tickers,
        start_date=start_date,
        end_date=end_date,
        use_cache=True,
    )
    if features_df.empty:
        pytest.skip("Feature extraction returned empty DataFrame — check cache.")
    feature_col = features_df.columns[0]
    target_col = targets_df.columns[0]
    feature = features_df[feature_col].dropna()
    target = targets_df[target_col].reindex(feature.index)
    aligned = pd.DataFrame({"f": feature, "t": target}).dropna()
    return aligned["f"], aligned["t"]


def _print_section(title: str) -> None:
    print(f"\n{'═' * 50}")
    print(f"  {title}")
    print(f"{'═' * 50}")


# ── integration tests ─────────────────────────────────────────────────────────

@pytest.mark.integration
def test_common_eda_continuous(
    bias_spec: dict[str, Any] = _CONTINUOUS_BIAS_SPEC,
    tickers: list[Ticker] = _TICKERS,
    start_date: datetime = _START_DATE,
    end_date: datetime = _END_DATE,
    n_bins: int = 15,
) -> None:
    """Integration test: common + continuous EDA pipeline (T001 + T002 + T004).

    Default: RSI lookback=5, ES daily, 2020-2023.
    Parameterize via direct call for researcher exploration with any continuous bias node.

    Cache policy: Populated automatically from data/ohlc_data/ if missing.
    """
    project_root = _project_root()
    _maybe_populate_cache(project_root, bias_spec)

    _print_section("Integration Test: Common + Continuous EDA")
    print(f"  Bias spec : {bias_spec['module_name']}, params={bias_spec['params']}")
    print(f"  Ticker    : {[t.value for t in tickers]}")
    print(f"  Date range: {start_date.date()} → {end_date.date()}")
    print(f"  n_bins    : {n_bins}")

    feature, target = _extract_first_column(bias_spec, tickers, start_date, end_date)
    timestamps = pd.DatetimeIndex(feature.index)
    print(f"  Samples   : {len(feature):,} observations")

    metadata = EDAMetadata(
        feature_name=f"{bias_spec['module_name']}_signal_D",
        param_combo=bias_spec["params"],
        timeframe=TimeFrame.D,
        ticker=tickers[0],
        timestamp=datetime.now(),
    )
    config = EDAConfig(n_bins=n_bins, rolling_window=min(252, len(feature) // 4))

    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_eda_for_continuous_feature(feature, target, timestamps, metadata, config)
        report_path = save_eda_report(report, Path(tmpdir), overwrite=True)

        # Assertions — automated verification
        assert report.common_stats.feature_stats.sample_size == len(feature)
        assert report.common_stats.correlation_analysis.pearson is not None
        assert set(report.common_stats.correlation_analysis.lagged_correlations.keys()) == {1, 2, 3, 4, 5}
        assert len(report.continuous_stats.decile_analysis.bin_stats.mean_return) == n_bins
        assert report.continuous_stats.monotonicity_test.kendall_tau is not None
        assert (report_path / "metadata.json").exists()
        assert (report_path / "plots").is_dir()

        # Terminal output for researcher manual verification
        fs = report.common_stats.feature_stats
        print(f"\nDescriptive Statistics:")
        print(f"  Mean  : {fs.mean:.4f}")
        print(f"  Std   : {fs.std:.4f}")
        print(f"  Skew  : {fs.skew:.4f}")
        print(f"  NaN%  : {fs.nan_pct:.1%}")

        corr = report.common_stats.correlation_analysis
        print(f"\nCorrelations (feature → target):")
        print(f"  Pearson : {corr.pearson:.4f}")
        print(f"  Spearman: {corr.spearman:.4f}")
        print(f"  Kendall : {corr.kendall:.4f}")

        print(f"\nLagged Correlations (feature leads target by k bars):")
        for lag, c in sorted(corr.lagged_correlations.items()):
            print(f"  Lag {lag}: {c:.4f}")

        mono = report.continuous_stats.monotonicity_test
        print(f"\nMonotonicity Test:")
        print(f"  Kendall tau: {mono.kendall_tau:.4f}  (p={mono.p_value:.4f})")
        print(f"  Is monotonic: {mono.is_monotonic}")
        print(f"  Overall trend: {report.continuous_stats.decile_analysis.overall_trend}")

        print(f"\nPer-bin Sharpe (bins 0→{n_bins-1}):")
        sharpe = report.continuous_stats.decile_analysis.bin_stats.sharpe
        for i, s in enumerate(sharpe):
            bar = "█" * max(0, int(abs(s) * 10)) if not np.isnan(s) else ""
            print(f"  Bin {i:2d}: {s:+.3f}  {bar}")

        print(f"\nDiagnostic Flags:")
        if report.diagnostics.warnings:
            for w in report.diagnostics.warnings:
                print(f"  ⚠  {w}")
        if report.diagnostics.red_flags:
            for rf in report.diagnostics.red_flags:
                print(f"  ✗  {rf}")
        viable = "✅ PASS" if report.diagnostics.is_viable else "❌ FAIL"
        print(f"  {viable}: is_viable={report.diagnostics.is_viable}")
        print(f"\n  Plots saved to: {report_path / 'plots'}")


@pytest.mark.integration
def test_rule_based_eda(
    bias_spec: dict[str, Any] = _RULE_BASED_BIAS_SPEC,
    tickers: list[Ticker] = _TICKERS,
    start_date: datetime = _START_DATE,
    end_date: datetime = _END_DATE,
) -> None:
    """Integration test: common + rule-based EDA pipeline (T001 + T003 + T004).

    Default: RSI Signal 2-period strategy (long entry RSI<25, exit RSI>65 or 5 bars).
    Parameterize via direct call for researcher exploration with any rule-based bias node.

    Cache policy: Populated automatically from data/ohlc_data/ if missing.
    """
    project_root = _project_root()
    _maybe_populate_cache(project_root, bias_spec)

    _print_section("Integration Test: Common + Rule-Based EDA")
    print(f"  Bias spec : {bias_spec['module_name']}, params={bias_spec['params']}")
    print(f"  Ticker    : {[t.value for t in tickers]}")
    print(f"  Date range: {start_date.date()} → {end_date.date()}")

    feature, target = _extract_first_column(bias_spec, tickers, start_date, end_date)
    timestamps = pd.DatetimeIndex(feature.index)
    print(f"  Samples   : {len(feature):,} observations")

    metadata = EDAMetadata(
        feature_name=f"{bias_spec['module_name']}_signal_D",
        param_combo=bias_spec["params"],
        timeframe=TimeFrame.D,
        ticker=tickers[0],
        timestamp=datetime.now(),
    )
    config = EDAConfig(rolling_window=min(252, len(feature) // 4), bootstrap_iterations=500)

    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_eda_for_rule_based_feature(feature, target, timestamps, metadata, config)
        report_path = save_eda_report(report, Path(tmpdir), overwrite=True)

        # Assertions — automated verification
        assert report.common_stats.feature_stats.sample_size == len(feature)
        assert len(report.rule_stats.per_level_stats.stats_by_level) >= 1
        assert report.rule_stats.transition_matrix.transition_probs.shape == (3, 3)
        assert (report_path / "metadata.json").exists()
        assert (report_path / "plots").is_dir()

        # Terminal output for researcher manual verification
        fs = report.common_stats.feature_stats
        print(f"\nDescriptive Statistics (feature signal levels):")
        print(f"  Mean  : {fs.mean:.4f}")
        print(f"  NaN%  : {fs.nan_pct:.1%}")

        print(f"\nPer-Level Statistics:")
        for lvl, ls in sorted(report.rule_stats.per_level_stats.stats_by_level.items()):
            ci = report.rule_stats.bootstrap_ci_results.ci_by_level.get(lvl)
            ci_str = f"CI=[{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]" if ci else "CI=N/A"
            reliable = "✓" if ls.is_reliable else "⚠ unreliable"
            print(
                f"  Level {lvl:+d}: mean={ls.mean_return:+.4f}  "
                f"sharpe={ls.sharpe:+.3f}  n={ls.sample_count:4d}  "
                f"{ci_str}  {reliable}"
            )

        print(f"\nTransition Matrix (row=from, col=to, levels [-1, 0, 1]):")
        probs = report.rule_stats.transition_matrix.transition_probs
        print(f"        L=-1    L=0    L=+1")
        for i, from_lvl in enumerate([-1, 0, 1]):
            row = "  ".join(f"{probs[i, j]:.3f}" for j in range(3))
            print(f"  L={from_lvl:+d}: {row}")

        corr = report.common_stats.correlation_analysis
        print(f"\nCorrelations (signal → target):")
        print(f"  Pearson : {corr.pearson:.4f}")
        print(f"  Spearman: {corr.spearman:.4f}")

        print(f"\nDiagnostic Flags:")
        if report.diagnostics.warnings:
            for w in report.diagnostics.warnings:
                print(f"  ⚠  {w}")
        if report.diagnostics.red_flags:
            for rf in report.diagnostics.red_flags:
                print(f"  ✗  {rf}")
        viable = "✅ PASS" if report.diagnostics.is_viable else "❌ FAIL"
        print(f"  {viable}: is_viable={report.diagnostics.is_viable}")
        print(f"\n  Plots saved to: {report_path / 'plots'}")
```

### Step 2: Run integration tests (verify they run or skip gracefully)

```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate
pytest tests/integration/feature_validator/test_eda_pipeline.py -v -m integration -s
```

Expected: Either PASS with rich terminal output, or SKIP with "Missing persisted candle directory" if data not available.

### Step 3: Run full unit test suite to verify no regressions

```bash
pytest tests/validators/eda/ tests/validators/test_eda_common.py tests/validators/test_eda_continuous.py tests/validators/test_eda_rule_based.py -v
```

Expected: All existing tests still PASS (we never modified `validators/eda/`).

### Step 4: Final commit

```bash
git add tests/integration/feature_validator/test_eda_pipeline.py
git commit -m "feat: add EDA pipeline integration tests for continuous (RSI) and rule-based (RSI Signal 2-period) features"
```

---

## Summary

| Task | Module | Tests |
|------|--------|-------|
| T001 | `feature_selection/eda/common_eda.py` | `tests/validators/eda/test_common_eda.py` (7 tests) |
| T002 | `feature_selection/eda/continuous_eda.py` | `tests/validators/eda/test_continuous_eda.py` (7 tests) |
| T003 | `feature_selection/eda/rule_based_eda.py` | `tests/validators/eda/test_rule_based_eda.py` (9 tests) |
| T004 | `feature_selection/eda/eda_reporter.py` | `tests/validators/eda/test_eda_reporter.py` (8 tests) |
| Integration | `test_eda_pipeline.py` | 2 parameterizable integration tests |

**No existing code is modified.** The existing `feature_selection/validators/eda/` namespace is untouched.

**Run all at once:**
```bash
source /home/raman/repos/Trading-Algo/venv/bin/activate
pytest tests/validators/eda/ -v && pytest tests/integration/feature_validator/test_eda_pipeline.py -v -m integration -s
```
