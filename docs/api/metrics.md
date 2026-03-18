# metrics

> **Path:** `metrics/`  
> **Status:** Stable  
> **Last updated:** 2026-02-13

## Purpose
`metrics` provides strategy evaluation primitives (performance, risk, equity) plus reporting/plotting utilities used by feature analysis, walk-forward evaluation, and portfolio testing.

## Public API policy (what we document)
This document covers public API used outside `metrics/`:

- Re-exports from `metrics/__init__.py`, `metrics/performance/__init__.py`, `metrics/risk/__init__.py`, and `metrics/equity/__init__.py`
- Reporting/plotting symbols imported by other packages (notably `eda/`, `ensemble/`, `feature_selection/`, and `utils/`)
- Entry-style helper functions that produce reports/figures

Not covered:
- Private helpers prefixed with `_`
- Plot styling internals and subplot layout internals

## Quickstart (minimal)

```python
import pandas as pd

from metrics.performance import SortinoRatio
from metrics.equity import cumulative_returns
from metrics.risk import max_drawdown
from metrics.plotting.graphing.quantstats_reports import generate_tearsheet

returns = pd.Series(
    [0.01, -0.004, 0.007, -0.01],
    index=pd.date_range("2025-01-01", periods=4, freq="D"),
)

metric = SortinoRatio(annualization_factor=252)
print(metric.compute(returns))

equity = cumulative_returns(returns, initial_value=1.0)
print(max_drawdown(equity=equity))

# Optional (requires quantstats installed)
# generate_tearsheet(strategy_returns=returns, feature_name="demo", mode="metrics")
```

## Data contracts
Input(s):

- **Return series (`pd.Series`)**
  - Decimal returns (`0.01 == 1%`), not percentage points
  - For reporting APIs, expected daily frequency
  - For tearsheets, must use `DatetimeIndex`
  - Time alignment rule: when combining feature, signal, target, or baseline return series, align by index first and only then compute metrics/equity
- **Equity series (`pd.Series`)**
  - Numeric equity values indexed by time or integer position
  - `drawdown_series`/`max_drawdown` accept either precomputed equity or returns input
- **Walk-forward results (`pd.DataFrame`)**
  - `compute_baseline_results` expects `step`, `test_start`, `test_end`
  - `all_returns` must cover all `[test_start, test_end]` ranges
- **Feature/target plotting frames**
  - Numeric feature columns required for most plotting helpers
  - Many helpers drop NaN rows pairwise before computing bins/correlations

Output(s):

- Scalar metrics (`float`), series (`pd.Series`), tabular summaries (`pd.DataFrame`), and plot handles (`matplotlib.figure.Figure` or `plotly.graph_objects.Figure`)

## Public API reference

### Performance

`ObjectiveMetric`
- Type: class (abstract)
- Signature: `class ObjectiveMetric(ABC)`
- Behavior: Base class for annualized risk-adjusted metrics; handles return validation, NaN removal, standard deviation floor, callable interface.
- Raises/Errors: no explicit custom raises in base methods.
- Constraints: `compute()` implementations should treat inputs as 1D return sequences.

`SharpeRatio`
- Type: class
- Signature: `SharpeRatio(annualization_factor: float = 252, min_std: float = 1e-6, risk_free_rate: float = 0.0)`
- Behavior: Computes annualized Sharpe using per-period risk-free adjustment.
- Returns: `compute(...) -> float`, `compute_volatility(...) -> float`
- Constraints: empty input returns `0.0`; NaNs are dropped before calculation.

`SortinoRatio`
- Type: class
- Signature: `SortinoRatio(annualization_factor: float = 252, min_std: float = 1e-6, target_return: float = 0.0)`
- Behavior: Computes annualized Sortino using downside deviation (`returns < target_return`).
- Returns: `compute(...) -> float`, `compute_downside_deviation(...) -> float`
- Constraints: if no downside observations and mean > target, returns annualized mean proxy.

Minimal example:

```python
from metrics.performance import SortinoRatio

metric = SortinoRatio(annualization_factor=252)
score = metric.compute(returns)
```

### Equity

`cumulative_returns`
- Type: function
- Signature: `cumulative_returns(returns: pd.Series, initial_value: float = 1.0) -> pd.Series`
- Behavior: Computes geometric cumulative equity via `(1 + returns).cumprod() * initial_value`.
- Errors: none explicitly raised.
- Constraints: assumes returns are ordered as intended; sort by timestamp before calling when chronology matters.

`equity_curve`
- Type: function
- Signature: `equity_curve(returns: pd.Series, initial_equity: float = 1.0) -> pd.Series`
- Behavior: Alias wrapper over `cumulative_returns`.

`equity_peak`
- Type: function
- Signature: `equity_peak(equity: pd.Series) -> pd.Series`
- Behavior: Running max via expanding window.

`equity_tracking`
- Type: function
- Signature: `equity_tracking(equity: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]`
- Behavior: Returns `(equity, peak, drawdown)` with drawdown `(equity - peak) / peak`.

### Risk

`drawdown_series`
- Type: function
- Signature: `drawdown_series(returns: pd.Series | None = None, equity: pd.Series | None = None) -> pd.Series`
- Behavior: Computes drawdown from either returns (via cumulative equity) or direct equity.
- Raises: `ValueError` if neither/both of `returns` and `equity` are provided.
- Constraints: for returns input, same ordering/alignment caveat as `cumulative_returns`.

`max_drawdown`
- Type: function
- Signature: `max_drawdown(returns: pd.Series | None = None, equity: pd.Series | None = None) -> float`
- Behavior: Minimum value of `drawdown_series`.

`trailing_drawdown_threshold`
- Type: function
- Signature: `trailing_drawdown_threshold(equity: pd.Series, trailing_drawdown_pct: float) -> pd.Series`
- Behavior: Computes threshold series from running peak.
- Constraint: implementation currently computes `peak - trailing_drawdown_pct` (absolute subtraction), so callers should use same equity scaling convention expected by simulator code.

`daily_drawdown`
- Type: function
- Signature: `daily_drawdown(returns: pd.Series) -> pd.Series`
- Behavior: Identity helper returning daily returns where negatives represent losses.

`check_drawdown_breach`
- Type: function
- Signature: `check_drawdown_breach(equity: pd.Series, max_drawdown_pct: float, returns: pd.Series | None = None, trailing_drawdown_pct: float | None = None, max_daily_drawdown_pct: float | None = None) -> tuple[bool, int, str]`
- Behavior: Sequential rule check: max drawdown -> trailing drawdown -> daily drawdown, returns first breach.
- Raises: `ValueError` when daily drawdown limit is configured but `returns` is missing.
- Returns: `(breached, index, reason)` where reason in `{"max_drawdown", "trailing_drawdown", "daily_drawdown", "none"}`.

Minimal example:

```python
from metrics.equity import cumulative_returns
from metrics.risk.drawdown import check_drawdown_breach

equity = cumulative_returns(returns, initial_value=1.0)
breached, idx, reason = check_drawdown_breach(equity=equity, max_drawdown_pct=0.1)
```

### Reporting / plotting APIs imported outside `metrics/`

`generate_tearsheet`
- Type: function
- Signature: `generate_tearsheet(strategy_returns: pd.Series, baseline_returns: pd.Series | None = None, feature_name: str = "Strategy", output_file: str | None = None, mode: str = "full", timeframe: TimeFrame = TimeFrame.D) -> None`
- Behavior: QuantStats wrapper for report generation (`html`/`full`/`basic`/`metrics`).
- Raises: `ImportError` if QuantStats missing, `TypeError` for invalid series/index, `ValueError` for unknown mode.
- Logging/side effects: emits warnings when QuantStats is unavailable at import time; prints save path in `html` mode.
- Constraints: expects `DatetimeIndex`; timezone info is stripped internally; uses `compounded=False`. For sub-daily (`H1`/`H4`) inputs, returns are additively resampled to daily before passing to QuantStats.

`compute_baseline_results`
- Type: function
- Signature: `compute_baseline_results(results_df: pd.DataFrame, all_returns: pd.Series, objective_metric: ObjectiveMetric) -> pd.DataFrame`
- Behavior: Builds always-in benchmark metrics by slicing returns over each walk-forward test window.
- Constraints: `results_df` must include `step`, `test_start`, `test_end`; index-based date filtering must be valid for `all_returns`.

`plot_parameter_sensitivity`
- Type: function
- Signature: `plot_parameter_sensitivity(df: pd.DataFrame, param_name: str, metric: str = "sortino", title: str | None = None, show_plot: bool = True) -> go.Figure`
- Behavior: 1D parameter-vs-metric line with optional sample-size bars.
- Raises: `ValueError` when no plottable points remain.

`plot_2d_parameter_surface`
- Type: function
- Signature: `plot_2d_parameter_surface(df: pd.DataFrame, param1: str, param2: str, metric: str = "sortino", title: str | None = None, show_plot: bool = True, plot_type: str = "surface") -> go.Figure`
- Behavior: 2-parameter surface/scatter/heatmap/contour/lines view.
- Raises: `ValueError` for insufficient grid or invalid `plot_type`.

`plot_3d_parameter_interactive`
- Type: function
- Signature: `plot_3d_parameter_interactive(df: pd.DataFrame, param_names: list[str], metric: str = "sortino", title: str | None = None, show_plot: bool = True, plot_type: str = "surface") -> go.Figure`
- Behavior: Dropdown+slider exploration with one fixed parameter and two free axes.
- Raises: `ValueError` when `param_names` length is not 3.

`plot_4d_parameter_interactive`
- Type: function
- Signature: `plot_4d_parameter_interactive(df: pd.DataFrame, param_names: list[str], metric: str = "sortino", title: str | None = None, show_plot: bool = True, plot_type: str = "surface") -> go.Figure`
- Behavior: Dropdown+slider exploration across fixed parameter pairs.
- Raises: `ValueError` when `param_names` length is not 4.

`plot_decile_analysis`, `plot_2bin_analysis`, `plot_uniform_binning`
- Type: functions
- Signatures:
  - `plot_decile_analysis(feature_data: pd.Series, target_data: pd.Series, feature_name: str, n_bins: int = 10, figsize: tuple[int, int] = (12, 8), plot_type: str = "bar", save_path: str | None = None, selected_bin: int | None = None, strategy: str | None = None) -> tuple[plt.Figure, pd.DataFrame]`
  - `plot_2bin_analysis(feature_data: pd.Series, target_data: pd.Series, feature_name: str, figsize: tuple[int, int] = (10, 6), save_path: str | None = None) -> tuple[plt.Figure, pd.DataFrame]`
  - `plot_uniform_binning(feature_data: pd.Series, target_data: pd.Series, feature_name: str, n_bins: int = 10, figsize: tuple[int, int] = (12, 8), plot_type: str = "bar", save_path: str | None = None) -> tuple[plt.Figure, pd.DataFrame]`
- Behavior: Binned feature-target analysis plots plus per-bin summary table.
- Raises: `ValueError` when binning cannot be formed or usable data is empty.
- Constraints: feature and target must be time-aligned before passing.

`plot_feature_distribution`, `plot_feature_timeseries`
- Type: functions
- Signatures:
  - `plot_feature_distribution(feature_data: pd.Series, feature_name: str, figsize: tuple[int, int] = (12, 6), bins: int = 50, show_stats: bool = True, save_path: str | None = None) -> plt.Figure`
  - `plot_feature_timeseries(feature_data: pd.Series, feature_name: str, figsize: tuple[int, int] = (14, 6), show_rolling_mean: bool = True, rolling_window: int = 20, show_rolling_std: bool = True, save_path: str | None = None) -> plt.Figure`
- Raises: `TypeError` for non-series/non-numeric inputs; `plot_feature_timeseries` also requires `DatetimeIndex`; `ValueError` for empty/all-NaN data.

`plot_all_feature_deciles`, `plot_feature_2bin`, `plot_all_feature_uniform_bins`, `plot_feature_target_correlations`, `plot_feature_correlation_matrix`, `plot_all_feature_distributions`, `plot_all_feature_timeseries`, `plot_feature_signal_cumsum`, `combine_decile_plots`, `combine_signal_cumsum_plots`, `combine_distribution_plots`, `combine_timeseries_plots`
- Type: functions
- Behavior: Higher-level batch plotting wrappers consumed by `eda.feature_explorer.FeatureExplorer`.
- Data contracts: expect aligned `features_df`/`targets_df` and valid feature names; most functions skip non-numeric features.
- Errors/logging: many wrappers are fail-soft (catch-and-continue) and print status when `verbose=True`; some raise `ValueError` for empty input collections.
- Time alignment constraint: `plot_feature_signal_cumsum` sorts by index before cumulative sum; upstream code should still align and sort signal-gated returns chronologically.

Minimal example:

```python
from metrics.plotting.parameter_plots import plot_2d_parameter_surface

fig = plot_2d_parameter_surface(
    df=results_df,
    param1="fast_span",
    param2="slow_span",
    metric="sortino",
    show_plot=False,
)
```

## Internal but required
- `metrics.performance.base.ObjectiveMetric` is the required interface for metric injection in APIs like `compute_baseline_results`.
- `metrics.equity.cumulative_returns` and `metrics.equity.equity_peak` are internal dependencies used by risk/reporting functions; behavior assumptions in those call paths rely on their formulas.

## Errors & logging
- Common exceptions: `ValueError` (invalid argument combinations, empty data, impossible bins), `TypeError` (wrong data types/index), `ImportError` (optional QuantStats dependency).
- Logging/printing is mostly stdout-based (progress/status/warnings); there is no structured logging contract in this package.

## Open questions
Q1: `trailing_drawdown_threshold` subtracts percentage as an absolute value (`peak - pct`) instead of multiplicative (`peak * (1 - pct)`) despite docstring wording; consumers currently depend on implemented behavior.

Q2: Some plotting wrappers are explicitly fail-soft (print and continue) while others raise; if these are treated as stable API, error handling semantics should be standardized.
