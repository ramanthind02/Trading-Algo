# T004 — Design: GlobalWeightLayer Interface Specification

## Goal
Produce a precise interface specification for `GlobalWeightLayer` — the new cross-timeframe weight layer that resamples per-TF forecast streams to daily, applies downside HRP across timeframes, and computes a cross-TF FDM. No code written.

## Context / References
- T002 findings
- `ensemble/weight_layer.py` — `BaseWeightLayer`, `DownsideHRPGroupedWeightLayer`, `_compute_downside_semi_covariance`, `_hrp_weights_from_semi_cov`, `_compute_fdm_from_corr_matrix`
- `ensemble/portfolio_tester.py` — `resample_positions_to_daily`, `aggregate_intraday_returns_to_daily` (resampling helpers that may be reused)
- `docs/library/Ensemble/weight_layer.md` — FDM formula, no-Sharpe-tilt principle

## Scope
In scope:
- Define `GlobalWeightLayer.__init__` (params; likely wraps `WeightLayerConfig` or a new `GlobalWeightLayerConfig`)
- Define `GlobalWeightLayer.fit(tf_forecast_streams, instrument_returns)` — receives a dict of `{TimeFrame: pd.DataFrame}` (forecast_score time series per TF, per ticker) and aligned daily returns per ticker; fits cross-TF downside HRP weights and cross-TF FDM
- Define `GlobalWeightLayer.combine(tf_forecast_streams)` — returns a single `forecast_score` per (ticker, datetime) on the daily grid
- Specify the resampling contract: how weekly/monthly `forecast_score` series are brought to daily (forward-fill; no lookahead)
- Specify the "group" structure: each timeframe is one group; within-TF signals are already aggregated by the per-TF WeightLayer before arriving here
- Specify cross-TF FDM formula (same as intra-TF FDM but on TF-level group return streams)
- Specify fallback when only one TF is present (FDM=1.0, weight=1.0)

Out of scope:
- Implementation.
- Per-TF intra-ensemble weighting (that stays inside `TFPortfolio`).
- IDM (that stays in `GlobalPortfolio`).

## Key Design Questions to Answer
1. Does `GlobalWeightLayer` subclass `BaseWeightLayer` or is it a standalone class? (Subclassing is tempting, but the input schema is different: dict of TF streams, not a flat `forecast_vectors` list.)
2. What is the input schema for `fit` and `combine`? Options:
   a. `Dict[TimeFrame, pd.DataFrame]` where each DataFrame is `[ticker, datetime, forecast_score]`
   b. A flat `pd.DataFrame` with a `timeframe` column added
3. How are daily returns aligned to the resampled forecast grid? (instrument_returns is already daily — just reindex)
4. Should `GlobalWeightLayer` produce per-ticker weights (like `BaseWeightLayer`) or one set of TF weights applied uniformly across tickers?

## Deliverable
Append a "Design Spec" section to this file with:
- Class hierarchy decision (subclass vs standalone)
- `__init__` params
- `fit(...)` input schema + state attributes set
- `combine(...)` input/output schema
- Resampling contract (step-by-step)
- Cross-TF FDM formula reference
- Fallback behaviour

## Dependencies
- T002 must be complete before this task begins.

## Acceptance Tests
- No code changes → no tests.
- Deliverable: spec section appended to this file.

## Notes
- The resampling must be strictly no-lookahead: a weekly bar's signal is only known at the close of that week; forward-fill to subsequent days is correct, but the first day of the week should NOT see next week's signal.
- Consider whether `GlobalWeightLayer` should expose the same `get_diagnostics()` interface as `BaseWeightLayer` for consistency.
- If `GlobalWeightLayer` is standalone (not subclassing `BaseWeightLayer`), it can have a cleaner API tailored to the TF-stream input — this is likely the better choice given the different input schema.

---

## Design Spec

### 1. Class Hierarchy: Standalone (not subclassing BaseWeightLayer)

`GlobalWeightLayer` is a **standalone class**. Reasoning:
- Input schema is `Dict[TimeFrame, pd.DataFrame]`, not `List[pd.DataFrame]`
- Resampling to daily is a domain-level concern not supported by `BaseWeightLayer`
- TF-level return proxy computation is unique to cross-TF weighting
- Pure helper functions (`_compute_downside_semi_covariance`, `_hrp_weights_from_semi_cov`, `_compute_fdm_from_corr_matrix`) are reused by direct import

### 2. `GlobalWeightLayerConfig`

```python
@dataclass(frozen=True)
class GlobalWeightLayerConfig:
    shrinkage: str = "ledoit_wolf"   # "ledoit_wolf" or "none"
    linkage: str = "ward"            # HRP linkage method
    fdm_max: float = 2.0             # cross-TF FDM cap
    resample_method: str = "forward_fill"  # only supported option
```

### 3. `__init__` Signature

```python
def __init__(self, config: Optional[GlobalWeightLayerConfig] = None) -> None:
    self._config = config or GlobalWeightLayerConfig()
    self.tf_weights_: Dict[TimeFrame, float] = {}
    self.fdm_: float = 1.0
    self.mean_cross_tf_correlation_: float = 0.0
    self.is_fitted_: bool = False
```

### 4. `fit(...)` Signature and State

```python
def fit(
    self,
    tf_forecast_streams: Dict[TimeFrame, pd.DataFrame],
    instrument_returns: pd.DataFrame,
) -> 'GlobalWeightLayer':
    """
    tf_forecast_streams: each value is DataFrame with columns ['ticker', 'datetime', 'forecast_score']
    instrument_returns: DataFrame with datetime index, ticker columns, daily return values

    Raises ValueError if len(tf_forecast_streams) < 1 or insufficient data.
    Single TF: weight=1.0, FDM=1.0 (fallback, no error).
    """
```

State set:
- `self.tf_weights_: Dict[TimeFrame, float]` — HRP weights summing to 1.0
- `self.fdm_: float` — cross-TF FDM
- `self.mean_cross_tf_correlation_: float`
- `self._daily_grid: pd.DatetimeIndex` — stored for combine()
- `self.is_fitted_: bool = True`

Fitting steps:
1. Build daily grid = union of all dates across all TF streams
2. Resample each TF stream to daily (forward-fill, then fill leading NaN with 0.0)
3. Derive TF-level return proxy: for each TF, `tf_return[t] = mean_over_tickers(forecast_score[tf,ticker,t] * instrument_return[ticker,t])`
4. Build `(T × K)` return matrix from TF return proxies
5. Compute downside semi-covariance (reuse `_compute_downside_semi_covariance`)
6. HRP weights (reuse `_hrp_weights_from_semi_cov`)
7. Cross-TF FDM (reuse `_compute_fdm_from_corr_matrix`)

### 5. `combine(...)` Signature

```python
def combine(
    self,
    tf_forecast_streams: Dict[TimeFrame, pd.DataFrame],
) -> pd.DataFrame:
    """
    Input: same schema as fit() — per-TF forecast streams
    Output: DataFrame with columns ['ticker', 'datetime', 'forecast_score'] on daily grid
    Formula: forecast_score[ticker,t] = FDM * sum_tf(tf_weight[tf] * resampled_score[tf,ticker,t])
    Clipped to [-2.0, 2.0].
    Raises ValueError if not fitted.
    """
```

### 6. Resampling Contract (No-Lookahead)

For each (TF, ticker):
1. Filter to ticker rows, set datetime as index, sort ascending
2. Reindex to `daily_grid` → produces NaN for missing dates
3. Forward-fill (`.ffill()`) → a weekly bar's value persists until next bar
4. Fill leading NaN with 0.0 (no signal before first observation)

**No-lookahead guarantee**: a weekly bar closing on Friday is first seen on Friday. It forward-fills to Monday, Tuesday, etc. It does NOT appear before Friday.

### 7. Cross-TF FDM Formula

Same as intra-TF FDM:
```
FDM = min(sqrt(1 / (mean_off_diagonal_downside_corr + 0.01)), fdm_max)
```
where `mean_off_diagonal_downside_corr` is the mean of the upper-triangle entries of the normalized downside semi-covariance matrix of TF return streams.

### 8. Fallback Behaviour

| Condition | Behaviour |
|---|---|
| Single TF in `fit()` | `tf_weights_ = {tf: 1.0}`, `fdm_ = 1.0`; no HRP computed |
| < 2 rows after alignment | `tf_weights_` = equal, `fdm_ = 1.0`; warning logged |
| Singular semi-covariance | Fall back to equal TF weights; warning logged |

### 9. `get_diagnostics()` Output

```python
{
    "is_fitted": bool,
    "num_timeframes": int,
    "timeframes": List[str],          # e.g. ["D", "W", "M"]
    "tf_weights": Dict[str, float],   # e.g. {"D": 0.50, "W": 0.35, "M": 0.15}
    "cross_tf_fdm": float,
    "mean_downside_correlation": float,
    "config": Dict[str, Any],
}
```
