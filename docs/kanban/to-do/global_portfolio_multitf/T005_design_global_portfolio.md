# T005 — Design: GlobalPortfolio Interface Specification

## Goal
Produce a precise interface specification for `GlobalPortfolio` — the new top-level wrapper that owns multiple `TFPortfolio` instances, a `GlobalWeightLayer`, and applies global IDM. No code written.

## Context / References
- T003 spec (TFPortfolio interface)
- T004 spec (GlobalWeightLayer interface)
- `ensemble/portfolio.py` — IDM computation and instrument weight logic to migrate here
- `docs/library/Ensemble/portfolio.md` — IDM formula, pipeline position

## Scope
In scope:
- Define `GlobalPortfolio.__init__` (params: list of `TFPortfolio`, `GlobalWeightLayer`, IDM config, instrument weights)
- Define `GlobalPortfolio.fit(candles_per_tf, returns)` — fits each `TFPortfolio`, then fits `GlobalWeightLayer`, then fits global IDM from instrument returns
- Define `GlobalPortfolio.predict(candles_per_tf)` — calls each `TFPortfolio.predict` → feeds into `GlobalWeightLayer.combine` → applies instrument weights + global IDM → returns position fractions
- Define output schema: must match current `Portfolio.predict` output (`['ticker', 'forecast_score', 'position_fraction']`) for drop-in compatibility with `PortfolioTester` and `PositionSizer`
- Specify degenerate case: `GlobalPortfolio` with a single `TFPortfolio` must produce the same result as calling that `TFPortfolio` directly (property test target)
- Specify how `trading_timeframe` is determined for the global output (use the finest-grained TF; or expose it as a config param)

Out of scope:
- Implementation.
- `PortfolioTester` changes (T009).
- `PositionSizer` — no changes needed, output schema is preserved.

## Key Design Questions to Answer
1. Does `GlobalPortfolio` expose a `trading_timeframe` attribute? (Yes — the output grid is daily, regardless of TF mix.)
2. Who owns instrument weights: each `TFPortfolio` independently, or only `GlobalPortfolio`? (Likely: `GlobalPortfolio` owns global instrument weights; `TFPortfolio` is instrument-agnostic for multi-TF use.)
3. Should `GlobalPortfolio` be the new public export named `Portfolio` (with `TFPortfolio` as the internal class)? This preserves the `from ensemble import Portfolio` import contract.
4. How does `GlobalPortfolio.fit` pass instrument returns to `GlobalWeightLayer.fit`? Does it align them itself, or expect pre-aligned inputs?
5. Should `GlobalPortfolio` have a `get_diagnostics()` that aggregates per-TF diagnostics + global WL diagnostics?

## Deliverable
Append a "Design Spec" section to this file with:
- Class signature and params
- `fit(...)` inputs, orchestration steps (ordered), state attributes set
- `predict(...)` inputs and output schema
- Degenerate single-TF equivalence contract
- Public export strategy (rename plan for `Portfolio` alias)
- `get_diagnostics()` structure

## Dependencies
- T003 and T004 must both be complete before this task begins.

## Acceptance Tests
- No code changes → no tests.
- Deliverable: spec section appended to this file.

## Notes
- The global IDM is computed from instrument return correlations (same formula as today), but now on the daily-resampled combined forecast returns, not per-TF returns. Clarify this in the spec.
- The `sector_allocation_config_path` feature (JSON-based sector weighting) should move to `GlobalPortfolio` since instrument weights are global.

---

## Design Spec

### 1. Class Signature — `GlobalPortfolio.__init__`

```python
class GlobalPortfolio:
    def __init__(
        self,
        tf_portfolios: List[TFPortfolio],
        global_weight_layer: GlobalWeightLayer,
        instrument_weights: Optional[Dict[str, float]] = None,
        sector_allocation_config_path: Optional[str] = None,
        idm_max: float = 2.5,
        max_position_pct: float = 2.0,
    ) -> None: ...
```

**Init-time state:**
- `tf_portfolios: List[TFPortfolio]`
- `global_weight_layer: GlobalWeightLayer`
- `instrument_weights: Optional[Dict[str, float]]` — resolved from sector JSON if path provided
- `sector_allocation_config_: Optional[Dict[str, Any]]` — loaded JSON tree
- `idm_max: float`
- `max_position_pct: float`

**Fitted state (after `fit()`):**
- `global_idm_: float`
- `mean_instrument_return_correlation_: float`
- `instruments_: List[str]`
- `is_fitted_: bool = True`

**Immutable property:**
- `trading_timeframe: TimeFrame = TimeFrame.D` — output grid is always daily

### 2. `fit(candles_per_tf, instrument_returns)` — Ordered Steps

```python
def fit(
    self,
    candles_per_tf: Dict[TimeFrame, pd.DataFrame],
    instrument_returns: pd.DataFrame,
) -> 'GlobalPortfolio': ...
```

Inputs:
- `candles_per_tf`: per-TF OHLCV candles (datetime index, ticker/OHLCV columns)
- `instrument_returns`: daily returns (datetime index, ticker columns)

Ordered steps:
1. For each `tf_p` in `tf_portfolios`: call `tf_p.fit_from_candles(candles_per_tf[tf_p.trading_timeframe])`
2. For each `tf_p`: call `tf_p.predict_from_candles_raw(...)` → collect `tf_forecast_streams: Dict[TimeFrame, pd.DataFrame]`
3. Call `global_weight_layer.fit(tf_forecast_streams, instrument_returns)`
4. Call `self._calculate_global_idm(instrument_returns)` → sets `global_idm_`, `mean_instrument_return_correlation_`, `instruments_`
5. Set `self.is_fitted_ = True`

### 3. `predict(candles_per_tf)` — Ordered Steps

```python
def predict(
    self,
    candles_per_tf: Dict[TimeFrame, pd.DataFrame],
) -> pd.DataFrame: ...
```

Output columns: `['ticker', 'datetime', 'forecast_score', 'position_fraction']`

Ordered steps:
1. For each `tf_p`: call `tf_p.predict_from_candles_raw(candles_per_tf[tf_p.trading_timeframe])` → `tf_forecast_streams`
2. Call `global_weight_layer.combine(tf_forecast_streams)` → daily `['ticker', 'datetime', 'forecast_score']`
3. Apply global `instrument_weights`: `position_weighted = forecast_score * instrument_weight[ticker]`
4. Apply `global_idm_`: `idm_scaled = position_weighted * global_idm_`
5. Clip to `[-max_position_pct, max_position_pct]` → `position_fraction`
6. Return sorted by (datetime, ticker)

### 4. Degenerate Single-TF Equivalence

When `tf_portfolios = [one_daily_tf_p]`:
- `GlobalWeightLayer` assigns `tf_weights_ = {TimeFrame.D: 1.0}`, `fdm_ = 1.0`
- `combine()` returns the raw daily forecast_score unchanged
- `global_idm_` is computed from the same `instrument_returns` as `one_daily_tf_p.idm_`
- Result: `GlobalPortfolio.predict()` is numerically identical to `one_daily_tf_p.predict_from_candles()`
- Test: `assert_allclose(tf_result[['position_fraction']], global_result[['position_fraction']], rtol=1e-10)`

### 5. Public Export Strategy

In `ensemble/__init__.py`:
```python
from ensemble.portfolio import GlobalPortfolio, TFPortfolio
Portfolio = GlobalPortfolio   # supersedes the T006 TFPortfolio alias
```

`from ensemble import Portfolio` → `GlobalPortfolio`
`from ensemble import TFPortfolio` → per-TF class
`from ensemble import GlobalPortfolio` → explicit multi-TF class

### 6. `get_diagnostics()` Output

```python
{
    "is_fitted": bool,
    "global": {
        "global_idm": float,
        "mean_instrument_return_correlation": float,
        "instruments": List[str],
        "idm_max": float,
        "max_position_pct": float,
    },
    "global_weight_layer": { ... },    # from GlobalWeightLayer.get_diagnostics()
    "per_tf_portfolios": [
        {
            "trading_timeframe": str,
            "is_fitted": bool,
            "idm": float,
            "instruments": List[str],
            "weight_layer_diagnostics": { ... },
        },
        ...
    ],
    "sector_allocation": {
        "enabled": bool,
        "config_path": Optional[str],
        "resolved_weights": Dict[str, float],
    },
}
```

### 7. `sector_allocation_config_path` Handling

- Parsed at `__init__` if provided; overwrites `instrument_weights` with resolved ticker-level weights
- Precedence: `sector_allocation_config_path` > `instrument_weights` > equal weight
- Same JSON schema as current `Portfolio` (flat or nested sector trees)
