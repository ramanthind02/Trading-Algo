# T003 — Design: TFPortfolio Interface Specification

## Goal
Produce a precise interface specification for `TFPortfolio` (the renamed, per-timeframe `Portfolio`) that defines exactly what it keeps, what it strips, and what its public API looks like after the refactor. No code written — output is a design doc / spec appended to this file.

## Context / References
- T002 findings (interface summary of current `Portfolio`)
- `ensemble/portfolio.py` — current implementation
- `docs/library/Ensemble/portfolio.md`
- `docs/library/Ensemble/weight_layer.md` — FDM is per-TF, stays inside TFPortfolio

## Scope
In scope:
- Define `TFPortfolio.__init__` signature (params, types, defaults)
- Define `TFPortfolio.fit` signature and what state it sets
- Define `TFPortfolio.predict` output schema
- Specify what `TFPortfolio` strips out (global IDM, cross-TF instrument weights)
- Define what `TFPortfolio.predict` returns to `GlobalPortfolio` — likely a `forecast_score` per (ticker, datetime) plus the vol-scaling metadata needed for GlobalPortfolio to compute global IDM
- Specify backward-compatibility guarantee: single-TF `GlobalPortfolio(tfs=[one_tf_portfolio])` must behave identically to current `Portfolio`

Out of scope:
- Implementation.
- `GlobalPortfolio` or `GlobalWeightLayer` design (T004/T005).

## Key Design Questions to Answer
1. Does `TFPortfolio` keep its own instrument weights and IDM for single-TF use, or does it always delegate to `GlobalPortfolio`? (Backward-compat says it should still work standalone.)
2. What does `TFPortfolio.predict` return when used inside `GlobalPortfolio` — raw `forecast_score` before IDM, or position fractions?
3. Does `TFPortfolio` keep vol scaling, or does that stay in `DiversifiedEnsemble`? (Per user: vol scaling stays in TFPortfolio.)
4. Should `TFPortfolio` be importable under the old name `Portfolio` as an alias for backward compat?

## Deliverable
Append a "Design Spec" section to this file with:
- Class signature (params + types)
- `fit(...)` signature and state attributes set
- `predict(...)` signature and return schema
- Explicit list of responsibilities kept vs stripped
- Backward-compatibility contract

## Dependencies
- T002 must be complete before this task begins.

## Acceptance Tests
- No code changes → no tests.
- Deliverable: spec section appended to this file.

## Notes
- Keep `TFPortfolio` self-contained — it should be testable in isolation with no `GlobalPortfolio` dependency.
- The `trading_timeframe: TimeFrame` attribute is the key identifier that `GlobalPortfolio` will use to route TF streams.

---

## Design Spec

### 1. Class Signature — `TFPortfolio.__init__`

```python
class TFPortfolio:
    def __init__(
        self,
        ensembles: Optional[List[DiversifiedEnsemble]] = None,
        ensemble_names: Optional[List[str]] = None,
        vault_root: str = "vault",
        trading_timeframe: TimeFrame = TimeFrame.D,
        target_volatility: Optional[float] = None,
        max_position_pct: float = 2.0,
        weight_layer: Optional[BaseWeightLayer] = None,
        instrument_weights: Optional[Dict[str, float]] = None,
        idm_max: float = 2.5,
        dm: Optional[float] = None,
        use_cache: bool = True,
        sector_allocation_config_path: Optional[str] = None,
    ) -> None: ...
```

All parameters are identical to the current `Portfolio.__init__` — this is a rename, not a signature change.

### 2. `fit(...)` Signature and State

```python
def fit(
    self,
    instrument_returns: pd.DataFrame,
    idm_override: Optional[float] = None,
) -> 'TFPortfolio': ...
```

State set:
- `self.idm_: Optional[float]` — per-TF IDM (used in standalone `predict()`)
- `self.mean_return_correlation_: Optional[float]`
- `self.instruments_: Optional[List[str]]`
- `self.is_fitted_: bool = True`

### 3. `predict(...)` and `predict_raw(...)` Signatures

**Standalone mode** (backward compatible — identical to current `Portfolio.predict`):
```python
def predict(self, combined_forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns: ['ticker', 'forecast_score'] (FDM already applied by WeightLayer)
    Output columns: ['ticker', 'forecast_score', 'position_fraction']
    Processing: instrument_weights → IDM → position cap
    """
```

**Raw mode** (for GlobalPortfolio integration — no IDM, no cap):
```python
def predict_raw(self, combined_forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns: ['ticker', 'forecast_score']
    Output columns: ['ticker', 'forecast_score', 'position_weighted']
    Processing: instrument_weights only (GlobalPortfolio applies global IDM and cap)
    position_weighted = forecast_score * instrument_weight
    """
```

### 4. Responsibilities Kept vs Stripped

| Responsibility | Kept in TFPortfolio | Stripped (moves to GlobalPortfolio) |
|---|---|---|
| Per-TF WeightLayer (intra-TF FDM) | Yes | — |
| Vol scaling (via DiversifiedEnsemble) | Yes (pass-through) | — |
| Per-TF instrument weights | Yes | — |
| Per-TF IDM (standalone use) | Yes (`predict()`) | Applied globally in `GlobalPortfolio` |
| Cross-TF forecast merging | — | GlobalPortfolio |
| Global IDM across TFs | — | GlobalPortfolio |
| Position cap (standalone use) | Yes (`predict()`) | Applied globally in `GlobalPortfolio` |
| `sector_allocation_config_path` | Yes (parsed at init) | — |

### 5. Backward-Compatibility Contract

- `Portfolio = TFPortfolio` alias at module level in `ensemble/portfolio.py`
- `from ensemble import Portfolio` continues to work and returns `TFPortfolio`
- `from ensemble import TFPortfolio` also works
- All existing `Portfolio(...)` call sites work unchanged — signature is identical
- `portfolio.predict(combined_forecasts)` returns identical schema: `['ticker', 'forecast_score', 'position_fraction']`
- No test changes required for existing tests

### 6. `trading_timeframe` Attribute

- Type: `TimeFrame`
- Set in `__init__`, immutable thereafter
- Used to filter candles in `fit_from_candles()` and `predict_from_candles()`
- Used by `GlobalPortfolio` to route per-TF forecast streams
- Exposed in `get_diagnostics()` output

### 7. Integration with GlobalPortfolio

`GlobalPortfolio` calls:
1. `tf_portfolio.fit(per_tf_returns)` — fits per-TF IDM (used only in standalone path)
2. `tf_portfolio.predict_raw(combined_forecasts)` — returns weight-scaled forecasts without IDM
3. `GlobalPortfolio` collects raw forecasts across TFs, applies `GlobalWeightLayer.combine`, then applies global IDM and position cap
