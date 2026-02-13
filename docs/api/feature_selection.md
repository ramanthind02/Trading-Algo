# feature_selection

> **Path:** `feature_selection/`  
> **Status:** Draft  
> **Last updated:** 2026-02-13

## Purpose
`feature_selection` provides the public runtime APIs for feature validation workflows: in-sample/out-of-sample selectors, walk-forward split utilities, and permutation/stability entrypoints that are reused by training, deployment, and utility modules.

## Public API policy (what we document)
This document covers public API only:

Public items include:
- Top-level classes/functions not prefixed with `_`
- Entry points imported from other packages/modules in this repo
- Report/config dataclasses used as contracts between validator stages

Not public in this doc:
- Deep implementation details of `feature_selection/base_models/*`
- Validator design-spec narrative in `docs/api/feature_validator_api.md`
- Private helpers prefixed with `_`

## Quickstart (minimal)
```python
from datetime import datetime
from feature_selection.os_feature_selector import OSFeatureSelector
from feature_selection.walkforward.walkforward_model import WalkForwardSplitter

# 1) Prepare aligned feature/target frames (DatetimeIndex required)
selector = OSFeatureSelector(features_df=features_df, targets_df=targets_df)

# 2) Build reusable walk-forward splits (no overlap leakage)
splitter = WalkForwardSplitter(
    train_start=datetime(2015, 1, 1),
    train_end=datetime(2020, 1, 1),
    test_step=252,
    num_steps=5,
)
splits = splitter.split(features_df.index)

# 3) Run model OOS evaluation for one feature
results_df, step_info, _ = selector.walkforward_test(
    model=base_model,
    objective_metric=metric,
    feature_cols="rsi_signal_D_lookback_14",
    train_start=datetime(2015, 1, 1),
    train_end=datetime(2020, 1, 1),
    target_col="log_return",
)
```

## Data contracts
Input(s):
- `features_df`: `pd.DataFrame` with feature columns and optional `ticker`; must share index with targets.
- `targets_df`/`target`: return-like series/columns (commonly `raw_return`, `log_return`, `log_return_atr`, `log_return_ewsd`).
- All walk-forward utilities require `pd.DatetimeIndex`.
- Portfolio validator paths additionally require `candles_df` with a `datetime` column.

Output(s):
- Selector methods return tuples/dicts containing per-fold DataFrames plus detail lists.
- Permutation entrypoints return `PermutationReport` dataclasses.
- Progressive/full validator flows return stage dataclasses (`EDAReport`, `StabilityReport`, `ValidationReport`).

## Public API reference

### `OSFeatureSelector`
Type: class  
Module: `feature_selection/os_feature_selector.py`

Signature:
```python
OSFeatureSelector(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
)
```
Description: multi-feature out-of-sample selector used in training/deployment flows.

Primary methods:
- `walkforward_test(...) -> Union[Tuple[pd.DataFrame, List[Dict], Optional[Any]], Dict[str, Tuple[...]]]`
- `cv_test(...) -> Union[Tuple[pd.DataFrame, List[Dict], Optional[Any]], Dict[str, Tuple[...]]]`
- `permutation_test(...) -> pd.DataFrame`
- `get_results(test_name: str) -> Dict[str, Any]`
- `get_summary() -> pd.DataFrame`

Notes / Constraints:
- Rejects misaligned feature/target indices.
- Uses time-ordered walk-forward windows; CV defaults to `shuffle=False`.
- Supports optional volatility normalization series per feature.

### `ISFeatureSelector`
Type: class  
Module: `feature_selection/is_feature_selector.py`

Signature:
```python
ISFeatureSelector(
    feature_name: str,
    feature_data: pd.Series,
    target_data: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
)
```
Description: single-feature in-sample selector focused on rolling decile stability and walk-forward model checks.

Primary methods:
- `plot_rolling_decile_whiskers(...) -> Tuple[plt.Figure, pd.DataFrame]`
- `walkforward_analysis(...) -> Tuple[pd.DataFrame, List[Dict], Optional[plt.Figure]]`

Notes / Constraints:
- Requires `raw_return` in targets for equity-curve construction.
- Drops NaNs during initialization; raises if no valid rows remain.

### `FeatureValidator` (legacy portfolio walk-forward validator)
Type: class  
Module: `feature_selection/feature_validator.py`

Signature:
```python
FeatureValidator(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
)
```
Description: portfolio-level walk-forward/permutation validator that reuses `Portfolio` and `PortfolioTester`.

Primary methods:
- `walkforward_test(...) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, float]]`
- `walkforward_permutation_test(...) -> pd.DataFrame`
- `get_results(test_name: str) -> Dict[str, Any]`

Notes / Constraints:
- Uses `logging.getLogger(__name__)` for fold/permutation failures.
- If all folds fail, raises `ValueError`.

### Walk-forward utility entrypoints
Type: class/functions  
Module: `feature_selection/walkforward/walkforward_model.py`

Public symbols:
- `WalkForwardSplitter`
- `WalkForwardModel`
- `generate_rolling_windows(...)`
- `apply_function_to_walkforward(...)`
- `apply_function_to_rolling_windows(...)`

Cross-package import surface:
- Canonical exports are re-imported by `utils/walkforward.py` for backward compatibility.
- `utils.walkforward.generate_walkforward_splits(...)` is deprecated; prefer `WalkForwardSplitter` directly.

### Validator-stage entrypoints
Type: dataclass/functions/class  
Modules:
- `feature_selection/validators/config.py`
- `feature_selection/validators/permutation.py`
- `feature_selection/validators/validator.py`
- `feature_selection/validators/reports/*.py`

Public symbols commonly used by integration/tests:
- `ValidationConfig`
- `run_vector_shuffle_test(...) -> PermutationReport`
- `run_feature_shuffle_test(...) -> PermutationReport`
- `feature_selection.validators.validator.FeatureValidator`
- Report contracts: `EDAReport`, `ContinuousEDAReport`, `RuleEDAReport`, `PermutationReport`, `FoldResult`, `StabilityReport`, `ValidationReport`

## Examples
```python
from feature_selection.validators.config import ValidationConfig
from feature_selection.validators.permutation import run_vector_shuffle_test

config = ValidationConfig(feature_type="continuous", n_permutations=500)
report = run_vector_shuffle_test(
    feature=feature_series,
    target=target_series,
    n_permutations=config.n_permutations,
    confidence_level=config.confidence_level,
    random_seed=config.random_seed,
)

assert report.stage == "stage1_vector_shuffle"
assert 0.0 <= report.p_value <= 1.0
```

## Internal but required
- `BinningModelBase`-compatible model contract is required by selector/walk-forward evaluators (`fit`, `predict`, optional normalization support).
- `Portfolio`/`PortfolioTester` are required to use legacy `feature_selection.feature_validator.FeatureValidator`.

## Errors & logging
- Common raised exceptions:
  - `TypeError` for wrong object types (e.g., non-DataFrame inputs, wrong portfolio type)
  - `ValueError` for index mismatch, missing target columns, missing required normalization/raw-return columns, or empty/invalid fold generation
  - `NotImplementedError` for not-yet-supported branches (e.g., some multi-feature validator paths, candle-shuffle stage)
- Logging behavior:
  - `feature_selection.feature_validator.FeatureValidator` logs fold and permutation criterion failures through module logger.
  - Selector APIs mostly print progress when `verbose=True` and store structured results in `self.results`.

## No-lookahead constraints
- Walk-forward splits are strictly sequential: each test window starts at prior train-window end (`train < test`, no future rows in training mask).
- Split generation is index-time based, not random; minimum sample gates prevent degenerate folds.
- CV APIs default to `shuffle=False`; turning `shuffle=True` is available but should be treated as non-time-series validation.
- Permutation routines shuffle feature/target associations but keep the observed target series used for scoring aligned to the test procedure.
- Equity-curve logic in `ISFeatureSelector.walkforward_analysis` uses out-of-sample periods only for baseline comparison.

## Open questions
Q1: Should `feature_selection/validators/__init__.py` re-export stable public symbols (currently empty), or is module-level import style the intended API?

Q2: Should legacy `feature_selection/feature_validator.py` and new `feature_selection/validators/validator.py` be explicitly versioned to clarify preferred public entrypoint?
