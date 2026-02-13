# feature_selection.validators

> **Path:** `feature_selection/validators/`  
> **Status:** Draft  
> **Last updated:** 2026-02-13

## Purpose
`feature_selection.validators` provides a `FeatureValidator` API that runs feature diagnostics and statistical validation in stages (EDA, permutation tests, and walk-forward stability) for continuous and rule-based features.

## Public API policy (what we document)
This document covers **public API** only:

Public items include:
- Top-level classes/types used directly by callers
- Methods on `FeatureValidator` intended for external use
- Return dataclasses consumed by downstream code

Not public (skip unless required to use a public API):
- `_run_continuous_eda`, `_run_rule_eda`
- helper functions prefixed with `_` in `feature_selection/validators/permutation.py`

If a “private” symbol is required to use a public API, document it under **Internal but required**.

## Quickstart (minimal)
```python
from pathlib import Path
import pandas as pd

from feature_selection.validators.config import ValidationConfig
from feature_selection.validators.validator import FeatureValidator

config = ValidationConfig(feature_type="continuous", n_permutations=500, random_seed=42)
validator = FeatureValidator(
    config=config,
    permutation_engine=None,   # currently accepted but not used by stage methods
    parameter_analyzer=None,   # currently accepted but not used by stage methods
    output_dir=Path("reports/validators"),
)

feature_data = pd.DataFrame({"rsi_14": ...}, index=...)
target = pd.Series(..., index=feature_data.index, name="target")

report = validator.run_full_validation(
    feature_data=feature_data,
    target=target,
    feature_name="rsi_14",
)
print(report.validation_status)
```

## Data contracts

Input(s):
- `feature_data`: `pd.DataFrame`; current implementation supports exactly one feature column for stage methods.
- `target`: `pd.Series` aligned to `feature_data` by index.
- `params_grid`: `dict[str, list]` for stage-3 metadata/counting (stability currently evaluates a single default combo).

Output(s):
- `EDAReport`, `PermutationReport`, `StabilityReport`, and top-level `ValidationReport` dataclasses.

## Public API reference

### `FeatureType`
Type: constant/type alias

Signature:
```python
FeatureType = Literal['continuous', 'rule_based']
```

Description: Enumerates supported feature families used by `ValidationConfig` and `ValidationReport`.

Notes / Constraints
- Any other string will fail type checking and may cause runtime branch behavior mismatch.

### `ValidationConfig`
Type: class (dataclass)

Signature:
```python
@dataclass(frozen=True)
class ValidationConfig:
    feature_type: FeatureType
    n_permutations: int = 1000
    confidence_level: float = 0.95
    min_sharpe_threshold: float = 0.5
    min_t_stat_threshold: float = 2.0
    neighbor_steps: int = 1
    top_k_per_fold: int = 5
    random_seed: int | None = None
```

Description: Immutable configuration for validator behavior.

Notes / Constraints
- `n_permutations`, `confidence_level`, `random_seed` are used directly by stage 1/2 permutation methods.
- Other fields are currently stored for forward compatibility and not fully consumed in current stage implementations.

### `FeatureValidator`
Type: class

Signature:
```python
class FeatureValidator:
    def __init__(
        self,
        config: ValidationConfig,
        permutation_engine: Any,
        parameter_analyzer: Any,
        output_dir: Path,
    ) -> None
```

Description: Orchestrates feature validation stages and report assembly.

Parameters
- `config (ValidationConfig)`: feature type + statistical settings.
- `permutation_engine (Any)`: accepted dependency handle.
- `parameter_analyzer (Any)`: accepted dependency handle.
- `output_dir (Path)`: directory is created on init (`mkdir(parents=True, exist_ok=True)`).

Returns
- `None`

Raises
- Filesystem exceptions if `output_dir` cannot be created.

Notes / Constraints
- Time alignment: all stage computations assume `feature_data` and `target` refer to the same timestamps.
- No-lookahead: stage 3 computes thresholds from training folds and applies them to held-out test folds.

### `FeatureValidator.run_eda`
Type: function (instance method)

Signature:
```python
def run_eda(self, feature_data: pd.DataFrame, target: pd.Series) -> EDAReport
```

Description: Runs common EDA (distribution, correlations, stationarity, rolling correlation) plus feature-type-specific EDA.

Observable behavior
- Requires one feature column; raises `NotImplementedError` for multi-column input.
- Sets `continuous_report` for `feature_type='continuous'`; sets `rule_report` for `feature_type='rule_based'`.
- Emits warnings for very low Pearson correlation (`abs(corr) < 0.05`) and non-stationary ADF result.

### `FeatureValidator.run_stage1_permutation`
Type: function (instance method)

Signature:
```python
def run_stage1_permutation(self, feature_data: pd.DataFrame, target: pd.Series) -> PermutationReport
```

Description: Runs vector-shuffle permutation test (`stage='stage1_vector_shuffle'`).

Observable behavior
- Internally aligns `feature` and `target` by timestamp and drops rows with NaN in either series.
- Uses top quartile feature threshold (`q=0.75`) to select returns for observed/permuted Sharpe and t-stat.
- One-sided p-value: proportion of permuted sharpes `>= observed_sharpe`.
- Pass criterion: `observed_sharpe > critical_value` where critical value is percentile at `confidence_level`.

### `FeatureValidator.run_stage2_permutation`
Type: function (instance method)

Signature:
```python
def run_stage2_permutation(
    self,
    feature_data: pd.DataFrame,
    target: pd.Series,
    permutation_type: str = 'feature_shuffle',
) -> PermutationReport
```

Description: Runs stage-2 permutation in feature-shuffle mode.

Observable behavior
- `permutation_type='feature_shuffle'`: returns `PermutationReport` with `stage='stage2_feature_shuffle'`.
- `permutation_type='candle_shuffle'`: raises `NotImplementedError`.
- Unknown permutation type: raises `ValueError`.

### `FeatureValidator.run_stage3_stability`
Type: function (instance method)

Signature:
```python
def run_stage3_stability(
    self,
    feature_data: pd.DataFrame,
    target: pd.Series,
    params_grid: dict[str, list],
    train_start: datetime,
    train_end: datetime,
    test_step: int,
    num_steps: int,
) -> StabilityReport
```

Description: Runs walk-forward stability evaluation over splitter folds.

Observable behavior
- Uses `WalkForwardSplitter` on `feature_data.index`; raises `ValueError` if no valid splits.
- For each fold:
  - computes train quantile threshold (`75%`) from `X_train` only,
  - applies threshold to `X_test`,
  - computes fold objective as Sharpe-like mean/std on selected `y_test` returns.
- Returns a `StabilityReport` with one synthetic `param_combo='default'` per fold and placeholder aggregate tables.
- `best_region_stability` is classified as `Stable|Moderate|Unstable|Unknown` from fold objective dispersion.

Notes / Constraints
- No-lookahead guarantee in current logic: test-fold selection uses only threshold estimated on the train fold.

### `FeatureValidator.run_full_validation`
Type: function (instance method)

Signature:
```python
def run_full_validation(
    self,
    feature_data: pd.DataFrame,
    target: pd.Series,
    feature_name: str,
    params_grid: dict[str, list] | None = None,
) -> ValidationReport
```

Description: Runs EDA then stage-1 permutation and returns a top-level report.

Observable behavior
- Always includes `eda_report` and `stage1_report`.
- If stage 1 fails, sets `validation_status='failed'` and `failure_stage='Stage 1: Vector Shuffle'`.
- If stage 1 passes, currently returns `validation_status='incomplete'` (stage 2/3 not wired into full pipeline yet).

### `FeatureValidator.run_progressive_validation`
Type: function (instance method)

Signature:
```python
def run_progressive_validation(
    self,
    feature_data: pd.DataFrame,
    target: pd.Series,
    params_grid: dict[str, list] | None = None,
) -> Generator[tuple[str, EDAReport | PermutationReport], None, None]
```

Description: Generator API for stage-by-stage execution.

Observable behavior
- Yields `("EDA", EDAReport)` then `("Stage 1: Vector Shuffle", PermutationReport)`.
- Stops early (returns) when stage 1 fails.

### `ValidationReport`
Type: class (dataclass)

Signature:
```python
@dataclass(frozen=True)
class ValidationReport:
    feature_name: str
    feature_type: Literal['continuous', 'rule_based']
    timestamp: datetime
    eda_report: EDAReport | None = None
    stage1_report: PermutationReport | None = None
    stage2_report: PermutationReport | None = None
    stage3_report: StabilityReport | None = None
    parameter_report: Any | None = None
    validation_status: Literal['passed', 'failed', 'incomplete'] = 'incomplete'
    failure_stage: str | None = None
    researcher_notes: str = ""
    ensemble_decision: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]
    def to_json(self) -> str
    def to_markdown(self) -> str
```

Description: Top-level report object returned by full pipeline API.

Observable behavior
- `to_dict()` includes status fields plus boolean presence flags (`has_eda_report`, etc.) rather than serializing nested objects.
- `to_markdown()` renders scalar summary sections for EDA and stage 1 when present.

## Internal but required
- `feature_selection.validators.permutation.run_vector_shuffle_test(...)` is the stage-1 backend.
- `feature_selection.validators.permutation.run_feature_shuffle_test(...)` is the stage-2 feature-shuffle backend.
- `feature_selection.walkforward.walkforward_model.WalkForwardSplitter` defines fold generation for stage 3; invalid date windows can produce zero splits and fail.

## Errors & logging
- Common exceptions:
  - `NotImplementedError` for multi-feature inputs in current EDA/permutation/stability stage methods.
  - `NotImplementedError` for `permutation_type='candle_shuffle'`.
  - `ValueError` for unknown permutation type and for zero walk-forward splits.
- Logging/metrics:
  - No module-level logger usage in `feature_selection/validators/validator.py` stage methods; diagnostics are returned via report fields.

## Open questions
Q1: Should `run_full_validation()` execute stage 2 and stage 3 when stage 1 passes (current behavior always returns `incomplete` on pass)?

Q2: The constructor accepts `permutation_engine` and `parameter_analyzer`, but current stage methods call local helper functions and do not use injected dependencies. Is dependency injection intentionally deferred?

Q3: `feature_selection/feature_validator.py` also defines a different `FeatureValidator` (walk-forward portfolio API). Which class should be treated as the canonical public API long term?
