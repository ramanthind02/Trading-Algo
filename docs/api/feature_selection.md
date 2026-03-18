# feature_selection

> **Path:** `feature_selection/`  
> **Status:** Draft  
> **Last updated:** 2026-02-18

## Purpose
`feature_selection` provides the public runtime APIs for feature validation workflows: in-sample/out-of-sample selectors, walk-forward split utilities, and permutation/stability entrypoints that are reused by training, deployment, and utility modules.
Train/validation/test cutover plan references: `docs/kanban/to-do/feature_research_train_val_test/`.

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
- Canonical exports are re-imported by `utils/evaluation/walkforward/__init__.py` for backward compatibility.
- `utils.evaluation.walkforward.generate_walkforward_splits(...)` is deprecated; prefer `WalkForwardSplitter` directly.

### Walk-forward research config and metrics
Type: dataclass/function  
Modules:
- `utils/evaluation/walkforward/config.py`
- `utils/evaluation/walkforward/metrics.py`
- `utils/evaluation/walkforward/runner.py`
- `utils/evaluation/walkforward/visualization.py`
- `utils/evaluation/walkforward/io.py`
- `feature_research/pipelines/in_sample.py`
- `feature_research/pipelines/validation.py`
- `feature_research/pipelines/oos.py`
- `feature_research/pipelines/permutation.py`

Compatibility surface:
- `feature_research/pipeline.py` remains the public façade that re-exports
  `run_eda_pipeline`, `run_validation_pipeline`, `run_oos_pipeline`,
  `run_permutation_pipeline`, and `write_permutation_summary`.

#### `WalkforwardResearchConfig`
Type: class

Signature:
```python
class WalkforwardResearchConfig:
    train_start: datetime
    train_end: datetime
    enabled: bool = False
    test_step: int = 252
    num_steps: int = 8
    top_k: int = 5
    objective_metric_name: str = "sortino"
    min_fold_samples: int = 10
    output_root: Path = Path("feature_research/shared_results")
    selection_method: WalkforwardSelectionMethod | str = WalkforwardSelectionMethod.TOP_K
    trade_freq_min: float = 0.01
```

Description: frozen configuration contract for walk-forward feature-research runs, including fold geometry, objective metric selection, and output location.

#### `resolve_objective_metric`
Type: function

Signature:
```python
def resolve_objective_metric(metric_name: str) -> Callable[[pd.Series], float]
```

Description: resolves a supported metric name to a deterministic scoring callable used during walk-forward fold evaluation.

Supported metric names:
- `"sharpe"`: `mean(returns) / std(returns, ddof=0)`
- `"sortino"`: `mean(returns) / std(returns[returns < 0], ddof=0)`
- `"mean_return"`: `mean(returns)`
 - `"t_stat"`: `mean(returns) / (std(returns, ddof=0) / sqrt(n_valid))`

Degenerate-input behavior:
- Returns `0.0` fallback when the computed metric would be undefined (`NaN`), including empty/all-`NaN` return series, zero-variance Sharpe denominator, and missing/zero downside deviation for Sortino.

#### `FoldScoreRow`
Type: dataclass

Signature:
```python
@dataclass(frozen=True)
class FoldScoreRow:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    param_label: str
    raw_objective: float
    oos_objective: float
    smoothed_objective: float
    rank: int
```

Description: fold-level scored parameter row contract for deterministic ranking outputs.

#### `WalkforwardRunReport`
Type: dataclass

Signature:
```python
@dataclass(frozen=True)
class WalkforwardRunReport:
    folds_df: pd.DataFrame
    fold_scores_df: pd.DataFrame
    selection_summary_df: pd.DataFrame
    portfolio_results_df: pd.DataFrame
```

Description: aggregate walkforward research output with fold boundaries, full per-fold scores, and selected-feature summaries.

#### `run_walkforward_research`
Type: function

Signature:
```python
def run_walkforward_research(
    candles_df: pd.DataFrame,
    target: pd.Series,
    feature_type: str,
    module_name: str,
    config: WalkforwardResearchConfig,
    param_grid: list[dict[str, object]],
    evaluate_param_combo: Callable[..., pd.Series],
    research_config: Any | None = None,
    portfolio_candles_df: pd.DataFrame | None = None,
) -> WalkforwardRunReport
```

Description: evaluates all parameter combinations per walkforward fold, ranks selections from in-sample (train-window) objectives, applies deterministic rank ordering (`smoothed_objective` desc, `raw_objective` desc, `param_label` asc), and emits fold/selection dataframes with separate out-of-sample objective tracking (`oos_objective`). When `research_config` is provided, a second portfolio simulation stage runs per fold using the production `Portfolio` stack and stores results in `portfolio_results_df`.

Validation behavior:
- Raises `ValueError` when `feature_type` or `module_name` is blank.
- Raises `ValueError` when index contracts fail or parameter grid is empty.

#### `plot_fold_timeline`
Type: function

Signature:
```python
def plot_fold_timeline(folds_df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]
```

Description: returns a timeline figure and normalized plotting frame with columns `fold_id`, `segment`, `start`, `end`.

#### `plot_selection_stability`
Type: function

Signature:
```python
def plot_selection_stability(
    selection_summary_df: pd.DataFrame,
    top_k: int,
) -> tuple[plt.Figure, pd.DataFrame]
```

Description: returns a selection-stability figure and deterministic summary frame with columns `fold_id`, `selected_feature`, `selected_rank`, `selected_smoothed_objective`.

Malformed `top_k_features` behavior:
- Invalid JSON or non-array JSON payloads are treated as unranked (`selected_rank = top_k + 1`) instead of raising parse errors.

#### `WalkforwardArtifactPaths`
Type: dataclass

Signature:
```python
@dataclass(frozen=True)
class WalkforwardArtifactPaths:
    output_dir: Path
    tearsheets_dir: Path
    report_json: Path
```

Description: immutable output-path contract for persisted walkforward artifacts. Only `report.json` is written by this module; tearsheets are produced by the walkforward runner and are the single source of truth for performance metrics.

#### `resolve_walkforward_output_dir`
Type: function

Signature:
```python
def resolve_walkforward_output_dir(
    feature_type: str,
    module_name: str,
    root_dir: Path = Path("feature_research/shared_results"),
    output_subdir: str = "walkforward",
) -> Path
```

Description: resolves deterministic output directory path as `{root_dir}/{feature_type}/{module_name}/{output_subdir}/` (e.g. `feature_research/shared_results/{feature_type}/{module_name}/validation/`).

Validation behavior:
- Raises `ValueError` when `feature_type` or `module_name` is blank.

#### `write_walkforward_artifacts`
Type: function

Signature:
```python
def write_walkforward_artifacts(
    report: WalkforwardRunReport,
    feature_type: str,
    module_name: str,
    root_dir: Path = Path("feature_research/shared_results"),
    research_context: dict[str, object] | None = None,
    output_subdir: str = "walkforward",
) -> WalkforwardArtifactPaths
```

Description: writes only `report.json` to the resolved output directory. Tearsheets (QuantStats HTML reports) are written by the walkforward runner under `tearsheets/` and are the single source of truth for performance metrics; this function does not write tables, figures, or summary files.

Output-file contract:
- `report.json` (only file written by this function)

`report.json` keys:
- `feature_type`, `module_name`, `output_dir`, `tearsheets_dir`, `tearsheet_files` (list of relative paths under tearsheets_dir), `objective_metric_name`, `timeframe`, `research_context` (includes `last_fold_test_end`, `aggregate_returns_last_date` when available)

Serialization guarantees:
- UTF-8 text output
- sorted JSON keys
- stable compact separators (`","` and `":"`)

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
- Binning diagnostics contracts:
  - `BinningSuccessCriteria`
  - `RegionMetadata`
  - `validate_binning_success(...)`
  - `extract_region_metadata(...)`
  - `detect_region_shape(...)`
  - `calculate_coverage(...)`

### EDA pipeline entrypoints
Type: dataclass/functions modules  
Modules:
- `feature_selection/eda/common_eda.py`
- `feature_selection/eda/continuous_eda.py`
- `feature_selection/eda/rule_based_eda.py`
- `feature_selection/eda/eda_reporter.py`
- `feature_selection/eda/eda_dataclasses.py`

Public symbols commonly used by integration/tests:
- Common EDA: `compute_descriptive_stats(...)`, `compute_temporal_stability(...)`, `compute_correlation_analysis(...)`, `compute_rolling_objective(...)`, `create_common_eda_plots(...)`
- Continuous EDA: `compute_decile_analysis(...)`, `compute_monotonicity_test(...)`, `compute_distribution_diagnostics(...)`, `create_continuous_eda_plots(...)`
- Rule-based EDA: `compute_per_level_stats(...)`, `compute_bootstrap_ci(...)`, `compute_transition_matrix(...)`, `create_rule_based_eda_plots(...)`
- Reporter/orchestration: `run_eda_for_continuous_feature(...)`, `run_eda_for_rule_based_feature(...)`, `compute_diagnostic_flags(...)`, `save_eda_report(...)`, `load_eda_report(...)`
- Contracts: `CommonEDAStats`, `ContinuousEDAStats`, `RuleBasedEDAStats`, `ContinuousEDAReport`, `RuleBasedEDAReport`, `EDAConfig`, `EDAMetadata`, `DiagnosticFlags`

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
