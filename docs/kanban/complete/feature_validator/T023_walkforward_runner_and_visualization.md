# T023 - Walkforward Runner and Visualization Slice

## Goal
Implement the shared walkforward fold runner and deterministic visualization helpers so feature selection per fold is reproducible and fold boundaries are auditable before adapter/integration wiring.

## Context / References
- `docs/library/Feature_selection/Walkforward/walkforward.md`
- `docs/library/Feature_selection/feature_validator.md`
- `docs/plans/2026-02-17-walkforward-shared-research-design.md`
- `docs/plans/2026-02-17-shared-walkforward-research-pipeline.md`
- `docs/kanban/in-progress/feature_validator/T022_shared_walkforward_research_pipeline.md`

## Scope
In scope:
- Implement fold orchestration and deterministic fold selection ranking in `feature_research/walkforward/runner.py`.
- Implement fold timeline and selection-summary visualization helpers in `feature_research/walkforward/visualization.py`.
- Add related unit tests for runner and visualization behavior only.

Out of scope:
- Artifact writing and file layout orchestration in `feature_research/walkforward/io.py` (follow-on task).
- Rule-based or continuous adapter wiring in `feature_research/rule_based/pipeline.py` and `feature_research/continuous_binning/pipeline.py` (follow-on task).
- Integration tests and persisted-data pipeline runs under `tests/integration/` (follow-on task).
- Any changes to Stage 1/2 permutation logic, validator report schemas, vault schemas, or deployment paths.

## Interfaces (must match)
- Add/modify: `feature_research/walkforward/runner.py`
  - `@dataclass(frozen=True)`
    - `class FoldScoreRow:`
      - `fold_id: int`
      - `train_start: pd.Timestamp`
      - `train_end: pd.Timestamp`
      - `test_start: pd.Timestamp`
      - `test_end: pd.Timestamp`
      - `param_label: str`
      - `raw_objective: float`
      - `smoothed_objective: float`
      - `rank: int`
  - `@dataclass(frozen=True)`
    - `class WalkforwardRunReport:`
      - `folds_df: pd.DataFrame`
      - `fold_scores_df: pd.DataFrame`
      - `selection_summary_df: pd.DataFrame`
  - `def run_walkforward_research(candles_df: pd.DataFrame, target: pd.Series, feature_type: str, module_name: str, config: WalkforwardResearchConfig, param_grid: list[dict[str, object]], evaluate_param_combo: Callable[[pd.DataFrame, pd.Series, dict[str, object]], pd.Series]) -> WalkforwardRunReport`
  - Deterministic ranking contract for each fold: sort by `smoothed_objective` descending, then `raw_objective` descending, then `param_label` ascending; rank-1 becomes `selected_feature`.

- Add/modify: `feature_research/walkforward/visualization.py`
  - `def plot_fold_timeline(folds_df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]`
    - Returns a figure and a normalized plotting frame with columns `fold_id`, `segment`, `start`, `end`.
  - `def plot_selection_stability(selection_summary_df: pd.DataFrame, top_k: int) -> tuple[plt.Figure, pd.DataFrame]`
    - Returns a figure and deterministic summary frame with columns `fold_id`, `selected_feature`, `selected_rank`, `selected_smoothed_objective`.

## Data Contracts
- `folds_df` columns (exact): `fold_id`, `train_start`, `train_end`, `test_start`, `test_end`, `train_samples`, `test_samples`.
- `fold_scores_df` columns (exact): `fold_id`, `param_label`, `raw_objective`, `smoothed_objective`, `rank`, `selected_feature`.
- `selection_summary_df` columns (exact): `fold_id`, `selected_feature`, `selected_raw_objective`, `selected_smoothed_objective`, `top_k_features`.
- `top_k_features` serialization must be deterministic JSON array text (stable ordering, no pretty-print variance).

## Dependencies
- Allowed production module edits (max 2 modules):
  - `feature_research/walkforward/runner.py`
  - `feature_research/walkforward/visualization.py`
- Allowed related unit test edits:
  - `tests/feature_research/walkforward/test_runner.py`
  - `tests/feature_research/walkforward/test_visualization.py`
- Required interface documentation sync:
  - `docs/api/feature_selection.md`

## Invariants / Constraints
- Deterministic: identical inputs (including `candles_df` ordering and param grid) produce byte-identical `selection_summary_df` values and the same selected feature per fold.
- No lookahead: for every fold, `train_end < test_start`, and scoring for fold N uses only fold N train window + test window timestamps.
- Minimum fold samples: folds with `train_samples < config.min_fold_samples` or `test_samples < config.min_fold_samples` are excluded with explicit reasoning in test assertions.
- Canonical labels: `param_label` formatting is stable across runs and independent of dict insertion order.

## Acceptance tests
1. `source venv/bin/activate && pytest tests/feature_research/walkforward/test_runner.py::test_run_walkforward_research_deterministic_selection_order -q` - deterministic tie-break and stable selected feature assertions.
2. `source venv/bin/activate && pytest tests/feature_research/walkforward/test_runner.py::test_run_walkforward_research_enforces_no_lookahead_fold_boundaries -q` - deterministic no-lookahead guard (`train_end < test_start`) for every returned fold.
3. `source venv/bin/activate && pytest tests/feature_research/walkforward/test_visualization.py::test_plot_fold_timeline_returns_deterministic_plot_frame -q` - deterministic visualization frame contract for timeline plotting.
4. `source venv/bin/activate && pytest tests/feature_research/walkforward/test_visualization.py::test_plot_selection_stability_returns_expected_columns -q` - deterministic summary columns and selected-rank consistency.

## Definition of done
- [ ] Production code changes are limited to `feature_research/walkforward/runner.py` and `feature_research/walkforward/visualization.py`.
- [ ] Unit tests are added/updated only under `tests/feature_research/walkforward/test_runner.py` and `tests/feature_research/walkforward/test_visualization.py`.
- [ ] `docs/api/feature_selection.md` is updated to document exact runner and visualization signatures.
- [ ] Acceptance test commands executed and passing:
  - `source venv/bin/activate && pytest tests/feature_research/walkforward/test_runner.py::test_run_walkforward_research_deterministic_selection_order -q`
  - `source venv/bin/activate && pytest tests/feature_research/walkforward/test_runner.py::test_run_walkforward_research_enforces_no_lookahead_fold_boundaries -q`
  - `source venv/bin/activate && pytest tests/feature_research/walkforward/test_visualization.py::test_plot_fold_timeline_returns_deterministic_plot_frame -q`
  - `source venv/bin/activate && pytest tests/feature_research/walkforward/test_visualization.py::test_plot_selection_stability_returns_expected_columns -q`
- [ ] This slice excludes io/adapters/integration work and leaves those for explicit follow-on tasks.

## Notes
- Follow-on task (io): artifact writers and output layout in `feature_research/walkforward/io.py`.
- Follow-on task (adapters): rule-based/continuous integration in `feature_research/rule_based/pipeline.py` and `feature_research/continuous_binning/pipeline.py`.
- Follow-on task (integration): persisted-data coverage under `tests/integration/feature_validator/` using explicit cache/data contracts.

## Result
- Implemented in: uncommitted changes on branch `feature/shared-walkforward-research`
- Tests: `pytest tests/feature_research/walkforward/test_runner.py -q` and `pytest tests/feature_research/walkforward/test_visualization.py -q` ✅
- Notes: Added robust input validation, deterministic tie-break behavior, and defensive malformed-JSON handling for `top_k_features`.
