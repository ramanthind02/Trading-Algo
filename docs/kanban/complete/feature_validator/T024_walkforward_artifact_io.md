# T024 - Walkforward Artifact IO Slice

## Goal
Add deterministic walkforward artifact writers so `WalkforwardRunReport` tables and generated figures are persisted in the shared walkforward output layout for downstream review and adapter consumption.

## Context / References
- `docs/library/Feature_selection/Walkforward/walkforward.md`
- `docs/plans/2026-02-17-walkforward-shared-research-design.md`
- `docs/kanban/complete/feature_validator/T023_walkforward_runner_and_visualization.md`
- `docs/api/feature_selection.md`

## Scope
In scope:
- Implement artifact IO in `feature_research/walkforward/io.py` for writing report tables, metadata JSON, and figures from walkforward runner/visualization outputs.
- Sync public interface documentation in `docs/api/feature_selection.md` for the new IO entrypoints and contracts.
- Add/update related unit tests for deterministic IO behavior.

Out of scope:
- Adapter wiring in `feature_research/rule_based/pipeline.py` or `feature_research/continuous_binning/pipeline.py`.
- Integration tests under `tests/integration/` and persisted-data smoke runs.
- Changes to walkforward fold construction, scoring, ranking, or visualization logic in `runner.py`/`visualization.py`.

## Interfaces (must match)
- Add: `feature_research/walkforward/io.py`
  - `@dataclass(frozen=True)`
    - `class WalkforwardArtifactPaths:`
      - `output_dir: Path`
      - `folds_csv: Path`
      - `fold_scores_csv: Path`
      - `selection_summary_csv: Path`
      - `report_json: Path`
      - `walkforward_stability_png: Path`
      - `fold_timeline_png: Path`
  - `def resolve_walkforward_output_dir(feature_type: str, module_name: str, root_dir: Path = Path("feature_research/shared_results")) -> Path`
  - `def write_walkforward_artifacts(report: WalkforwardRunReport, walkforward_stability_figure: Figure, fold_timeline_figure: Figure, feature_type: str, module_name: str, root_dir: Path = Path("feature_research/shared_results")) -> WalkforwardArtifactPaths`
    - Writes exactly:
      - `folds.csv`
      - `fold_scores.csv`
      - `selection_summary.csv`
      - `report.json`
      - `walkforward_stability.png`
      - `fold_timeline.png`
    - Returns all resolved output paths via `WalkforwardArtifactPaths`.

- Modify: `docs/api/feature_selection.md`
  - Add API docs for `WalkforwardArtifactPaths`, `resolve_walkforward_output_dir(...)`, and `write_walkforward_artifacts(...)` with exact signatures and output-file contract.

## Data Contracts
- Output path contract:
  - `feature_research/shared_results/{feature_type}/{module_name}/walkforward/`
- CSV schemas (exact columns):
  - `folds.csv`: `fold_id`, `train_start`, `train_end`, `test_start`, `test_end`, `train_samples`, `test_samples`
  - `fold_scores.csv`: `fold_id`, `param_label`, `raw_objective`, `smoothed_objective`, `rank`, `selected_feature`
  - `selection_summary.csv`: `fold_id`, `selected_feature`, `selected_raw_objective`, `selected_smoothed_objective`, `top_k_features`
- `report.json` minimum keys (exact):
  - `feature_type`
  - `module_name`
  - `output_dir`
  - `artifact_files`
  - `row_counts`
- JSON serialization must be deterministic (sorted keys, stable separators, UTF-8 text).

## Dependencies
- Allowed production module edits (max 2 modules):
  - `feature_research/walkforward/io.py`
  - `docs/api/feature_selection.md`
- Allowed related unit test edits:
  - `tests/feature_research/walkforward/test_io.py`

## Invariants / Constraints
- Deterministic: identical `WalkforwardRunReport` input frames and identical figure inputs produce byte-identical `report.json` and identical CSV row ordering/content.
- No lookahead: IO layer must not mutate or recompute fold boundaries; persisted timestamps must match report inputs exactly.
- Idempotent writes: re-running write API to same output folder overwrites deterministically without creating extra versioned files.
- Explicit validation: raise `ValueError` for blank `feature_type`/`module_name`.

## Acceptance tests
1. `source venv/bin/activate && pytest tests/feature_research/walkforward/test_io.py::test_write_walkforward_artifacts_writes_required_files_and_columns -q` - verifies required files exist and CSV schemas exactly match contract.
2. `source venv/bin/activate && pytest tests/feature_research/walkforward/test_io.py::test_write_walkforward_artifacts_is_deterministic_for_same_inputs -q` - writes twice from identical inputs and asserts byte-identical `report.json` plus equal CSV payloads.
3. `source venv/bin/activate && pytest tests/feature_research/walkforward/test_io.py::test_resolve_walkforward_output_dir_returns_expected_layout -q` - validates shared-results directory layout contract and deterministic path resolution.

## Definition of done
- [ ] Production code changes are limited to `feature_research/walkforward/io.py`.
- [ ] Interface docs are updated only in `docs/api/feature_selection.md`.
- [ ] Unit tests are added/updated only in `tests/feature_research/walkforward/test_io.py`.
- [ ] Acceptance test commands executed and passing:
  - `source venv/bin/activate && pytest tests/feature_research/walkforward/test_io.py::test_write_walkforward_artifacts_writes_required_files_and_columns -q`
  - `source venv/bin/activate && pytest tests/feature_research/walkforward/test_io.py::test_write_walkforward_artifacts_is_deterministic_for_same_inputs -q`
  - `source venv/bin/activate && pytest tests/feature_research/walkforward/test_io.py::test_resolve_walkforward_output_dir_returns_expected_layout -q`
- [ ] Adapters and integration work remain explicitly deferred to follow-on tasks.

## Notes
- Follow-on task (adapters): integrate IO invocation in `feature_research/rule_based/pipeline.py` and `feature_research/continuous_binning/pipeline.py`.
- Follow-on task (integration): add persisted-data pipeline tests under `tests/integration/feature_validator/` with explicit cache/data contracts.

## Result
- Implemented in: uncommitted changes on branch `feature/shared-walkforward-research`
- Tests: `pytest tests/feature_research/walkforward/test_io.py -q` ✅
- Notes: Normalized identifiers are now consistent between output directory structure and `report.json` metadata.
