# T026 — Integrate Shared Walkforward Into Continuous Binning Research Pipeline

## Goal
Integrate the shared walkforward research flow into the continuous binning pipeline so continuous runs can optionally produce standardized fold analytics, plots, and artifact files alongside existing EDA outputs.

## Context / References
- `docs/api/data_pipeline.md`
- `feature_research/continuous_binning/config.py`
- `feature_research/continuous_binning/pipeline.py`
- `feature_research/walkforward/config.py`
- `feature_research/walkforward/runner.py`
- `feature_research/walkforward/visualization.py`
- `feature_research/walkforward/io.py`
- `tests/integration/feature_validator/test_continuous_eda_pipeline.py`

## Scope
In scope:
- Add a walkforward config field to `ResearchConfig` and initialize it in `load_config()`.
- Update continuous pipeline behavior so it invokes shared walkforward runner + visualization + artifact IO when walkforward is enabled.
- Persist walkforward outputs under `feature_research/shared_results/continuous/{module_name}/walkforward/`.
- Ensure `selection_summary.csv` includes `selected_feature`.

Out of scope:
- Any changes to shared walkforward internals under `feature_research/walkforward/*`.
- Any edits outside the two production modules listed in `Dependencies`.
- CI workflow/job-matrix edits (tracked as follow-on).

## Interfaces (must match)
- Modify: `feature_research/continuous_binning/config.py`
  - `@dataclass(frozen=True) class ResearchConfig:`
    - Add field: `walkforward: WalkforwardResearchConfig`
  - `def load_config() -> ResearchConfig`
    - Must construct and pass a `WalkforwardResearchConfig` instance.
    - Default output root must be `Path("feature_research/shared_results")`.

- Modify: `feature_research/continuous_binning/pipeline.py`
  - Keep public entrypoint signature unchanged:
    - `def run_continuous_eda_pipeline(config: ResearchConfig, output_dir: Path) -> dict[str, Path]`
  - When `config.walkforward.enabled` is `True`, invoke shared interfaces exactly:
    - `run_walkforward_research(candles_df: pd.DataFrame, target: pd.Series, feature_type: str, module_name: str, config: WalkforwardResearchConfig, param_grid: list[dict[str, object]], evaluate_param_combo: Callable[[pd.DataFrame, pd.Series, dict[str, object]], pd.Series]) -> WalkforwardRunReport`
    - `plot_selection_stability(selection_summary_df: pd.DataFrame, top_k: int) -> tuple[Figure, pd.DataFrame]`
    - `plot_fold_timeline(folds_df: pd.DataFrame) -> tuple[Figure, pd.DataFrame]`
    - `write_walkforward_artifacts(report: WalkforwardRunReport, walkforward_stability_figure: Figure, fold_timeline_figure: Figure, feature_type: str, module_name: str, root_dir: Path = Path("feature_research/shared_results")) -> WalkforwardArtifactPaths`
  - `feature_type` passed to shared IO must be exactly `"continuous"`.

## Data Contracts
- Walkforward output directory must resolve to:
  - `feature_research/shared_results/continuous/{module_name}/walkforward/`
- Required files written by shared IO:
  - `folds.csv`
  - `fold_scores.csv`
  - `selection_summary.csv`
  - `report.json`
  - `walkforward_stability.png`
  - `fold_timeline.png`
- `selection_summary.csv` must contain column: `selected_feature`.

## Dependencies
- Production modules (max 1-2, hard cap):
  - `feature_research/continuous_binning/config.py`
  - `feature_research/continuous_binning/pipeline.py`
- Tests/docs allowed:
  - Unit tests under `tests/feature_research/`
  - Integration tests under `tests/integration/feature_validator/`
  - API docs update in `docs/api/data_pipeline.md`

## Invariants / Constraints
- Deterministic behavior: fixed inputs + deterministic scoring must produce stable artifact schema and stable selected columns.
- No-lookahead guarantee remains enforced by shared runner (`train_end < test_start` per fold).
- If `config.walkforward.enabled` is `False`, continuous EDA behavior and outputs remain unchanged.
- Integration smoke tests must skip with explicit reason when persisted data and/or required cache artifacts are missing.
- No unrelated production-file edits outside scoped modules.

## Acceptance tests
1. `pytest tests/feature_research/test_config.py::test_load_config_includes_walkforward_defaults -q`
   - Verifies `ResearchConfig.walkforward` exists and is initialized by `load_config()`.
2. `pytest tests/feature_research/test_continuous_pipeline_walkforward.py::test_walkforward_disabled_skips_shared_runner -q`
   - Deterministic unit test with monkeypatched shared APIs asserting no walkforward calls when disabled.
3. `pytest tests/feature_research/test_continuous_pipeline_walkforward.py::test_walkforward_enabled_writes_selected_feature_artifacts -q`
   - Deterministic unit test with fixed synthetic data asserting path includes `continuous/{module_name}/walkforward` and `selection_summary.csv` includes `selected_feature`.
4. `pytest tests/integration/feature_validator/test_continuous_eda_pipeline.py::test_continuous_eda_pipeline_walkforward_enabled_smoke -q`
   - Integration smoke verifying shared walkforward artifacts for persisted-data pipeline run.
   - Must `pytest.skip(...)` with explicit reason if persisted data directory and/or required cache is unavailable.

### Integration Test Data Contract (required when integration tests are in scope)
- Data source path: `data/ohlc_data`
- Tickers: `[Ticker.ES]`
- Timeframe: `[TimeFrame.D]`
- Date range: `2020-01-01` to `2023-12-31`
- Bias node spec / model config: `module_name="rsi"`, params include fixed `lookback=5`
- Cache mode: `USE_CACHE=True`, `POPULATE_CACHE=False`; if cache/data missing, test must `pytest.skip(...)` with explicit reason

## Definition of done
- [x] `ResearchConfig` includes `walkforward: WalkforwardResearchConfig` and `load_config()` initializes it.
- [x] Continuous pipeline invokes shared walkforward runner, visualization, and IO only when enabled.
- [x] Walkforward artifacts are saved under `feature_research/shared_results/continuous/{module_name}/walkforward/`.
- [x] `selection_summary.csv` includes `selected_feature`.
- [x] Deterministic acceptance tests and integration smoke skip policy are implemented as specified.
- [x] `docs/api/data_pipeline.md` updated to document continuous walkforward integration and output location.

### DoD verification commands
- `source venv/bin/activate && pytest tests/feature_research/test_config.py::test_load_config_includes_walkforward_defaults -q`
- `source venv/bin/activate && pytest tests/feature_research/test_continuous_pipeline_walkforward.py::test_walkforward_disabled_skips_shared_runner -q`
- `source venv/bin/activate && pytest tests/feature_research/test_continuous_pipeline_walkforward.py::test_walkforward_enabled_writes_selected_feature_artifacts -q`
- `source venv/bin/activate && pytest tests/integration/feature_validator/test_continuous_eda_pipeline.py::test_continuous_eda_pipeline_walkforward_enabled_smoke -q -rs`

## Notes
- Follow-on: wire the new continuous walkforward tests into CI once deterministic cache/data handling is finalized for the integration environment.
