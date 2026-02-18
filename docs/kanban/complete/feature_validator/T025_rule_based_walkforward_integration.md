# T025 — Integrate Shared Walkforward Into Rule-Based Research Pipeline

## Goal
Integrate the shared walkforward research pipeline into the existing rule-based feature research flow so rule-based runs can optionally produce standardized fold analytics, plots, and artifact files alongside EDA outputs.

## Context / References
- `docs/api/data_pipeline.md`
- `feature_research/rule_based/config.py`
- `feature_research/rule_based/pipeline.py`
- `feature_research/walkforward/config.py`
- `feature_research/walkforward/runner.py`
- `feature_research/walkforward/visualization.py`
- `feature_research/walkforward/io.py`
- `tests/integration/feature_validator/test_rule_based_eda_pipeline.py`

## Scope
In scope:
- Add a walkforward config field to `RuleBasedResearchConfig` and default it in `load_config()`.
- Update the rule-based pipeline to invoke shared walkforward runner + visualization + artifact IO when walkforward is enabled.
- Persist walkforward outputs under `feature_research/shared_results/rule_based/{module_name}/walkforward/`.
- Ensure selection summary artifacts include `selected_feature`.

Out of scope:
- Any changes to walkforward shared module internals (`feature_research/walkforward/*`).
- Any changes to rule-based data loader module contracts.
- CI pipeline wiring or job matrix edits (tracked as a follow-on task).

## Interfaces (must match)
- Modify: `feature_research/rule_based/config.py`
  - `@dataclass(frozen=True) class RuleBasedResearchConfig:`
    - Add field: `walkforward: WalkforwardResearchConfig`
  - `def load_config() -> RuleBasedResearchConfig`
    - Must construct and pass a `WalkforwardResearchConfig` instance.
    - Default output root for this task pathing: `Path("feature_research/shared_results")`.

- Modify: `feature_research/rule_based/pipeline.py`
  - Keep public entrypoint signature unchanged:
    - `def run_rule_based_eda_pipeline(config: RuleBasedResearchConfig, output_dir: Path) -> dict[str, Path]`
  - When `config.walkforward.enabled` is `True`, pipeline must call shared interfaces exactly:
    - `run_walkforward_research(candles_df: pd.DataFrame, target: pd.Series, feature_type: str, module_name: str, config: WalkforwardResearchConfig, param_grid: list[dict[str, object]], evaluate_param_combo: Callable[[pd.DataFrame, pd.Series, dict[str, object]], pd.Series]) -> WalkforwardRunReport`
    - `plot_selection_stability(selection_summary_df: pd.DataFrame, top_k: int) -> tuple[Figure, pd.DataFrame]`
    - `plot_fold_timeline(folds_df: pd.DataFrame) -> tuple[Figure, pd.DataFrame]`
    - `write_walkforward_artifacts(report: WalkforwardRunReport, walkforward_stability_figure: Figure, fold_timeline_figure: Figure, feature_type: str, module_name: str, root_dir: Path = Path("feature_research/shared_results")) -> WalkforwardArtifactPaths`
  - Feature type string passed to shared IO must be exactly `"rule_based"`.

## Data Contracts
- Walkforward output directory must resolve to:
  - `feature_research/shared_results/rule_based/{module_name}/walkforward/`
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
  - `feature_research/rule_based/config.py`
  - `feature_research/rule_based/pipeline.py`
- Tests allowed:
  - Unit tests under `tests/feature_research/`
  - Integration tests under `tests/integration/feature_validator/`

## Invariants / Constraints
- Deterministic behavior: with fixed input data/config and deterministic objective metric, artifact schema and selected columns are stable.
- No-lookahead guarantee remains enforced by shared runner (`train_end < test_start` per fold).
- If `config.walkforward.enabled` is `False`, rule-based EDA behavior and outputs remain unchanged.
- No unrelated production-file edits outside the two scoped modules.

## Acceptance tests
1. `pytest tests/feature_research/test_rule_based_config.py::test_load_config_includes_walkforward_defaults -q`
   - Verifies `RuleBasedResearchConfig.walkforward` exists and defaults are initialized via `load_config()`.
2. `pytest tests/feature_research/test_rule_based_pipeline_walkforward.py::test_walkforward_disabled_skips_shared_runner -q`
   - Deterministic unit test with monkeypatched shared APIs asserting no walkforward calls when disabled.
3. `pytest tests/feature_research/test_rule_based_pipeline_walkforward.py::test_walkforward_enabled_writes_selected_feature_artifacts -q`
   - Deterministic unit test (fixed synthetic frame + fixed scorer) asserting write path includes `rule_based/{module_name}/walkforward` and that `selection_summary.csv` contains `selected_feature`.
4. `pytest tests/integration/feature_validator/test_rule_based_eda_pipeline.py::test_rule_based_eda_pipeline_walkforward_enabled_smoke -q`
   - Integration smoke path validating shared walkforward artifacts are emitted for a persisted-data run.

### Integration Test Data Contract (required when integration tests are in scope)
- Data source path: `data/ohlc_data`
- Tickers: `[Ticker.ES]`
- Timeframe: `[TimeFrame.D]`
- Date range: `2020-01-01` to `2023-12-31`
- Bias node spec / model config: `module_name="rsi_signal"`, params include fixed `rsi_period=2`, `oversold=25.0`, `overbought=65.0`, `strategy_mode="long"`, `exit_policy="threshold_or_bars"`, `exit_bars=5`
- Cache mode: `USE_CACHE=True`, `POPULATE_CACHE=False`; if cache/data missing, test must `pytest.skip(...)` with explicit reason.

## Definition of done
- [ ] `RuleBasedResearchConfig` includes `walkforward: WalkforwardResearchConfig` and `load_config()` initializes it.
- [ ] Rule-based pipeline calls shared walkforward runner, visualization, and IO only when enabled.
- [ ] Walkforward artifacts are saved under `feature_research/shared_results/rule_based/{module_name}/walkforward/`.
- [ ] `selection_summary.csv` includes `selected_feature`.
- [ ] `docs/api/data_pipeline.md` updated to document rule-based walkforward integration and output location.
- [ ] Acceptance tests pass with exact commands listed below.

### DoD verification commands
- `source venv/bin/activate && pytest tests/feature_research/test_rule_based_config.py::test_load_config_includes_walkforward_defaults -q`
- `source venv/bin/activate && pytest tests/feature_research/test_rule_based_pipeline_walkforward.py::test_walkforward_disabled_skips_shared_runner -q`
- `source venv/bin/activate && pytest tests/feature_research/test_rule_based_pipeline_walkforward.py::test_walkforward_enabled_writes_selected_feature_artifacts -q`
- `source venv/bin/activate && pytest tests/integration/feature_validator/test_rule_based_eda_pipeline.py::test_rule_based_eda_pipeline_walkforward_enabled_smoke -q`

## Notes
- Follow-on task: add/extend continuous integration coverage so rule-based walkforward integration tests run in CI with deterministic cache/data handling.

## Result
- Implemented in: uncommitted changes on branch `feature/shared-walkforward-research`
- Tests: `pytest tests/feature_research/test_rule_based_pipeline_walkforward.py -q` and `pytest tests/integration/feature_validator/test_rule_based_eda_pipeline.py::test_rule_based_eda_pipeline_walkforward_enabled_smoke -q -rs` ✅
- Notes: Added timezone normalization for walkforward alignment to handle tz-aware persisted data safely.
