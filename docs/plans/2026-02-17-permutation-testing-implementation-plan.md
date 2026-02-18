# Permutation Testing Suite (3-Phase) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a shared, config-driven permutation testing suite for continuous and rule-based research pipelines with in-sample, walkforward, and out-of-sample phases, vector-first compute gating, and researcher-ready reports.

**Architecture:** Keep one core implementation in `feature_selection/validation/` and wire thin adapters from both `feature_research/continuous_binning/` and `feature_research/rule_based/`. Extend current stage logic with an out-of-sample phase and per-combo decision records while preserving existing deterministic seed behavior and early stopping. Add optional candle override support to bias-node helper loading so candle-shuffle tests can reuse normal extraction paths.

**Tech Stack:** Python 3, dataclasses, pandas, numpy, pytest, matplotlib, project cache manager and bias-node extraction helpers.

---

Execution note: follow @superpowers:test-driven-development during implementation, run @superpowers:verification-before-completion before any success claim, and request review via @superpowers:requesting-code-review before branch completion.

### Task 1: Add Objective Metric Resolver + Nested Permutation Config

**Files:**
- Create: `feature_selection/validation/objective_metrics.py`
- Modify: `feature_selection/validation/config.py`
- Modify: `feature_selection/validation/__init__.py`
- Test: `tests/unit-tests/validators/permutation/test_objective_metrics_unit.py`
- Test: `tests/unit-tests/validators/permutation/test_orchestration_unit.py`

**Step 1: Write the failing test**

```python
def test_metric_spec_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError):
        ObjectiveMetricSpec(builtin=None, callable_path=None)


def test_metric_resolver_builtin_sharpe() -> None:
    returns = pd.Series([0.02, -0.01, 0.01])
    fn = resolve_objective_metric(ObjectiveMetricSpec(builtin="sharpe"))
    assert isinstance(fn(returns), float)
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/validators/permutation/test_objective_metrics_unit.py -v`
Expected: FAIL with missing `ObjectiveMetricSpec`/`resolve_objective_metric`.

**Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class ObjectiveMetricSpec:
    builtin: str | None = None
    callable_path: str | None = None
    kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.builtin is None) == (self.callable_path is None):
            raise ValueError("Set exactly one of builtin or callable_path")
```

```python
def resolve_objective_metric(spec: ObjectiveMetricSpec) -> Callable[[pd.Series], float]:
    if spec.builtin is not None:
        return BUILTIN_METRICS[spec.builtin]
    module_name, func_name = spec.callable_path.split(":", maxsplit=1)
    return cast(Callable[[pd.Series], float], getattr(import_module(module_name), func_name))
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/validators/permutation/test_objective_metrics_unit.py tests/unit-tests/validators/permutation/test_orchestration_unit.py::test_permutation_test_config_defaults -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add feature_selection/validation/objective_metrics.py feature_selection/validation/config.py feature_selection/validation/__init__.py tests/unit-tests/validators/permutation/test_objective_metrics_unit.py tests/unit-tests/validators/permutation/test_orchestration_unit.py
git commit -m "feat: add objective metric resolver and structured permutation config"
```

### Task 2: Add Candle Override Support to Bias-Node Data Loading Helpers

**Files:**
- Modify: `feature_extraction/feature_extractor.py`
- Modify: `feature_research/continuous_binning/data_loader.py`
- Modify: `feature_research/rule_based/data_loader.py`
- Test: `tests/unit-tests/validators/permutation/test_data_loader_candle_override_unit.py`

**Step 1: Write the failing test**

```python
def test_load_features_for_combo_uses_candles_override(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {"used": False}

    def fake_extract_features_for_bias_node(*, candles_override=None, **kwargs):
        called["used"] = candles_override is not None
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        features = pd.DataFrame({"feat": np.arange(5.0)}, index=idx)
        targets = pd.DataFrame({"log_return": np.arange(5.0)}, index=idx)
        return features, targets

    monkeypatch.setattr(
        "feature_research.continuous_binning.data_loader.extract_features_for_bias_node",
        fake_extract_features_for_bias_node,
    )
    # call loader with candles_override and assert called["used"] is True
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/validators/permutation/test_data_loader_candle_override_unit.py -v`
Expected: FAIL because loader/extractor signatures do not accept `candles_override`.

**Step 3: Write minimal implementation**

```python
def extract_features_for_bias_node(..., candles_override: pd.DataFrame | None = None, ...) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candles_override is None:
        return extract_features_with_forward_returns(...)
    return extract_features_with_forward_returns_from_candles(
        candles_df=candles_override,
        module_name=bias_spec["module_name"],
        params=bias_spec.get("params", {}),
        timeframes=bias_spec.get("timeframes", [TimeFrame.D]),
        target_col=target_col,
    )
```

```python
def load_features_for_combo(..., candles_override: pd.DataFrame | None = None) -> tuple[pd.Series, pd.Series, str] | None:
    features_df, targets_df = extract_features_for_bias_node(..., candles_override=candles_override)
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/validators/permutation/test_data_loader_candle_override_unit.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add feature_extraction/feature_extractor.py feature_research/continuous_binning/data_loader.py feature_research/rule_based/data_loader.py tests/unit-tests/validators/permutation/test_data_loader_candle_override_unit.py
git commit -m "feat: support candles override in bias-node data loaders"
```

### Task 3: Extend Report Models for OOS + Per-Combo Decision Records

**Files:**
- Modify: `feature_selection/validation/reports.py`
- Test: `tests/unit-tests/validators/permutation/test_orchestration_unit.py`
- Test: `tests/unit-tests/validators/permutation/test_report_generation_unit.py`

**Step 1: Write the failing test**

```python
def test_permutation_suite_includes_phase3_oos_and_combo_records() -> None:
    suite = PermutationTestSuite(...)
    assert hasattr(suite, "phase3_oos_reports")
    assert hasattr(suite, "combo_decisions")
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/validators/permutation/test_orchestration_unit.py::test_permutation_test_suite_fields -v`
Expected: FAIL on missing dataclass fields.

**Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class ComboDecisionRecord:
    param_combo: str
    stage1_passed: bool
    stage2_passed: bool
    walkforward_stable: bool
    oos_passed: bool
    final_status: Literal["candidate", "rejected", "needs_review"]
```

```python
@dataclass(frozen=True)
class OutOfSamplePermutationReport:
    param_combo: str
    vector_report: VectorShuffleReport
    candle_report: PipelinePermutationReport | None
    passed: bool
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/validators/permutation/test_orchestration_unit.py tests/unit-tests/validators/permutation/test_report_generation_unit.py::test_report_bundle_fields -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add feature_selection/validation/reports.py tests/unit-tests/validators/permutation/test_orchestration_unit.py tests/unit-tests/validators/permutation/test_report_generation_unit.py
git commit -m "feat: add OOS and combo decision report models"
```

### Task 4: Implement Out-of-Sample Permutation Runner (Vector-First Gate)

**Files:**
- Modify: `feature_selection/validation/permutation_tests.py`
- Test: `tests/unit-tests/validators/permutation/test_pipeline_permutation_unit.py`
- Create: `tests/unit-tests/validators/permutation/test_oos_permutation_unit.py`

**Step 1: Write the failing test**

```python
def test_oos_permutation_skips_candle_when_vector_fails() -> None:
    report = run_oos_permutation_for_param(...)
    assert report.vector_report.passed is False
    assert report.candle_report is None
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/validators/permutation/test_oos_permutation_unit.py -v`
Expected: FAIL with missing `run_oos_permutation_for_param`.

**Step 3: Write minimal implementation**

```python
def run_oos_permutation_for_param(... ) -> OutOfSamplePermutationReport:
    vector_report = run_vector_shuffle_test(...)
    if not vector_report.passed:
        return OutOfSamplePermutationReport(..., candle_report=None, passed=False)

    candle_report = run_pipeline_permutation_continuous(...)  # or rule-based variant
    return OutOfSamplePermutationReport(
        param_combo=param_combo,
        vector_report=vector_report,
        candle_report=candle_report,
        passed=bool(vector_report.passed and candle_report.passed),
    )
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/validators/permutation/test_oos_permutation_unit.py tests/unit-tests/validators/permutation/test_pipeline_permutation_unit.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add feature_selection/validation/permutation_tests.py tests/unit-tests/validators/permutation/test_oos_permutation_unit.py tests/unit-tests/validators/permutation/test_pipeline_permutation_unit.py
git commit -m "feat: add out-of-sample permutation runner with vector-first gating"
```

### Task 5: Refactor Orchestration to 3 Phases + Candidate Source Policy

**Files:**
- Modify: `feature_selection/validation/orchestration.py`
- Modify: `feature_selection/validation/config.py`
- Test: `tests/unit-tests/validators/permutation/test_orchestration_unit.py`

**Step 1: Write the failing test**

```python
def test_suite_runs_oos_on_stage2_passers_by_default() -> None:
    suite = run_permutation_test_suite(...)
    assert len(suite.phase3_oos_reports) <= suite.funnel_stats.stage2_pass
```

```python
def test_suite_can_switch_oos_source_to_stable_intersection() -> None:
    config = PermutationTestConfig(..., out_of_sample=OutOfSampleConfig(locked_selection_source="stable_intersection"))
    suite = run_permutation_test_suite(..., config=config)
    assert isinstance(suite.phase3_oos_reports, dict)
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/validators/permutation/test_orchestration_unit.py -v`
Expected: FAIL for missing OOS fields/source policy behavior.

**Step 3: Write minimal implementation**

```python
if config.out_of_sample.locked_selection_source == "stage2_passers":
    oos_candidates = stage2_passers
else:
    oos_candidates = stage2_passers & stable_params

phase3_oos_reports = {
    combo: run_oos_permutation_for_param(...)
    for combo in sorted(oos_candidates)
}
```

```python
combo_decisions[combo] = ComboDecisionRecord(
    param_combo=combo,
    stage1_passed=combo in stage1_passers,
    stage2_passed=combo in stage2_passers,
    walkforward_stable=combo in stable_params,
    oos_passed=combo in oos_passers,
    final_status="candidate" if combo in final_candidates else "rejected",
)
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/validators/permutation/test_orchestration_unit.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add feature_selection/validation/orchestration.py feature_selection/validation/config.py tests/unit-tests/validators/permutation/test_orchestration_unit.py
git commit -m "feat: extend permutation orchestration with out-of-sample phase"
```

### Task 6: Extend Report Generation for OOS Artifacts + Decision Table Export

**Files:**
- Modify: `feature_selection/validation/report_generator.py`
- Modify: `feature_selection/validation/reports.py`
- Test: `tests/unit-tests/validators/permutation/test_report_generation_unit.py`

**Step 1: Write the failing test**

```python
def test_generate_reports_writes_combo_decision_table() -> None:
    bundle = generate_permutation_reports(suite, tmp_path)
    decision_csv = tmp_path / "combo_decision_table.csv"
    assert decision_csv.exists()


def test_generate_reports_writes_oos_plots() -> None:
    bundle = generate_permutation_reports(suite, tmp_path)
    assert (tmp_path / "phase3_oos").exists()
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/validators/permutation/test_report_generation_unit.py -v`
Expected: FAIL because OOS outputs/decision table are not generated.

**Step 3: Write minimal implementation**

```python
decision_df = pd.DataFrame([asdict(r) for r in suite.combo_decisions.values()])
decision_csv = output_dir / "combo_decision_table.csv"
decision_df.to_csv(decision_csv, index=False)

for combo, oos_report in suite.phase3_oos_reports.items():
    plot_null_distribution(oos_report.vector_report, output_dir / "phase3_oos" / f"vector_{combo}.png")
    if oos_report.candle_report is not None:
        plot_null_distribution(oos_report.candle_report, output_dir / "phase3_oos" / f"candle_{combo}.png")
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/validators/permutation/test_report_generation_unit.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add feature_selection/validation/report_generator.py feature_selection/validation/reports.py tests/unit-tests/validators/permutation/test_report_generation_unit.py
git commit -m "feat: add OOS reporting artifacts and combo decision export"
```

### Task 7: Wire Shared Permutation Suite into Continuous and Rule-Based Pipelines

**Files:**
- Modify: `feature_research/continuous_binning/config.py`
- Modify: `feature_research/continuous_binning/pipeline.py`
- Modify: `feature_research/continuous_binning/data_loader.py`
- Modify: `feature_research/rule_based/config.py`
- Modify: `feature_research/rule_based/pipeline.py`
- Modify: `feature_research/rule_based/data_loader.py`
- Test: `tests/integration/feature_validator/test_continuous_eda_pipeline.py`
- Test: `tests/integration/feature_validator/test_rule_based_eda_pipeline.py`

**Step 1: Write the failing test**

```python
def test_continuous_pipeline_can_run_permutation_suite_mode(...) -> None:
    result = run_continuous_eda_pipeline(config_with_permutation_enabled, output_dir)
    assert "permutation_suite" in result
```

```python
def test_rule_based_pipeline_can_run_permutation_suite_mode(...) -> None:
    result = run_rule_based_eda_pipeline(config_with_permutation_enabled, output_dir)
    assert "permutation_suite" in result
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/integration/feature_validator/test_continuous_eda_pipeline.py tests/integration/feature_validator/test_rule_based_eda_pipeline.py -k permutation -v`
Expected: FAIL because pipelines do not expose suite mode.

**Step 3: Write minimal implementation**

```python
def run_continuous_permutation_pipeline(config: ResearchConfig, output_dir: Path) -> PermutationTestSuite:
    # adapter: build candles, param grid, extractor_func, binning_model_factory
    return run_permutation_test_suite(..., feature_type="continuous")
```

```python
def run_rule_based_permutation_pipeline(config: RuleBasedResearchConfig, output_dir: Path) -> PermutationTestSuite:
    # adapter: build candles, param grid, extractor_func
    return run_permutation_test_suite(..., feature_type="rule_based")
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/integration/feature_validator/test_continuous_eda_pipeline.py tests/integration/feature_validator/test_rule_based_eda_pipeline.py -k permutation -v`
Expected: PASS or SKIP (cache/data unavailable) with clear skip reason.

**Step 5: Commit**

```bash
git add feature_research/continuous_binning/config.py feature_research/continuous_binning/pipeline.py feature_research/continuous_binning/data_loader.py feature_research/rule_based/config.py feature_research/rule_based/pipeline.py feature_research/rule_based/data_loader.py tests/integration/feature_validator/test_continuous_eda_pipeline.py tests/integration/feature_validator/test_rule_based_eda_pipeline.py
git commit -m "feat: wire shared permutation suite into research pipeline adapters"
```

### Task 8: Expand Integration Coverage for Full 3-Phase Suite

**Files:**
- Modify: `tests/integration/feature_validator/test_permutation_testing.py`
- Modify: `feature_selection/validation/orchestration.py` (only if test-driven fixes needed)

**Step 1: Write the failing test**

```python
@pytest.mark.integration
def test_permutation_suite_runs_in_sample_walkforward_oos() -> None:
    suite = run_permutation_test_suite(...)
    assert len(suite.stage1_reports) > 0
    assert suite.stage3_report is not None
    assert isinstance(suite.phase3_oos_reports, dict)
```

```python
@pytest.mark.integration
def test_permutation_suite_respects_vector_first_gate_in_oos() -> None:
    suite = run_permutation_test_suite(...)
    for r in suite.phase3_oos_reports.values():
        if not r.vector_report.passed:
            assert r.candle_report is None
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/integration/feature_validator/test_permutation_testing.py -k "oos or three_phase" -v`
Expected: FAIL on missing OOS fields/logic.

**Step 3: Write minimal implementation adjustments**

```python
# Keep integration helper config fast
config = PermutationTestConfig(
    in_sample=InSampleConfig(nreps=100, alpha=0.10),
    out_of_sample=OutOfSampleConfig(enabled=True, ...),
    random_seed=42,
)
```

```python
# Assert report artifacts include OOS outputs
bundle = generate_permutation_reports(suite, output_dir)
assert (output_dir / "phase3_oos").exists()
assert (output_dir / "combo_decision_table.csv").exists()
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/integration/feature_validator/test_permutation_testing.py -v`
Expected: PASS or explicit SKIP with cache guidance.

**Step 5: Commit**

```bash
git add tests/integration/feature_validator/test_permutation_testing.py feature_selection/validation/orchestration.py
git commit -m "test: cover full three-phase permutation suite integration"
```

### Task 9: Update Documentation for New Public Interfaces

**Files:**
- Modify: `docs/library/Feature_selection/Permutation Testing/in-sample_pt.md`
- Modify: `docs/library/Feature_selection/Permutation Testing/candle_permutation_specs.md`
- Modify: `docs/kanban/README.md` (only if workflow references need updates)
- Create or Modify: `docs/api/feature_selection/validation.md`

**Step 1: Write the failing doc check (manual)**

```python
def test_docs_reference_three_phase_suite() -> None:
    # pseudo-check in review: docs must mention in-sample/walkforward/oos and vector-first gating
    assert True
```

**Step 2: Run quick doc sanity check**

Run: `source venv/bin/activate && pytest tests/unit-tests/validators/permutation/test_orchestration_unit.py::test_permutation_test_suite_fields -v`
Expected: PASS (code unchanged), then manually review docs for API correctness.

**Step 3: Write minimal documentation updates**

```markdown
- Added `ObjectiveMetricSpec` config contract.
- Added out-of-sample phase and candidate-source policy.
- Added `candles_override` helper behavior for candle-shuffle extraction.
```

**Step 4: Verify docs render and links are valid (manual + grep)**

Run: `source venv/bin/activate && pytest tests/unit-tests/validators/permutation/test_report_generation_unit.py::test_markdown_contains_required_sections -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/library/Feature_selection/Permutation\ Testing/in-sample_pt.md docs/library/Feature_selection/Permutation\ Testing/candle_permutation_specs.md docs/api/feature_selection/validation.md docs/kanban/README.md
git commit -m "docs: document three-phase permutation suite interfaces"
```

### Task 10: Final Verification Sweep

**Files:**
- Modify: as needed from failures only
- Test: existing targeted + integration suites

**Step 1: Run targeted validator unit tests**

Run: `source venv/bin/activate && pytest tests/unit-tests/validators/permutation -v`
Expected: PASS.

**Step 2: Run relevant integration tests**

Run: `source venv/bin/activate && pytest tests/integration/feature_validator/test_permutation_testing.py -v`
Expected: PASS or documented SKIP if cache is unavailable.

**Step 3: Run broader suite if cross-layer edits were extensive**

Run: `source venv/bin/activate && pytest tests/ -v`
Expected: PASS (or investigate and fix regressions before completion).

**Step 4: Verify report artifacts from integration run**

```python
assert (output_dir / "suite_summary.md").exists()
assert (output_dir / "suite_summary.json").exists()
assert (output_dir / "combo_decision_table.csv").exists()
assert (output_dir / "phase3_oos").exists()
```

**Step 5: Commit final stabilization fixes**

```bash
git add .
git commit -m "chore: finalize permutation suite verification and test stability"
```
