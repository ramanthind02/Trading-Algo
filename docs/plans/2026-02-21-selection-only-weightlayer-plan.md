# Selection-Only Walkforward and WeightLayer-First Flow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert walkforward selection methods into selector-only outputs and make WeightLayer the explicit combiner for selected member signals.

**Architecture:** Standardize selection outputs in walkforward as selected member identities with metadata only. Push selected member signals unchanged into portfolio evaluation and delegate final combination to `WeightLayer` using enum-backed researcher configuration.

**Tech Stack:** Python 3, dataclasses, Enum, pandas, pytest, existing walkforward and ensemble modules.

---

### Task 1: Add enum-backed walkforward method and weight-layer algorithm config

**Files:**
- Modify: `feature_research/walkforward/config.py`
- Test: `tests/feature_research/walkforward/test_config.py`

**Step 1: Write the failing test**

```python
def test_walkforward_config_rejects_unknown_methods() -> None:
    from feature_research.walkforward.config import WalkforwardConfig

    WalkforwardConfig(selection_method="unknown")
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/feature_research/walkforward/test_config.py -v`
Expected: FAIL with validation/coercion error mismatch.

**Step 3: Write minimal implementation**

```python
class SelectionMethod(str, Enum):
    TOP_K = "top_k"
    ENHANCED = "enhanced"
    STABLE_REGION = "stable_region"
```

Add matching enum for weight-layer algorithm and coercion from `str`.

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/feature_research/walkforward/test_config.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/feature_research/walkforward/test_config.py feature_research/walkforward/config.py
git commit -m "feat: add enum-backed walkforward method and weighting config"
```

### Task 2: Expose researcher controls in continuous-binning config

**Files:**
- Modify: `feature_research/continuous_binning/config.py`
- Test: `tests/feature_research/test_config.py`

**Step 1: Write the failing test**

```python
def test_load_config_exposes_selection_and_weight_algorithm() -> None:
    cfg = load_config()
    assert cfg.walkforward_config.selection_method is not None
    assert cfg.walkforward_config.weight_layer_config is not None
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/feature_research/test_config.py -v`
Expected: FAIL due to missing default/plumbing.

**Step 3: Write minimal implementation**

Set researcher-editable defaults for selection and weighting in `load_config()`.

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/feature_research/test_config.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/feature_research/test_config.py feature_research/continuous_binning/config.py
git commit -m "feat: expose selection and weight-layer controls in research config"
```

### Task 3: Refactor walkforward runner to selection-only outputs

**Files:**
- Modify: `feature_research/walkforward/runner.py`
- Test: `tests/feature_research/walkforward/test_runner.py`

**Step 1: Write the failing test**

```python
def test_selection_methods_return_selected_members_only() -> None:
    result = run_walkforward_research(...)
    assert "top_k_features" in result
    assert "averaged_signal" not in result
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/feature_research/walkforward/test_runner.py -k "selection_only" -v`
Expected: FAIL due to legacy semantics.

**Step 3: Write minimal implementation**

Unify method branches to emit selected member identifiers/params only and metadata.

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/feature_research/walkforward/test_runner.py -k "selection_only" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/feature_research/walkforward/test_runner.py feature_research/walkforward/runner.py
git commit -m "refactor: make walkforward methods selection-only"
```

### Task 4: Ensure portfolio evaluator delegates combination to WeightLayer

**Files:**
- Modify: `feature_research/walkforward/portfolio_evaluator.py`
- Modify: `feature_research/walkforward/runner.py`
- Test: `tests/feature_research/walkforward/test_portfolio_evaluator.py`

**Step 1: Write the failing test**

```python
def test_portfolio_evaluator_uses_weight_layer_for_selected_members() -> None:
    result = run_portfolio_simulation(...)
    assert result.weight_layer_method == "inverse_correlation"
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/feature_research/walkforward/test_portfolio_evaluator.py -v`
Expected: FAIL due to missing config propagation.

**Step 3: Write minimal implementation**

Map enum algorithm to `WeightLayerConfig` and pass through evaluator construction.

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/feature_research/walkforward/test_portfolio_evaluator.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/feature_research/walkforward/test_portfolio_evaluator.py feature_research/walkforward/portfolio_evaluator.py feature_research/walkforward/runner.py
git commit -m "feat: route selected-member combination through weight layer"
```

### Task 5: Update docs for selection-only and weight-layer-first behavior

**Files:**
- Modify: `docs/library/Feature_selection/Parameter Sensitivity/top_k_ensemble_selection.md`
- Modify: `docs/library/Ensemble/weight_layer.md`
- Test: `tests/unit-tests/docs/test_walkforward_selection_weightlayer_docs.py`

**Step 1: Write the failing test**

```python
def test_docs_state_selection_only_and_weightlayer_combiner() -> None:
    assert "selection-only" in top_k_doc
    assert "WeightLayer is the combiner" in weight_layer_doc
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/docs/test_walkforward_selection_weightlayer_docs.py -v`
Expected: FAIL until docs updated.

**Step 3: Write minimal implementation**

Update both docs with contract language, remove selection-stage averaging wording.

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/docs/test_walkforward_selection_weightlayer_docs.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/unit-tests/docs/test_walkforward_selection_weightlayer_docs.py docs/library/Feature_selection/Parameter\ Sensitivity/top_k_ensemble_selection.md docs/library/Ensemble/weight_layer.md
git commit -m "docs: define selection-only walkforward and weight-layer-first combining"
```

### Task 6: Validate touched walkforward slice end-to-end

**Files:**
- Test only: `tests/feature_research/walkforward/`

**Step 1: Run focused suite**

Run: `source venv/bin/activate && pytest tests/feature_research/walkforward -v`
Expected: PASS.

**Step 2: Run config + pipeline slice**

Run: `source venv/bin/activate && pytest tests/feature_research/test_config.py tests/feature_research/test_continuous_pipeline_walkforward.py -v`
Expected: PASS.

**Step 3: Commit verification checkpoint**

```bash
git add -A
git commit -m "test: verify selection-only and weight-layer-first walkforward path"
```
