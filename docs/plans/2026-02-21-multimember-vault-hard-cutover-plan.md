# Multi-Member Base Models and Vault Hard-Cutover Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Hard-cut to a multi-member base-model architecture with flattened member signals and vault persistence of selection method/hyperparameters (including top-k) plus member metadata.

**Architecture:** Replace single-member base-model assumptions with deterministic member containers that emit flattened member-level signals. Propagate those signals through diversified ensemble and portfolio layers, and enforce strict new vault/control schema requirements with no legacy compatibility paths.

**Tech Stack:** Python 3, dataclasses, enums, pandas/numpy, pytest, ensemble/vault modules.

---

### Task 1: Define deterministic multi-member naming contract

**Files:**
- Modify: `feature_selection/base_models/feature_base_model.py`
- Test: `tests/unit-tests/feature_selection/base_models/test_model_naming_cutover.py`

**Step 1: Write the failing test**

```python
def test_member_model_name_is_deterministic() -> None:
    assert build_member_model_name("ewmac", "spanFast=32,spanSlow=128") == "ewmac::spanFast=32,spanSlow=128"
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/feature_selection/base_models/test_model_naming_cutover.py -v`
Expected: FAIL because helper/contract missing.

**Step 3: Write minimal implementation**

Add deterministic naming helper and route all member names through it.

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/feature_selection/base_models/test_model_naming_cutover.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/unit-tests/feature_selection/base_models/test_model_naming_cutover.py feature_selection/base_models/feature_base_model.py
git commit -m "feat: add deterministic flattened member naming contract"
```

### Task 2: Refactor BaseModel to own and emit multiple members

**Files:**
- Modify: `feature_selection/base_models/feature_base_model.py`
- Modify: `feature_selection/base_models/base_model.py`
- Modify: `feature_selection/base_models/continuous_binning.py`
- Modify: `feature_selection/base_models/rule_based.py`
- Test: `tests/unit-tests/feature_selection/base_models/test_multi_member_contract.py`

**Step 1: Write the failing test**

```python
def test_base_model_emits_member_level_predictions() -> None:
    preds = base_model.predict(X)
    assert set(preds.columns) == {"feature::member_a", "feature::member_b"}
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/feature_selection/base_models/test_multi_member_contract.py -v`
Expected: FAIL because current model is single-member.

**Step 3: Write minimal implementation**

Implement member collection fit/predict and flattened output in base model classes.

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/feature_selection/base_models/test_multi_member_contract.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/unit-tests/feature_selection/base_models/test_multi_member_contract.py feature_selection/base_models/feature_base_model.py feature_selection/base_models/base_model.py feature_selection/base_models/continuous_binning.py feature_selection/base_models/rule_based.py
git commit -m "refactor: convert base models to multi-member signal emitters"
```

### Task 3: Enforce hard-cut control-file schema for memberized models

**Files:**
- Modify: `ensemble/ensemble_utils.py`
- Test: `tests/unit-tests/ensemble/test_control_file_multi_member_schema.py`

**Step 1: Write the failing test**

```python
def test_legacy_control_file_schema_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_control_file(legacy_payload)
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/ensemble/test_control_file_multi_member_schema.py -v`
Expected: FAIL because legacy payload is still accepted.

**Step 3: Write minimal implementation**

Require member arrays and selection metadata in validator; reject legacy structure.

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/ensemble/test_control_file_multi_member_schema.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/unit-tests/ensemble/test_control_file_multi_member_schema.py ensemble/ensemble_utils.py
git commit -m "feat: enforce hard-cut multi-member control-file schema"
```

### Task 4: Propagate flattened member signals through diversified ensemble

**Files:**
- Modify: `ensemble/diversified_ensemble.py`
- Test: `tests/unit-tests/ensemble/test_diversified_ensemble.py`

**Step 1: Write the failing test**

```python
def test_diversified_ensemble_tracks_member_level_signals() -> None:
    preds = ensemble.predict_from_candles(candles)
    assert "feature::member_a" in preds.model_name.unique()
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/ensemble/test_diversified_ensemble.py -v`
Expected: FAIL due to single-model assumptions.

**Step 3: Write minimal implementation**

Update fit/predict metadata and exposure accounting to use flattened member names.

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/ensemble/test_diversified_ensemble.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/unit-tests/ensemble/test_diversified_ensemble.py ensemble/diversified_ensemble.py
git commit -m "feat: propagate flattened member signals in diversified ensemble"
```

### Task 5: Align weight layer and portfolio to flattened member contract

**Files:**
- Modify: `ensemble/weight_layer.py`
- Modify: `ensemble/portfolio.py`
- Test: `tests/unit-tests/ensemble/test_weight_layer.py`
- Test: `tests/unit-tests/ensemble/test_portfolio_flattened_member_pipeline.py`

**Step 1: Write the failing test**

```python
def test_portfolio_combines_flattened_member_forecasts_with_weight_layer() -> None:
    forecast = portfolio.predict_from_candles(candles)
    assert forecast is not None
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/ensemble/test_weight_layer.py tests/unit-tests/ensemble/test_portfolio_flattened_member_pipeline.py -v`
Expected: FAIL due to naming/path mismatches.

**Step 3: Write minimal implementation**

Normalize flattened model-name handling and remove legacy assumptions in portfolio-weight-layer path.

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/ensemble/test_weight_layer.py tests/unit-tests/ensemble/test_portfolio_flattened_member_pipeline.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/unit-tests/ensemble/test_weight_layer.py tests/unit-tests/ensemble/test_portfolio_flattened_member_pipeline.py ensemble/weight_layer.py ensemble/portfolio.py
git commit -m "fix: align weight-layer portfolio path to flattened member signals"
```

### Task 6: Persist selection hyperparameters and member metadata in vault

**Files:**
- Modify: `ensemble/vault_manager.py`
- Modify: `ensemble/ensemble_utils.py`
- Test: `tests/unit-tests/vault/test_vault_system.py`
- Test: `tests/unit-tests/vault/test_vault_member_schema_cutover.py`

**Step 1: Write the failing test**

```python
def test_vault_persists_topk_hyperparams_and_member_metadata() -> None:
    artifact = save_and_load_strategy(...)
    assert artifact["selection"]["method"] == "top_k"
    assert "top_k" in artifact["selection"]["hyperparams"]
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/vault/test_vault_member_schema_cutover.py -v`
Expected: FAIL due to missing fields.

**Step 3: Write minimal implementation**

Persist required selection block and member metadata, and require them on load.

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/vault/test_vault_member_schema_cutover.py tests/unit-tests/vault/test_vault_system.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/unit-tests/vault/test_vault_member_schema_cutover.py tests/unit-tests/vault/test_vault_system.py ensemble/vault_manager.py ensemble/ensemble_utils.py
git commit -m "feat: persist selection hyperparams and member metadata in vault"
```

### Task 7: Update vault and API documentation for hard cutover

**Files:**
- Modify: `docs/library/Vault/vault_user_guide.md`
- Modify: `docs/api/ensemble.md`
- Modify: `docs/api/base_models.md`

**Step 1: Write the failing doc test**

```python
def test_vault_docs_include_selection_and_member_schema() -> None:
    assert "selection" in vault_doc
    assert "member" in vault_doc
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/docs -k "vault or base_models or ensemble" -v`
Expected: FAIL due to stale docs.

**Step 3: Write minimal implementation**

Update docs to show hard-cut member schema and required selection metadata.

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/docs -k "vault or base_models or ensemble" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/library/Vault/vault_user_guide.md docs/api/ensemble.md docs/api/base_models.md
git commit -m "docs: update vault and api contracts for multi-member hard cutover"
```

### Task 8: Final verification and cutover checks

**Files:**
- Test only: touched test suites

**Step 1: Run targeted unit suites**

Run: `source venv/bin/activate && pytest tests/unit-tests/feature_selection tests/unit-tests/ensemble tests/unit-tests/vault -v`
Expected: PASS.

**Step 2: Run relevant integration suite**

Run: `source venv/bin/activate && pytest tests/integration -k "portfolio or ensemble" -v`
Expected: PASS or explicit skip reason due to cache/data prerequisites.

**Step 3: Run full suite**

Run: `source venv/bin/activate && pytest tests/ -v`
Expected: PASS.

**Step 4: Commit verification checkpoint**

```bash
git add -A
git commit -m "test: verify multi-member hard-cutover across feature, ensemble, and vault"
```

## Risks and Validation Checklist

- Deterministic naming collisions are prevented by including full member identity in `model_name`.
- Legacy schema is intentionally rejected; tests must assert rejection explicitly.
- Vault writer and validator stay aligned via round-trip tests.
- Portfolio/weight-layer integration tests verify no hidden single-member assumptions remain.
- Doc assertions keep user guides and API docs synchronized with cutover behavior.
