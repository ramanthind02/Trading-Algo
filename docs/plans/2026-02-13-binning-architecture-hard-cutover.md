# Binning Architecture Hard Cutover Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace legacy best-bin binning with the new continuous/rule-based architecture, perform a global naming cutover to `continuous_binning` and `rule_based`, and ship schema-breaking fitted-state persistence.

**Architecture:** Rewrite `BinningModelBase` and both concrete binning models in place while renaming files/classes and replacing persistence payloads with explicit v2 fields (`bin_stats`, regions, multipliers, fit config). Update ensemble/vault factories and save/load paths to consume only new identifiers and schema keys. Preserve the fit/predict interface shape, but change runtime behavior to adjusted-Sharpe continuous multipliers and explicit region filtering.

**Tech Stack:** Python 3.12, pandas, numpy, pytest, existing ensemble/vault infrastructure.

---

**Execution notes:**
- Use `@superpowers/test-driven-development` for every behavior change.
- Use `@superpowers/systematic-debugging` for any failing tests outside expected RED state.
- Use `@superpowers/verification-before-completion` before claiming completion.
- Activate environment before all commands: `source venv/bin/activate`.

### Task 1: Global Naming Cutover (Files, Classes, Imports)

**Files:**
- Modify: `feature_selection/base_models/quantile_binning.py`
- Modify: `feature_selection/base_models/rule_based_binning.py`
- Modify: `feature_selection/base_models/__init__.py`
- Modify: `ensemble/ensemble_utils.py`
- Modify: `deployment/production_training_pipeline.py`
- Modify: `ensemble/vault_manager.py`
- Modify: `tests/test_ensemble_base_models.py`
- Modify: `tests/test_vault_system.py`

**Step 1: Write failing import tests for new names**

```python
# tests/base_models/test_model_naming_cutover.py
from feature_selection.base_models import ContinuousBinningModel, RuleBasedModel


def test_new_model_names_exported() -> None:
    assert ContinuousBinningModel.__name__ == "ContinuousBinningModel"
    assert RuleBasedModel.__name__ == "RuleBasedModel"
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/base_models/test_model_naming_cutover.py::test_new_model_names_exported -v`
Expected: FAIL with import error for missing names.

**Step 3: Implement minimal rename layer**

```python
# feature_selection/base_models/continuous_binning.py
from feature_selection.base_models.base_model import BinningModelBase


class ContinuousBinningModel(BinningModelBase):
    ...
```

```python
# feature_selection/base_models/rule_based.py
from feature_selection.base_models.base_model import BinningModelBase


class RuleBasedModel(BinningModelBase):
    ...
```

```python
# feature_selection/base_models/__init__.py
from feature_selection.base_models.continuous_binning import ContinuousBinningModel
from feature_selection.base_models.rule_based import RuleBasedModel
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/base_models/test_model_naming_cutover.py::test_new_model_names_exported -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add feature_selection/base_models/__init__.py \
        feature_selection/base_models/continuous_binning.py \
        feature_selection/base_models/rule_based.py \
        tests/base_models/test_model_naming_cutover.py
git commit -m "refactor: cut over binning model naming to continuous_binning and rule_based"
```

### Task 2: Replace Legacy Best-Bin State Model in Binning Base

**Files:**
- Modify: `feature_selection/base_models/base_model.py`
- Create: `tests/base_models/test_binning_base_stats.py`

**Step 1: Write failing tests for new stats payload and removed best-bin semantics**

```python
import pandas as pd
import numpy as np
from feature_selection.base_models.continuous_binning import ContinuousBinningModel


def test_fit_builds_v2_stats_without_best_bin_fields() -> None:
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    feature = pd.Series(np.linspace(0, 1, 200), index=idx, name="rsi_signal_D_lookback_14")
    target = pd.Series(np.random.normal(0.0, 0.01, 200), index=idx)

    model = ContinuousBinningModel(n_bins=10)
    model.fit(feature, target)

    assert model.is_fitted_ is True
    assert isinstance(model.bin_stats_, dict)
    assert hasattr(model, "position_multipliers_by_strategy_")
    assert not hasattr(model, "best_long_bin_")
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/base_models/test_binning_base_stats.py::test_fit_builds_v2_stats_without_best_bin_fields -v`
Expected: FAIL because old attributes/fields still exist and new fields do not.

**Step 3: Implement minimal v2 fitted-state shape in base class**

```python
# in BinningModelBase.__init__
self.bin_edges_ = None
self.significant_regions_ = []
self.active_bins_by_strategy_ = {"long": [], "short": [], "long_short": []}
self.position_multipliers_by_strategy_ = {"long": {}, "short": {}, "long_short": {}}
self.fit_config_ = {}
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/base_models/test_binning_base_stats.py::test_fit_builds_v2_stats_without_best_bin_fields -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add feature_selection/base_models/base_model.py tests/base_models/test_binning_base_stats.py
git commit -m "refactor: replace best-bin state with v2 binning state containers"
```

### Task 3: Implement Continuous Stats + Adjusted-Sharpe Multipliers

**Files:**
- Modify: `feature_selection/base_models/base_model.py`
- Modify: `feature_selection/base_models/continuous_binning.py`
- Create: `tests/base_models/test_continuous_binning_multipliers.py`

**Step 1: Write failing tests for sharpe, adjusted_sharpe, t_stat, clipping**

```python
import numpy as np
import pandas as pd
from feature_selection.base_models.continuous_binning import ContinuousBinningModel


def test_continuous_predict_returns_clipped_multipliers() -> None:
    idx = pd.date_range("2021-01-01", periods=300, freq="D")
    feature = pd.Series(np.linspace(-2, 2, 300), index=idx, name="rsi_signal_D_lookback_14")
    target = pd.Series(np.where(feature > 0.5, 0.02, -0.01), index=idx)

    model = ContinuousBinningModel(n_bins=15, selection_metric="sharpe", metric_threshold=0.0)
    model.fit(feature, target)
    pred = model.predict(feature, strategy="long_short")

    assert pred.max() <= 2.0
    assert pred.min() >= -2.0
    assert np.any(pred > 0)
    assert np.any(pred < 0)
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/base_models/test_continuous_binning_multipliers.py::test_continuous_predict_returns_clipped_multipliers -v`
Expected: FAIL because legacy predict returns binary bins.

**Step 3: Implement minimal continuous multiplier pipeline**

```python
# base_model.py helper
adjusted = sharpe * np.sqrt(n) / (np.sqrt(n) + self.shrinkage_k)

# long multiplier
long_mult = float(np.clip(1.0 + adjusted, self.long_clip_min, self.long_clip_max))
# short multiplier
short_mult = -float(np.clip(1.0 + adjusted, self.short_clip_min, self.short_clip_max))
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/base_models/test_continuous_binning_multipliers.py::test_continuous_predict_returns_clipped_multipliers -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add feature_selection/base_models/base_model.py \
        feature_selection/base_models/continuous_binning.py \
        tests/base_models/test_continuous_binning_multipliers.py
git commit -m "feat: implement continuous adjusted-sharpe multiplier output"
```

### Task 4: Add Region Detection + Direction-Aware Threshold Filter

**Files:**
- Modify: `feature_selection/base_models/base_model.py`
- Create: `tests/base_models/test_continuous_regions.py`

**Step 1: Write failing tests for contiguous region selection**

```python
import numpy as np
import pandas as pd
from feature_selection.base_models.continuous_binning import ContinuousBinningModel


def test_regions_require_min_consecutive_bins() -> None:
    idx = pd.date_range("2022-01-01", periods=400, freq="D")
    feature = pd.Series(np.linspace(0, 100, 400), index=idx, name="rsi_signal_D_lookback_14")
    target = pd.Series(np.where((feature > 20) & (feature < 35), 0.03, 0.0), index=idx)

    model = ContinuousBinningModel(
        n_bins=15,
        t_threshold=1.75,
        min_region_width=2,
        selection_metric="sharpe",
        metric_threshold=0.0,
    )
    model.fit(feature, target)

    assert len(model.significant_regions_) >= 1
    assert all((r[1] - r[0] + 1) >= 2 for r in model.significant_regions_)
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/base_models/test_continuous_regions.py::test_regions_require_min_consecutive_bins -v`
Expected: FAIL due missing region logic.

**Step 3: Implement minimal region detector + threshold filter**

```python
# pseudo in base_model.py
active = [idx for idx, stat in stats.items() if abs(stat["t_stat"]) >= self.t_threshold]
regions = _group_consecutive(active)
regions = [r for r in regions if (r[1] - r[0] + 1) >= self.min_region_width]
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/base_models/test_continuous_regions.py::test_regions_require_min_consecutive_bins -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add feature_selection/base_models/base_model.py tests/base_models/test_continuous_regions.py
git commit -m "feat: add contiguous region detection and threshold filtering"
```

### Task 5: Rewrite Rule-Based Model to Stats-Driven Multipliers

**Files:**
- Modify: `feature_selection/base_models/rule_based.py`
- Modify: `feature_selection/base_models/base_model.py`
- Create: `tests/base_models/test_rule_based_multipliers.py`

**Step 1: Write failing tests for -1/0/+1 mapping to multipliers**

```python
import pandas as pd
import numpy as np
from feature_selection.base_models.rule_based import RuleBasedModel


def test_rule_based_maps_levels_to_signed_multipliers() -> None:
    idx = pd.date_range("2020-01-01", periods=240, freq="D")
    feature = pd.Series(np.tile([-1, 0, 1, 1], 60), index=idx, name="breakout_signal_D_lookback_20")
    target = pd.Series(np.where(feature == 1, 0.02, np.where(feature == -1, -0.015, 0.0)), index=idx)

    model = RuleBasedModel(selection_metric="sharpe", metric_threshold=0.0)
    model.fit(feature, target)
    pred = model.predict(feature, strategy="long_short")

    assert set(np.sign(pred.unique())) <= {-1.0, 0.0, 1.0}
    assert (pred[feature == 0] == 0).all()
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/base_models/test_rule_based_multipliers.py::test_rule_based_maps_levels_to_signed_multipliers -v`
Expected: FAIL because current model is pass-through.

**Step 3: Implement rule-based fit/predict using shared stats path**

```python
# rule_based.py
def _create_bins(self, feature_data: pd.Series, target_data: pd.Series) -> pd.Series:
    level_map = {-1: 0, 0: 1, 1: 2}
    return feature_data.map(level_map)
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/base_models/test_rule_based_multipliers.py::test_rule_based_maps_levels_to_signed_multipliers -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add feature_selection/base_models/rule_based.py \
        feature_selection/base_models/base_model.py \
        tests/base_models/test_rule_based_multipliers.py
git commit -m "feat: replace rule-based pass-through with stats-driven multipliers"
```

### Task 6: Implement New Fitted Schema in BaseModel Snapshots

**Files:**
- Modify: `feature_selection/base_models/feature_base_model.py`
- Create: `tests/base_models/test_feature_base_model_fitted_schema_v2.py`

**Step 1: Write failing test for v2 fitted payload keys**

```python
def test_feature_base_model_exports_v2_fitted_payload(base_model):
    payload = base_model.get_fitted_params()
    assert "model_version" in payload
    assert "position_multipliers_by_strategy" in payload
    assert "fit_config" in payload
    assert "best_long_bin" not in payload
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/base_models/test_feature_base_model_fitted_schema_v2.py::test_feature_base_model_exports_v2_fitted_payload -v`
Expected: FAIL because old keys are still returned.

**Step 3: Implement minimal v2 snapshot structure**

```python
return {
    "model_version": "binning_v2",
    "model_type": self.binning_model.model_type,
    "bin_edges": self.binning_model.bin_edges_,
    "bin_stats": self.binning_model.bin_stats_,
    "significant_regions": self.binning_model.significant_regions_,
    "active_bins_by_strategy": self.binning_model.active_bins_by_strategy_,
    "position_multipliers_by_strategy": self.binning_model.position_multipliers_by_strategy_,
    "fit_config": self.binning_model.fit_config_,
}
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/base_models/test_feature_base_model_fitted_schema_v2.py::test_feature_base_model_exports_v2_fitted_payload -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add feature_selection/base_models/feature_base_model.py \
        tests/base_models/test_feature_base_model_fitted_schema_v2.py
git commit -m "refactor: emit v2 fitted schema from feature base model"
```

### Task 7: Cut Over Ensemble Factory/Loader to New `model_type` IDs

**Files:**
- Modify: `ensemble/ensemble_utils.py`
- Modify: `ensemble/diversified_ensemble.py`
- Create: `tests/ensemble/test_model_factory_v2_ids.py`

**Step 1: Write failing tests for `model_type` IDs**

```python
from ensemble.ensemble_utils import create_base_model_from_config


def test_factory_accepts_continuous_binning_id(config_fixture):
    config_fixture["model_type"] = "continuous_binning"
    model = create_base_model_from_config(config_fixture)
    assert model.binning_model.model_type == "continuous_binning"
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/ensemble/test_model_factory_v2_ids.py::test_factory_accepts_continuous_binning_id -v`
Expected: FAIL with unsupported model type.

**Step 3: Implement new model-type routing and strict rejection of old IDs**

```python
if model_type == "continuous_binning":
    binning_model = ContinuousBinningModel(**constructor_params)
elif model_type == "rule_based":
    binning_model = RuleBasedModel(**constructor_params)
else:
    raise ValueError("Unsupported model_type. Expected 'continuous_binning' or 'rule_based'.")
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/ensemble/test_model_factory_v2_ids.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add ensemble/ensemble_utils.py ensemble/diversified_ensemble.py tests/ensemble/test_model_factory_v2_ids.py
git commit -m "refactor: switch ensemble factory to v2 model_type identifiers"
```

### Task 8: Cut Over Control/Vault Persistence and Old-Schema Rejection

**Files:**
- Modify: `ensemble/diversified_ensemble.py`
- Modify: `ensemble/vault_manager.py`
- Modify: `ensemble/ensemble_utils.py`
- Create: `tests/ensemble/test_fitted_schema_v2_roundtrip.py`

**Step 1: Write failing tests for v2 round-trip and old-schema rejection**

```python
def test_control_file_roundtrip_uses_v2_keys(tmp_path, fitted_ensemble):
    path = tmp_path / "control.json"
    fitted_ensemble.save_control_file(str(path))
    loaded = json.loads(path.read_text())

    payload = next(iter(loaded["fitted_base_models"].values()))
    assert payload["model_version"] == "binning_v2"
    assert "position_multipliers_by_strategy" in payload


def test_old_schema_fails_to_load(tmp_path, old_schema_control_file):
    with pytest.raises(ValueError, match="binning_v2"):
        DiversifiedEnsemble(target_volatility=0.2, control_file_path=str(old_schema_control_file))
```

**Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && pytest tests/ensemble/test_fitted_schema_v2_roundtrip.py -v`
Expected: FAIL because loaders still accept old schema / writers emit old fields.

**Step 3: Implement v2 write/read schema and explicit migration errors**

```python
if fitted_payload.get("model_version") != "binning_v2":
    raise ValueError("Unsupported fitted schema. Regenerate fitted models with binning_v2.")
```

**Step 4: Run tests to verify they pass**

Run: `source venv/bin/activate && pytest tests/ensemble/test_fitted_schema_v2_roundtrip.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add ensemble/diversified_ensemble.py ensemble/vault_manager.py ensemble/ensemble_utils.py \
        tests/ensemble/test_fitted_schema_v2_roundtrip.py
git commit -m "feat: enforce binning_v2 fitted schema in control and vault persistence"
```

### Task 9: Update Existing Integration Tests to New Names and Behavior

**Files:**
- Modify: `tests/test_ensemble_base_models.py`
- Modify: `tests/test_vault_system.py`
- Modify: `tests/test_buy_hold_volatility_scaling.py`

**Step 1: Write/adjust failing assertions for renamed models and v2 payload keys**

```python
assert model_config["model_type"] == "continuous_binning"
assert "position_multipliers_by_strategy" in fitted_model
assert "best_long_bin" not in fitted_model
```

**Step 2: Run targeted tests to verify failures**

Run: `source venv/bin/activate && pytest tests/test_ensemble_base_models.py -v`
Expected: FAIL on old model names and old fitted schema assertions.

**Step 3: Implement minimal test and fixture updates**

```python
# replace constructor usage
model = ContinuousBinningModel(n_bins=3, selection_metric="sharpe", strategy="long")
```

**Step 4: Run targeted tests to verify pass**

Run: `source venv/bin/activate && pytest tests/test_ensemble_base_models.py -v`
Run: `source venv/bin/activate && pytest tests/test_vault_system.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_ensemble_base_models.py tests/test_vault_system.py tests/test_buy_hold_volatility_scaling.py
git commit -m "test: align ensemble and vault tests with v2 binning architecture"
```

### Task 10: Final Verification Sweep

**Files:**
- Modify: `feature_selection/base_models/base_model.py`
- Modify: `feature_selection/base_models/continuous_binning.py`
- Modify: `feature_selection/base_models/rule_based.py`
- Modify: `ensemble/ensemble_utils.py`
- Modify: `ensemble/diversified_ensemble.py`
- Modify: `ensemble/vault_manager.py`

**Step 1: Run targeted suites**

Run: `source venv/bin/activate && pytest tests/base_models -v`
Expected: PASS.

**Step 2: Run cross-layer integration suites**

Run: `source venv/bin/activate && pytest tests/test_ensemble_base_models.py tests/test_vault_system.py -v`
Expected: PASS.

**Step 3: Run permutation-related regression checks if touched by interface changes**

Run: `source venv/bin/activate && pytest tests/test_permutation_candle_shuffle.py tests/test_permutation_engine_bar_strategy.py -v`
Expected: PASS.

**Step 4: Run broader suite for confidence**

Run: `source venv/bin/activate && pytest tests/ -v`
Expected: PASS (or documented known failures unrelated to this branch).

**Step 5: Commit final cleanup and docs sync**

```bash
git add -A
git commit -m "refactor: complete hard cutover to continuous/rule_based binning architecture"
```

### Task 11: Documentation Sync (Post-Implementation)

**Files:**
- Modify: `docs/library/Feature_selection/features/Continuous_binning.md`
- Modify: `docs/library/Feature_selection/features/rule_based.md`
- Modify: `docs/library/Feature_selection/base_model/base_model.md`
- Modify: `docs/library/Feature_selection/features/base_feature.md`

**Step 1: Write failing docs-consistency test (optional linter/grep check)**

```python
# tests/docs/test_binning_naming_consistency.py
from pathlib import Path


def test_docs_do_not_reference_legacy_model_names() -> None:
    docs = Path("docs/library/Feature_selection").rglob("*.md")
    text = "\n".join(p.read_text() for p in docs)
    assert "QuantileBinningModel" not in text
    assert "RuleBasedBinningModel" not in text
```

**Step 2: Run to verify failure**

Run: `source venv/bin/activate && pytest tests/docs/test_binning_naming_consistency.py -v`
Expected: FAIL before doc updates.

**Step 3: Update docs to match shipped behavior and naming**

```markdown
- model_type: `continuous_binning` or `rule_based`
- fitted schema version: `binning_v2`
```

**Step 4: Run docs consistency test again**

Run: `source venv/bin/activate && pytest tests/docs/test_binning_naming_consistency.py -v`
Expected: PASS.

**Step 5: Commit docs changes**

```bash
git add docs/library/Feature_selection tests/docs/test_binning_naming_consistency.py
git commit -m "docs: align feature-selection docs with continuous/rule_based hard cutover"
```
