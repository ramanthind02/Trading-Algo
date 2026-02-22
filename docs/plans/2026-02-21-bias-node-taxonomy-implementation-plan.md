# Bias Node Taxonomy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize active bias nodes into strategy/family subfolders while preserving current flat loading and import behavior.

**Architecture:** Introduce a canonical categorized node layout, keep root-level compatibility shims for legacy imports, and add deterministic recursive resolver support in `create_bias_node`. Use a small taxonomy mapping for canonical resolution and alias/reference placements so multi-category membership does not duplicate implementation code.

**Tech Stack:** Python 3, `importlib`, `pathlib`, `pytest`

---

### Task 1: Add taxonomy registry and resolver helpers

**Files:**
- Create: `nodes/_taxonomy.py`
- Modify: `utils/helpers.py`
- Test: `tests/unit-tests/nodes/test_taxonomy_resolution.py`

**Step 1: Write the failing test**

```python
from utils.helpers import _resolve_bias_node_module


def test_resolve_bias_node_module_uses_taxonomy_mapping() -> None:
    resolved = _resolve_bias_node_module("rsi")
    assert resolved == "nodes.mean_reversion.rsi.rsi"
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/nodes/test_taxonomy_resolution.py::test_resolve_bias_node_module_uses_taxonomy_mapping -v`
Expected: FAIL because `_resolve_bias_node_module` does not exist.

**Step 3: Write minimal implementation**

```python
# nodes/_taxonomy.py
CANONICAL_MODULES = {
    "rsi": "nodes.mean_reversion.rsi.rsi",
}


# utils/helpers.py
def _resolve_bias_node_module(module_name: str) -> str:
    from nodes._taxonomy import CANONICAL_MODULES
    base_name = module_name.replace(".py", "")
    return CANONICAL_MODULES.get(base_name, f"nodes.{base_name}")
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/nodes/test_taxonomy_resolution.py::test_resolve_bias_node_module_uses_taxonomy_mapping -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nodes/_taxonomy.py utils/helpers.py tests/unit-tests/nodes/test_taxonomy_resolution.py
git commit -m "feat(nodes): add taxonomy-backed module resolver"
```

### Task 2: Add recursive fallback and ambiguity guard

**Files:**
- Modify: `utils/helpers.py`
- Test: `tests/unit-tests/nodes/test_taxonomy_resolution.py`

**Step 1: Write the failing test**

```python
import pytest
from utils.helpers import _resolve_bias_node_module


def test_resolve_bias_node_module_raises_on_ambiguous_match(monkeypatch) -> None:
    monkeypatch.setattr(
        "utils.helpers._find_node_module_candidates",
        lambda _: ["nodes.a.rsi", "nodes.b.rsi"],
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        _resolve_bias_node_module("rsi")
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/nodes/test_taxonomy_resolution.py::test_resolve_bias_node_module_raises_on_ambiguous_match -v`
Expected: FAIL because ambiguity handling is not implemented.

**Step 3: Write minimal implementation**

```python
def _find_node_module_candidates(base_module_name: str) -> list[str]:
    # recursively scan nodes/ for files named <base_module_name>.py (excluding archive)
    ...


def _resolve_bias_node_module(module_name: str) -> str:
    ...
    candidates = _find_node_module_candidates(base_name)
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous module '{base_name}': {candidates}")
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(...)
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/nodes/test_taxonomy_resolution.py::test_resolve_bias_node_module_raises_on_ambiguous_match -v`
Expected: PASS

**Step 5: Commit**

```bash
git add utils/helpers.py tests/unit-tests/nodes/test_taxonomy_resolution.py
git commit -m "feat(nodes): add recursive fallback and ambiguity guard"
```

### Task 3: Rewire create_bias_node to use resolver

**Files:**
- Modify: `utils/helpers.py`
- Test: `tests/unit-tests/nodes/test_taxonomy_resolution.py`

**Step 1: Write the failing test**

```python
from utils.enums import TimeFrame, Ticker
from utils.helpers import create_bias_node


def test_create_bias_node_resolves_canonical_module_path() -> None:
    node = create_bias_node("rsi", Ticker.ES, TimeFrame.D, {"lookback": 14})
    assert node.module_name == "rsi"
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/nodes/test_taxonomy_resolution.py::test_create_bias_node_resolves_canonical_module_path -v`
Expected: FAIL if import path resolution still assumes `nodes/<module>.py` root file only.

**Step 3: Write minimal implementation**

```python
def create_bias_node(module_name: str, ticker: Ticker, tf: TimeFrame, params: Dict) -> Any:
    ...
    resolved_module_path = _resolve_bias_node_module(base_module_name)
    module = importlib.import_module(resolved_module_path)
    ...
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/nodes/test_taxonomy_resolution.py::test_create_bias_node_resolves_canonical_module_path -v`
Expected: PASS

**Step 5: Commit**

```bash
git add utils/helpers.py tests/unit-tests/nodes/test_taxonomy_resolution.py
git commit -m "refactor(nodes): resolve bias nodes through taxonomy resolver"
```

### Task 4: Create category package skeletons and move first family (RSI)

**Files:**
- Create: `nodes/mean_reversion/__init__.py`
- Create: `nodes/mean_reversion/rsi/__init__.py`
- Move: `nodes/rsi.py -> nodes/mean_reversion/rsi/rsi.py`
- Create: `nodes/rsi.py` (compatibility shim)
- Test: `tests/unit-tests/nodes/test_new_bias_nodes.py`

**Step 1: Write the failing test**

```python
def test_root_rsi_import_remains_compatible() -> None:
    from nodes.rsi import RSI
    assert RSI.__name__ == "RSI"
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/nodes/test_new_bias_nodes.py::test_root_rsi_import_remains_compatible -v`
Expected: FAIL while files are moved and shim is not yet present.

**Step 3: Write minimal implementation**

```python
# nodes/rsi.py
from nodes.mean_reversion.rsi.rsi import RSI

__all__ = ["RSI"]
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/nodes/test_new_bias_nodes.py::test_root_rsi_import_remains_compatible -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nodes/mean_reversion/__init__.py nodes/mean_reversion/rsi/__init__.py nodes/mean_reversion/rsi/rsi.py nodes/rsi.py tests/unit-tests/nodes/test_new_bias_nodes.py
git commit -m "refactor(nodes): move rsi to mean_reversion with compatibility shim"
```

### Task 5: Migrate remaining active nodes to canonical categories

**Files:**
- Modify/Create: categorized node paths under `nodes/` for all active root modules
- Modify/Create: root-level compatibility shims for each migrated module
- Modify/Create: optional alias/reference modules for multi-category membership
- Modify: `nodes/_taxonomy.py`
- Test: `tests/unit-tests/nodes/test_cython_bias_nodes.py`
- Test: `tests/unit-tests/nodes/test_new_bias_nodes.py`
- Test: `tests/unit-tests/nodes/test_basic_breakout_basic_mr.py`

**Step 1: Write the failing test**

```python
from utils.enums import TimeFrame, Ticker
from utils.helpers import create_bias_node


def test_all_active_modules_still_construct_nodes() -> None:
    module_names = ["rsi", "momentum", "atr", "buy_hold", "basic_breakout"]
    for module_name in module_names:
        node = create_bias_node(module_name, Ticker.ES, TimeFrame.D, {})
        assert node is not None
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/unit-tests/nodes/test_taxonomy_resolution.py::test_all_active_modules_still_construct_nodes -v`
Expected: FAIL for unmigrated or unmapped modules.

**Step 3: Write minimal implementation**

```python
# nodes/_taxonomy.py
CANONICAL_MODULES = {
    "rsi": "nodes.mean_reversion.rsi.rsi",
    "stochastic_rsi": "nodes.mean_reversion.rsi.stochastic_rsi",
    "momentum": "nodes.momentum.core.momentum",
    "simple_momentum": "nodes.momentum.core.simple_momentum",
    "atr": "nodes.volatility.atr.atr",
    ...
}
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/unit-tests/nodes/test_taxonomy_resolution.py::test_all_active_modules_still_construct_nodes -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nodes/ utils/helpers.py tests/unit-tests/nodes/test_taxonomy_resolution.py
git commit -m "refactor(nodes): migrate active node modules into taxonomy layout"
```

### Task 6: Validate extraction/research integration remains unchanged

**Files:**
- Test: `tests/integration/test_cache_integration.py`
- Test: add `tests/integration/test_bias_node_taxonomy_extraction.py`

**Step 1: Write the failing test**

```python
def test_extract_features_still_accepts_flat_module_name() -> None:
    from feature_extraction.feature_extractor import extract_features
    from utils.enums import Ticker

    features_df, targets_df = extract_features(
        module_name="rsi",
        params={"lookback": 14},
        ticker=Ticker.ES,
    )
    assert not features_df.empty
    assert not targets_df.empty
```

**Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/integration/test_bias_node_taxonomy_extraction.py::test_extract_features_still_accepts_flat_module_name -v`
Expected: FAIL if loader compatibility regressed.

**Step 3: Write minimal implementation**

```python
# If failing, fix only resolver/shim/taxonomy gaps. No call-site API changes.
```

**Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/integration/test_bias_node_taxonomy_extraction.py::test_extract_features_still_accepts_flat_module_name -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/integration/test_bias_node_taxonomy_extraction.py nodes/ utils/helpers.py
git commit -m "test(nodes): verify flat module extraction compatibility after taxonomy migration"
```

### Task 7: Full regression checks and docs sync

**Files:**
- Modify: `docs/api/` pages that mention node locations (if any)
- Modify: `docs/kanban/` task artifacts for traceability

**Step 1: Write the failing test**

```python
# N/A (verification and docs task)
```

**Step 2: Run checks**

Run: `source venv/bin/activate && pytest tests/unit-tests/nodes -v`
Expected: PASS

Run: `source venv/bin/activate && pytest tests/integration/test_cache_integration.py -v`
Expected: PASS or explicit skip with cache reason

Run: `source venv/bin/activate && pytest tests/ -v`
Expected: PASS for full suite

**Step 3: Write minimal implementation**

```python
# If failures occur, make smallest compatible fixes and re-run until green.
```

**Step 4: Verify all checks pass**

Run: `source venv/bin/activate && pytest tests/ -v`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/api docs/kanban
git commit -m "docs(nodes): sync taxonomy reorganization docs and traceability"
```
