# Shared Walkforward Research Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a shared walkforward research workflow used by both rule-based and continuous feature research pipelines, with fold visualization, metric-name model selection, and per-fold selected-feature outputs in a shared results layout.

**Architecture:** Introduce a new `feature_research/walkforward` shared package for config, metric registry, fold orchestration, plotting, and artifact writing. Keep rule-based and continuous modules as thin adapters that pass feature extraction hooks and local config into the shared runner. Reuse existing walkforward splitter and Stage 3 stability utilities where possible.

**Tech Stack:** Python, pandas, NumPy, matplotlib, pytest.

---

### Task 1: Create Kanban Task Contract

**Files:**
- Create: `docs/kanban/to-do/feature_validator/T022_shared_walkforward_research_pipeline.md`
- Modify: `docs/kanban/in-progress/` (move task here before code changes)

**Step 1: Draft kanban task from template**

Use `docs/kanban/templates/feature.md` and include:
- Context refs: `docs/library/Feature_selection/Walkforward/walkforward.md`, `docs/library/Feature_selection/feature_validator.md`, `utils/walkforward.py`, `docs/plans/2026-02-17-walkforward-shared-research-design.md`
- Scope: shared walkforward package + rule-based/continuous integration
- Interfaces: exact files in Tasks 2-8 below
- Acceptance tests: commands listed in Task 8

**Step 2: Move task to in-progress before implementation**

Move the kanban file from `to-do` to `in-progress` before any production code edits.

**Step 3: Commit**

```bash
git add docs/kanban/to-do/feature_validator/T022_shared_walkforward_research_pipeline.md docs/kanban/in-progress/
git commit -m "Add kanban task for shared walkforward research pipeline"
```

---

### Task 2: Add Shared Walkforward Config Dataclass (TDD)

**Files:**
- Create: `feature_research/walkforward/config.py`
- Create: `feature_research/walkforward/__init__.py`
- Create: `tests/feature_research/test_walkforward_config.py`

**Step 1: Write failing tests**

```python
from datetime import datetime
from pathlib import Path

import pytest

from feature_research.walkforward.config import WalkforwardResearchConfig


def test_walkforward_config_defaults():
    cfg = WalkforwardResearchConfig.default(
        output_root=Path("feature_research/shared_results"),
    )
    assert cfg.enabled is True
    assert cfg.test_step > 0
    assert cfg.num_steps > 0
    assert cfg.top_k >= 1
    assert cfg.objective_metric_name == "sharpe"


def test_walkforward_config_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        WalkforwardResearchConfig(
            enabled=True,
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2022, 1, 1),
            test_step=252,
            num_steps=5,
            top_k=0,
            objective_metric_name="sharpe",
            min_fold_samples=10,
            output_root=Path("feature_research/shared_results"),
        )
```

**Step 2: Run test to verify failure**

Run: `pytest tests/feature_research/test_walkforward_config.py -v`

Expected: FAIL (`ModuleNotFoundError` / missing class)

**Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class WalkforwardResearchConfig:
    ...

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/feature_research/test_walkforward_config.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add feature_research/walkforward/config.py feature_research/walkforward/__init__.py tests/feature_research/test_walkforward_config.py
git commit -m "Add shared walkforward config dataclass"
```

---

### Task 3: Add Objective Metric Registry (TDD)

**Files:**
- Create: `feature_research/walkforward/metrics.py`
- Create: `tests/feature_research/test_walkforward_metrics.py`

**Step 1: Write failing tests**

```python
import pandas as pd
import pytest

from feature_research.walkforward.metrics import get_objective_metric


def test_get_objective_metric_sharpe():
    fn = get_objective_metric("sharpe")
    s = pd.Series([0.01, -0.005, 0.02, 0.003])
    assert isinstance(fn(s), float)


def test_get_objective_metric_unknown_raises():
    with pytest.raises(ValueError, match="Unknown objective metric"):
        get_objective_metric("not_real")
```

**Step 2: Run test to verify failure**

Run: `pytest tests/feature_research/test_walkforward_metrics.py -v`

Expected: FAIL (missing module/function)

**Step 3: Write minimal implementation**

```python
def get_objective_metric(name: str) -> Callable[[pd.Series], float]:
    registry = {"sharpe": sharpe, "sortino": sortino, "mean_return": mean_return}
    if name not in registry:
        raise ValueError(f"Unknown objective metric: {name}")
    return registry[name]
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/feature_research/test_walkforward_metrics.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add feature_research/walkforward/metrics.py tests/feature_research/test_walkforward_metrics.py
git commit -m "Add walkforward objective metric registry"
```

---

### Task 4: Implement Shared Runner + Selection Summary (TDD)

**Files:**
- Create: `feature_research/walkforward/runner.py`
- Create: `tests/feature_research/test_walkforward_runner.py`

**Step 1: Write failing test for selected feature per fold**

```python
def test_runner_emits_selected_feature_for_each_fold():
    report = run_walkforward_research(...)
    assert len(report.selection_rows) > 0
    assert all(row["selected_feature"] for row in report.selection_rows)
```

**Step 2: Run targeted test and confirm failure**

Run: `pytest tests/feature_research/test_walkforward_runner.py::test_runner_emits_selected_feature_for_each_fold -v`

Expected: FAIL (missing runner/report)

**Step 3: Implement minimal runner using shared utilities**

Core behavior:
- Build splits via `WalkForwardSplitter`.
- Evaluate full param grid per fold via adapter callback.
- Compute neighbor-smoothed objectives.
- Rank deterministically and set `selected_feature` from rank-1.

```python
selected = ranked[0]["param_label"]
selection_rows.append({"fold": fold_id, "selected_feature": selected, ...})
```

**Step 4: Run runner tests**

Run: `pytest tests/feature_research/test_walkforward_runner.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add feature_research/walkforward/runner.py tests/feature_research/test_walkforward_runner.py
git commit -m "Implement shared walkforward runner with per-fold selection"
```

---

### Task 5: Add Fold Timeline Plot + Artifact Writers (TDD)

**Files:**
- Create: `feature_research/walkforward/visualization.py`
- Create: `feature_research/walkforward/io.py`
- Create: `tests/feature_research/test_walkforward_outputs.py`

**Step 1: Write failing output test**

```python
def test_write_outputs_creates_required_files(tmp_path):
    write_walkforward_outputs(..., output_dir=tmp_path)
    assert (tmp_path / "folds.csv").exists()
    assert (tmp_path / "selection_summary.csv").exists()
    assert (tmp_path / "fold_timeline.png").exists()
```

**Step 2: Run test and confirm failure**

Run: `pytest tests/feature_research/test_walkforward_outputs.py -v`

Expected: FAIL (writers/plotter missing)

**Step 3: Implement writers and timeline visualization**

`write_walkforward_outputs` must emit:
- `folds.csv`
- `fold_scores.csv`
- `selection_summary.csv`
- `report.json`
- `walkforward_stability.png`
- `fold_timeline.png`

**Step 4: Run tests**

Run: `pytest tests/feature_research/test_walkforward_outputs.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add feature_research/walkforward/visualization.py feature_research/walkforward/io.py tests/feature_research/test_walkforward_outputs.py
git commit -m "Add walkforward fold visualization and artifact writers"
```

---

### Task 6: Wire Rule-Based Config + Pipeline Into Shared Runner (TDD)

**Files:**
- Modify: `feature_research/rule_based/config.py`
- Modify: `feature_research/rule_based/pipeline.py`
- Modify: `feature_research/rule_based/__init__.py`
- Modify: `tests/feature_research/test_rule_based_config.py`

**Step 1: Add failing config test for walkforward params**

```python
def test_rule_based_config_includes_walkforward_block():
    cfg = load_config()
    assert cfg.walkforward.objective_metric_name == "sharpe"
    assert cfg.walkforward.top_k >= 1
```

**Step 2: Run test and confirm failure**

Run: `pytest tests/feature_research/test_rule_based_config.py -v`

Expected: FAIL (`walkforward` field missing)

**Step 3: Implement config + optional pipeline invocation**

Pipeline behavior:
- Keep existing EDA behavior unchanged.
- If `config.walkforward.enabled`, run shared runner and write outputs under:
  `feature_research/shared_results/rule_based/{module_name}/walkforward/`

**Step 4: Run tests**

Run: `pytest tests/feature_research/test_rule_based_config.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add feature_research/rule_based/config.py feature_research/rule_based/pipeline.py feature_research/rule_based/__init__.py tests/feature_research/test_rule_based_config.py
git commit -m "Wire rule-based research config and pipeline to shared walkforward"
```

---

### Task 7: Wire Continuous Pipeline To Same Shared Runner (TDD)

**Files:**
- Modify: `feature_research/continuous_binning/config.py`
- Modify: `feature_research/continuous_binning/pipeline.py`
- Modify: `feature_research/continuous_binning/__init__.py`
- Modify: `tests/feature_research/test_config.py`

**Step 1: Add failing continuous config test for shared walkforward config use**

```python
def test_continuous_config_has_walkforward_block():
    cfg = load_config()
    assert cfg.walkforward.enabled is True
```

**Step 2: Run test and confirm failure**

Run: `pytest tests/feature_research/test_config.py -v`

Expected: FAIL (`walkforward` field missing)

**Step 3: Implement continuous adapter wiring**

Ensure output root is:
`feature_research/shared_results/continuous/{module_name}/walkforward/`

**Step 4: Run tests**

Run: `pytest tests/feature_research/test_config.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add feature_research/continuous_binning/config.py feature_research/continuous_binning/pipeline.py feature_research/continuous_binning/__init__.py tests/feature_research/test_config.py
git commit -m "Reuse shared walkforward runner in continuous research pipeline"
```

---

### Task 8: Add Integration Coverage + API Docs

**Files:**
- Create: `tests/integration/feature_validator/test_walkforward_research_pipeline.py`
- Modify: `docs/api/feature_selection.md`
- Modify: `docs/kanban/in-progress/T022_shared_walkforward_research_pipeline.md`

**Step 1: Add integration tests (rule-based + continuous smoke paths)**

Test assertions:
- generated shared walkforward output directory exists,
- `selection_summary.csv` exists with `selected_feature` column,
- `fold_timeline.png` exists,
- one row per valid fold in `folds.csv`.

**Step 2: Run targeted tests**

Run:
- `pytest tests/feature_research/test_walkforward_config.py -v`
- `pytest tests/feature_research/test_walkforward_metrics.py -v`
- `pytest tests/feature_research/test_walkforward_runner.py -v`
- `pytest tests/feature_research/test_walkforward_outputs.py -v`
- `pytest tests/integration/feature_validator/test_walkforward_research_pipeline.py -v`

Expected: PASS (integration may skip if persisted data/cache missing).

**Step 3: Update API docs**

Document new shared walkforward research entrypoints/config in `docs/api/feature_selection.md`.

**Step 4: Update kanban Result block and move to complete**

Append:

```markdown
## Result
- Implemented in: <commit hash / PR link>
- Tests: `pytest ...` ✅
- Notes: <unexpected findings>
```

Then move task file to `docs/kanban/complete/feature_validator/`.

**Step 5: Commit**

```bash
git add tests/integration/feature_validator/test_walkforward_research_pipeline.py docs/api/feature_selection.md docs/kanban/in-progress/T022_shared_walkforward_research_pipeline.md docs/kanban/complete/feature_validator/
git commit -m "Add shared walkforward research integration coverage and docs"
```

---

### Task 9: Final Verification

**Files:**
- No additional file changes expected.

**Step 1: Run full feature_research test slice**

Run: `pytest tests/feature_research -v`

Expected: PASS

**Step 2: Run integration slice for feature validator research pipelines**

Run: `pytest tests/integration/feature_validator -v`

Expected: PASS or explicit skip reasons for missing persisted data/cache.

**Step 3: Commit verification notes (if any doc/test metadata changed)**

```bash
git add -A
git commit -m "Verify shared walkforward research pipeline"
```
