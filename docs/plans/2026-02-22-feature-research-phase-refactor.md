# Feature Research Phase Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Hard-cutover refactor of `feature_research/` into phase-first packages (`in_sample/`, `walkforward/`, `oos/`), update all imports/tests/docs, add import smoke tests, and delete legacy `feature_research/rule_based` and `feature_research/continuous_binning`.

**Architecture:** Phase packages contain type-specific in-sample logic; walkforward keeps a shared engine plus type-specific entrypoints; OOS adds type-specific entrypoints. No backward compatibility shims.

**Tech Stack:** Python 3.12, pytest.

---

### Task 1: Baseline Snapshot and Safety Checks

**Files:**
- None

**Step 1: Confirm we are in the refactor worktree**

Run: `git branch --show-current`
Expected: prints `feature-research-phase-refactor`

**Step 2: Confirm `.worktrees/` is ignored**

Run: `git check-ignore -q .worktrees && echo OK`
Expected: `OK`

**Step 3: Record current failing tests for `tests/feature_research` (expected pre-work)**

Run: `pytest -q tests/feature_research`
Expected: May fail (config default expectations + walkforward portfolio evaluator validation); we will fix/align as part of later tasks.

---

### Task 2: Create Phase Skeleton Packages (T002)

**Files:**
- Create: `feature_research/in_sample/__init__.py`
- Create: `feature_research/in_sample/rule_based/__init__.py`
- Create: `feature_research/in_sample/continuous_binning/__init__.py`
- Create: `feature_research/walkforward/rule_based/__init__.py`
- Create: `feature_research/walkforward/continuous_binning/__init__.py`
- Create: `feature_research/oos/__init__.py`
- Create: `feature_research/oos/rule_based/__init__.py`
- Create: `feature_research/oos/continuous_binning/__init__.py`

**Step 1: Write minimal package markers**

Each `__init__.py` should be minimal (empty or `"""..."""`).

**Step 2: Run import checks**

Run:
`python -c "import feature_research.in_sample.rule_based, feature_research.in_sample.continuous_binning, feature_research.walkforward.rule_based, feature_research.oos.continuous_binning"`
Expected: no output, exit 0

---

### Task 3: Migrate In-Sample Rule-Based Modules (T003)

**Files:**
- Move: `feature_research/rule_based/config.py` -> `feature_research/in_sample/rule_based/config.py`
- Move: `feature_research/rule_based/data_loader.py` -> `feature_research/in_sample/rule_based/data_loader.py`
- Move: `feature_research/rule_based/pipeline.py` -> `feature_research/in_sample/rule_based/pipeline.py`
- Move: `feature_research/rule_based/run_eda.py` -> `feature_research/in_sample/rule_based/run_eda.py`
- Move: `feature_research/rule_based/param_sensitivity.ipynb` -> `feature_research/in_sample/rule_based/param_sensitivity.ipynb`

**Step 1: Move files without changing semantics**

Preserve public functions/classes and config behavior.

**Step 2: Update internal imports within these moved files**

Replace internal references from:
- `feature_research.rule_based.*` -> `feature_research.in_sample.rule_based.*`

Keep walkforward engine imports pointing to `feature_research.walkforward.*`.

**Step 3: Fix `run_eda.py` repo-root path shim (if present)**

Replace fixed-depth `parents[N]` usage with a robust upward search for repo root.

**Step 4: Verify compile/import**

Run: `python -m compileall feature_research/in_sample/rule_based`
Expected: exit 0

Run: `python -c "from feature_research.in_sample.rule_based.config import load_config; load_config()"`
Expected: no output, exit 0

---

### Task 4: Migrate In-Sample Continuous-Binning Modules (T004)

**Files:**
- Move: `feature_research/continuous_binning/config.py` -> `feature_research/in_sample/continuous_binning/config.py`
- Move: `feature_research/continuous_binning/data_loader.py` -> `feature_research/in_sample/continuous_binning/data_loader.py`
- Move: `feature_research/continuous_binning/pipeline.py` -> `feature_research/in_sample/continuous_binning/pipeline.py`
- Move: `feature_research/continuous_binning/run_eda.py` -> `feature_research/in_sample/continuous_binning/run_eda.py`
- Move: `feature_research/continuous_binning/binning_analysis.py` -> `feature_research/in_sample/continuous_binning/binning_analysis.py`
- Move: `feature_research/continuous_binning/run_binning_analysis.py` -> `feature_research/in_sample/continuous_binning/run_binning_analysis.py`
- Move: `feature_research/continuous_binning/param_sensitivity.ipynb` -> `feature_research/in_sample/continuous_binning/param_sensitivity.ipynb`

**Step 1: Move files without changing semantics**

**Step 2: Update internal imports**

Replace:
- `feature_research.continuous_binning.*` -> `feature_research.in_sample.continuous_binning.*`

Keep engine imports at `feature_research.walkforward.*`.

**Step 3: Fix path shims in `run_eda.py` and `run_binning_analysis.py`**

Same approach as Task 3.

**Step 4: Verify compile/import**

Run: `python -m compileall feature_research/in_sample/continuous_binning`
Expected: exit 0

Run: `python -c "from feature_research.in_sample.continuous_binning.config import load_config; load_config()"`
Expected: no output, exit 0

---

### Task 5: Walkforward Entrypoints for Rule-Based (T005)

**Files:**
- Move/Rewrite: `feature_research/rule_based/run_walkforward.py` -> `feature_research/walkforward/rule_based/run_walkforward.py`

**Step 1: Update imports**

Entrypoint should import:
- in-sample config/pipeline from `feature_research.in_sample.rule_based.*`
- engine runner from `feature_research.walkforward.runner`

**Step 2: Fix repo-root path shim**

As in Task 3.

**Step 3: Verify import**

Run: `python -c "import feature_research.walkforward.rule_based.run_walkforward"`
Expected: exit 0

---

### Task 6: Walkforward Entrypoints for Continuous-Binning (T006)

**Files:**
- Move/Rewrite: `feature_research/continuous_binning/run_walkforward.py` -> `feature_research/walkforward/continuous_binning/run_walkforward.py`

**Step 1: Update imports**

Entrypoint should import:
- in-sample config/pipeline from `feature_research.in_sample.continuous_binning.*`
- engine runner from `feature_research.walkforward.runner`

**Step 2: Fix repo-root path shim**

As in Task 4.

**Step 3: Verify import**

Run: `python -c "import feature_research.walkforward.continuous_binning.run_walkforward"`
Expected: exit 0

---

### Task 7: Add Strict OOS Phase Scaffolding (T007)

**Files:**
- Create: `feature_research/oos/rule_based/run_oos.py`
- Create: `feature_research/oos/continuous_binning/run_oos.py`

**Step 1: Add minimal entrypoint code**

Implement thin scripts that load the corresponding in-sample config and call a walkforward/oos runner.

Suggested minimal shape (adjust to actual engine API):

```python
from __future__ import annotations

def main() -> None:
    # Load in-sample config and run strict OOS evaluation
    raise NotImplementedError("Wire strict OOS runner")


if __name__ == "__main__":
    main()
```

**Step 2: Verify imports**

Run:
`python -c "import feature_research.oos.rule_based.run_oos; import feature_research.oos.continuous_binning.run_oos"`
Expected: exit 0

---

### Task 8: Update Imports Across Repo + Fix/Align Tests (T008)

**Files:**
- Modify: all python modules/tests importing legacy packages

**Step 1: Replace legacy imports**

Update every occurrence:
- `feature_research.rule_based.*` -> `feature_research.in_sample.rule_based.*` (or `feature_research.walkforward.rule_based.*` for entrypoints)
- `feature_research.continuous_binning.*` -> `feature_research.in_sample.continuous_binning.*` (or `feature_research.walkforward.continuous_binning.*`)

**Step 2: Update `tests/feature_research/*` import paths**

Ensure tests reference phase-first modules.

**Step 3: Fix the currently-failing `tests/feature_research` expectations**

Align tests with the actual config defaults (keep config semantics unchanged per kanban):
- Update the expected `start/end` values in `tests/feature_research/test_rule_based_config.py`.
- Fix `tests/feature_research/walkforward/test_portfolio_evaluator.py` to satisfy the multi-member control schema validation (ensure members include required `member_name` field or adjust fixture to match current validator requirements).

**Step 4: Verify**

Run: `python -m compileall feature_research`
Expected: exit 0

Run: `pytest -q tests/feature_research`
Expected: PASS

---

### Task 9: Add Import/Packaging Smoke Tests (T011)

**Files:**
- Create: `tests/feature_research/test_import_smoke.py`

**Step 1: Write import smoke test**

```python
def test_feature_research_phase_packages_import() -> None:
    import feature_research.in_sample.rule_based
    import feature_research.in_sample.continuous_binning
    import feature_research.walkforward.runner
    import feature_research.walkforward.rule_based.run_walkforward
    import feature_research.walkforward.continuous_binning.run_walkforward
    import feature_research.oos.rule_based
    import feature_research.oos.continuous_binning
```

**Step 2: Run smoke test**

Run: `pytest -q tests/feature_research -k import`
Expected: PASS

---

### Task 10: Delete Legacy Research Directories (T009)

**Files:**
- Delete: `feature_research/rule_based/`
- Delete: `feature_research/continuous_binning/`

**Step 1: Delete directories**

Remove the directories after all imports have been updated.

**Step 2: Verify no references remain**

Run: `git grep -n "feature_research\.rule_based" || true`
Expected: no matches

Run: `git grep -n "feature_research\.continuous_binning" || true`
Expected: no matches

**Step 3: Verify filesystem**

Run:
`python -c "import pathlib; assert not pathlib.Path('feature_research/rule_based').exists(); assert not pathlib.Path('feature_research/continuous_binning').exists()"`
Expected: exit 0

---

### Task 11: Update Docs to Match New Layout (T010 + follow-on doc fixes)

**Files:**
- Modify: `docs/library/Feature_selection/pipeline_overview.md` (only if it references legacy paths)
- Modify: `docs/library/Feature_selection/feature_validator.md` (only if it references legacy paths)
- Modify: `docs/api/data_pipeline.md`
- Modify: older plan docs referencing legacy imports under `docs/plans/` (as-needed)

**Step 1: Update any legacy path references**

Replace old references to:
- `feature_research/continuous_binning/run_binning_analysis.py` -> `feature_research/in_sample/continuous_binning/run_binning_analysis.py`
- `feature_research/rule_based/pipeline.py` -> `feature_research/in_sample/rule_based/pipeline.py`

Also update any `docs/plans/*.md` that reference `feature_research.continuous_binning.*` / `feature_research.rule_based.*` to their phase-first equivalents.

**Step 2: Verify no legacy strings in docs**

Run:
`git grep -n "feature_research\\.(rule_based|continuous_binning)" -- '*.py' || true`
Expected: no matches

---

### Task 12: Final Verification

**Files:**
- None

**Step 1: Compile**

Run: `python -m compileall feature_research`
Expected: exit 0

**Step 2: Tests**

Run: `pytest -q tests/feature_research`
Expected: PASS

**Step 3: Import sanity**

Run: `python -c "import feature_research; import feature_research.walkforward.runner"`
Expected: exit 0
