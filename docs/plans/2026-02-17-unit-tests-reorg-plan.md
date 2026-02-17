# Unit Tests Module Reorg Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize every non-integration test under `tests/` into a module-aligned tree rooted at `tests/unit-tests/` so newcomers can find the right suite without digging through a flat namespace.

**Architecture:** Mirror the production module boundaries (nodes, feature extraction/selection, ensemble, execution, validators, deployment, etc.) inside `tests/unit-tests/` and relocate each test file into the directory that owns the functionality it verifies. Keep `tests/integration/` untouched and let pytest keep discovering both trees.

**Tech Stack:** Python 3.12, pytest, standard filesystem operations (`mkdir`, `mv`, `find`).

---

### Task 1: Create the module-aligned hierarchy

**Files:**
- Create: `tests/unit-tests/nodes/`
- Create: `tests/unit-tests/feature_extraction/`
- Create: `tests/unit-tests/feature_selection/`
- Create: `tests/unit-tests/ensemble/`
- Create: `tests/unit-tests/execution/`
- Create: `tests/unit-tests/validators/` and its subfolders (binning, eda, permutation)
- Create: `tests/unit-tests/deployment/`, `tests/unit-tests/vault/`, `tests/unit-tests/utils/`, `tests/unit-tests/metrics/` as needed for the leftover suites

**Step 1:** Run `mkdir -p tests/unit-tests/{nodes,feature_extraction,feature_selection,ensemble,execution,validators,deployment,vault,utils,metrics}` to stage the base directories.

**Step 2:** Add pytest package markers if necessary (empty `__init__.py` files) to ensure imports work when tests live in packages.

**Step 3:** Verify the directories exist using `ls tests/unit-tests/` before moving files.

### Task 2: Relocate module-specific directories

**Files:**
- Move: `tests/base_models/` → `tests/unit-tests/feature_selection/base_models/` (or `ensemble` if the tests target the ensemble layer).
- Move: `tests/ensemble/` → `tests/unit-tests/ensemble/`
- Move existing validator subfolders from `tests/validators/` into `tests/unit-tests/validators/`

**Step 1:** For each folder above, run `mv tests/<folder> tests/unit-tests/<target>/<optional-subdir>/`.

**Step 2:** If the target path needs a new subfolder (e.g., `feature_selection/base_models/`), create it with `mkdir -p` before moving.

**Step 3:** List moved files (e.g., `ls tests/unit-tests/ensemble/`) to confirm everything relocated.

### Task 3: Relocate remaining root-level tests into modules

**Files:**
- Move node-related tests (`test_bias_node_cache.py`, `test_new_bias_nodes.py`, etc.) → `tests/unit-tests/nodes/`
- Move execution/portfolio-related tests (`test_portfolio_manager.py`, `test_execution.py?` none, but `test_position_sizer.py`) → `tests/unit-tests/execution/`
- Place validator-like suites (`test_feature_validator.py`, `test_permutation_candle_shuffle.py`, etc.) under `tests/unit-tests/validators/`
- Map service/ops tests (`test_forecast_server.py`, `test_production_training_pipeline.py`) to `tests/unit-tests/deployment/` or `tests/unit-tests/utils/` as appropriate

**Step 1:** For each file, run `mv tests/<file> tests/unit-tests/<module>/` and resolve collisions by keeping filenames identical.

**Step 2:** If a file tests cross-module behavior (e.g., `test_diversified_ensemble.py`), choose the module that orchestrates that behavior (`ensemble`) and document the mapping in a TODO comment inside `docs/plans/2026-02-17-tests-organization-design.md` if unsure.

**Step 3:** After moving, update references in the README or other docs that link to the old path if there are any.

### Task 4: Update pytest discovery and packaging

**Files:**
- Modify `pytest.ini` or `pyproject.toml` if `tests/` was explicitly listed so that `tests/unit-tests/` is included.

**Step 1:** Open the pytest config (`pyproject.toml` or `pytest.ini`, whichever defines `testpaths`).

**Step 2:** Add `tests/unit-tests` to `testpaths` or ensure `python_files = test_*.py` is still valid.

**Step 3:** Run `pytest tests/unit-tests/ --maxfail=1 -q` to ensure pytest picks up the new tree.

### Task 5: Verify and document success

**Files:**
- Document the move in `docs/plans/2026-02-17-tests-organization-design.md` (already done) and optionally note the new layout in `README.md` if it references the old paths.

**Step 1:** Run module slices (`pytest tests/unit-tests/nodes/`, `pytest tests/unit-tests/validators/`, `pytest tests/unit-tests/ensemble/ tests/unit-tests/execution/`).

**Step 2:** Confirm no imports broke by running `pytest tests/unit-tests/...` for each module before concluding.

**Step 3:** Stage the reorganized files, `docs/plans/2026-02-17-tests-organization-design.md`, and any config changes, then commit with a descriptive message (e.g., `git commit -am "refactor: module-aligned unit tests"`).

---

Plan complete and saved to `docs/plans/2026-02-17-unit-tests-reorg-plan.md`. Two execution options:

1. **Subagent-Driven (this session)** - dispatch fresh subagents per task, review between tasks, fast iteration (@superpowers:subagent-driven-development is required).
2. **Parallel Session (separate worktree)** - open a new session using `superpowers:executing-plans`, batch execute with checkpoints.

Which approach would you like?@Module
