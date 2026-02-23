# Utils Reorg Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize `utils/` into clear subpackages, remove tracked Cython build artifacts, and update all imports across the repo to the new module paths (no shims).

**Architecture:** Keep `utils/` as a Python package but convert it from a flat module dump into purpose-named subpackages (`core`, `compute`, `cache`, `memory`, `data`, `evaluation`, `simulation`, `dev`). Update code and tests to import from the new explicit module paths. Treat generated Cython outputs (`.so/.c/.html`) as build artifacts.

**Tech Stack:** Python 3.12, pytest, Cython extensions, git worktrees.

---

### Task 0: Worktree + Baseline Snapshot

**Files:**
- None

**Step 1: Confirm you are in the worktree**

Run: `git worktree list`
Expected: an entry like `.worktrees/utils-reorg`.

**Step 2: Capture baseline failures (known unrelated issues)**

Run: `pytest -q`
Expected: collection errors due to missing optional deps (e.g. `MetaTrader5`, `statsmodels`) and a few missing fixtures under `tests/unit-tests/ensemble/`.

---

### Task 1: Add New Package Folders

**Files:**
- Create: `utils/core/__init__.py`
- Create: `utils/compute/__init__.py`
- Create: `utils/compute/cython/__init__.py`
- Create: `utils/cache/__init__.py`
- Create: `utils/memory/__init__.py`
- Create: `utils/data/__init__.py`
- Create: `utils/evaluation/__init__.py`
- Create: `utils/simulation/__init__.py`
- Create: `utils/dev/__init__.py`

**Step 1: Create empty `__init__.py` files**

Expected: each subfolder is a valid Python package.

**Step 2: Compile check**

Run: `python -m compileall -q .`
Expected: exit code 0.

---

### Task 2: Stop Tracking Cython Build Artifacts

**Files:**
- Modify: `.gitignore`
- Delete (tracked): `utils/cython_nodes.c`
- Delete (tracked): `utils/cython_nodes.html`
- Delete (tracked): `utils/cython_optimized.c`
- Delete (tracked): `utils/cython_optimized.html`

**Step 1: Ensure ignores exist**

Add/ensure patterns in `.gitignore`:

```gitignore
*.so
*.pyd
*.dll
*.dylib
*.c
*.html
```

If those patterns are too broad for this repo, scope them to `utils/compute/cython/`.

**Step 2: Remove tracked artifacts**

Run:

```bash
git rm utils/cython_nodes.c utils/cython_nodes.html utils/cython_optimized.c utils/cython_optimized.html
```

**Step 3: Sanity check**

Run: `git status --porcelain`
Expected: those files are removed; `.gitignore` changed.

---

### Task 3: Move Core Modules + Rewrite Imports

**Files:**
- Location: `utils/core/enums.py`
- Location: `utils/core/models.py`
- Location: `utils/core/helpers.py`
- Location: `utils/core/logger.py`
- Location: `utils/core/functime.py`
- Modify: `utils/__init__.py`
- Modify: repo-wide imports from `utils.core.enums`, `utils.core.models`, `utils.core.helpers`, `utils.core.logger`, `utils.core.functime`

**Step 1: Ensure files reside under `utils/core/`**

Use `git mv` if any core modules remain at the top level.

**Step 2: Rewrite imports repo-wide**

Rewrite rules:
- `utils.core.enums` -> `utils.core.enums`
- `utils.core.models` -> `utils.core.models`
- `utils.core.helpers` -> `utils.core.helpers`
- `utils.core.logger` -> `utils.core.logger`
- `utils.core.functime` -> `utils.core.functime`

Also eliminate `from utils.core import helpers` style imports (rewrite to `from utils.core import helpers`).

**Step 3: Fix dynamic string refs**

Update string literal `'utils.core.functime'` in `utils/core/helpers.py` to `'utils.core.functime'`.

**Step 4: Compile + import sanity**

Run:
- `python -m compileall -q .`
- `python -c "from utils.core.enums import TimeFrame"`
- `python -c "from utils.core.models import Candle"`

Expected: no ImportError.

**Step 5: Optional commit**

```bash
git add .
git commit -m "refactor(utils): move core modules into utils/core"
```

---

### Task 4: Move Compute Modules + Rewrite Imports

**Files:**
- Move: `utils/fast_candle.py` -> `utils/compute/fast_candle.py`
- Move: `utils/fast_nodes.py` -> `utils/compute/fast_nodes.py`
- Move: `utils/fast_stats.py` -> `utils/compute/fast_stats.py`
- Move: `utils/fast_volatility.py` -> `utils/compute/fast_volatility.py`
- Move: `utils/grid_smoothing.py` -> `utils/compute/grid_smoothing.py`
- Move: `utils/rsi_helpers.py` -> `utils/compute/rsi_helpers.py`
- Modify: imports inside moved modules (e.g., relative imports)
- Modify: repo-wide imports from `utils.fast_*`, `utils.compute.grid_smoothing`, `utils.compute.rsi_helpers`

**Step 1: Move files**

```bash
git mv utils/fast_candle.py utils/fast_nodes.py utils/fast_stats.py utils/fast_volatility.py utils/grid_smoothing.py utils/rsi_helpers.py utils/compute/
```

**Step 2: Rewrite imports repo-wide**

Rewrite rules:
- `utils.compute.fast_candle` -> `utils.compute.fast_candle`
- `utils.compute.fast_nodes` -> `utils.compute.fast_nodes`
- `utils.compute.fast_stats` -> `utils.compute.fast_stats`
- `utils.compute.fast_volatility` -> `utils.compute.fast_volatility`
- `utils.compute.grid_smoothing` -> `utils.compute.grid_smoothing`
- `utils.compute.rsi_helpers` -> `utils.compute.rsi_helpers`

**Step 3: Fix intra-utils imports**

Examples:
- `from .fast_nodes ...` in volatility becomes `from utils.compute.fast_nodes ...` (or relative from the new package).
- Any `from utils.compute.rsi_helpers ...` becomes `from utils.compute.rsi_helpers ...`.

**Step 4: Compile check**

Run: `python -m compileall -q .`
Expected: exit code 0.

**Step 5: Optional commit**

```bash
git add .
git commit -m "refactor(utils): move fast math into utils/compute"
```

---

### Task 5: Move Cache + Memory + Data Modules + Rewrite Imports

**Files:**
- Move: `utils/bias_node_cache.py` -> `utils/cache/bias_node_cache.py`
- Move: `utils/cache_manager.py` -> `utils/cache/cache_manager.py`
- Move: `utils/cognitive_memory.py` -> `utils/memory/cognitive_memory.py`
- Move: `utils/memory_service.py` -> `utils/memory/memory_service.py`
- Move: `utils/memory_commands.py` -> `utils/memory/memory_commands.py`
- Move: `utils/candle_fetcher.py` -> `utils/data/candle_fetcher.py`
- Modify: tests that patch these modules via string paths

**Step 1: Move files**

```bash
git mv utils/bias_node_cache.py utils/cache_manager.py utils/cache/
git mv utils/cognitive_memory.py utils/memory_service.py utils/memory_commands.py utils/memory/
git mv utils/candle_fetcher.py utils/data/
```

**Step 2: Rewrite imports repo-wide**

Rewrite rules:
- `utils.cache.bias_node_cache` -> `utils.cache.bias_node_cache`
- `utils.cache.cache_manager` -> `utils.cache.cache_manager`
- `utils.memory.cognitive_memory` -> `utils.memory.cognitive_memory`
- `utils.memory.memory_service` -> `utils.memory.memory_service`
- `utils.memory.memory_commands` -> `utils.memory.memory_commands`
- `utils.data.candle_fetcher` -> `utils.data.candle_fetcher`

**Step 3: Rewrite `@patch("utils.*")` strings in tests**

Examples to update:
- `@patch('utils.memory.cognitive_memory.CognitiveMemory')` -> `@patch('utils.memory.cognitive_memory.CognitiveMemory')`
- `@patch('utils.memory.memory_service.MemoryService')` -> `@patch('utils.memory.memory_service.MemoryService')`
- `@patch('utils.memory.memory_commands.asyncio.run')` -> `@patch('utils.memory.memory_commands.asyncio.run')`

**Step 4: Compile + import sanity**

Run:
- `python -m compileall -q .`
- `python -c "from utils.cache.cache_manager import CacheManager"`

Expected: no ImportError.

**Step 5: Optional commit**

```bash
git add .
git commit -m "refactor(utils): move cache/memory/data modules into subpackages"
```

---

### Task 6: Move Evaluation + Simulation Subpackages

**Files:**
- Location: `utils/evaluation/walkforward.py`
- Location: `utils/evaluation/objective_metric/` (placeholder if/when added)
- Location: `utils/evaluation/permutation_test/`
- Location: `utils/evaluation/robustness_test/`
- Location: `utils/simulation/prop_firm_simulator/`
- Modify: repo-wide imports referencing those subpackages

**Step 1: Move**

Relocate any remaining evaluation/simulation modules into the target folders above using `git mv`.

**Step 2: Rewrite imports repo-wide**

Rewrite rules:
- `utils.evaluation.walkforward` -> `utils.evaluation.walkforward`
- `utils.evaluation.objective_metric` -> `utils.evaluation.objective_metric`
- `utils.evaluation.permutation_test` -> `utils.evaluation.permutation_test`
- `utils.evaluation.robustness_test` -> `utils.evaluation.robustness_test`
- `utils.simulation.prop_firm_simulator` -> `utils.simulation.prop_firm_simulator`

**Step 3: Compile check**

Run: `python -m compileall -q .`
Expected: exit code 0.

**Step 4: Optional commit**

```bash
git add .
git commit -m "refactor(utils): move evaluation and simulation subpackages"
```

---

### Task 7: Move Dev Modules

**Files:**
- Move: `utils/debug_helpers.py` -> `utils/dev/debug_helpers.py`
- Move: `utils/index_repo.py` -> `utils/dev/index_repo.py`
- Modify: any imports referencing them

**Step 1: Move**

```bash
git mv utils/debug_helpers.py utils/index_repo.py utils/dev/
```

**Step 2: Rewrite imports**

- `utils.dev.debug_helpers` -> `utils.dev.debug_helpers`
- `utils.dev.index_repo` -> `utils.dev.index_repo`

**Step 3: Compile check**

Run: `python -m compileall -q .`
Expected: exit code 0.

---

### Task 8: Move Cython Sources + Update Extension Names + Imports

**Files:**
- Move: `utils/cython_nodes.pyx` -> `utils/compute/cython/cython_nodes.pyx`
- Move: `utils/cython_optimized.pyx` -> `utils/compute/cython/cython_optimized.pyx`
- Move: `utils/setup_cython.py` -> `utils/compute/cython/setup_cython.py`
- Modify: `utils/compute/fast_nodes.py`
- Modify: `utils/compute/fast_stats.py`
- Modify: any nodes/scripts importing the Cython modules under `utils.compute.cython`

**Step 1: Move sources**

```bash
git mv utils/cython_nodes.pyx utils/cython_optimized.pyx utils/setup_cython.py utils/compute/cython/
```

**Step 2: Update extension module names**

In `utils/compute/cython/setup_cython.py`, update Extension names:
- Point imports to `utils.compute.cython.cython_nodes`
- Point imports to `utils.compute.cython.cython_optimized`

**Step 3: Update imports**

Rewrite rules:
- Update imports to `utils.compute.cython.cython_nodes`
- Update imports to `utils.compute.cython.cython_optimized`

**Step 4: Rebuild extensions in place**

Run:

```bash
python utils/compute/cython/setup_cython.py build_ext --inplace
```

Expected: compiled extensions appear under `utils/compute/cython/`.

**Step 5: Import sanity**

Run:
- `python -c "from utils.compute.cython.cython_nodes import compute_rsi"` (or another exported symbol)
- `python -c "from utils.compute.cython.cython_optimized import fast_mean"` (or another exported symbol)

Expected: no ImportError.

---

### Task 9: Clean Remaining `utils/` Root + Update `utils/__init__.py`

**Files:**
- Modify: `utils/__init__.py`

**Step 1: Ensure `utils/__init__.py` does not re-export moved modules**

Keep it minimal (package marker only) so callers must use explicit imports.

**Step 2: Verify `utils/` root is only subpackages**

Run: `ls utils`
Expected: `__init__.py` and the new subfolders; no leftover `.py` modules.

---

### Task 10: Update Docs That Reference Old Paths

**Files:**
- Modify: any `docs/**/*.md` that mentions `utils.*` old paths

**Step 1: Search**

Run: `rg "\butils\.(enums|models|helpers|logger|fast_|cache_manager|bias_node_cache|cognitive_memory|memory_service|memory_commands|walkforward|permutation_test|robustness_test|prop_firm_simulator|cython_nodes|cython_optimized)\b" docs`

**Step 2: Rewrite to new module paths**

Expected: docs reflect the new imports.

---

### Task 11: Verification

**Files:**
- None

**Step 1: Compileall**

Run: `python -m compileall -q .`
Expected: exit code 0.

**Step 2: Focused pytest run (exclude known-missing optional deps)**

Run a narrow selection that doesn’t require `MetaTrader5` or `statsmodels`. Example:

```bash
pytest -q tests/unit-tests/nodes
```

If that suite fails due to unrelated issues, downgrade to import-only checks + compileall and record the reasons.

**Step 3: Final diff review**

Run:
- `git status --porcelain`
- `git diff`
Expected: only refactor/move/import changes.

---

## Execution Handoff

Plan saved to `docs/plans/2026-02-22-utils-reorg-implementation-plan.md`.

Two execution options:
1) Subagent-Driven (this session) - dispatch a fresh subagent per task and review between tasks
2) Parallel Session (separate) - run with superpowers:executing-plans in a new session
