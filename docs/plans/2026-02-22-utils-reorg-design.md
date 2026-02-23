# Utils Refactor: Subpackages + Repo-Wide Import Updates

Date: 2026-02-22

## Goal

Make `utils/` navigable by grouping modules into purposeful subpackages, and update imports across the repository to the new locations. No compatibility shims remain at `utils/*.py`.

## Non-goals

- Do not change runtime behavior, algorithms, or public semantics of the utilities.
- Do not address unrelated failing tests or missing optional dependencies.

## Constraints

- Refactor occurs in an isolated git worktree.
- All call sites are updated to new import paths.
- Cython build outputs are treated as build artifacts (not tracked in git).

## Target Package Layout

After the move, `utils/` should be mostly folders (plus `utils/__init__.py`).

- `utils/core/`
  - `enums.py`
  - `models.py`
  - `helpers.py`
  - `logger.py`
  - `functime.py`

- `utils/compute/`
  - `fast_candle.py`
  - `fast_nodes.py`
  - `fast_stats.py`
  - `fast_volatility.py`
  - `grid_smoothing.py`
  - `rsi_helpers.py`
  - `cython/`
    - `cython_nodes.pyx`
    - `cython_optimized.pyx`
    - `setup_cython.py`

- `utils/cache/`
  - `bias_node_cache.py`
  - `cache_manager.py`

- `utils/memory/`
  - `cognitive_memory.py`
  - `memory_service.py`
  - `memory_commands.py`

- `utils/data/`
  - `candle_fetcher.py`

- `utils/evaluation/`
  - `walkforward.py`
  - `objective_metric/` (moved as-is)
  - `permutation_test/` (moved as-is)
  - `robustness_test/` (moved as-is)

- `utils/simulation/`
  - `prop_firm_simulator/` (moved as-is)

- `utils/dev/`
  - `debug_helpers.py`
  - `index_repo.py`

## Import Rewrite Rules

All imports are updated repo-wide.

- `utils.core.enums` -> `utils.core.enums`
- `utils.core.models` -> `utils.core.models`
- `utils.core.helpers` -> `utils.core.helpers`
- `utils.core.logger` -> `utils.core.logger`
- `utils.core.functime` -> `utils.core.functime`

- `utils.fast_*` -> `utils.compute.fast_*`
- `utils.compute.grid_smoothing` -> `utils.compute.grid_smoothing`
- `utils.compute.rsi_helpers` -> `utils.compute.rsi_helpers`

- `utils.cache.cache_manager` -> `utils.cache.cache_manager`
- `utils.cache.bias_node_cache` -> `utils.cache.bias_node_cache`

- `utils.memory.memory_service` -> `utils.memory.memory_service`
- `utils.memory.memory_commands` -> `utils.memory.memory_commands`
- `utils.memory.cognitive_memory` -> `utils.memory.cognitive_memory`

- `utils.data.candle_fetcher` -> `utils.data.candle_fetcher`

- `utils.evaluation.walkforward` -> `utils.evaluation.walkforward`
- `utils.evaluation.objective_metric.*` -> `utils.evaluation.objective_metric.*`
- `utils.evaluation.permutation_test.*` -> `utils.evaluation.permutation_test.*`
- `utils.evaluation.robustness_test.*` -> `utils.evaluation.robustness_test.*`
- `utils.simulation.prop_firm_simulator.*` -> `utils.simulation.prop_firm_simulator.*`

`from utils.core import helpers` and similar star exports are removed; callers must import the concrete module path.

## Cython Artifacts Policy

The following are treated as build artifacts and removed from git tracking (if currently tracked):

- `*.so`
- `*.c`
- `*.html`

Only `*.pyx` and the build helper (`setup_cython.py`) remain as source.

The Cython extension module names are updated to match the new package path under `utils.compute.cython.*`.

## Verification

- `python -m compileall .` passes.
- Targeted pytest runs that do not require optional external dependencies succeed (selection defined during implementation).
- Key import sanity checks pass:
  - `python -c "from utils.core.enums import TimeFrame"`
  - `python -c "from utils.core.models import Candle"`
  - `python -c "from utils.compute.fast_nodes import compute_rsi"` (or equivalent entrypoint used in repo)

## Notes

Baseline `pytest` currently fails in this repo due to missing optional dependencies (e.g. `MetaTrader5`, `statsmodels`) and some missing test fixtures. This refactor will not attempt to resolve those unrelated issues.
