# Tests Organization Design

## Context

- Current unit tests under `tests/` mix module-specific suites, validators, and standalone files in a flat namespace while the production code lives in clearly scoped modules (`nodes`, `feature_selection`, `ensemble`, etc.).
- Integration tests already sit under `tests/integration/` and must remain untouched.
- Pytest discovery implicitly walks `tests/`, which makes the structure hard to navigate for both contributors and automated tooling.

## Scope

- Rehome every unit test currently outside `tests/integration/` into a new module-aligned tree rooted at `tests/unit-tests/`.
- Keep validator suites under `tests/unit-tests/validators/` and ensure any helpers remain grouped there.
- Do not touch `tests/integration/` or artifacts outside `tests/` unless necessary for configuration adjustments (for example, `pytest.ini`).

## Module-aligned layout

- Create `tests/unit-tests/<module>/` folders that mirror production modules:
  - `nodes` (bias node signal generation)
  - `feature_extraction`
  - `feature_selection` and `base_models` variants if they belong to selection/ensemble layers
  - `ensemble`
  - `execution`
  - `validators` (existing validator bundles such as `binning`, `eda`, `permutation`, and generic validator tests)
  - `deployment` / `vault` / `utils` / `metrics` as needed for the remaining standalone tests
- For base-model focused tests, place them under the module that coordinates base modeling (likely `feature_selection` or `ensemble`) so their location maps to the production entry point.
- Standalone files like `test_forecast_server.py`, `test_portfolio_manager.py`, and `test_position_sizer.py` move under the module that owns the functionality (e.g., `ensemble` or `execution`).

## Migration steps

1. Create `tests/unit-tests/` and the necessary subdirectories, replicating the module structure described above.
2. Move each non-integration test file from the current `tests/` root or subdirectories (`base_models/`, `ensemble/`, `validators/`, etc.) into its new module folder, preserving filenames to limit test discovery changes.
3. Update any pytest-related configs (`pytest.ini` / `pyproject.toml`) so they discover tests under `tests/unit-tests/` in addition to `tests/integration/` if not already generic.
4. Verify the module relocation doesn’t break imports (rare since tests usually reference modules via absolute imports); adjust `sys.path` hacks only if necessary.
5. Run targeted pytest invocations (see Verification) to ensure each module’s suite still passes after the move.

## Verification (Acceptance tests)

- Run `pytest tests/unit-tests/nodes/` to confirm node-level suites still behave.
- Run `pytest tests/unit-tests/validators/` to cover validator helpers that were reorganized.
- Run `pytest tests/unit-tests/ensemble/ tests/unit-tests/execution/` to cover orchestration and sizing units.

## Definition of done

- Every non-integration test file resides under `tests/unit-tests/<module>`.
- `tests/integration/` remains untouched.
- Pytest discovery still finds all tests via `pytest tests/unit-tests/` (plus existing integration commands).
- `docs/plans/2026-02-17-tests-organization-design.md` recorded the plan.
- Verifying pytest commands pass and the restructuring is documented for future contributors.
