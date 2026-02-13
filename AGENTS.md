# AGENTS.md

## Repository Context

- This repository is a systematic trading framework inspired by Carver-style forecasting, diversification, and risk scaling.
- Core flow is: candles -> bias nodes -> base models -> diversified ensemble -> weight layer -> portfolio -> position sizing.
- Main domains are feature extraction/selection, ensemble construction, execution sizing, and production deployment helpers.

## Environment

- Use the project virtual environment for all Python work: `source venv/bin/activate`.
- Keep dependencies scoped to `venv`; do not rely on system Python packages.
- Primary test runner is `pytest`.

## Architecture Map

- `nodes/`: bias node feature generation from OHLCV data.
- `feature_selection/`: base model and feature validation logic.
- `ensemble/`: forecast diversification, weighting, and portfolio combination layers.
- `execution/`: conversion from forecast fractions to tradeable contract quantities.
- `utils/`: shared data models, enums, cache manager, and common utilities.
- `vault/`: persisted validated features and control artifacts.
- `deployment/`: production-facing forecast and training pipeline components.

## Base Model And Validation Workflow

- Treat a base model as an ensemble of binning models over parameter variants, not a single model.
- Fit each member independently on its own feature series and shared target; aggregate predictions by simple mean.
- Keep permutation testing stages explicit: vector shuffle screen, pipeline permutation test, walkforward stability analysis.
- Keep manual researcher ensemble selection as the final step using both statistical validation and temporal stability.

## Coding Rules

- Prefer functional core, imperative shell: pure logic in pure functions, I/O at boundaries.
- Prefer immutable data structures (`@dataclass(frozen=True)`) unless mutation is required and justified.
- Use strong domain typing (`NewType`, `Enum`, Pydantic) instead of raw primitives in domain interfaces.
- Avoid boolean flags in public/domain APIs; prefer explicit enums/modes.
- Prefer composition and `Protocol` interfaces over inheritance-heavy designs.
- Keep type hints complete and strict; avoid `Any` unless unavoidable.
- Use ADT-style unions and `match/case` where it improves correctness of state handling.

## Functional Control Flow

- Avoid raw `for`/`while` loops when a comprehension, generator, builtin reduction, or `itertools` expression is clearer.
- Model data processing as filter -> map -> reduce pipelines with small, explicit transformation steps.
- Use higher-order functions for configurable behavior instead of duplicating loop/control-flow structure.
- Use decorators for cross-cutting concerns (logging, timing, caching) instead of inlining repetitive wrapper logic.

## Type And API Design

- Maintain 100% argument/return type hints in production code; keep interfaces checker-friendly.
- Prefer `TypeVar`/bounded generics over duplicated typed implementations.
- Use `Optional[T]` or `Result`-style return models for expected failures; avoid exceptions for normal control flow.
- For builder/config APIs, prefer fluent chaining and return `Self`.

## Data And Signal Conventions

- Preserve feature naming consistency for generated bias-node columns.
- Keep target alignment and scaling assumptions explicit when modifying feature selection logic.
- Do not silently change control-file schema semantics used by ensembles, vault artifacts, or deployment readers.

## Validation Expectations

- Run targeted tests for changed modules first (for example `pytest tests/test_ensemble_base_models.py -v`).
- Run relevant integration tests when cross-layer behavior changes.
- Run full suite (`pytest tests/`) before finalizing substantial architecture or pipeline changes.
- If numerical formulas or scaling behavior changes, update/add tests that pin expected outputs.

## DRY And SRP

- Keep each function/class focused on one concern and one reason to change.
- Split mixed-responsibility functions (especially names containing `and`) into composable units.
- Distinguish logic duplication from superficial similarity: deduplicate shared business rules, not unrelated code.
- Preserve decorator safety: use `functools.wraps` and typed signatures (`ParamSpec`/`TypeVar`) for wrappers.

## Codex Skills

- Codex now discovers the `superpowers` skill catalog via `~/.codex/superpowers/skills` and the symlink `~/.agents/skills/superpowers`.
- Before starting creative work, run the `using-superpowers` skill flow to confirm which skills apply; refer to the relevant `SKILL.md` under the symlink.
- The project-level instructions above (especially around planning, testing, and architecture) assume those skills are available; mention specific skill requirements when you open a `SKILL.md`.

## Docs Landscape

- `docs/api/` - auto-generated API docs guided by `docs/api/_template.md` and `_scope.md`; update relevant module pages whenever you touch public interfaces.
- `docs/kanban/` - the new kanban workflow, with `README.md` enforcing scope/interfaces/tests, plus templates under `docs/kanban/templates/` for feature, bugfix, and docs tasks; put every coding intent here before modifying code.
- `docs/library/`, `docs/methodology/`, `docs/plans/`, `docs/complete/`, and related subfolders hold domain research, validation philosophy, operational playbooks, and project plans—cite them when describing designs or documenting decisions.
- Keep `docs/to-do/` (existing specs) and `docs/methodology/` in sync with new kanban tasks so implementation artifacts remain traceable.
