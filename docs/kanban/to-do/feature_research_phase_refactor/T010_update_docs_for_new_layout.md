# T010 — Update Docs to Match New feature_research Layout

## Goal
Update library docs to describe the new 3-phase research folder organization and the new canonical script locations.

## Context / References
- `docs/library/Feature_selection/pipeline_overview.md`
- `docs/library/Feature_selection/feature_validator.md`

## Target Docs
- `docs/library/Feature_selection/pipeline_overview.md` (add a “Repo layout / where to run” section)
- `docs/library/Feature_selection/feature_validator.md` (update any paths to EDA / permutation entrypoints)

## Source of Truth
- Code locations under `feature_research/in_sample/`, `feature_research/walkforward/`, `feature_research/oos/` after migration.

## Scope
In scope:
- Replace references to legacy locations with new canonical paths.
- Add a short mapping of “Phase -> scripts/entrypoints” for both feature types.

Out of scope:
- Changing the theoretical/statistical spec content.

## Interfaces
- Documentation only (no runtime interfaces).

## Definition of Done
- Docs reference the new package layout.
- No doc references remain to removed legacy folders.

## Notes
- If the docs/kanban README references `docs/kanban/todo/` but the repo uses `docs/kanban/to-do/`, consider a separate docs task (optional) to align wording; do not mix into this task unless requested.
