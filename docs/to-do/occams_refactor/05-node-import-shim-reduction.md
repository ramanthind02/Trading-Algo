# Ticket 05: Reduce Node Import Shims And `sys.modules` Re-Exports

## Why

Many `nodes/*` modules act as compatibility shims that re-export canonical implementations (often with `sys.modules` manipulation).
This duplicates module identity and complicates debugging/import tracing.

## Goal

Move to one canonical import surface for bias nodes, driven by taxonomy mapping and explicit compatibility policy.

## Scope

- Inventory shim modules and classify: keep, deprecate, remove.
- Replace dynamic module replacement with static compatibility wrappers where still required.
- Provide migration map from legacy flat imports to canonical paths.

## Out Of Scope

- Rewriting node math internals.
- Immediate hard-break for all legacy imports.

## Acceptance Criteria

- Shim count is measurably reduced.
- Canonical import paths are documented as the default.
- Deprecation warnings (or docs notices) exist for retained legacy paths.

## Risks

- External notebooks/scripts may rely on old imports.

## Dependencies

- Prefer after packaging cleanup (Ticket 06) to simplify import behavior globally.
