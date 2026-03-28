# Ticket 06: Package Entry Points And Remove `sys.path` Bootstrapping

## Why

Deployment modules, scripts, and tests frequently mutate `sys.path` with directory-depth assumptions.
This is repeated glue code and a frequent source of fragile behavior.

## Goal

Run the project as an installable package (`pip install -e .`) and replace path hacks with package imports and CLI entrypoints.

## Scope

- Standardize project entrypoints for deployment and scripts.
- Remove ad-hoc `sys.path` edits from Python modules where possible.
- Introduce shared test import setup via root `conftest.py` or pytest config.

## Out Of Scope

- Full CLI redesign.
- Behavioral changes in strategy logic.

## Acceptance Criteria

- Core scripts and deployment modules execute without local `sys.path` mutations.
- Tests run with deterministic import resolution.
- Docs reference package-first invocation paths.

## Risks

- Some environments currently depend on direct script execution from arbitrary cwd.

## Dependencies

- None. High leverage and early value.
