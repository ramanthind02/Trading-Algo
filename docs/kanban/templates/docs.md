# TXXX — <Short Docs Title>

## Goal
Explain which section of the docs we are improving and why.

## Context / References
- Existing docs file(s) (links to `docs/api`, `docs/methodology`, etc.)
- Source code or behaviors that justify the change

## Target docs
- `docs/api/<module>.md`
- `docs/methodology/<topic>.md`

## Source of truth
- Code paths that should be referenced to keep accuracy (e.g., `nodes/data_loader.py`, `platform/runner.py`)

## Examples to add/update
- Snippets, expected CLI output, config examples that must change

## Scope
In scope:
- Specific sections, tables, or diagrams being rewritten

Out of scope:
- Anything that requires code changes or new tools

## Interfaces (must match)
- Documented API surfaces tied to this change (e.g., CLI options, config fields)
- Mention any implicit contracts (e.g., “This config value maps 1:1 to `ExecutionConfig.max_leverage`”)

## Acceptance tests
1. `python -m mkdocs build` (or repo’s docs build command) completes without warnings for the changed pages.
2. `pytest tests/docs/test_examples.py::test_example_snippets -q` (if available) succeeds.
3. Manual review: changed files render correctly in preview (describe how to run if not automated).

## Definition of done
- [ ] Updated docs checked into `docs/api/...`
- [ ] Related code path referenced/linked where appropriate
- [ ] `mkdocs build` (or equivalent) passes

## Notes
- Mention follow-up doc cleanups or verification steps.
