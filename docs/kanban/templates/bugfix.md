# TXXX — <Short Bugfix Title>

## Goal
Describe the concrete bug being fixed and its impact.

## Context / References
- Link to docs/api or bug reports
- Related code files or tests

## Observed behavior
- Step-by-step description of what currently fails (include outputs, exceptions, timestamps).

## Expected behavior
- The target state (what should happen instead).

## Reproduction
1. `python scripts/run_example.py --mode <X>`
2. `pytest tests/...::test_failing_case -q`

## Regression window
- Last known good version/commit (if known) or “unknown (needs investigation).”

## Scope
In scope:
- <Responsibility A>

Out of scope:
- <Related but separate fixes>

## Interfaces (must match)
- Modify: `<path.py>` — include signature updates
- Add: `<config/class>` where the fix surfaces

## Constraints / Risk
- Determinism / no lookahead requirements
- Risk: what may break if this fix is incorrect (hints for regression tests)

## Acceptance tests
1. `pytest tests/...::test_repro_failure -q` — currently failing until fix applied.
2. `pytest tests/...::test_regression_guard -q` — ensures the old behavior stays intact.
3. `pytest tests/...::test_alignment -q` — deterministic alignment/no-lookahead pin.

## Definition of done
- [ ] Tests updated/added under `tests/<path>`
- [ ] Docs (if interfaces changed) under `docs/api/<module>.md`
- [ ] `pytest tests/... -q` passes

## Notes
- Any follow-up verification or cautionary reminders.
