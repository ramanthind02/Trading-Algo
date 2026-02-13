# TXXX — <Short Feature Title>

## Goal
Describe in one sentence what this feature builds, why it matters, and how it connects to the trading workflow.

## Context / References
- `docs/api/<module>.md` or other docs
- Relevant code files or existing designs

## Scope
In scope:
- <Responsibility A>
- <Responsibility B>

Out of scope:
- <Adjacent work to keep separate>

## Interfaces (must match)
- Modify: `<path/to/file.py>` — describe the signature and behavior expected
- Add: `<config/class/enum>` — detail inputs, outputs, and where it plugs into the pipeline

## Data Contracts
- Schema (DataFrame columns, dataclass fields, API payload shape)
- Alignment expectations (timestamps, time zones, OOS windows)

## Dependencies
- Modules or packages this task touches (keep to 1–2 to preserve small scope)

## Invariants / Constraints
- Deterministic: same inputs ⇒ same outputs
- No lookahead: any signal/trade timestamp ≤ bar timestamp
- Units: specify bps vs decimals, risk units, etc.

## Acceptance tests
1. `<pytest path::test_name>` — deterministic check (e.g., fixed seed yields identical forecast sequence).
2. `<pytest path::alignment_test>` — ensures features/trades align (no lookahead).
3. `<pytest path::smoke_test>` — end-to-end smoke run that validates new behavior (e.g., `pytest tests/...::test_run -q`).

## Definition of done
- [ ] Tests added under `tests/<path>`
- [ ] Docs updated under `docs/api/<module>.md`
- [ ] `pytest <path>::<test> -q` passes

## Notes
- Edge cases, optional follow-ups, or hints for future work.
