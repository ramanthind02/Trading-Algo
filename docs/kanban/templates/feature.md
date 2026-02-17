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
1. `<pytest path::unit_test_name>` — unit check with isolated/synthetic fixtures (must NOT live under `tests/integration/`).
2. `<pytest path::alignment_test>` — deterministic no-lookahead/alignment check.
3. `<pytest tests/integration/...::integration_test_name>` — pipeline-backed integration test using persisted repo data/cache (no synthetic-only fixtures).

### Integration Test Data Contract (required when integration tests are in scope)
- Data source path: `<e.g., data/ohlc_data>`
- Tickers: `<e.g., [Ticker.ES, Ticker.NQ]>`
- Timeframe: `<e.g., TimeFrame.D>`
- Date range: `<start/end datetimes>`
- Bias node spec / model config: `<exact dict/config>`
- Cache mode: `USE_CACHE=<bool>`, `POPULATE_CACHE=<bool>` with expected behavior if cache missing

## Definition of done
- [ ] Tests added under `tests/<path>`
- [ ] Docs updated under `docs/api/<module>.md`
- [ ] `pytest <path>::<test> -q` passes

## Notes
- Edge cases, optional follow-ups, or hints for future work.
