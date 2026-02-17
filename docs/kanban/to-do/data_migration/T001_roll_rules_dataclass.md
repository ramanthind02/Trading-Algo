# T001 — Roll Rules Dataclass and Lookup

## Goal
Define fixed-date roll rules for all futures tickers using type-safe dataclasses and provide lookup functions.

## Context / References
- Roll rules from old data provider (fixed-date schedule: ES 5d before expiration, CL 3d before, GC 2d from month end, etc.)
- `utils/enums.py` - Ticker enum
- `docs/plans/2026-02-15-data-migration-design.md` - Architecture overview
- `docs/library/Data/Norgate.md` - Norgate data structure and migration context

## Data Source
Norgate Data is the migration target and comparison baseline. See `docs/library/Data/Norgate.md` for schema, formats, and constraints. These roll rules represent the legacy fixed-date schedule; Norgate continuous futures roll is volume-based, so roll dates are expected to diverge in validation.

## Scope
- In scope:
  - Create `RollRule` frozen dataclass with ticker, rollover_offset, reference_point, description
  - Define `ROLL_RULES` dict mapping all ~25 tickers to their rules
  - Implement `get_roll_rule(ticker)` and `get_all_roll_rules()` functions
- Out of scope:
  - Roll date detection (Task 2)
  - Back-adjustment logic (Tasks 3-4)

## Interfaces (must match)
- Add: `data_cleaning/back_adjustment/roll_rules.py`
  ```python
  @dataclass(frozen=True)
  class RollRule:
      ticker: Ticker
      rollover_offset: int
      reference_point: Literal["expiration", "month_end"]
      description: str

  def get_roll_rule(ticker: Ticker) -> RollRule
  def get_all_roll_rules() -> Dict[Ticker, RollRule]
  ```

## Data Contracts
- `RollRule`: Immutable dataclass representing fixed-date roll schedule for one ticker
- `ROLL_RULES`: Dict[Ticker, RollRule] - complete mapping for all instruments
- `rollover_offset`: Negative for "days before", positive for "days from month end"
- `reference_point`: Either "expiration" or "month_end"

## Dependencies
- `utils/enums.py` (Ticker enum)
- `dataclasses`, `typing`

## Invariants / Constraints
- All dataclasses must be frozen (immutable)
- 100% type hints (no `Any`)
- All tickers in Ticker enum must have a roll rule defined
- Rule lookups must be O(1) (dictionary-based)

## Acceptance tests
1. `pytest tests/back_adjustment/test_roll_rules.py::test_all_tickers_have_rules -v` – verifies every Ticker has a RollRule
2. `pytest tests/back_adjustment/test_roll_rules.py::test_get_roll_rule_lookup -v` – validates dictionary lookup works
3. `pytest tests/back_adjustment/test_roll_rules.py::test_roll_rule_immutability -v` – confirms frozen dataclass

## Definition of done
- [ ] Tests added under `tests/back_adjustment/test_roll_rules.py`
- [ ] Implementation in `data_cleaning/back_adjustment/roll_rules.py`
- [ ] All ~25 tickers have rules defined in ROLL_RULES dict
- [ ] `pytest tests/back_adjustment/test_roll_rules.py -v` passes

## Notes
- Reference roll rules table from old data provider specification
- ES: 5 days before expiration, CL: 3 days before expiration, GC: 2 days from month end (example rules)
