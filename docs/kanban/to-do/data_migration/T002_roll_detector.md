# T002 — Roll Event Detection

## Goal
Detect actual roll dates in historical 1-minute data by identifying large price gaps that align with expected fixed-date roll windows.

## Context / References
- `data_cleaning/back_adjustment/roll_rules.py` - RollRule lookup (Task 1)
- `data/intraday_1min_original/` - Unadjusted source data
- `docs/plans/2026-02-15-data-migration-design.md` - Gap detection algorithm

## Scope
- In scope:
  - Create `RollEvent` frozen dataclass (roll_date, old_contract_close, new_contract_close, gap_points)
  - Implement `detect_roll_dates(df, rule)` to find price gaps exceeding threshold
  - Verify detected gaps occur within expected roll windows based on RollRule
- Out of scope:
  - Cumulative adjustment calculation (Task 3)
  - Applying adjustments to data (Task 4)

## Interfaces (must match)
- Add: `data_cleaning/back_adjustment/roll_detector.py`
  ```python
  @dataclass(frozen=True)
  class RollEvent:
      roll_date: datetime
      old_contract_close: float
      new_contract_close: float
      gap_points: float

  def detect_roll_dates(df: pd.DataFrame, rule: RollRule) -> List[RollEvent]
  ```

## Data Contracts
- Input DataFrame: columns `['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume']`
- `RollEvent.gap_points` = new_contract_close - old_contract_close
- Returns list sorted chronologically (oldest to newest)

## Dependencies
- `data_cleaning/back_adjustment/roll_rules.py` (RollRule)
- `pandas`, `datetime`, `typing`

## Invariants / Constraints
- Gap detection threshold: 1% of price OR 3 standard deviations of daily changes
- Detected gaps must align with expected roll window (±3 days tolerance)
- Returns empty list if no rolls detected (valid for recent contracts)
- Deterministic: same input data + rule → same roll events

## Acceptance tests
1. `pytest tests/back_adjustment/test_roll_detector.py::test_detect_known_roll -v` – synthetic data with known gap
2. `pytest tests/back_adjustment/test_roll_detector.py::test_no_rolls_found -v` – data with no significant gaps
3. `pytest tests/back_adjustment/test_roll_detector.py::test_roll_event_ordering -v` – chronological sort verification
4. `pytest tests/back_adjustment/test_roll_detector.py::test_determinism -v` – same input → same output

## Definition of done
- [ ] Tests added under `tests/back_adjustment/test_roll_detector.py`
- [ ] Implementation in `data_cleaning/back_adjustment/roll_detector.py`
- [ ] Gap detection logic handles edge cases (small datasets, no gaps)
- [ ] `pytest tests/back_adjustment/test_roll_detector.py -v` passes

## Notes
- May need to tune threshold parameters (1% vs 3σ) based on real data
- Consider logging detected gaps for manual review
- Roll window tolerance (±3 days) accounts for holidays/early rolls
