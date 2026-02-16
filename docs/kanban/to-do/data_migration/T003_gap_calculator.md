# T003 — Gap Calculator and Adjustment Factors

## Goal
Calculate cumulative adjustment factors for back-adjustment from detected roll events using arithmetic back-adjustment principle.

## Context / References
- `data_cleaning/back_adjustment/roll_detector.py` - RollEvent dataclass (Task 2)
- `docs/plans/2026-02-15-data-migration-design.md` - Cumulative adjustment algorithm

## Scope
- In scope:
  - Create `AdjustmentFactor` frozen dataclass (roll_date, gap_points, cumulative_adjustment)
  - Implement `calculate_adjustments(roll_events)` to compute cumulative adjustments
  - Ensure cumulative_adjustment increases as we go back in time (newer rolls don't affect older data)
- Out of scope:
  - Applying adjustments to DataFrame (Task 4)
  - Roll detection (Task 2)

## Interfaces (must match)
- Add: `data_cleaning/back_adjustment/gap_calculator.py`
  ```python
  @dataclass(frozen=True)
  class AdjustmentFactor:
      roll_date: datetime
      gap_points: float
      cumulative_adjustment: float

  def calculate_adjustments(roll_events: List[RollEvent]) -> List[AdjustmentFactor]
  ```

## Data Contracts
- Input: List[RollEvent] sorted chronologically (oldest to newest)
- Output: List[AdjustmentFactor] same length as input, sorted chronologically
- `cumulative_adjustment` = sum of all gaps from this roll forward in time
- For the most recent roll: cumulative_adjustment = 0 (no adjustment needed)
- For older rolls: cumulative_adjustment increases as we go back in time

## Dependencies
- `data_cleaning/back_adjustment/roll_detector.py` (RollEvent)
- `dataclasses`, `typing`, `datetime`

## Invariants / Constraints
- Cumulative adjustments must be monotonic (non-decreasing as we go back in time)
- Most recent roll has cumulative_adjustment = 0
- Input roll_events must be chronologically sorted (oldest first)
- Deterministic: same roll events → same adjustments

## Acceptance tests
1. `pytest tests/back_adjustment/test_gap_calculator.py::test_cumulative_calculation -v` – verify cumulative sum logic
2. `pytest tests/back_adjustment/test_gap_calculator.py::test_most_recent_roll_zero -v` – latest roll adjustment is 0
3. `pytest tests/back_adjustment/test_gap_calculator.py::test_monotonic_adjustments -v` – adjustments increase backward
4. `pytest tests/back_adjustment/test_gap_calculator.py::test_determinism -v` – same input → same output

## Definition of done
- [ ] Tests added under `tests/back_adjustment/test_gap_calculator.py`
- [ ] Implementation in `data_cleaning/back_adjustment/gap_calculator.py`
- [ ] Algorithm handles empty roll_events list (returns empty list)
- [ ] `pytest tests/back_adjustment/test_gap_calculator.py -v` passes

## Notes
- Back-adjustment principle: newer rolls don't affect older data
- Example: 3 rolls with gaps [+10, -5, +3] → cumulative adjustments [8, 3, 0]
- Adjustment is added to OHLC prices (NOT volume)
