# T004 — Back Adjuster Implementation

## Goal
Apply arithmetic back-adjustment to OHLC data using cumulative adjustment factors, ensuring immutability and data integrity.

## Context / References
- `data_cleaning/back_adjustment/gap_calculator.py` - AdjustmentFactor dataclass (Task 3)
- `docs/plans/2026-02-15-data-migration-design.md` - Adjustment application algorithm
- `docs/library/Data/Norgate.md` - Norgate data structure and migration context

## Data Source
Back-adjusted outputs are produced from legacy fixed-date roll events. See `docs/library/Data/Norgate.md` for schema, formats, and constraints. Norgate continuous futures data uses volume-based rolling, so comparisons focus on level alignment and roll-date differences rather than exact parity.

## Scope
- In scope:
  - Implement `apply_back_adjustment(df, adjustments)` to add cumulative adjustments to OHLC columns
  - Implement `validate_adjusted_data(original, adjusted, adjustments)` for post-processing checks
  - Ensure volume and timestamp columns remain unchanged
  - Return new DataFrame (immutability)
- Out of scope:
  - Calculating adjustments (Task 3)
  - Orchestration and I/O (Task 5)

## Interfaces (must match)
- Add: `data_cleaning/back_adjustment/back_adjuster.py`
  ```python
  def apply_back_adjustment(df: pd.DataFrame, adjustments: List[AdjustmentFactor]) -> pd.DataFrame
  def validate_adjusted_data(original: pd.DataFrame, adjusted: pd.DataFrame, adjustments: List[AdjustmentFactor]) -> bool
  ```

## Data Contracts
- Input DataFrame: columns `['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume']`
- Output DataFrame: same columns, same row count, same order
- Adjustments applied to: open, high, low, close (NOT volume, timestamp, datetime)
- `roll_date` denotes the first timestamp of the new contract; apply adjustments to rows with `T < roll_date`
- Add cumulative_adjustment to OHLC prices
- For rows earlier than the earliest roll_date, apply the sum of all gaps (pre-first-roll segment)

## Dependencies
- `data_cleaning/back_adjustment/gap_calculator.py` (AdjustmentFactor)
- `pandas`, `typing`

## Invariants / Constraints
- Must maintain immutability (return new DataFrame, don't modify input)
- Output row count == input row count
- No negative prices after adjustment (validation check)
- Volume column unchanged (exact equality)
- Timestamp/datetime columns unchanged
- Deterministic: same df + adjustments → same output

## Acceptance tests
1. `pytest tests/back_adjustment/test_back_adjuster.py::test_apply_adjustment_correctness -v` – verify OHLC adjustment math
2. `pytest tests/back_adjustment/test_back_adjuster.py::test_volume_unchanged -v` – volume not modified
3. `pytest tests/back_adjustment/test_back_adjuster.py::test_no_negative_prices -v` – validation catches negative prices
4. `pytest tests/back_adjustment/test_back_adjuster.py::test_immutability -v` – input df not modified
5. `pytest tests/back_adjustment/test_back_adjuster.py::test_determinism -v` – same input → same output

## Definition of done
- [ ] Tests added under `tests/back_adjustment/test_back_adjuster.py`
- [ ] Implementation in `data_cleaning/back_adjustment/back_adjuster.py`
- [ ] Validation function raises ValueError if checks fail
- [ ] `pytest tests/back_adjustment/test_back_adjuster.py -v` passes

## Notes
- For row at time T, find latest roll_date < T, use that cumulative_adjustment
- If T is on/after the most recent roll_date, adjustment = 0 (most recent data period)
- If T is earlier than the earliest roll_date, apply the sum of all gaps (pre-first-roll segment)
- Validation checks: same row count, no negative prices, adjustments match expected cumulative sums
