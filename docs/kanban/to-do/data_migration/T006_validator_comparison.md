# T006 — Validator and Comparison Tools

## Goal
Build comparison tools to analyze differences between back-adjusted old data (fixed-date rolling) and Norgate data (volume-based rolling).

## Context / References
- `data_cleaning/back_adjustment/orchestrator.py` - AdjustmentMetadata (Task 5)
- `data/intraday_1min_adjusted/` - Back-adjusted old data
- Norgate daily data location (placeholder: `data/norgate/continuous_futures/` — update when ingestion path is finalized)
- `docs/plans/2026-02-15-data-migration-design.md` - Comparison methodology
- `docs/library/Data/Norgate.md` - Norgate data structure and migration context

## Data Source
Norgate continuous futures data uses volume-based rolling; legacy data uses fixed-date roll rules. See `docs/library/Data/Norgate.md` for schema, formats, and constraints. Comparisons should expect roll-date differences and focus on aligned level statistics. Daily Norgate series should be aligned with resampled legacy 1-minute data for fair comparison.

## Scope
- In scope:
  - Implement `compare_price_levels(old_data, new_data)` to calculate correlation, mean difference, max divergence
  - Implement `compare_roll_dates(old_metadata, new_data)` to show where roll dates differ
  - Implement `generate_comparison_report(ticker)` to create markdown report with statistics
  - Save comparison reports to `docs/library/Data/comparisons/{ticker}_comparison.md`
- Out of scope:
  - Automated chart generation (future enhancement)
  - Re-rolling old data using Norgate dates (Option B, future work)

## Interfaces (must match)
- Add: `data_cleaning/back_adjustment/validator.py`
  ```python
  def compare_price_levels(old_data: pd.DataFrame, new_data: pd.DataFrame) -> Dict[str, float]
  def compare_roll_dates(old_metadata: Path, new_data: pd.DataFrame) -> pd.DataFrame
  def generate_comparison_report(ticker: Ticker, old_data_path: Path, new_data_path: Path, metadata_path: Path) -> str
  ```

## Data Contracts
- `compare_price_levels()` returns dict with keys: `correlation`, `mean_diff`, `max_divergence`, `rmse`
- `compare_roll_dates()` returns DataFrame with columns: `roll_date_old`, `roll_date_new`, `delta_days`
- `generate_comparison_report()` returns markdown string and saves to file
- Comparison report includes: price statistics, roll date comparison table, summary findings
- Norgate input path placeholder: `data/norgate/continuous_futures/` (daily). Update once ingestion path is finalized.
- `roll_date_new` is the first `Date` (daily close) where `Delivery Month` changes in the Norgate continuous futures series; if the column is unavailable, report `None` and note the missing field in the report.

## Dependencies
- `data_cleaning/back_adjustment/orchestrator.py` (AdjustmentMetadata)
- `pandas`, `numpy`, `pathlib`, `json`, `typing`

## Invariants / Constraints
- Align timestamps between old and new data (may need resampling if frequencies differ)
- Handle missing data gracefully (report gaps)
- Correlation calculation requires overlapping time periods
- Deterministic: same data → same statistics

## Acceptance tests
1. `pytest tests/back_adjustment/test_validator.py::test_price_level_comparison -v` – correlation/diff calculations
2. `pytest tests/back_adjustment/test_validator.py::test_roll_date_comparison -v` – roll date alignment
3. `pytest tests/back_adjustment/test_validator.py::test_report_generation -v` – markdown report created
4. `pytest tests/back_adjustment/test_validator.py::test_determinism -v` – same input → same output

## Definition of done
- [ ] Tests added under `tests/back_adjustment/test_validator.py`
- [ ] Implementation in `data_cleaning/back_adjustment/validator.py`
- [ ] Comparison reports saved to `docs/library/Data/comparisons/`
- [ ] `pytest tests/back_adjustment/test_validator.py -v` passes

## Notes
- May need to aggregate 1-minute old data to daily for fair comparison with Norgate daily data
- Roll date differences expected due to fixed-date vs volume-based methodologies
- Price level differences may reveal systematic bias or drift from roll methodology
- Future: Add chart generation (price overlay, roll date timeline, difference plot)
