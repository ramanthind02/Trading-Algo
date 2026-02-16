# T005 — Orchestrator and CLI

## Goal
Build CLI orchestrator to execute the full back-adjustment pipeline for one or all tickers, with parallel processing support and metadata logging.

## Context / References
- `data_cleaning/back_adjustment/roll_rules.py` - get_roll_rule() (Task 1)
- `data_cleaning/back_adjustment/roll_detector.py` - detect_roll_dates() (Task 2)
- `data_cleaning/back_adjustment/gap_calculator.py` - calculate_adjustments() (Task 3)
- `data_cleaning/back_adjustment/back_adjuster.py` - apply_back_adjustment() (Task 4)
- `docs/plans/2026-02-15-data-migration-design.md` - Pipeline orchestration

## Scope
- In scope:
  - Create `AdjustmentMetadata` dataclass for audit trail (ticker, processing_date, roll_events, adjustments, source_file, output_file, etc.)
  - Implement `process_ticker(ticker, input_dir, output_dir)` to run full pipeline for one ticker
  - Implement CLI with argparse: --ticker, --all, --parallel, --input-dir, --output-dir
  - Save adjustment metadata JSON for each ticker
  - Progress logging (ticker, start/end time, rolls detected, adjustment range)
- Out of scope:
  - Validation/comparison (Task 6)

## Interfaces (must match)
- Add: `data_cleaning/back_adjustment/orchestrator.py`
  ```python
  @dataclass(frozen=True)
  class AdjustmentMetadata:
      ticker: Ticker
      processing_date: datetime
      roll_events: List[RollEvent]
      adjustments: List[AdjustmentFactor]
      source_file: str
      output_file: str
      num_rolls_detected: int
      total_adjustment_range: float

  def process_ticker(ticker: Ticker, input_dir: Path, output_dir: Path) -> AdjustmentMetadata

  # CLI:
  # python -m data_cleaning.back_adjustment.orchestrator --ticker ES
  # python -m data_cleaning.back_adjustment.orchestrator --all --parallel
  ```

## Data Contracts
- Input files: `{input_dir}/{ticker}.parquet` (1-minute OHLCV data)
- Output files: `{output_dir}/{ticker}.parquet` (back-adjusted)
- Metadata files: `data/adjustment_metadata/{ticker}.json`
- AdjustmentMetadata serialized to JSON for audit trail

## Dependencies
- All prior tasks (roll_rules, roll_detector, gap_calculator, back_adjuster)
- `argparse`, `pathlib`, `json`, `datetime`, `logging`, `concurrent.futures` (for parallel mode)

## Invariants / Constraints
- Never modify original input files (read-only)
- Create output directories if they don't exist
- Log errors but don't crash on single ticker failure (when processing --all)
- Parallel mode: each ticker processed independently (no shared state)
- Deterministic: same input → same output (given same timestamp for processing_date)

## Acceptance tests
1. `pytest tests/back_adjustment/test_orchestrator.py::test_process_single_ticker -v` – end-to-end for one ticker
2. `pytest tests/back_adjustment/test_orchestrator.py::test_metadata_saved -v` – JSON metadata written correctly
3. `pytest tests/back_adjustment/test_orchestrator.py::test_cli_single_ticker -v` – CLI argument parsing for --ticker
4. `pytest tests/back_adjustment/test_orchestrator.py::test_cli_all_tickers -v` – CLI --all flag
5. `pytest tests/back_adjustment/test_orchestrator.py::test_determinism -v` – same input → same output

## Definition of done
- [ ] Tests added under `tests/back_adjustment/test_orchestrator.py`
- [ ] Implementation in `data_cleaning/back_adjustment/orchestrator.py`
- [ ] CLI accepts --ticker, --all, --parallel, --input-dir, --output-dir arguments
- [ ] Metadata JSON files saved to `data/adjustment_metadata/`
- [ ] `pytest tests/back_adjustment/test_orchestrator.py -v` passes

## Notes
- Default input_dir: `data/intraday_1min_original/`
- Default output_dir: `data/intraday_1min_adjusted/`
- Parallel mode: use ProcessPoolExecutor for CPU-bound work
- Log format: "Processing {ticker}: {num_rolls} rolls detected, adjustment range {min}-{max}"
