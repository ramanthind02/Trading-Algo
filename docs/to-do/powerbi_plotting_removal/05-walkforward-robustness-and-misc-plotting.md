# Ticket 05: Walkforward, Robustness, And Miscellaneous Plotting

## Why

Evaluation utilities still embed matplotlib for fold timelines, Monte Carlo–style charts, and similar. These duplicate BI dashboards and increase headless dependency weight.

## Goal

Remove visualization from evaluation engines while keeping **numeric outputs** and tearsheet generation (tearsheets stay per README decisions).

## Scope

- `utils/evaluation/walkforward/visualization.py` — delete module or reduce to data-only helpers; update `runner.py` if it imports visualization (keep `generate_tearsheet` path intact).
- `utils/evaluation/robustness_test/robustness_engine.py` (and `plotting/robustness.py` if used) — remove `savefig` paths; retain returned statistics structures.
- Grep for remaining `matplotlib` / `plotly` under `utils/evaluation/` and clean up.

## Out Of Scope

- Removing **QuantStats** tearsheet hooks in `walkforward/runner.py` (keep-list).

## Acceptance Criteria

- Walkforward and robustness tests pass without image artifacts unless explicitly testing tearsheets.
- No duplicate plotting entrypoints under `utils/evaluation/` except tearsheet imports.

## Risks

- Some integration tests may assert file existence for PNGs; update to assert DataFrame columns or JSON payloads.

## Dependencies

- **02** for any shared typing or imports from old `metrics.plotting`.
