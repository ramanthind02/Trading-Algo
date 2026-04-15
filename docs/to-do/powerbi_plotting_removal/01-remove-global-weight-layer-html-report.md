# Ticket 01: Remove Global Weight Layer HTML Report

## Why

Portfolio research currently writes a matplotlib-backed **`report.html`** under each phase’s `global_weight_layer/` directory via `export_global_weight_layer_report`. That artifact overlaps with the move to **Power BI**; we are explicitly **dropping** this HTML report while keeping tearsheets and prop-firm HTML.

## Goal

Remove the weight-layer HTML report module and all pipeline/test hooks so no phase output depends on `report.html` for weight layers.

## Scope

- Delete or fully retire `portfolio_research/weight_layer_report.py` (and any re-export shims if present).
- Remove the call from `portfolio_research/pipelines/portfolio_test.py` (currently after `global_portfolio.save_to_vault`, around `export_global_weight_layer_report(...)`).
- Remove `from portfolio_research.weight_layer_report import ...` and adjust comments that mention the HTML export.
- Delete `tests/portfolio_research/test_weight_layer_report.py` or replace with tests that only assert **vault/materialization** behavior if needed.
- Update `tests/portfolio_research/test_run_portfolio_test_multitimeframe.py` (monkeypatches and expectations around `export_global_weight_layer_report`).
- Grep for `weight_layer_report`, `export_global_weight_layer_report`, `export_weight_layer_report`, and `global_weight_layer/report.html`; eliminate remaining references in docs or scripts.

## Out Of Scope

- Changing `ensemble/weight_layer.py` fitting or prediction math.
- Adding Power BI export formats or new parquet/csv contracts (separate initiative).

## Acceptance Criteria

- `pytest tests/portfolio_research/` passes with no stub required for weight-layer HTML export.
- No file named or documented as the canonical weight-layer **`report.html`** remains in the research pipeline.
- Git-ignored or historical `portfolio_research/results/**` HTML artifacts may remain on disk but are not produced by current code paths.

## Risks

- Researchers who bookmarked `global_weight_layer/report.html` lose that URL; communicate the Power BI replacement when available.

## Dependencies

- None (good first ticket).
