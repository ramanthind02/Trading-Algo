# Ticket 03: Feature Selection EDA And Validation Plotting

## Why

`feature_selection/eda/` and `feature_selection/validation/report_generator.py` build matplotlib figures, embed PNGs, and save report artifacts tied to exploratory and permutation workflows. Those visuals should migrate to **Power BI** (or plain tabular exports) rather than in-repo rendering.

## Goal

Remove plotting from feature-selection EDA and validation reporting while preserving **data outputs** (statistics tables, JSON, pickles, CSV) needed for research and tests.

## Scope (indicative files)

- `feature_selection/eda/eda_reporter.py` — strip `savefig` / figure assembly; keep serializable payloads.
- `feature_selection/eda/continuous_eda.py`, `common_eda.py`, `rule_based_eda.py` — remove figure construction or gate behind explicit deprecation removal.
- `feature_selection/eda/eda_dataclasses.py` — drop `matplotlib.figure.Figure` from types if figures are removed.
- `feature_selection/validation/report_generator.py` — remove heatmaps and similar; emit numeric summaries or file paths for downstream BI only if still required.
- `feature_selection/validation/binning/plots.py` and related stubs — delete or collapse to typed placeholders without matplotlib.
- Callers such as `feature_research/pipelines/in_sample.py` (cumsum PNG / Agg backend) — align with non-plotting outputs.

## Out Of Scope

- Changing statistical definitions of EDA or permutation tests (only visualization delivery).

## Acceptance Criteria

- Unit/integration tests under `tests/unit-tests/validators/eda/`, `tests/integration/feature_validator/`, and feature-research validation tests either use **tabular fixtures** or skip plot assertions.
- No matplotlib imports remain in the scoped modules **unless** required for a keep-list exception (none here).

## Risks

- Golden-file or image-based tests will need explicit replacement assertions.

## Dependencies

- **02** should be done or in progress so `metrics.plotting` is not re-expanded to compensate.
