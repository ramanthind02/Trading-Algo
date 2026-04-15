# Ticket 02: Trim `metrics/plotting` To Tearsheet Support

## Why

`metrics/plotting` re-exports a large matplotlib/Plotly surface (deciles, distributions, correlations, parameter surfaces, feature explorer composites). Most of that will move to **Power BI**; we should keep only what **QuantStats tearsheets** require.

## Goal

Reduce `metrics/plotting` so the supported public API is essentially **`metrics.plotting.graphing.quantstats_reports`** (`generate_tearsheet`, `compute_baseline_results` and any helpers those modules need internally).

## Scope

- Remove or relocate (then delete) modules such as:
  - `decile_plots.py`, `distribution.py`, `correlation.py`, `cumulative_plot.py`, `rolling_decile.py`
  - `feature_explorer_plots.py`, `parameter_plots.py`
- Remove **`portfolio_analytics.py`** from `metrics/plotting/graphing/` (matplotlib portfolio figures; **not** tearsheets—grep shows no external callers outside package `__init__` re-exports).
- Rewrite `metrics/plotting/__init__.py` and `metrics/plotting/graphing/__init__.py` to export only tearsheet-related symbols; prefer **direct imports** of `quantstats_reports` at call sites where practical.
- Update any imports from `metrics.plotting` that expected the old barrel exports (see grep hits in `eda/feature_explorer.py`, `eda/parameter_analysis.py`, tests under `tests/unit-tests/feature_extraction/`, `tests/integration/feature_validator/param_sens/`).

## Out Of Scope

- Rewriting QuantStats tearsheet internals unless required for import hygiene.
- Prop-firm HTML (lives under `prop_firms/`, not this package).

## Acceptance Criteria

- `from metrics.plotting.graphing.quantstats_reports import generate_tearsheet` (and equivalent) still works everywhere tearsheets are generated: `portfolio_research/pipelines/portfolio_test.py`, `ensemble/portfolio_tester.py`, `utils/evaluation/walkforward/runner.py`, `research/bias_node_helpers.py`, integration tests.
- No remaining production imports of deleted `metrics.plotting.*` plot helpers, or those tickets explicitly own the follow-up (see tickets 03–04).

## Risks

- Broad import surface: expect to coordinate with ticket 03–04 or do them immediately after this change.

## Dependencies

- Recommended after **01** (orthogonal) but before heavy EDA refactors if you want a clean package boundary.
