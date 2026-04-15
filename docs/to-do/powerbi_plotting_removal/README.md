# Power BI Migration: Remove In-Repo Plotting

## Purpose

Retire Python-side charting and HTML report generation that duplicates what we will do in **Power BI**. This backlog is **scoping and sequencing only**; designing export pipelines, semantic models, or Power BI workspaces is explicitly **out of scope** for these tickets.

## Decisions Locked

- **Keep** QuantStats **HTML tearsheets** (`generate_tearsheet`, walkforward/portfolio research callers, bias-node tearsheet helpers, and tests that pin tearsheet behavior).
- **Keep** **prop firm** report flows that produce **HTML** (and any matplotlib usage **inside** those reports to embed figures), e.g. `prop_firms/reporting.py`, `prop_firms/optimization_reporting.py`, and their runners under `prop_firms/`.
- **Keep** **Norgate migration QA plotting**: `scripts/validate_norgate_migration.py` and its Plotly `write_html` outputs (e.g. per-ticker overlays under the report directory), plus any tests or docs tied to that workflow.
- **Remove** the **global weight layer HTML report** (`portfolio_research/weight_layer_report.py`, `report.html` under phase `global_weight_layer/`, direct callers, and dedicated tests). Weight-layer **logic** in `ensemble/weight_layer.py` stays; only the research HTML artifact path goes away.
- **Remove** (or replace with tabular-only outputs) other matplotlib / Plotly / seaborn research and validation plotting unless it falls under the keep buckets above (tearsheets, prop-firm HTML, Norgate migration).

## Tickets

- [01-remove-global-weight-layer-html-report.md](./01-remove-global-weight-layer-html-report.md)
- [02-trim-metrics-plotting-package.md](./02-trim-metrics-plotting-package.md)
- [03-feature-selection-eda-and-validation-plots.md](./03-feature-selection-eda-and-validation-plots.md)
- [04-eda-explorer-parameter-and-research-pipelines.md](./04-eda-explorer-parameter-and-research-pipelines.md)
- [05-walkforward-robustness-and-misc-plotting.md](./05-walkforward-robustness-and-misc-plotting.md)
- [06-scripts-top-level-plotting-package-and-optional-ui.md](./06-scripts-top-level-plotting-package-and-optional-ui.md)
- [07-tests-requirements-docs-cleanup.md](./07-tests-requirements-docs-cleanup.md)

## Suggested Execution Order

1. `01-remove-global-weight-layer-html-report.md` (isolated, clear contract change).
2. `02-trim-metrics-plotting-package.md` (many downstream imports depend on this package shape).
3. `03-feature-selection-eda-and-validation-plots.md`
4. `04-eda-explorer-parameter-and-research-pipelines.md`
5. `05-walkforward-robustness-and-misc-plotting.md`
6. `06-scripts-top-level-plotting-package-and-optional-ui.md`
7. `07-tests-requirements-docs-cleanup.md`

## Definition Of Done

- No code path writes `global_weight_layer/report.html` or depends on `weight_layer_report` exports.
- `metrics.plotting` exposes only what tearsheets need (primarily `quantstats_reports`); removed modules are gone or replaced with non-plotting APIs.
- Feature validation and EDA pipelines either emit **tables/files only** suitable for external BI, or documented stubs; no stray `savefig` / `write_html` for research dashboards except the keep-list (tearsheets, prop-firm HTML, Norgate migration QA).
- Test suite and optional dependencies (`matplotlib`, `plotly`, `seaborn`, etc.) reflect the reduced surface; CI stays green.

## Notes From Repo Inventory (April 2026)

Inventory was cross-checked with sub-agent exploration: main plotting clusters are `metrics/plotting/`, `feature_selection/eda/` and `validation/report_generator.py`, `eda/feature_explorer.py` and `eda/parameter_analysis.py`, `utils/evaluation/walkforward/visualization.py`, robustness plotting, `feature_research` pipelines, `scripts/*demo*`, and top-level `plotting/` (e.g. prop-firm **PNG** dashboards for demos—not the same as `prop_firms/*` HTML reports). **`scripts/validate_norgate_migration.py` is a keep-list exception** (Plotly HTML for data migration QA).
