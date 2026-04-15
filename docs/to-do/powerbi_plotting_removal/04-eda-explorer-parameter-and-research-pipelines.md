# Ticket 04: Top-Level EDA Explorer, Parameter Analysis, And Research Pipelines

## Why

`eda/feature_explorer.py` and `eda/parameter_analysis.py` are large consumers of `metrics.plotting` (Plotly and matplotlib), `write_html`, `savefig`, and optional `plt.show()`. In-sample and related research scripts orchestrate PNG/HTML outputs. These are prime candidates to replace with **exported tables** and Power BI visuals.

## Goal

Remove interactive and static plotting from the `eda/` package and from research entrypoints that exist only to generate charts, without breaking non-visual workflows.

## Scope

- `eda/feature_explorer.py` — remove Plotly/matplotlib figure generation and `write_html`; retain data prep functions if still useful, or split into a thin “export only” module.
- `eda/parameter_analysis.py` — remove imports from `metrics.plotting.parameter_plots` and any `fig.show()` paths; replace with CSV/parquet summaries where researchers still need sensitivity tables.
- `feature_research/in_sample/run_is.py` and `feature_research/pipelines/in_sample.py` (and related) — stop writing plot artifacts to `reports_dir` unless replaced with data files.
- Research notebooks under `research/*.ipynb` — either archive, strip plot cells, or document as legacy (optional; do not block code removal).

## Out Of Scope

- Building Power BI datasets or gateway flows.

## Acceptance Criteria

- No `plotly`, `matplotlib`, or `seaborn` imports in the refactored `eda/` modules covered above.
- Any CLI or script that used to emit HTML/PNG either **emits structured data** or prints a clear “use Power BI” deprecation message if the script is retained stub-only.

## Risks

- Notebook users may rely on inline plots; document migration path in ticket 07 if needed.

## Dependencies

- **02** (trimmed `metrics.plotting`) and ideally **03** (EDA reporter alignment).
