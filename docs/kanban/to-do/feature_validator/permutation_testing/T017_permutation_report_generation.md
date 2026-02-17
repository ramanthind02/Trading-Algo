# T017 — Permutation Report Generation and Visualization

## Goal
Implement comprehensive report generation and visualization layer for permutation testing results, providing researcher-friendly outputs that support manual ensemble formation decisions with rich diagnostics, plots, and structured data exports.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — Phase 4: Permutation Testing (lines 257-352), Researcher Ensemble Formation (lines 355-376)
- `docs/library/Feature_selection/Permutation Testing/in-sample_pt.md` — §4 Researcher Ensemble Formation (lines 172-224)
- Tasks T013 (Stage 1), T014 (Stage 2), T015 (Stage 3), T016 (Orchestration)
- `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md` — Unit vs integration test standards, default config, cache policy

## Scope
In scope:
- Generate structured reports for each validation stage:
  - **VectorShuffleReport**: p-value, null distribution histogram, pass/fail verdict
  - **PipelinePermutationReport**: p-value, null distribution, no-trade permutation count
  - **WalkforwardStabilityReport**: per-fold top-K selections, consistency metrics, stability verdict
  - **PermutationTestSuite**: aggregate funnel statistics, ensemble candidates, researcher summary
- Create diagnostic visualizations:
  - Stage 1: null distribution histogram with original metric and critical value overlay
  - Stage 2: null distribution histogram, comparison of feature_shuffle vs candle_shuffle nulls (continuous only)
  - Stage 3: per-fold top-K heatmap, parameter landscape evolution across folds, stability timeline
  - Funnel flow: Sankey diagram showing param flow through stages (N → N1 → N2 → candidates)
- Export reports in multiple formats: JSON (structured data), Markdown (readable), HTML (interactive)
- Provide researcher guidance: interpretation notes, decision criteria, red flags

Out of scope:
- Individual stage implementations (T013-T016) — already implemented
- Automated ensemble formation — researcher makes final decision using reports
- Real-time/interactive dashboards — static reports only
- Vault integration for saving ensembles — separate workflow after researcher decision

## Interfaces (must match)
- Add: `feature_selection/validation/report_generator.py::generate_permutation_reports()`
  - Signature: `generate_permutation_reports(suite: PermutationTestSuite, output_dir: Path, format: Literal['json', 'markdown', 'html'] = 'markdown') -> ReportBundle`
  - Generates all plots and text reports for the suite
  - Returns `ReportBundle` with file paths to generated artifacts

- Add: `feature_selection/validation/report_generator.py::plot_null_distribution()`
  - Signature: `plot_null_distribution(report: Union[VectorShuffleReport, PipelinePermutationReport], output_path: Path) -> Path`
  - Creates histogram of null distribution with original metric, critical value, and p-value annotation
  - Returns path to saved figure

- Add: `feature_selection/validation/report_generator.py::plot_walkforward_stability()`
  - Signature: `plot_walkforward_stability(stability_report: WalkforwardStabilityReport, output_path: Path) -> Path`
  - Creates multi-panel plot: (1) per-fold top-K heatmap, (2) parameter landscape evolution, (3) stability timeline
  - Returns path to saved figure

- Add: `feature_selection/validation/report_generator.py::plot_funnel_diagram()`
  - Signature: `plot_funnel_diagram(funnel_stats: FunnelStatistics, output_path: Path) -> Path`
  - Creates Sankey diagram or bar chart showing param flow through stages
  - Returns path to saved figure

- Add: `feature_selection/validation/reports.py::ReportBundle`
  - Fields: `suite_json: Path`, `suite_markdown: Path`, `suite_html: Optional[Path]`, `stage1_plots: Dict[str, Path]`, `stage2_plots: Dict[str, Path]`, `stage3_plot: Path`, `funnel_plot: Path`, `timestamp: datetime`

## Data Contracts
- **Report directory structure**:
  ```
  output_dir/
  ├── suite_summary.md               # Researcher-readable summary
  ├── suite_summary.json             # Structured data export
  ├── suite_summary.html             # Interactive HTML (optional)
  ├── stage1/
  │   ├── null_dist_RSI_3.png
  │   ├── null_dist_RSI_4.png
  │   └── ...
  ├── stage2/
  │   ├── null_dist_RSI_3.png
  │   ├── null_dist_RSI_4.png
  │   └── mode_comparison.png       # feature_shuffle vs candle_shuffle
  ├── stage3/
  │   └── walkforward_stability.png  # Multi-panel stability plot
  └── funnel/
      └── funnel_diagram.png         # Param flow through stages
  ```

- **Markdown summary format**:
  - Feature overview: name, type, parameter grid
  - Funnel statistics: params tested/passed per stage
  - Ensemble candidates: list with justification (passed tests + stable)
  - Interpretation guidance: what the results mean, decision criteria
  - Red flags: warnings if feature shows instability or borderline p-values

- **JSON export schema** (structured data for programmatic access):
  - Suite metadata: feature_name, feature_type, timestamp
  - Stage 1 results: per-param p-values, pass/fail verdicts
  - Stage 2 results: per-param p-values, permutation modes, no-trade counts
  - Stage 3 results: per-fold top-K, consistency metrics, stability verdict
  - Funnel statistics: computational savings, ensemble candidates

## Dependencies
- `feature_selection/validation/reports.py` — All report dataclasses (T013-T016)
- `matplotlib`, `seaborn` for static plots
- `plotly` (optional) for interactive HTML plots
- `pandas` for tabular data formatting
- `json` for JSON export
- `jinja2` (optional) for HTML templating
- `pathlib` for file path handling

## Invariants / Constraints
- Deterministic: same suite → same plots and reports (fixed random seeds for any stochastic elements)
- Self-contained: reports include all information needed for researcher decision (no external dependencies)
- Versioned: include timestamp and configuration hash for reproducibility
- Readable: markdown reports should be human-readable without tools (plain text tables, ASCII art if needed)
- Exportable: JSON format should be parseable for downstream tools (e.g., automated meta-analysis)

## Acceptance tests

**Unit tests:**
- `test_markdown_contains_required_sections()` — build a synthetic PermutationTestSuite with known values, call `generate_permutation_reports(..., format='markdown')`, assert the output markdown string contains: "Feature Overview", "Funnel Statistics", "Ensemble Candidates", "Interpretation Guidance", "Red Flags", "Next Steps"
- `test_json_export_round_trips()` — build synthetic PermutationTestSuite, export to JSON, deserialize, assert all numeric fields (p-values, savings_pct, consistency metrics) round-trip without loss
- `test_null_distribution_plot_elements()` — mock VectorShuffleReport with known null_distribution, original_metric, and critical_value, call `plot_null_distribution()`, assert figure has histogram bars, a vertical line for original_metric, a vertical line for critical_value, and a p-value annotation
- `test_walkforward_stability_plot_panels()` — mock WalkforwardStabilityReport with 2 folds, call `plot_walkforward_stability()`, assert figure has 3 subplots (fold heatmap, landscape evolution, stability timeline)
- `test_funnel_diagram_counts()` — build FunnelStatistics with known counts (total=10, stage1_pass=7, stage2_pass=5, stable=3, candidates=3), call `plot_funnel_diagram()`, assert displayed counts match
- `test_report_bundle_fields()` — assert ReportBundle contains all required fields: `suite_json`, `suite_markdown`, `suite_html`, `stage1_plots`, `stage2_plots`, `stage3_plot`, `funnel_plot`, `timestamp`
- `test_report_bundle_paths_exist()` — call `generate_permutation_reports()` with synthetic suite into a temp directory, assert all Path fields in the returned ReportBundle point to existing files
- `test_red_flags_borderline_pvalue()` — build suite with a Stage 2 param having p=0.12 (borderline), assert markdown output contains a "WARNING" or "borderline" mention for that param
- `test_red_flags_unstable_feature()` — build suite with `is_stable=False` in Stage 3 report, assert markdown output contains an instability warning

**Integration tests:**
- There is no dedicated integration test function for T017 in isolation; report generation is exercised as part of the full permutation testing pipeline. The integration test entry point is `tests/integration/feature_validator/test_permutation_testing.py::test_early_stopping_orchestration()` (T016), which runs the full suite including report generation.
- If standalone report generation integration verification is needed: use default config RSI lookback 5, ES daily, 2020-2023 to produce a real PermutationTestSuite, then call `generate_permutation_reports()` and assert ReportBundle files exist and are non-empty.
- Customizable: `output_format` ('json', 'markdown', 'html'), `output_dir`, and suite configuration are exposed as function parameters.
- Researcher manual verification:
  - After running the orchestration test, open the generated markdown summary and confirm all sections are present and coherent
  - Inspect null distribution plots: histogram should be visible, original metric and critical value lines should be clearly annotated
  - Inspect walkforward stability plot: fold heatmap rows should correspond to walkforward folds, stable regions should be highlighted
  - Open JSON export and verify it is valid and contains p-values, consistency metrics, and ensemble candidates

**Cache policy (integration):**
- Depends on T016 orchestration test cache: RSI lookback [5], ES, D, 2020-2023
- Use existing cache: `USE_CACHE=True`
- If cache missing: skip with message "Run CacheManager.populate_cache() first"

## Definition of done
- [ ] Unit tests added under `tests/validators/permutation/test_report_generation_unit.py`
- [ ] Integration coverage provided by `tests/integration/feature_validator/test_permutation_testing.py::test_early_stopping_orchestration()` (T016), which exercises full report generation end-to-end
- [ ] `ReportBundle` dataclass defined in `feature_selection/validation/reports.py`
- [ ] Report generation functions implemented in `feature_selection/validation/report_generator.py`
- [ ] Plotting functions implemented (null distribution, walkforward stability, funnel diagram)
- [ ] Markdown and JSON export formats implemented
- [ ] `pytest tests/validators/permutation/test_report_generation_unit.py -v` passes
- [ ] Docs updated: example reports in docs/examples/, docstrings with usage examples

## Notes
- **Markdown summary template** (key sections):
  1. Feature Overview: name, type, parameter grid size
  2. Configuration: nreps, alpha, metric_threshold, fold structure
  3. Funnel Statistics: params tested/passed per stage, computational savings
  4. Ensemble Candidates: list with justification (passed tests + stable region)
  5. Interpretation Guidance: what results mean, decision criteria
  6. Red Flags: warnings for instability, borderline p-values, low sample sizes
  7. Next Steps: recommended actions (form ensemble, reject feature, investigate further)

- **Visualization best practices**:
  - Null distribution plots: annotate p-value, critical value, original metric clearly
  - Color coding: green for pass, red for fail, yellow for borderline (p ∈ [0.05, 0.15])
  - Walkforward stability: use heatmap for top-K evolution, highlight stable regions
  - Funnel diagram: show param counts at each stage, highlight early stopping savings

- **Researcher guidance examples**:
  - "Feature shows strong stability (top-K consistent across 4/4 folds). Recommend selecting 2-3 params from {RSI_3, RSI_4, RSI_5} for ensemble."
  - "WARNING: Feature unstable (top-K jumps between {3,4,5} and {14,20,10} across folds). Consider rejecting or investigating regime-dependence."
  - "Borderline Stage 2 p-values (p=0.12 for candle_shuffle). Consider increasing nreps or using stricter α=0.05 for production."

- **JSON export use cases**:
  - Automated meta-analysis across multiple features
  - Version control for validation results (track changes over time)
  - Integration with downstream portfolio construction tools
  - Reproducibility: export includes full configuration and results

- **HTML interactive reports (optional enhancement)**:
  - Plotly charts with hover tooltips (show exact p-values, metric values)
  - Collapsible sections (expand/collapse stage details)
  - Embedded plots and tables
  - Click to navigate between stages
