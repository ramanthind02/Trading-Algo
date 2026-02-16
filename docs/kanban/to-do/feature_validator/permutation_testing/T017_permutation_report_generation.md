# T017 — Permutation Report Generation and Visualization

## Goal
Implement comprehensive report generation and visualization layer for permutation testing results, providing researcher-friendly outputs that support manual ensemble formation decisions with rich diagnostics, plots, and structured data exports.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — Phase 4: Permutation Testing (lines 257-352), Researcher Ensemble Formation (lines 355-376)
- `docs/library/Feature_selection/Permutation Testing/in-sample_pt.md` — §4 Researcher Ensemble Formation (lines 172-224)
- Tasks T013 (Stage 1), T014 (Stage 2), T015 (Stage 3), T016 (Orchestration)

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
1. `pytest tests/integration/feature_validator/permutation_testing/test_report_generation.py::test_markdown_summary` — Verify markdown report includes all required sections (feature overview, funnel stats, ensemble candidates, guidance)
2. `pytest tests/integration/feature_validator/permutation_testing/test_report_generation.py::test_json_export` — Verify JSON export is valid and contains all suite data (can round-trip deserialize)
3. `pytest tests/integration/feature_validator/permutation_testing/test_report_generation.py::test_null_distribution_plot` — Verify plot has histogram, original metric line, critical value line, p-value annotation
4. `pytest tests/integration/feature_validator/permutation_testing/test_report_generation.py::test_walkforward_stability_plot` — Verify multi-panel plot with fold heatmap, landscape evolution, stability timeline
5. `pytest tests/integration/feature_validator/permutation_testing/test_report_generation.py::test_funnel_diagram` — Verify funnel plot shows correct param counts at each stage
6. `pytest tests/integration/feature_validator/permutation_testing/test_report_generation.py::test_report_bundle_completeness` — Verify ReportBundle contains paths to all expected artifacts (markdown, JSON, plots)
7. `pytest tests/integration/feature_validator/permutation_testing/test_report_generation.py::test_rsi_full_suite_report` — End-to-end test: run full permutation suite on RSI [2,3,4,5,6,7,8,9,10], generate reports, verify completeness

## Definition of done
- [ ] Tests added under `tests/integration/feature_validator/permutation_testing/test_report_generation.py`
- [ ] `ReportBundle` dataclass defined in `feature_selection/validation/reports.py`
- [ ] Report generation functions implemented in `feature_selection/validation/report_generator.py`
- [ ] Plotting functions implemented (null distribution, walkforward stability, funnel diagram)
- [ ] Markdown and JSON export formats implemented
- [ ] `pytest tests/integration/feature_validator/permutation_testing/test_report_generation.py -v` passes
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
