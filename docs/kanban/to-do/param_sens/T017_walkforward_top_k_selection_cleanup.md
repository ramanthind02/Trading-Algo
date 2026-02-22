# T017 — Walkforward Top-K Selection Cleanup (No Robustness + Precise Selected Params)

## Goal
Fix walkforward selection/reporting so selected rows show exact parameter combos including bin count for continuous features, and remove robustness scoring from enhanced top-k selection.

## Context / References
- `feature_research/walkforward/top_k_selection.py`
- `feature_research/walkforward/runner.py`
- `feature_research/walkforward/io.py`
- `feature_research/walkforward/config.py`
- `docs/library/Feature_selection/Parameter Sensitivity/top_k_ensemble_selection.md`
- `docs/api/feature_selection.md`
- `feature_research/shared_results/continuous/rsi/walkforward/selected_params_detailed.csv`

## Scope
In scope:
- Remove robustness term from enhanced top-k quality scoring and outputs.
- Rebalance selection toward strong smoothed performance by config defaults.
- Export `selected_params_detailed.csv` as one row per top-k selected member per fold (`selected_by_diversity=True`).
- Ensure selected parameter label includes bin count for continuous features.
- Update walkforward docs/API docs for changed schema and algorithm.

Out of scope:
- Changes to upstream binning model internals.
- Changes to unrelated feature-selection pipelines.

## Interfaces (must match)
- Modify: `feature_research/walkforward/top_k_selection.py`
  - Remove robustness API/logic from `EnhancedSelectionResult` and `run_enhanced_selection(...)`.
  - `compute_quality_scores(...)` uses smoothed objectives only.
- Modify: `feature_research/walkforward/config.py`
  - Remove robustness config fields.
  - Add/adjust fields for stronger smoothed-objective emphasis (e.g., `quality_exponent`) and diversity behavior.
- Modify: `feature_research/walkforward/runner.py`
  - Add canonical label builder that can include `bin_count` when provided.
  - Remove `robustness_score` from fold score outputs.
- Modify: `feature_research/walkforward/io.py`
  - Remove robustness column handling.
  - Build selected-params rows from `selected_by_diversity=True` when available.
- Modify docs: `docs/library/Feature_selection/Parameter Sensitivity/top_k_ensemble_selection.md`, `docs/api/feature_selection.md`.

## Data Contracts
- `fold_scores.csv` enhanced columns: `trade_frequency`, `quality_score`, `selected_by_diversity`.
- `selected_params_detailed.csv` contains top-k selected rows per fold (not only primary rank-1 row).
- Continuous-feature row labels include `bin_count=<n>` as part of `param_label`.

## Invariants / Constraints
- Deterministic ordering/ranking unchanged outside the requested schema/algorithm updates.
- No-lookahead guarantees unchanged.
- Same inputs yield identical CSV outputs.

## Acceptance tests
1. `pytest tests/feature_research/walkforward/test_top_k_selection.py -q`
2. `pytest tests/feature_research/walkforward/test_runner.py -q`
3. `pytest tests/feature_research/walkforward/test_io.py -q`
4. `pytest tests/feature_research/walkforward/test_config.py -q`

## Definition of done
- [ ] Unit tests added/updated for no-robustness algorithm and selected-rows export behavior.
- [ ] Walkforward code updated and tests pass.
- [ ] `docs/api/feature_selection.md` updated.
- [ ] `docs/library/Feature_selection/Parameter Sensitivity/top_k_ensemble_selection.md` updated.

## Notes
- Keep API changes constrained to walkforward modules only.
