# T013 — Joint n_bins Param Sensitivity Notebook

## Goal
Treat `n_bins` as an optimizable parameter in the continuous-binning sensitivity notebook so rankings are computed jointly across feature params and bin counts.

## Context / References
- `feature_research/continuous_binning/param_sensitivity.ipynb`
- `docs/library/Feature_selection/feature_validator.md`
- `docs/plans/2026-02-20-nbins-joint-param-sensitivity-design.md`

## Scope
In scope:
- Update notebook config, grid loop, and report parameter list for joint ranking.

Out of scope:
- Rule-based notebook changes.
- API changes in `eda/parameter_analysis.py`.

## Interfaces (must match)
- Modify: `feature_research/continuous_binning/param_sensitivity.ipynb`
  - Replace `N_BINS` scalar with `N_BINS_GRID` iterable.
  - Add `n_bins` output column to sensitivity grid rows.
  - Pass `[...varying_params, "n_bins"]` to report generation.

## Invariants / Constraints
- Deterministic notebook behavior for fixed config/cached data.
- No lookahead or target alignment assumptions changed.

## Acceptance tests
1. `python - <<'PY'` JSON-load the notebook successfully.
2. `python - <<'PY'` verify updated tokens are present in relevant cells (`N_BINS_GRID`, `row["n_bins"]`, `report_param_names`).

## Definition of done
- [ ] Notebook updated for joint `n_bins` optimization.
- [ ] Structural checks pass.
- [ ] Plan docs added under `docs/plans/`.

## Notes
- Existing plotting/report utilities should work unchanged with expanded `param_names`.
