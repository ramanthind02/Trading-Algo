# Ticket 07: Tests, Requirements, And Documentation Cleanup

## Why

Removing plotting changes import graphs, optional dependencies, and developer docs. A final pass keeps the repo honest about what still requires `matplotlib`, `plotly`, `kaleido`, `quantstats`, etc.

## Goal

Align `requirements*.txt` / `pyproject` extras, CI, and docs with the reduced visualization surface.

## Scope

- Run full `pytest tests/` and fix collection errors from deleted modules.
- Remove or rewrite tests that only validated plot APIs (`tests/integration/feature_validator/param_sens/test_visualizations.py`, `tests/unit-tests/feature_extraction/test_parameter_sensitivity.py`, visualization sections in feature-research engine tests, etc.).
- Audit dependencies: if **QuantStats tearsheets** still need matplotlib indirectly, keep minimal deps; drop `plotly` / `seaborn` only if nothing in the keep-list imports them ( **`scripts/validate_norgate_migration.py` requires Plotly** ).
- Update `docs/api/` or library docs that referenced removed plot functions (only where those docs are already maintained for public API).
- Add a short pointer from `CLAUDE.md` / `AGENTS.md` or internal playbook: **research charts → Power BI**; **tearsheets, prop-firm HTML, and Norgate migration QA plots remain in Python**.

## Out Of Scope

- Authoring Power BI semantic-model documentation.

## Acceptance Criteria

- Strict test suite green; no orphaned imports of deleted `metrics.plotting` symbols.
- Dependency files reflect actual imports (optional: run `pipdeptree` or a static checker where the repo already does).
- This `powerbi_plotting_removal` README’s Definition Of Done is satisfied.

## Risks

- Transitive deps from `quantstats` may still pull matplotlib; document that as acceptable until tearsheet stack changes.

## Dependencies

- Completes after **01–06** (or incrementally as each ticket merges).
