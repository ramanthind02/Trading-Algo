# T007 — Add Strict OOS Phase Scaffolding (Rule-Based + Continuous)

## Goal
Create the strict OOS phase packages under `feature_research/oos/` for both feature types, with entrypoints for OOS evaluation and an OOS permutation-test entrypoint.

## Context / References
- OOS concept: `docs/library/Feature_selection/pipeline_overview.md` (strict holdout)
- Existing configs: in-sample configs under `feature_research/in_sample/.../config.py` (after T003/T004)

## Scope
In scope:
- Create `feature_research/oos/rule_based/` and `feature_research/oos/continuous_binning/` with:
  - A small config surface (date range / tickers / feature spec reference).
  - An OOS evaluation entrypoint.
  - An OOS permutation-test entrypoint (may be placeholder if wiring is non-trivial).

Out of scope:
- Enforcing that strict OOS is a formal “gate” (keep as evaluation/sanity by default).

## Interfaces
- Canonical entrypoints:
  - `python -m feature_research.oos.rule_based.run_oos`
  - `python -m feature_research.oos.continuous_binning.run_oos`

## Invariants / Constraints
- Treat strict OOS as read-only: do not reuse OOS results to tune configs.

## Acceptance Tests
- `python -c "import feature_research.oos.rule_based"`
- `python -c "import feature_research.oos.continuous_binning"`

## Definition of Done
- OOS phase packages exist for both types and are importable.
