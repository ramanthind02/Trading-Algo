# T002 — Create Phase Skeleton Packages

## Goal
Create the new phase-first package skeleton under `feature_research/` and prepare `feature_research/walkforward/` to host both shared engine modules and phase/type subpackages.

## Context / References
- `docs/kanban/README.md` (task contract)
- `feature_research/walkforward/config.py` (shared config)

## Scope
In scope:
- Create directories:
  - `feature_research/in_sample/{rule_based,continuous_binning}/`
  - `feature_research/walkforward/{rule_based,continuous_binning}/`
  - `feature_research/oos/{rule_based,continuous_binning}/`
- Add minimal `__init__.py` files so imports are unambiguous.
- Add placeholder `README`-style module docstrings only if needed to clarify purpose (keep minimal).

Out of scope:
- Moving any existing modules (that is covered by T003–T007).

## Interfaces
New packages must be importable:
- `feature_research.in_sample`
- `feature_research.in_sample.rule_based`
- `feature_research.in_sample.continuous_binning`
- `feature_research.walkforward.rule_based`
- `feature_research.walkforward.continuous_binning`
- `feature_research.oos.rule_based`
- `feature_research.oos.continuous_binning`

## Invariants / Constraints
- Do not change existing engine module filenames under `feature_research/walkforward/*.py`.
- Keep `feature_research/walkforward/__init__.py` “thin” (no eager imports of heavy modules).

## Acceptance Tests
- `python -c "import feature_research.in_sample.rule_based"`
- `python -c "import feature_research.walkforward.rule_based"`
- `python -c "import feature_research.oos.continuous_binning"`

## Definition of Done
- All new directories exist and are importable.
- `feature_research/walkforward/` safely supports both engine modules and subpackages.

## Notes
- If `feature_research/walkforward/` is currently a namespace package, adding `__init__.py` is preferred here to avoid ambiguity once subpackages are introduced.
