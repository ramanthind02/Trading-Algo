# T006 — Walkforward Phase: Continuous-Binning Entrypoints

## Goal
Create/migrate the continuous-binning walkforward phase entrypoints under `feature_research/walkforward/continuous_binning/` while keeping the shared walkforward engine modules in `feature_research/walkforward/*.py`.

## Context / References
- Engine modules (shared): `feature_research/walkforward/runner.py`, `feature_research/walkforward/io.py`, `feature_research/walkforward/config.py`
- Existing runner entrypoint (to move): `feature_research/continuous_binning/run_walkforward.py`
- Existing config (after T004): `feature_research/in_sample/continuous_binning/config.py`

## Scope
In scope:
- Move/replace continuous-binning walkforward entry script into `feature_research/walkforward/continuous_binning/run_walkforward.py`.
- Add a placeholder (or thin entrypoint) for “walkforward permutation test” under the same folder.

Out of scope:
- Changing binning logic or walkforward selection behavior.

## Interfaces
- `python -m feature_research.walkforward.continuous_binning.run_walkforward` should be the canonical entry.

## Invariants / Constraints
- Engine modules remain stable under `feature_research.walkforward.*`.

## Acceptance Tests
- `python -c "import feature_research.walkforward.continuous_binning.run_walkforward"`
- `python -c "import feature_research.walkforward.io"`

## Definition of Done
- Continuous-binning walkforward entrypoints exist under `feature_research/walkforward/continuous_binning/` and import cleanly.
