# T005 — Walkforward Phase: Rule-Based Entrypoints

## Goal
Create/migrate the rule-based walkforward phase entrypoints under `feature_research/walkforward/rule_based/` while keeping the shared walkforward engine modules in `feature_research/walkforward/*.py`.

## Context / References
- Engine modules (shared): `feature_research/walkforward/runner.py`, `feature_research/walkforward/io.py`, `feature_research/walkforward/config.py`
- Existing runner entrypoint (to move): `feature_research/rule_based/run_walkforward.py`
- Existing config (after T003): `feature_research/in_sample/rule_based/config.py`

## Scope
In scope:
- Move/replace rule-based walkforward entry script into `feature_research/walkforward/rule_based/run_walkforward.py`.
- Add a placeholder (or thin entrypoint) for “walkforward permutation test” under the same folder, even if it is not fully wired yet.

Out of scope:
- Changing the walkforward engine algorithm.

## Interfaces
- `python -m feature_research.walkforward.rule_based.run_walkforward` should remain the canonical entry.
- Any config access should be explicit (prefer importing in-sample config + embedded `WalkforwardResearchConfig`).

## Invariants / Constraints
- Engine must not import `feature_research.walkforward.rule_based.*` (avoid cycles).
- Keep output root defaults unchanged unless explicitly scoped.

## Acceptance Tests
- `python -c "import feature_research.walkforward.rule_based.run_walkforward"`
- `python -c "import feature_research.walkforward.runner"`

## Definition of Done
- Rule-based walkforward entrypoints exist under `feature_research/walkforward/rule_based/` and import cleanly.
