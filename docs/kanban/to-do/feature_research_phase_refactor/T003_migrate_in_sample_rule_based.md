# T003 — Migrate In-Sample Rule-Based Research

## Goal
Move rule-based in-sample research modules (EDA, parameter sensitivity helpers, IS permutation entrypoints) from `feature_research/rule_based/` into `feature_research/in_sample/rule_based/` and update intra-package imports.

## Context / References
- Existing (to be migrated): `feature_research/rule_based/config.py`, `feature_research/rule_based/data_loader.py`, `feature_research/rule_based/pipeline.py`, `feature_research/rule_based/run_eda.py`
- Shared walkforward engine remains in `feature_research/walkforward/*.py`
- `docs/library/Feature_selection/feature_validator.md` (Phase 1 & 2 descriptions)

## Scope
In scope:
- Relocate rule-based in-sample code into `feature_research/in_sample/rule_based/`.
- Ensure `config.py` remains the single researcher-editable surface for the rule-based in-sample phase.
- If IS permutation entrypoints currently live outside, add phase entrypoints under `feature_research/in_sample/rule_based/` that call into the existing permutation infrastructure.

Out of scope:
- Walkforward scripts/entrypoints (handled by T005).

## Interfaces
New canonical imports (examples):
- `from feature_research.in_sample.rule_based.config import load_config`
- `from feature_research.in_sample.rule_based.data_loader import ...`

## Invariants / Constraints
- Do not change config field semantics; only move/update import paths.
- Keep output directory behavior unchanged unless explicitly addressed by a dedicated outputs task.

## Acceptance Tests
- `python -c "from feature_research.in_sample.rule_based.config import load_config; load_config()"`
- `python -c "import feature_research.in_sample.rule_based.run_eda"`
- `python -m compileall feature_research/in_sample/rule_based`

## Definition of Done
- Rule-based in-sample modules live under `feature_research/in_sample/rule_based/`.
- Internal imports updated; package imports cleanly.
