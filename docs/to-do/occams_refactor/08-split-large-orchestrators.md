# Ticket 08: Split Large Orchestrators (`feature_base_model.py`, `vault_manager.py`)

## Why

Some files mix orchestration, I/O policies, and transformation logic in one place:

- `feature_selection/base_models/feature_base_model.py`
- `ensemble/vault_manager.py`

This raises cognitive load and makes isolated testing difficult.

## Goal

Apply SRP-driven decomposition while preserving current public behavior.

## Scope

- For `feature_base_model.py`: separate extraction, alignment, cache access, and model orchestration concerns.
- For `vault_manager.py`: separate path resolution, JSON persistence, migration/compatibility, and high-level orchestration wrappers.
- Add thin composition layers that orchestrate focused helper modules.

## Out Of Scope

- New modeling features.
- Vault schema changes unless required for decomposition.

## Acceptance Criteria

- Each extracted module has one dominant reason to change.
- Unit tests can target focused helpers without heavy integration setup.
- Public API remains backward compatible during transition.

## Risks

- Temporary import churn and partial split states.

## Dependencies

- Ticket 07 recommended before splitting `vault_manager.py`.
