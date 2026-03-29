# Ticket 05: Simplify Vault, Ensemble, And Control-File Loading

## Why

The current persistence and loader path is built around model factories, fitted-state restore, and schema fields that only exist because features used to fit bin geometry.

## Goal

Rewrite the vault and ensemble choke points to load and save only node-backed domain-discrete features.

## Scope

- Update ensemble/control-file loading so base models are created from `bias_node_spec` only.
- Remove binning-model factory and fitted-state restore paths.
- Remove feature-entry persistence fields tied to fitted feature state.
- Keep `metadata.is_fit` for ensemble weights only.
- Hard-fail on legacy feature entries that declare:
  - `continuous_binning`
  - `rule_based`
  - base-model `fitted_params`
  - control-file `fitted_base_models`
- Update save paths so new vault artifacts persist only the frozen node contract and ensemble-level fitted state.

## Out Of Scope

- Walkforward evaluation logic.
- Research-time diagnostics.

## Acceptance Criteria

- New control files and vault entries use only the domain-discrete schema.
- Ensemble loading no longer instantiates or restores fitted binning models.
- Legacy fitted feature artifacts fail with one clear migration error path.
- The loader surface is simpler than before: one feature contract, one factory path.

## Risks

- Synthetic tests and old local control files will break immediately and must be updated together.

## Dependencies

- Tickets 01 and 04.
