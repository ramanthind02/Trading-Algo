# Ticket 01: Define Domain-Discrete Contract And Schema

## Why

The codebase still has multiple feature contracts: continuous binning, rule-based binning, and vault/control-file payloads tied to fitted state.
That ambiguity prevents a clean cutover.

## Goal

Define one canonical feature contract based on a frozen domain-discrete spec and update all persisted schemas to match it.

## Scope

- Define `DomainDiscreteSpec` as the canonical frozen contract.
- Lock required fields:
  - `source_bias_node_spec`
  - `ticker_scope`
  - `edges`
  - `n_bins`
  - `long_bins`
  - `short_bins`
  - `direction`
  - `spec_version`
- Lock validation rules:
  - `edges` must be absolute and strictly monotone
  - `n_bins == len(edges) + 1`
  - selected bins must be in range
  - `direction` must match selected bins
  - `ticker_scope` must be explicit
- Define naming/versioning policy:
  - `spec_version` is mandatory
  - changing cutpoints, scope, selected bins, or source recipe creates a new version
- Define control-file and vault schema changes:
  - canonical feature `model_type` is `domain_discrete`
  - feature behavior is stored in `bias_node_spec`
  - remove base-model `constructor_params`, `requires_fit`, `is_fitted`, and `fitted_params`
  - remove control-file `fitted_base_models`
  - keep `metadata.is_fit` only for ensemble-level fitted state
- Define the migration-error policy for legacy fitted feature payloads.

## Out Of Scope

- Implementing the node runtime.
- Reworking research plots or evaluators.

## Acceptance Criteria

- There is one written contract for frozen domain-discrete features.
- Persisted schema examples show no base-model fitted payloads.
- Legacy `continuous_binning` and `rule_based` payloads are explicitly rejected in the new contract.
- The contract is specific enough that implementation does not need to invent missing fields or edge-case behavior.

## Risks

- Schema churn will break old local artifacts immediately.

## Dependencies

- None. This ticket defines the branch contract.
