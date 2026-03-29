# Ticket 02: Implement Generic Domain-Discrete Node Framework

## Why

The target architecture requires deterministic node execution, but one-off generated Python classes for every frozen feature would create a second complexity problem.

## Goal

Implement one generic node family that composes an existing source node, applies absolute cutpoints, and emits final signed signal `-1/0/+1`.

## Scope

- Add a dedicated `nodes/domain_discrete/` package.
- Implement a generic wrapper node that:
  - instantiates the source node from `source_bias_node_spec`
  - validates `ticker_scope` at construction time
  - assigns bins from absolute edges
  - maps bins to long, short, or neutral output
  - emits only `-1`, `0`, or `+1`
- Keep the spec inside `bias_node_spec.params` rather than generating new Python modules per feature.
- Standardize node metadata:
  - `module_name`
  - `output_features`
  - `params`
  - `front_bad`
  - standardized column naming including `spec_version`
- Define failure behavior:
  - invalid ticker scope fails fast
  - invalid spec fails fast
  - warmup emits neutral output

## Out Of Scope

- Feature research automation.
- Ensemble or vault loader changes.

## Acceptance Criteria

- A domain-discrete node can wrap an existing bounded indicator node and emit signed output deterministically.
- Column names remain stable for the same spec and change when `spec_version` changes.
- No per-feature code generation is required to add a new frozen spec.
- The node contract is compatible with the existing bias-node cache and naming conventions.

## Risks

- Wrapped-node lookback and warmup behavior can be mis-sized if not propagated correctly.

## Dependencies

- Ticket 01.
