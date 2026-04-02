# Ticket 06: Cut Validation, Walkforward, And Deployment Over To Signed Signals

## Why

Even after the feature and persistence layers are simplified, the research and runtime stack will remain inconsistent if permutation, walkforward, and deployment still instantiate or fit removed feature-model classes.

## Goal

Update every downstream consumer to treat the feature layer as deterministic signed-signal generation with no fit step.

## Scope

- Update validation and permutation paths to consume raw signed signals.
- Update walkforward evaluators and portfolio evaluators to use the frozen node contract.
- Update research save/load helpers to materialize domain-discrete specs rather than fitted feature models.
- Update deployment training/runtime flows so they never instantiate or fit a binning model.
- Ensure production and research share the same feature contract and signal semantics.

## Out Of Scope

- Reworking weight-layer or portfolio formulas.
- Supporting old feature-model paths in parallel.

## Acceptance Criteria

- Validation and walkforward paths run with no feature-layer fit step.
- Deployment/runtime paths can load and predict from node-backed domain-discrete features only.
- No production call site imports or instantiates removed binning-model classes.
- The downstream storyline matches the docs: frozen spec in research, same spec in validation/test/live.

## Risks

- Some evaluation helpers currently assume fitted multipliers or bin-count metadata and will need explicit simplification.

## Dependencies

- Tickets 01, 04, and 05.
