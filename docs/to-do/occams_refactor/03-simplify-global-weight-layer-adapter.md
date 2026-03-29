# Ticket 03: Simplify Global Weight-Layer Adapter (`__GLOBAL__`)

## Why

Global timeframe combination is currently adapted through synthetic ticker encoding (`__GLOBAL__`) and decode steps.
This works but introduces a second conceptual API around `WeightLayer`.

## Goal

Choose one explicit architecture:

1. keep `WeightLayer` reuse but isolate adapter mechanics in one small module, or
2. introduce a dedicated global combiner component with a clear interface.

## Scope

- Document the selected approach with one canonical data contract.
- Remove spread-out encode/decode helpers from portfolio orchestration.
- Keep single-timeframe fallback behavior unchanged.

## Out Of Scope

- Redesigning weighting math itself.
- Changing output schemas consumed by downstream execution.

## Acceptance Criteria

- `GlobalPortfolio` fit/predict flow reads linearly without hidden stream-id hacks.
- Global combination has one implementation boundary and one test surface.
- Existing integration outputs are unchanged.

## Risks

- Hidden assumptions in current adapter naming and feature keys.

## Dependencies

- Strongly benefits from Ticket 02 module split.
