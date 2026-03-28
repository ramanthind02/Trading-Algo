# Ticket 01: Unify Correlation Multiplier Logic (FDM/IDM)

## Why

Correlation-to-multiplier math is duplicated across `ensemble/weight_layer.py` and `ensemble/portfolio.py`.
This increases drift risk and makes formula changes expensive.

## Goal

Create one canonical pure function for:

- correlation matrix normalization
- mean off-diagonal correlation extraction
- multiplier transform (`sqrt(1 / (mean_corr + epsilon))`)
- configurable cap and floor policy

## Scope

- Refactor FDM path in `WeightLayer` to use shared helper.
- Refactor IDM path in `TFPortfolio` and `GlobalPortfolio` to use shared helper.
- Keep behavior identical for current defaults.

## Out Of Scope

- Changing portfolio formulas or business semantics.
- Altering control-file schema.

## Acceptance Criteria

- All current FDM/IDM tests pass with no expected-value drift.
- There is one production implementation of correlation-to-multiplier math.
- Call sites only provide parameters (cap, epsilon, handling mode), not custom math.

## Risks

- Minor numeric changes from matrix pre-processing differences.

## Dependencies

- None. Safe as a first refactor.
