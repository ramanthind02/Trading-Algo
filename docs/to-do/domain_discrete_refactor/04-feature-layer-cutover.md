# Ticket 04: Collapse The Feature Layer To Node-Backed Signed Signals

## Why

`BaseModel` and the binning-model abstraction currently exist to support fitted geometry, fit/predict branching, and vault state updates.
Once the feature contract is a frozen signed-signal node, that machinery becomes unnecessary.

## Goal

Reduce the feature layer to a thin node-backed adapter and remove the fitted binning abstraction entirely.

## Scope

- Remove:
  - `BinningModelBase`
  - `ContinuousBinningModel`
  - `RuleBasedModel`
- Collapse `feature_selection/base_models/feature_base_model.py` into a thin compatibility shell that:
  - owns node specs
  - instantiates nodes
  - extracts signed signals
  - predicts deterministically
- Remove feature-layer responsibilities tied to fitting:
  - fit-time geometry learning
  - fit/predict binning branches
  - fitted-state update APIs
  - base-model fitted payload generation
- Replace feature-type branching with one signed discrete signal contract.

## Out Of Scope

- Vault/control-file persistence mechanics.
- Walkforward/deployment-specific call-site rewrites.

## Acceptance Criteria

- The feature layer no longer owns a binning model.
- There is no feature-layer API that learns, restores, or persists bin geometry.
- The remaining compatibility shell is clearly a node runner, not a hidden fitted model.
- Production code does not import removed binning-model classes.

## Risks

- External callers may still expect `.fit()` or binning-model-specific attributes.

## Dependencies

- Tickets 01 through 03.
