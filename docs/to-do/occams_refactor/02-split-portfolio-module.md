# Ticket 02: Split `ensemble/portfolio.py` By Responsibility

## Why

`ensemble/portfolio.py` currently mixes orchestration, math helpers, cache adapters, and global-layer encoding logic in one large module.
This slows onboarding and raises regression risk for unrelated changes.

## Goal

Split into smaller modules with explicit boundaries:

- domain math and transforms
- `TFPortfolio` orchestration
- `GlobalPortfolio` orchestration
- cache query/adapters
- serialization or snapshot helpers

## Scope

- Preserve public API during first pass with compatibility imports.
- Move private helpers to the nearest bounded module.
- Keep tests green while reducing file size and coupling.

## Out Of Scope

- Behavior changes in fit/predict outputs.
- Immediate removal of all compatibility imports.

## Acceptance Criteria

- No single portfolio module exceeds agreed complexity threshold.
- Public imports for `TFPortfolio` and `GlobalPortfolio` remain stable.
- Internal helpers are discoverable by concern instead of by historical location.

## Risks

- Circular imports when splitting without dependency direction rules.

## Dependencies

- Recommended after Ticket 01.
