# Ticket 04: Unify `PortfolioManager` And `GlobalPortfolio` Orchestration

## Why

The codebase has two top-level orchestration paths for multi-timeframe portfolio flow.
Parallel abstractions increase maintenance and make docs/tests ambiguous.

## Goal

Converge to one canonical orchestrator with explicit operation modes (for example, per-timeframe only vs cross-timeframe global weighting).

## Scope

- Compare responsibilities and actual call sites.
- Keep one public orchestration entrypoint.
- Mark legacy path as deprecated (or remove if unused in production paths).

## Out Of Scope

- Breaking existing consumers without migration notes.
- Formula or sizing changes.

## Acceptance Criteria

- One recommended orchestrator in docs and examples.
- Tests target one primary path, with legacy coverage only where needed for migration.
- Duplicate orchestration logic is removed.

## Risks

- Hidden use in local scripts or external tooling.

## Dependencies

- Can run after Tickets 02 and 03.
