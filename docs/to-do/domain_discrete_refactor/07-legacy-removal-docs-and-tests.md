# Ticket 07: Remove Legacy Paths And Rewrite Docs/Tests

## Why

The cutover is not complete until the old mental model is removed from code, docs, and tests.
Leaving the old path half-documented would preserve the same ambiguity the refactor is trying to eliminate.

## Goal

Delete legacy fitted feature-model paths, update docs to one canonical workflow, and add tests that lock the new contract.

## Scope

- Delete legacy continuous/rule-based feature-model code and stale references.
- Rewrite library and API docs to describe one frozen signed-signal path.
- Add or update tests for:
  - `DomainDiscreteSpec` validation
  - node bin mapping and signed output
  - ticker-scope enforcement
  - schema/loading behavior
  - migration errors for legacy artifacts
  - one repo-backed end-to-end integration path using the frozen node contract
- Add one acceptance sweep that verifies no production references remain to removed feature-model symbols outside migration-error tests.

## Out Of Scope

- Preserving behavior of legacy fitted feature artifacts.
- Maintaining dual docs for old and new workflows.

## Acceptance Criteria

- Docs point to one canonical feature workflow.
- Tests cover the new contract at unit and integration level.
- Legacy symbols are absent from production code.
- The branch contains an explicit grep-based acceptance check for removed terms.

## Risks

- Test churn will be large because many current assertions encode the old schema directly.

## Dependencies

- Tickets 01 through 06.
