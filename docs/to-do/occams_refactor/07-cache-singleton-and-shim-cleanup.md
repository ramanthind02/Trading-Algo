# Ticket 07: Cache Singleton And Shim Cleanup

## Why

`utils/cache` exposes multiple overlapping paths and uses singleton state mutation (`CentralCacheStore._instance`) in orchestration paths.
This increases hidden coupling and test complexity.

## Goal

Simplify cache architecture to one clear public entrypoint and explicit store dependency injection.

## Scope

- Reduce or remove thin re-export shims where unnecessary.
- Replace direct singleton mutation with injected store/factory boundaries.
- Normalize one path-resolution strategy across cache/vault integrations.

## Out Of Scope

- Replacing parquet layout or cache schema in this ticket.
- Functional redesign of cache coverage policy.

## Acceptance Criteria

- Cache consumers use one canonical API surface.
- Tests no longer rely on patching private singleton internals.
- Duplicate path-resolution logic is removed or centralized.

## Risks

- Existing tests may rely on singleton semantics.

## Dependencies

- Ticket 06 helps by stabilizing import boundaries first.
