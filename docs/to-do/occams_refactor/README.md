# Occam Refactor Ticket Backlog

## Purpose

This backlog tracks large simplifications that reduce conceptual surface area across the repository.

Guiding principle: if two code paths explain the same behavior, keep one and make it explicit.

## Tickets

- [01-unify-correlation-multiplier.md](./01-unify-correlation-multiplier.md)
- [02-split-portfolio-module.md](./02-split-portfolio-module.md)
- [03-simplify-global-weight-layer-adapter.md](./03-simplify-global-weight-layer-adapter.md)
- [04-unify-portfolio-orchestrators.md](./04-unify-portfolio-orchestrators.md)
- [05-node-import-shim-reduction.md](./05-node-import-shim-reduction.md)
- [06-package-entrypoints-path-cleanup.md](./06-package-entrypoints-path-cleanup.md)
- [07-cache-singleton-and-shim-cleanup.md](./07-cache-singleton-and-shim-cleanup.md)
- [08-split-large-orchestrators.md](./08-split-large-orchestrators.md)

## Suggested Execution Order

1. `01-unify-correlation-multiplier.md`
2. `06-package-entrypoints-path-cleanup.md`
3. `07-cache-singleton-and-shim-cleanup.md`
4. `02-split-portfolio-module.md`
5. `03-simplify-global-weight-layer-adapter.md`
6. `04-unify-portfolio-orchestrators.md`
7. `05-node-import-shim-reduction.md`
8. `08-split-large-orchestrators.md`

## Definition Of Done (Backlog Level)

- Each completed ticket leaves fewer public entrypoints than before.
- Any compatibility layer has an explicit deprecation plan and owner.
- Unit and integration tests reflect the new primary path.
- Docs in `docs/library/` and `docs/api/` point to one canonical workflow per concern.
