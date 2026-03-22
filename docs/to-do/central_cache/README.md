# Central Cache Refactor Plan

## Purpose

This folder breaks the `docs/library/bias_nodes/central_cache_architecture.md` refactor into concrete workstreams. The target is a single source of truth for candles and cacheable pipeline artifacts, datetime-driven portfolio queries, explicit cache coverage rules, and fail-fast semantics on cache miss.

This is not just a cache rewrite. It is a contract rewrite across:

- storage ownership
- artifact extensibility
- public API simplification
- node lookup semantics
- feature extraction
- volatility sourcing
- ensemble and portfolio APIs
- live and training orchestration
- tests and docs

## Plan Files

- [00-overview.md](./00-overview.md): current-vs-target gap, scope, principles, and success criteria
- [01-storage-and-contracts.md](./01-storage-and-contracts.md): storage ownership, coverage rules, cache miss contracts, invalidation/versioning
- [02-node-and-feature-access.md](./02-node-and-feature-access.md): `BiasNode`, cross-ticker access, feature extraction, walkforward/research readers
- [03-portfolio-api-cutover.md](./03-portfolio-api-cutover.md): `DiversifiedEnsemble`, `TFPortfolio`, `GlobalPortfolio`, wrappers, and public API migration
- [04-deployment-tests-and-docs.md](./04-deployment-tests-and-docs.md): live/training orchestration, verification, and documentation cutover
- [05-sequencing-and-open-questions.md](./05-sequencing-and-open-questions.md): phased implementation order, decision gates, rollback, and exit criteria

## Recommended Implementation Order

1. Define central cache contracts and typed errors.
2. Build central candle/bias store and coverage metadata.
3. Move node and feature extraction reads onto the new store.
4. Add cache-native ensemble and portfolio query paths.
5. Migrate deployment and training orchestration.
6. Write permanent design, usage, and API docs under `docs/library` and `docs/api`.
7. Remove candle-frame backdoors after test and doc cutover.

## Critical Constraint

Do not start by changing `GlobalPortfolio.fit` / `predict` signatures alone. That would only wrap the existing candle-frame internals and preserve the wrong coupling. The migration has to move from storage contracts upward.

Many internal modules are acceptable. Many public cache APIs are not. The finished system should expose one simple cache facade for callers, even if internal helpers remain modular.
