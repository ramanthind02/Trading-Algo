# Sequencing And Open Questions

## Recommended Phases

## Phase 0: decisions before coding

Lock the following decisions first:

- confirm EWSD cache as the canonical volatility source and remove `daily_volatility_df` from downstream APIs
- define the artifact descriptor model so new artifact families can be added without changing the cache contract
- define dependency tracking and refresh policy so OHLCV updates automatically invalidate dependent artifacts safely
- define artifact retention/scope policy so vault-selected bias nodes are the live source of truth and research caches remain disposable
- will ensemble/base-model vectors be materialized or derived on read
- does `CrossTickerDataStore` disappear or become an internal adapter
- will `GlobalPortfolio` support dual APIs during migration
- what is the live failure policy for stale forecasts vs hard failure
- which existing modules become the shared primitives for the new infra so we do not duplicate extraction or cache-orchestration code

## Phase 1: central cache contract

Deliverables:

- typed key/range models
- central cache interface
- typed miss/coverage errors
- coverage metadata schema
- invalidation/versioning policy
- extensible artifact namespace/descriptor model
- artifact lifecycle state model and dependency metadata
- artifact scope/retention model for `research` vs `vault/live`

Dependency:

- no higher-layer API work should land before this phase is stable

## Phase 2: storage adapters

Deliverables:

- candle storage implementation
- artifact storage implementation
- adapters around `BiasNodeCache` and `CrossTickerDataStore`
- `CacheManager` evolution into central population orchestration
- shared reusable cache-population primitives extracted instead of copied
- automatic stale marking and managed refresh plumbing for dependent artifacts
- research-cache prune and promotion plumbing

Dependency:

- needed before any caller can migrate without hidden candle backdoors

## Phase 3: node and feature cutover

Deliverables:

- `BiasNode` storage concerns removed or isolated
- feature extraction rewritten around central cache reads
- cross-ticker and as-of lookup semantics unified
- reuse-first refactor of `feature_extraction/feature_extractor.py` rather than a second extraction implementation

Dependency:

- required before ensemble layers can be truly cache-native

## Phase 4: ensemble and portfolio cutover

Deliverables:

- cache-native `DiversifiedEnsemble` query path
- cache-native `TFPortfolio` query path
- `GlobalPortfolio.fit/predict` datetime or range-driven contracts
- temporary adapters for remaining candle callers

Dependency:

- must follow phases 1 to 3

## Phase 5: deployment and training migration

Deliverables:

- `ForecastServer` ingest/query flow on the central cache
- production training on cache-backed artifacts
- live policy on operational misses implemented

## Phase 6: cleanup

Deliverables:

- delete deprecated candle-frame paths
- delete permissive lazy-load fallbacks
- finalize permanent design docs, usage docs, and API docs in `docs/library` and `docs/api`
- remove temporary compatibility shims

## Open Questions

### EWSD volatility lineage

- Should EWSD remain a dedicated artifact namespace or be represented as a standard bias-derived artifact under the same contract?
- What exact keying and coverage metadata should downstream consumers rely on?

### Materialization policy

- Are forecast vectors stored, or only derived from bias artifacts on demand?
- If stored, which layer owns invalidation when source artifacts change?
- Should portfolio predictions be a first-class persisted artifact type in the same contract?
- If yes, what identity and retention model should they use?

### Research vs vault retention

- How exactly do artifacts become vault-backed source-of-truth artifacts: explicit promotion, vault reference detection, or both?
- Should research artifacts live in a separate directory, namespace, or database partition?
- What cleanup command shape should researchers use to prune disposable artifacts safely?
- Which cache paths should be gitignored because they are local research state only?

### Fit contract

- Should `fit` take a date range, an explicit datetime grid, or a query object that can express both?
- Where do training returns live: persisted artifact, derived view, or external argument?

### Live runtime

- Can live carry stale forecasts after an ingest failure?
- If yes, is that a query-layer exception policy or an orchestration-layer retry policy?
- On artifact refresh after OHLCV update, do queries wait, fail, or use a documented stale-read mode?

### Compatibility window

- How long should dual APIs exist?
- Which callers get adapter shims first: research, deployment, or tests?

### DRY and reuse

- Which logic should be extracted from `feature_extraction/feature_extractor.py` into shared cache/query helpers?
- Is any orchestration code in `feature_extraction/backtest.py` worth lifting into shared query-grid or timeframe-processing utilities?
- Which behaviors from `tests/unit-tests/utils/test_cache_manager.py` are still authoritative under the new infra and should remain pinned?

## Rollback Strategy

During migration:

- keep new and old codepaths behind explicit adapters rather than hidden fallback
- do not delete current candle-frame APIs until equivalent integration coverage exists
- keep test fixtures for both old and new contracts only while the compatibility window is active

## Final Exit Criteria

The migration can be considered complete when:

- old candle-frame portfolio entrypoints are removed
- central cache coverage and miss semantics are enforced consistently
- live, training, and research all read through the same cache contract
- new artifact families can be added without redesigning the base cache interface
- duplicated extraction or cache-population stacks have been collapsed into shared primitives
- permanent design, usage, and API documentation exists for the refactor in `docs/library` and `docs/api`
- OHLCV updates automatically invalidate dependent artifacts and refresh behavior is explicit and documented
- vault-selected bias-node artifacts are the live source of truth and research artifacts are safely disposable
- docs present the new architecture as the default and only primary path
