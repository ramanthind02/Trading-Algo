# QuantFoundry SaaS Technical Design

## 1. Purpose

This document captures the current technical plan for turning the existing `Trading-Algo` research system into the QuantFoundry SaaS platform. It complements:

- `docs/SaaS/data_flow.md` - research, portfolio, deployment, and zone lifecycle.
- `docs/SaaS/strategy_spec.md` - user strategy contract and runtime expectations.
- `docs/SaaS/ui_ux.md` - product navigation and MVP user surfaces.

The guiding product goal is to let independent traders build, validate, combine, and deploy systematic strategies without needing to build the surrounding infrastructure themselves.

## 2. Repository Strategy

The existing repository should remain operational as the internal research workbench while stable product-grade pieces are extracted into QuantFoundry packages and services.

| Repository | Role |
|---|---|
| `Trading-Algo` | Internal research/backtesting/training lab. Keeps local data, cache, vault, experimental workflows, and current quant research code operational. |
| `QuantFoundry-Core` | Shared Python library for strategy contracts, validation, candle/runtime models, cache keys, artifact schemas, and reusable engine components. |
| `QuantFoundry-API` | SaaS backend: auth integration, user/project/strategy metadata, job submission, ACA Job orchestration, status/results APIs, deployment APIs. |
| `QuantFoundry-Worker` | Private batch worker image for executing strategy validation/backtests/signal generation jobs. It imports `QuantFoundry-Core`. |
| `QuantFoundry-Web` | Public web application for dashboard, research workspace, strategy library, portfolio builder, and deployment UI. |

Dependency direction must stay one-way:

```text
Trading-Algo / QuantFoundry-Research  ─┐
QuantFoundry-API                       ├── imports quantfoundry_core
QuantFoundry-Worker                    ┘

QuantFoundry-Web talks to QuantFoundry-API over HTTP.
QuantFoundry-Core never imports from API, Worker, Web, or Trading-Algo.
```

Migration should be incremental, not a big-bang rewrite:

1. Keep `Trading-Algo` operational.
2. Move stable contracts and portable logic into `QuantFoundry-Core`.
3. Update `Trading-Algo` to import those contracts where useful.
4. Build the cloud API and worker around the same Core contract.
5. Add parity tests proving local and cloud runners produce equivalent outputs for the same strategy, data, and parameters.

## 3. MVP Architecture

```text
User Browser
  -> QuantFoundry-Web on Vercel
  -> QuantFoundry-API on Azure Container Apps
  -> Azure Database for PostgreSQL for product metadata
  -> Azure Blob Storage for source packages, candle Parquet, and result artifacts
  -> QuantFoundry-Worker on Azure Container Apps Jobs, started directly by the API for MVP
```

### 3.1 Hosting Decisions

| Surface | MVP choice | Reason |
|---|---|---|
| Frontend | Vercel | Simplest frontend deploys, preview environments, rollbacks, custom domains, GitHub integration. |
| API | Azure Container Apps | Container-native FastAPI deployment with scale controls and managed ingress. |
| Workers | Azure Container Apps Jobs | Finite direct-start jobs, scale-to-zero economics, container-based Python stack. |
| Queue | Deferred | Azure Service Bus can be added before private beta if direct ACA Job startup needs better backpressure/retries. |
| Metadata DB | Azure Database for PostgreSQL Flexible Server | Relational ownership/versioning/audit model with JSONB escape hatch. |
| Artifacts | Azure Blob Storage | Cheap durable storage for Parquet, JSON, source bundles, and backtest outputs. |
| Secrets | Azure Key Vault | OAuth secrets, API keys, storage credentials if not fully using managed identity. |
| Observability | Azure Monitor/Application Insights | API/worker logs, job failures, latency, cost and health telemetry. |
| Infrastructure as Code | Azure Bicep | Native Azure IaC without AKS/Helm complexity. |
| CI/CD | GitHub Actions | Build, test, provision, and deploy from repo workflows using OIDC federation. |

Avoid for MVP unless required:

- Azure Front Door.
- AKS/Kubernetes.
- Per-strategy container images.
- Hosted Jupyter notebooks.
- Broker order routing.
- User-supplied `requirements.txt`.
- Private endpoints/NAT/Azure Firewall until security and revenue justify the cost.
- Helm charts unless the platform moves to Kubernetes later.

### 3.2 CI/CD and Infrastructure as Code

The MVP should avoid manual Azure portal setup as much as possible. Azure resources should be created and updated through Bicep modules executed by GitHub Actions.

Recommended infrastructure repository layout for `QuantFoundry-API`:

```text
infra/
  bicep/
    main.bicep
    modules/
      container_apps.bicep
      storage.bicep
      postgres.bicep
      key_vault.bicep
      monitoring.bicep
  scripts/
    bootstrap_azure.ps1
    deploy_infra.ps1
```

Infrastructure should live in the repos that own the deployable surface:

| Repository | Infrastructure/config ownership |
|---|---|
| `QuantFoundry-API` | Azure Bicep, API/worker container deployment, Postgres migrations, worker job definitions, storage, Key Vault, monitoring. |
| `QuantFoundry-Web` | Vercel project config, frontend environment variable docs, preview/prod deploy settings. |
| `QuantFoundry-Core` | No cloud infrastructure; Python package build/test/release only. |
| `Trading-Algo` | Internal/local research scripts and data publishing scripts until moved into a dedicated data/ops package. |

A separate `QuantFoundry-Infra` repo is not needed for MVP. It can be introduced later only if infrastructure ownership spans many services and the API repo becomes cluttered.

Recommended GitHub Actions workflows:

```text
.github/workflows/
  ci-core.yml
  ci-api.yml
  ci-worker.yml
  ci-web.yml
  deploy-dev.yml
  deploy-prod.yml
  data-publish-dev.yml
  data-publish-prod.yml
```

Pipeline responsibilities:

| Pipeline | Responsibility |
|---|---|
| `ci-core.yml` | Test and package `QuantFoundry-Core`. |
| `ci-api.yml` | Test API, build API container image, push to registry. |
| `ci-worker.yml` | Test worker, build shared worker image, push to registry. |
| `ci-web.yml` | Typecheck/build frontend, create Vercel preview. |
| `deploy-dev.yml` | Deploy Bicep to dev, run Alembic migrations, deploy API/job image revisions. |
| `deploy-prod.yml` | Same as dev but requires manual GitHub Environment approval. |
| `data-publish-dev.yml` | Upload validated candle dataset to dev Blob container. |
| `data-publish-prod.yml` | Promote an already validated dataset version to prod after approval. |

Authentication should use GitHub Actions OIDC federation into Azure, not long-lived Azure credentials stored as GitHub secrets. Vercel can be connected directly to the `QuantFoundry-Web` repository for preview/prod deploys, or driven through GitHub Actions if tighter release coordination is needed.

Initial one-time bootstrap may still require a small manual step:

1. Create Azure subscription/resource group naming convention.
2. Create a deployment service principal or user-assigned managed identity with federated GitHub credentials.
3. Store non-Azure deployment references in GitHub environment variables.
4. Run `infra/scripts/bootstrap_azure.ps1`.

After bootstrap, normal resource changes should flow through Bicep and CI/CD.

## 4. Strategy Execution Model

The MVP should support custom Python strategies through a constrained, stateless function contract rather than unconstrained Python applications.

```python
def compute(
    candles: dict[str, CandleData],
    params: StrategyParams,
) -> dict[str, float]:
    ...
```

Execution properties:

- The platform owns the event loop.
- The strategy receives all available candles up to the current bar.
- The strategy does not hold server-side state between calls.
- Output is a per-ticker forecast score clipped to `[-2.0, 2.0]`.
- Multi-timeframe access, custom dependencies, and user-defined indicator dependency graphs are deferred.

### 4.1 Shared Worker Image, Per-Job Runtime Isolation

The MVP should use one shared worker image, not one image per strategy.

```text
Shared image:
  quantfoundry-worker:<core_version>

Job A:
  starts container from shared image
  loads User A strategy source from Blob
  runs User A backtest
  writes User A artifacts
  exits

Job B:
  starts container from shared image
  loads User B strategy source from Blob
  runs User B backtest
  writes User B artifacts
  exits
```

This keeps runtime isolation without the operational cost of building, scanning, pushing, and managing a container image for every strategy version.

Per-strategy images can be revisited later for enterprise users or custom dependency support.

### 4.2 Security Boundary

Static validation is a fast-fail quality gate, not the primary security boundary. The primary boundary is isolated worker execution with resource limits.

MVP defense-in-depth:

| Layer | Requirement |
|---|---|
| Contract | Pure `compute()` function, no platform state mutation. |
| Static validation | AST checks for banned imports/calls and required signature. |
| Dependency policy | Only platform-approved libraries in the shared worker image. No custom pip installs. |
| Runtime isolation | User code never runs in API or orchestrator process. It runs in worker job containers. |
| Resource limits | CPU, memory, wall-clock timeout, max bars, max tickers, max lookback, max concurrent jobs. |
| Data scoping | Job receives only the source, candle data, and artifact paths it needs. |
| Output validation | Missing, NaN, undeclared, and out-of-range outputs are corrected with surfaced warnings. |

Initial banned capabilities:

- `open`, `eval`, `exec`, `compile`, `__import__`.
- `os`, `sys`, `subprocess`, `socket`, `requests`, `urllib`, `ctypes`, `importlib`.
- Dunder/reflection access patterns where practical.
- User-provided external dependencies.

## 5. Local Research Story

Local research remains first-class to control cost and preserve the existing quant workflow.

```text
Write strategy locally
  -> validate against QuantFoundry-Core
  -> run local backtest against Trading-Algo daily data/cache
  -> generate same artifact schema as cloud workers
  -> promote/upload only when ready
```

Local and cloud execution should share:

- `CandleData`.
- `StrategyParams`.
- Strategy metadata.
- Output validation.
- Cache key generation.
- Artifact schemas.

They should differ only in execution backend:

| Concern | Local/internal | Cloud/user-facing |
|---|---|---|
| Strategy source | Local file | Immutable Blob object/version. |
| Data | `Trading-Algo\data\ohlc_data` or local Parquet | Curated Parquet in Blob Storage. |
| Execution | Local Python process or local worker container | Azure Container Apps Job. |
| Artifacts | Local filesystem/vault | Blob Storage + Postgres metadata. |
| Metadata | Optional manifest | Postgres product records. |

## 6. Core Product Models

The MVP should make these constructs first-class.

### 6.1 User and Identity

QuantFoundry should not store passwords. Auth should be OAuth/OIDC only.

```text
User:
  id
  oauth_provider
  provider_subject
  email
  display_name
  created_at
  last_login_at
```

Google OAuth should be implemented through Clerk for MVP. The API should verify Clerk-issued JWTs and map the Clerk subject to internal users.

### 6.2 Research Project

```text
ResearchProject:
  id
  owner_user_id
  name
  description
  default_timeframe
  created_at
  updated_at
```

### 6.3 Zone

The backend should support arbitrary non-overlapping zones. The UI can initially present a guided Train / Validation / Test layout.

```text
Zone:
  id
  project_id
  name
  zone_type        # Train | OutOfSample
  start_date
  end_date
  created_at
```

Rules:

- Date range must be valid and non-empty.
- Zones in the same project must not overlap.
- No enforced count or ordering in the backend.

### 6.4 Strategy and Strategy Version

```text
Strategy:
  id
  project_id
  owner_user_id
  name
  description
  created_at
  updated_at

StrategyVersion:
  id
  strategy_id
  version_number
  semver
  source_code_hash
  source_blob_uri
  metadata_json
  params_schema_json
  training_config_json
  zone_snapshot_json
  core_version
  worker_image_version
  committed_at
```

Committed strategy versions are immutable.

### 6.5 Backtest Run

```text
BacktestRun:
  id
  owner_user_id
  project_id
  strategy_version_id
  selected_zone_ids
  ticker_set_hash
  parameter_hash
  date_range_start
  date_range_end
  status
  cache_policy
  request_json
  result_summary_json
  warnings_json
  artifact_root_uri
  created_at
  started_at
  completed_at
```

Recommended statuses:

- `queued`
- `running`
- `completed`
- `completed_with_warnings`
- `failed`
- `cancelled`

### 6.6 Portfolio and Portfolio Version

```text
Portfolio:
  id
  owner_user_id
  project_id
  name
  created_at
  updated_at

PortfolioVersion:
  id
  portfolio_id
  version_number
  strategy_weights_json
  zone_snapshot_json
  result_summary_json
  artifact_root_uri
  created_at
```

Portfolio versions should be immutable even if rollback UI is deferred.

### 6.7 Deployment

```text
Deployment:
  id
  owner_user_id
  portfolio_version_id
  status             # running | stopped
  output_mode        # forecast_score | position_fraction | contracts
  created_at
  started_at
  stopped_at

SignalApiKey:
  id
  owner_user_id
  deployment_id       # nullable only if later supporting user-wide keys
  name
  key_prefix          # safe display prefix, e.g. qf_sig_live_abc123
  key_hash
  scopes              # e.g. read:signals
  last_used_at
  expires_at
  revoked_at
  created_at
```

MVP deployment means daily signal generation and retrieval, not broker order routing.

Signal API keys should be deployment-scoped by default, not one global key per user. MVP should allow one active key per deployment for simplicity. Rotation means creating a replacement key and automatically revoking the prior active key for that deployment. A later advanced option can allow multiple concurrent keys or user-level keys with explicit scopes.

### 6.8 Rate Limit and Quota Models

Rate limiting and quota state should be first-class product data so users cannot accidentally or intentionally create unbounded compute cost.

```text
UserPlan:
  id
  name
  monthly_backtest_limit
  monthly_compute_seconds_limit
  max_concurrent_jobs
  max_tickers_per_run
  max_bars_per_run
  max_lookback
  max_worker_vcpu
  max_worker_memory_gib
  max_wall_clock_seconds

UsageLedger:
  id
  user_id
  period_start
  period_end
  backtest_count
  worker_vcpu_seconds
  worker_gib_seconds
  signal_api_request_count

RateLimitEvent:
  id
  user_id
  event_type
  limit_name
  observed_value
  request_id
  created_at
```

MVP can start with a simple built-in plan table rather than a billing integration. Stripe/billing can map onto these plans later.

Default limits should live in a versioned config file, for example:

```text
QuantFoundry-API/config/plans.yml
```

Initial editable defaults:

| Plan | Monthly backtests | Concurrent jobs | Max tickers/run | Max bars/run | Max lookback | Worker limit | Wall clock |
|---|---:|---:|---:|---:|---:|---|---:|
| `internal` | 1000 | 5 | 50 | 250000 | 1000 | 2 vCPU / 4 GiB | 60 min |
| `beta` | 100 | 2 | 20 | 100000 | 500 | 1 vCPU / 2 GiB | 30 min |
| `free_preview` | 20 | 1 | 10 | 50000 | 250 | 1 vCPU / 2 GiB | 15 min |

These are starting guesses, not product pricing decisions. They should be easy to edit without schema changes.

### 6.9 Billing and Payment Models

The exact packages and limits require market research, but the technical design should assume subscription/billing will become part of request admission. Stripe is the likely default payment gateway because it is the most common SaaS path and has mature Checkout, Customer Portal, subscriptions, invoices, webhooks, and tax integrations.

```text
BillingCustomer:
  id
  user_id
  provider                 # stripe
  provider_customer_id
  billing_email
  created_at

Subscription:
  id
  user_id
  provider_subscription_id
  plan_id
  status                   # trialing | active | past_due | canceled | unpaid
  current_period_start
  current_period_end
  cancel_at_period_end
  created_at
  updated_at

BillingEvent:
  id
  provider
  provider_event_id
  event_type
  payload_json
  processed_at
```

Frontend billing flow:

```text
User chooses plan
  -> Web calls API to create Stripe Checkout Session
  -> User completes payment in Stripe-hosted checkout
  -> Stripe webhook updates Subscription/UserPlan state
  -> Web reads current plan/limits from API
```

Backend admission checks should not trust the frontend. Every compute-triggering API should check the user's current plan/subscription/usage state before validation or worker dispatch:

```text
request authenticated
  -> load user plan and subscription status
  -> check plan allows requested feature
  -> check monthly quota and concurrent job limits
  -> reserve usage or create queued job
  -> reconcile actual usage after worker completion
```

Until billing launches, the same code path can use internal/free/beta plans managed manually in Postgres.

## 7. Storage Design

### 7.1 PostgreSQL

Postgres stores relational product state:

- Users and OAuth identity mapping.
- Projects and zones.
- Strategy metadata and immutable versions.
- Backtest job state and summaries.
- Portfolio versions and deployments.
- Cache metadata and artifact URIs.
- API key hashes and audit records.

Use JSONB for snapshots and flexible metadata, but keep ownership, status, versioning, and foreign keys relational.

### 7.2 Blob Storage

Blob Storage stores large or immutable artifacts:

- User strategy source bundles.
- Curated candle Parquet.
- Forecast streams.
- Equity curves.
- Trade/position logs.
- Tearsheet/report JSON or HTML.
- Worker logs if not retained solely in observability tooling.

Suggested container layout:

```text
candles/
  manifests/dataset_version=<version>/manifest.json
  partitions/timeframe=D/ticker=ES/year=2026/part.parquet
  adjusted/timeframe=D/ticker=ES/adjustment_set=<id>/year=2026/part.parquet

strategy-source/
  user=<user_id>/strategy=<strategy_id>/version=<version_id>/source.py

backtest-results/
  user=<user_id>/run=<backtest_run_id>/forecast_stream.parquet
  user=<user_id>/run=<backtest_run_id>/summary.json
  user=<user_id>/run=<backtest_run_id>/warnings.json

portfolio-results/
  user=<user_id>/portfolio_version=<portfolio_version_id>/summary.json

deployment-signals/
  deployment=<deployment_id>/date=<yyyy-mm-dd>/signals.json
```

## 8. Cache Design

Cache entries should be metadata rows pointing to Blob artifacts.

Canonical cache key inputs:

```text
strategy_id
strategy_version_id
source_code_hash
ticker_set_hash
timeframe
parameter_hash
date_range_start
date_range_end
dataset_version
core_version
worker_image_version
```

Cache tiers:

| Cache | Lifecycle |
|---|---|
| Research cache | User opt-in, TTL-based, visible and manually invalidatable. |
| Portfolio cache | Persistent for selected/committed strategies, invalidated when strategy source/hash changes or strategy is deselected. |

The orchestrator should check cache before dispatching worker jobs. A cache hit reuses the forecast stream artifact and avoids compute.

## 9. Market Data Operations

MVP data should be daily futures data only. Cloud workers should read curated Parquet from Blob Storage, while internal research can continue using local data in `Trading-Algo`.

### 9.1 Dataset Versioning

Every published candle dataset should be versioned, but this should not mean duplicating the full historical dataset every week. Dataset versions should be lightweight manifests that point to immutable Parquet partitions and metadata.

```text
CandleDatasetVersion:
  id
  version_name              # e.g. 2026-W20 or 2026Q2-roll-adjusted
  source                    # TWS, Norgate, manual import, etc.
  timeframe                 # D for MVP
  tickers
  start_date
  end_date
  adjustment_policy
  validation_summary_json
  manifest_blob_uri
  status                    # candidate | dev_published | prod_published | retired
  created_at
  promoted_to_dev_at
  promoted_to_prod_at
```

Backtest runs, cache keys, strategy versions, portfolio versions, and deployments should record the dataset version used. This prevents historical results from silently changing when data is republished.

The manifest should define exactly which physical files belong to the logical dataset:

```json
{
  "dataset_version": "2026-W20",
  "active_from": "2026-05-13",
  "timeframe": "D",
  "tickers": {
    "ES": [
      {
        "start": "2010-01-01",
        "end": "2025-12-31",
        "uri": "candles/adjusted/timeframe=D/ticker=ES/adjustment_set=2026Q2/year=*/part.parquet",
        "content_hash": "..."
      },
      {
        "start": "2026-01-01",
        "end": "2026-05-08",
        "uri": "candles/partitions/timeframe=D/ticker=ES/year=2026/part.parquet",
        "content_hash": "..."
      }
    ]
  }
}
```

This gives reproducibility without weekly full-copy storage growth.

Recommended physical storage model:

| Data type | Storage behavior |
|---|---|
| Raw TWS pulls | Append-only, retained for audit/debug, moved to cool/archive tier after validation. |
| Clean canonical daily partitions | Partitioned by ticker/timeframe/year or month. Daily updates append into a small open partition and are compacted into larger files on a schedule. |
| Dataset version manifests | Small immutable JSON files. One per published dataset version. |
| Quarterly backadjusted partitions | Rewrite only affected ticker/date partitions; retain recent prior adjustment sets according to retention policy. |
| Active dataset pointer | Small DB/config value pointing to the current manifest. |

Retention policy should be explicit:

- Keep all dataset manifests because they are tiny.
- Keep raw weekly TWS pulls in hot/cool storage for a limited operational window, then archive or delete based on legal/data-license needs.
- Keep the current adjusted dataset and at least one previous adjusted dataset hot.
- Move older adjusted partitions to cool/archive or delete if no backtests/deployments reference them.
- For intraday future data, partition by ticker/timeframe/date or month and use lifecycle policies aggressively.

### 9.2 Daily Data Update Flow

Hosted deployments require fresh data whenever signals are expected. For daily strategies, the normal operating target is a daily TWS import after the relevant market close/data availability window. During an internal alpha, this can be manually triggered; before users depend on hosted deployments, it should become a scheduled or checklist-driven daily operation.

```text
TWS daily pull
  -> local raw data landing folder
  -> normalization/cleaning
  -> append/update current open daily futures partition
  -> validation report
  -> write/update only changed open Parquet partitions locally
  -> write candidate dataset manifest locally
  -> publish to dev Blob
  -> run smoke/parity checks in dev
  -> promote same dataset version to prod after approval
  -> update active dataset pointer
  -> trigger hosted signal scheduler
```

Suggested scripts:

```text
scripts/data/import_tws_daily.py
scripts/data/build_continuous_futures.py
scripts/data/validate_candle_dataset.py
scripts/data/publish_dataset.py --target dev --dataset-version <version>
scripts/data/promote_dataset.py --from dev --to prod --dataset-version <version>
scripts/data/compact_open_partitions.py --target dev --through <date>
```

The daily update should not create a separate one-day file forever. It should maintain an open partition for the current period, such as current month or current quarter, and write a new manifest pointing to that updated open partition. Compaction periodically rewrites many tiny daily appends into a small number of larger Parquet files.

If the current-year partition is mutable during data assembly, mutation should happen only in a candidate/dev area. Once promoted, the production partition referenced by a manifest should be treated as immutable.

For MVP daily data, use one of these partition policies:

| Policy | Recommendation |
|---|---|
| One file per ticker/year | Simple for daily data, but each daily append rewrites the current-year file. Fine at MVP scale. |
| One file per ticker/month | Better balance: smaller rewrites, no tiny daily-file explosion. Recommended default. |
| One file per ticker/day | Avoid for daily data unless using a table format with compaction; creates too many tiny files. |

For future intraday data, use month/day partitions plus scheduled compaction or adopt a table format such as Delta Lake/Iceberg if append/update/query complexity justifies it.

### 9.3 Quarterly Backadjustment Flow

Quarterly futures backadjustment should also produce a new dataset manifest, but only physically rewrite partitions affected by rollover/backadjustment changes.

```text
Quarterly rollover review
  -> run backadjustment script locally
  -> compare prior active dataset vs new adjusted dataset
  -> produce adjustment report by ticker/contract/roll date
  -> write changed adjusted partitions under a new adjustment_set
  -> write new manifest pointing to reused unchanged partitions plus changed adjusted partitions
  -> publish candidate to dev
  -> rerun smoke backtests and data quality checks
  -> promote to prod
  -> update active dataset pointer for new runs
```

Existing historical backtests should remain tied to their original dataset version. New runs should use the current active version unless the user explicitly selects an older dataset version.

If storage pressure grows, the system can retain manifests while pruning old physical partitions that no active deployment, committed portfolio version, paid retention policy, or recent backtest references. A pruned dataset version remains visible for audit but is marked non-rerunnable unless restored from archive.

Daily manifests do not require retaining duplicate full datasets. Most daily manifests should point to the same historical partitions plus the latest current-month/current-quarter open partition. At quarter-end backadjustment, the system writes a consolidated adjustment set, updates the active manifest, and can retire intermediate daily open partitions after their retention window if no referenced runs require them.

### 9.4 Data Validation Gates

Before publishing to dev or prod, validation should check:

- Required OHLCV columns and timestamp dtype.
- Monotonic dates per ticker.
- No duplicate dates per ticker.
- Expected trading calendar coverage.
- Missing bar counts and gap report.
- Non-negative volume.
- High/low consistency.
- Suspicious return outliers.
- Roll dates and adjustment factors for continuous futures.
- Parquet schema compatibility.

Validation artifacts should be stored in Blob next to the dataset and summarized in Postgres.

## 10. API Shape

API paths are illustrative and can change during implementation. During MVP, app-internal APIs used only by `QuantFoundry-Web` do not need URL versioning. Versioning should be reserved for public/external APIs where third-party clients may depend on stable contracts.

Recommended convention:

| API surface | Path style | Reason |
|---|---|---|
| Web app API | `/api/...` | Frontend and backend ship together; avoiding `/v1` reduces churn while the product is evolving. |
| External Signal API | `/api/v1/signals/...` | Users may automate against it, so breaking changes need explicit versioning. |
| Internal worker/admin APIs | Not public or separately authenticated | Prefer queue/job contracts over public endpoints. |

Breaking app API changes should be handled by deploying compatible Web and API revisions together. If a public app API emerges later, add versioning then.

### 10.1 Health

```http
GET /api/health
```

### 10.2 Projects and Zones

```http
POST /api/projects
GET  /api/projects
GET  /api/projects/{project_id}

POST /api/projects/{project_id}/zones
GET  /api/projects/{project_id}/zones
PUT  /api/projects/{project_id}/zones/{zone_id}
DELETE /api/projects/{project_id}/zones/{zone_id}
```

### 10.3 Strategies

```http
POST /api/projects/{project_id}/strategies
GET  /api/projects/{project_id}/strategies
GET  /api/strategies/{strategy_id}

POST /api/strategies/{strategy_id}/versions/validate
POST /api/strategies/{strategy_id}/versions/commit
GET  /api/strategies/{strategy_id}/versions
GET  /api/strategy-versions/{strategy_version_id}
```

Validation response:

```json
{
  "is_valid": true,
  "issues": [],
  "warnings": []
}
```

### 10.4 Backtests

```http
POST /api/backtests
GET  /api/backtests/{backtest_run_id}
GET  /api/backtests/{backtest_run_id}/results
POST /api/backtests/{backtest_run_id}/cancel
```

Submission response:

```json
{
  "backtest_run_id": "uuid",
  "status": "queued"
}
```

Backtests should support both single-strategy and portfolio scopes.

```json
{
  "scope": "strategy",
  "strategy_version_id": "uuid",
  "parameters": {},
  "zone_ids": ["uuid"],
  "cache_policy": "use_cache"
}
```

```json
{
  "scope": "portfolio",
  "portfolio_version_id": "uuid",
  "zone_ids": ["uuid"],
  "cache_policy": "use_cache"
}
```

Single-strategy backtests are the main research loop. Portfolio backtests are required before deployment because users need to validate combined weights, correlations, and aggregate drawdown. Internally, a portfolio backtest can start simple as one worker job for the whole portfolio, then evolve to one job per strategy plus a combine step when scale requires it.

### 10.5 Portfolios

```http
POST /api/projects/{project_id}/portfolios
GET  /api/projects/{project_id}/portfolios
POST /api/portfolios/{portfolio_id}/versions
GET  /api/portfolio-versions/{portfolio_version_id}
```

### 10.6 Deployments and Signals

```http
POST /api/deployments
GET  /api/deployments
POST /api/deployments/{deployment_id}/stop
POST /api/deployments/{deployment_id}/signal-api-key
GET  /api/deployments/{deployment_id}/signal-api-keys
DELETE /api/deployments/{deployment_id}/signal-api-keys/{key_id}
GET  /api/v1/signals/latest
GET  /api/v1/signals?deployment_id=...&date=...
```

Signal API authentication should use hashed API keys scoped to a deployment or portfolio.

The Signal API is a pull-based machine interface for retrieving the latest hosted portfolio output. Users create an API key in the QuantFoundry portal, copy it once, store it in their own script/pipeline/bot, and call the Signal API to fetch the latest signal snapshot.

MVP key management:

| Action | API |
|---|---|
| Create or rotate key | `POST /api/deployments/{deployment_id}/signal-api-key`; creates a new key and revokes the prior active deployment key. |
| List keys | `GET /api/deployments/{deployment_id}/signal-api-keys` returns current and historical metadata only, never full secrets. |
| Revoke key | `DELETE /api/deployments/{deployment_id}/signal-api-keys/{key_id}` sets `revoked_at`. |

The full key secret is only shown once at creation. The database stores a hash and a safe prefix for display. MVP should enforce one active key per deployment. This is simpler than multi-key management and still supports emergency revocation/replacement.

Signal API does not place trades. It only returns the latest signal snapshot for a running deployment.

### 10.7 Rate Limiting and Abuse Controls

Rate limiting must happen before expensive work is queued.

Recommended layers:

| Layer | Control |
|---|---|
| Edge/frontend | Basic bot protection through Vercel and OAuth-required app access. |
| API request rate | Per-user/IP limits for validation, backtest submission, results polling, and Signal API requests. |
| Job admission | Check plan quota before starting ACA Jobs. |
| Queue concurrency | Enforce max active/running jobs per user and global worker concurrency. |
| Worker runtime | Enforce CPU, memory, wall-clock timeout, max tickers, max bars, and max lookback. |
| Monthly ledger | Track approximate vCPU/GiB seconds and backtest count by billing period. |

MVP implementation should avoid expensive rate-limit infrastructure if possible:

- Use Postgres-backed quota checks for job admission and monthly usage.
- Use in-process/API middleware for coarse short-window limits in dev.
- Add Redis/Valkey or Azure API Management later if API traffic requires distributed high-throughput rate limiting.
- Keep Signal API key limits separate from web app OAuth user limits.

Abuse response should be explicit:

```http
429 Too Many Requests       # short-window API limit
402 Payment Required        # paid plan quota exceeded, if billing is active
403 Forbidden               # plan does not allow requested resource size
409 Conflict                # concurrent job limit reached
```

## 11. Worker Flow

```text
1. API receives backtest request.
2. API validates ownership, strategy version, parameters, zones, and quotas.
3. API writes BacktestRun(status='queued').
4. API checks cache; if complete hit exists, it attaches artifact and marks run complete.
5. API starts an ACA Job execution with the run ID and job payload reference.
6. ACA Job starts worker container.
7. Worker loads job payload and strategy source.
8. Worker loads candle Parquet for tickers/timeframe/date range.
9. Worker converts data to CandleData windows and calls compute() per bar.
10. Worker validates outputs and records warnings.
11. Worker writes forecast stream and summary artifacts to Blob.
12. Worker updates BacktestRun status and artifact URIs.
13. Frontend polls API for status/results.
```

Workers should be idempotent. Retrying the same job should either overwrite a deterministic staging path safely or create a new attempt path and atomically mark the successful attempt in Postgres.

The initial implementation should use direct ACA Job startup:

| Style | Flow | Pros | Cons |
|---|---|---|---|
| API starts ACA Job directly | API validates, writes run row, calls Azure to start an ACA Job. | Fewer moving pieces for first prototype. | API is coupled to Azure job API; burst handling/retries/backpressure are weaker. |
| Queue-first controller | API validates, writes run row, sends Service Bus message; a small controller or event process starts ACA Jobs. | Better backpressure, retries, auditability, burst absorption, and future portability. | One extra component to deploy/observe. |

Recommendation: start with direct ACA Job startup for MVP simplicity, but preserve queue-friendly status fields (`queued`, `running`, `attempts`, idempotency keys). Add Service Bus/controller later only if bursts, retries, or API coupling become painful.

## 12. Deployment and Hosted Signal Automation

MVP deployment means hosted daily signal generation for a committed `PortfolioVersion`. It does not mean broker order routing.

### 12.1 Deployment Lifecycle

```text
User creates portfolio version
  -> user clicks Deploy
  -> API creates Deployment(status='running')
  -> scheduler includes deployment in daily signal run
  -> worker computes latest signals after data update
  -> signals are stored and exposed through UI/API/notifications
```

Users should not need to press a browser button every day. Once a deployment is running, scheduled backend automation should produce signals.

### 12.2 Scheduled Signal Runs

Use a scheduled ACA Job, Azure Container Apps scheduled job, or GitHub Actions/Azure workflow for the first MVP. The job should run after the expected daily data update window.

Recommended daily flow:

```text
Daily market data import/promote completes
  -> active dataset pointer updated or confirmed
  -> signal scheduler starts
  -> fetch all running deployments
  -> group by portfolio version/dataset/timeframe where possible
  -> run strategy/portfolio signal jobs
  -> write deployment-signals artifacts
  -> update latest_signal records
```

Hosted signal automation depends on daily data availability. If data is not published for a trading day, the scheduler should not silently produce stale signals. It should mark affected deployments as `waiting_for_data` or `data_unavailable` and expose that status in the dashboard and API.

### 12.3 Signal Delivery Options

The MVP should support pull-based delivery through the website and Signal API only. Push channels such as Telegram, email, SMS, and broker execution are deferred.

| Delivery | MVP recommendation |
|---|---|
| Website dashboard | Required. Show latest signal, timestamp, dataset version, and deployment status. |
| Signal API | Required. Users can pull latest signals with deployment-scoped API keys. |
| CSV/JSON download | Useful and cheap. |
| Email | Deferred. |
| Telegram | Deferred. |
| Broker order routing | Deferred. |

Signal API example:

```http
GET /api/v1/signals/latest?deployment_id=...
Authorization: Bearer qf_sig_...
```

```json
{
  "deployment_id": "uuid",
  "portfolio_version_id": "uuid",
  "as_of": "2026-05-13",
  "dataset_version": "2026-W20",
  "signals": [
    {
      "ticker": "ES",
      "forecast_score": 1.25,
      "position_fraction": 0.42,
      "contracts": 1
    }
  ]
}
```

### 12.4 Hosted Portfolio Runtime

Hosted portfolio signal generation should reuse the same worker image and Core strategy contract as backtests.

Differences from backtesting:

| Concern | Backtest | Hosted signal run |
|---|---|---|
| Date range | Historical range | Latest required rolling window only. |
| Output | Full forecast stream and metrics | Latest signal snapshot plus optional recent history. |
| Trigger | User action or API request | Scheduled automation after data update. |
| Storage | Backtest artifacts | Deployment signal artifacts and latest pointer. |
| Notifications | None in MVP | User pulls through dashboard or Signal API. |

## 13. Frontend Structure

Vercel should host `QuantFoundry-Web`.

MVP navigation:

```text
Dashboard
Research Workspace
  Zone Manager
  Strategy Editor
  Backtest Runner
Strategy Library
Portfolio Builder
Deployment
```

Frontend responsibilities:

- OAuth login flow.
- Project and zone management UI.
- Strategy editor and schema validation display.
- Backtest submission and polling.
- Result summaries and warning surfacing.
- Strategy library and committed version browsing.
- Portfolio composition and deployment controls.
- Signal API key display/rotation flow.

The frontend must not hold provider secrets or execute strategy code.

## 14. OAuth and Authorization

MVP auth should be OAuth/OIDC only.

Recommended choices:

1. Clerk for fastest SaaS auth integration.
2. Microsoft Entra External ID only if Azure-native identity becomes strategically important.
3. Direct Google OAuth only if minimizing vendor abstraction is more important than speed.

Current product decision: use Clerk for MVP unless review finds a material downside. Clerk is optimized for modern frontend SaaS flows, has straightforward hosted auth UI, Google OAuth support, webhooks, and good Vercel ergonomics. The user experience should remain Google/OAuth-first, with no QuantFoundry-managed passwords.

API requirements:

- Verify JWT issuer, audience, signature, and expiry.
- Map provider subject to internal `users.id`.
- Enforce ownership on every project, strategy, backtest, portfolio, deployment, artifact, and API key.
- Store only provider identity metadata and no passwords.

## 15. Deployment Environments

Suggested environments should be kept lean:

| Environment | Purpose |
|---|---|
| Local | Developer machine, local API, local worker runner, local/readonly data. |
| Dev | Shared cloud development environment with low quotas. |
| Prod | Customer-facing environment. |

Do not create a standing staging environment for MVP if cost is a concern. Instead:

- Use local integration tests for fast validation.
- Use Vercel preview deployments for frontend review.
- Use dev as the cloud integration environment.
- Use GitHub Environment approvals before prod deployment.
- Optionally create short-lived preview resources later if needed.

Local development should support:

- `QuantFoundry-Core` editable install.
- API running locally.
- Worker running locally against a sample job payload.
- Optional local Postgres.
- Filesystem artifact adapter for local runners. Azurite/direct Blob can be added later when testing cloud storage behavior.

## 16. Cost Controls

MVP cost controls:

- ACA Jobs scale to zero.
- No per-strategy image builds.
- User quotas for concurrent jobs, monthly backtests, tickers, bars, memory, CPU, and timeout.
- API rate limits prevent polling/submission abuse before jobs are queued.
- Cache hits skip worker dispatch.
- Logs sampled and retained conservatively.
- Vercel for frontend instead of Azure Front Door.
- Small Postgres tier until usage requires scaling.

Worker compute should be close to serverless economics. A 1 vCPU / 2 GiB worker running for 10 minutes is expected to cost only cents before free grants. Fixed platform costs will usually dominate early: Postgres, Vercel Pro, monitoring, registry, and baseline API availability.

## 17. Testing Strategy

| Layer | Tests |
|---|---|
| Core unit tests | Strategy contract, metadata validation, cache key determinism, output validation. |
| Worker unit tests | Job payload parsing, source loading, result writing, warning behavior. |
| API tests | Auth mapping, ownership checks, CRUD, job submission, status transitions. |
| Local integration tests | Run sample strategies against local daily data and produce artifacts. |
| Cloud parity tests | Same strategy/data/params produce equivalent local and worker outputs. |
| Security tests | Banned imports/calls, suspicious AST patterns, timeout/memory/quota behavior. |
| Data publishing tests | Dataset validation gates, dev publish, prod promotion dry-run, backadjustment reports. |
| Deployment automation tests | Scheduled signal generation, latest signal API, notification formatting. |

## 18. Remaining Technical Decisions

These are the main items still worth hashing out before deeper implementation. The broad platform choices above are now fixed for MVP.

1. Final `QuantFoundry-Core` strategy contract details: exact `CandleData`, `StrategyParams`, metadata schema, and validation strictness.
2. Data operations v1: daily TWS import, validation, active manifest update, dev/prod promotion, and compaction details.
3. Result artifact format: Parquet-only for time series plus JSON summaries, or Arrow IPC for some paths.
4. Strategy source package shape: single `source.py` plus metadata JSON vs zipped package with controlled modules.
5. Warning thresholds: when a run becomes `completed_with_warnings` vs `failed`.
6. Signal API response format and rate limits.
7. Exact path for extracting reusable logic from `Trading-Algo` into `QuantFoundry-Core`.
8. Rate limit counters: which counters should be exact vs approximate.
9. Whether the daily hosted signal scheduler is ACA scheduled jobs, GitHub Actions, or an internal API-triggered timer.

Closed MVP decisions:

- Auth provider: Clerk.
- Database/migrations: SQLAlchemy 2 + Alembic.
- Local artifacts: filesystem adapter first.
- Worker orchestration: API directly starts ACA Jobs for MVP.
- Quotas: editable defaults in `QuantFoundry-API/config/plans.yml`.
- Signal API keys: one active deployment-scoped key at a time.
- Portfolio backtests: support both single-strategy and portfolio backtests; implement whole-portfolio worker first, matching the SaaS specs, with fanout later.
- IaC: Bicep/GitHub Actions live in the owning repos, mainly `QuantFoundry-API` for backend infrastructure and `QuantFoundry-Web` for Vercel config.

## 19. Near-Term Implementation Order

1. Update `QuantFoundry-Core` to match the `compute(candles, params)` strategy contract.
2. Add output validation, metadata models, cache key generation, and artifact schemas to Core.
3. Add a local runner that can execute the Core contract against local daily data.
4. Add API models/endpoints for projects, zones, strategy validation, and strategy version commit.
5. Add SQLAlchemy 2 models and Alembic migrations for the core product schema.
6. Add Worker package/image scaffold using shared-image runtime-loaded source.
7. Add backtest submission/status/result APIs.
8. Update Web to match the Dashboard / Research Workspace / Strategy Library / Portfolio Builder / Deployment structure.
9. Add cloud deployment scripts for Vercel + Azure Container Apps + ACA Jobs + Blob + Postgres.
10. Add rate limit/quota admission checks before queued compute.
11. Add market data publishing scripts for weekly TWS updates and quarterly backadjusted dataset versions.
12. Add hosted deployment scheduler and Signal API for latest portfolio signals.

