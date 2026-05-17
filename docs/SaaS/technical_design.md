# QuantFoundry SaaS Technical Design

## 1. Purpose

This document captures the current technical plan for turning the existing `Trading-Algo` research system into the QuantFoundry SaaS platform. It complements:

- `docs/SaaS/data_flow.md` - research, portfolio, deployment, and zone lifecycle.
- `docs/SaaS/zone_manager.md` - zone types, UTC slicing contract, Core vs API responsibilities, snapshots.
- `docs/SaaS/strategy_spec.md` - user strategy contract and runtime expectations.
- `docs/SaaS/ui_ux.md` - product navigation and MVP user surfaces.
- `docs/SaaS/metrics_library.md` - canonical return conventions, KPI computation boundaries, UTC series, QuantStats posture, NumPy-centric implementation.

The guiding product goal is to let independent traders build, validate, combine, and deploy systematic strategies without needing to build the surrounding infrastructure themselves.

## 2. Repository Strategy

The long-term goal is to avoid duplicate research/runtime code. `Trading-Algo` should not remain a parallel implementation of the product engine. Instead, QuantFoundry should become the single local and cloud runtime, with `Trading-Algo` treated as the source to migrate from and as a temporary reference while stable logic is extracted.

| Repository | Role |
|---|---|
| `Trading-Algo` | Existing research/backtesting codebase and migration source. Keep operational during transition, but do not build new product-runtime features here. |
| `QuantFoundry-Core` | Shared Python library for strategy contracts, validation, candle/runtime models, artifact schemas, zone splitting (`zone_manager` module), and reusable engine components. |
| `QuantFoundry-API` | SaaS backend: auth integration, user/project/strategy metadata, job submission, ACA Job orchestration, status/results APIs, deployment APIs. |
| `QuantFoundry-Worker` | Private batch worker image for executing strategy validation/backtests/signal generation jobs. It imports `QuantFoundry-Core`. |
| `QuantFoundry-Web` | Public web application for dashboard, research workspace, strategy library, portfolio builder, and deployment UI. |

Dependency direction must stay one-way:

```text
QuantFoundry-API     ─┐
QuantFoundry-Worker  ├── imports quantfoundry_core
Local research stack ┘

QuantFoundry-Web talks to QuantFoundry-API over HTTP.
QuantFoundry-Core never imports from API, Worker, Web, or Trading-Algo.
```

Migration should be incremental, not a big-bang rewrite:

1. Keep `Trading-Algo` operational only as a transition safety net.
2. Move stable contracts and portable logic into `QuantFoundry-Core`.
3. Build local and cloud execution through `QuantFoundry-API` + `QuantFoundry-Worker`.
4. Stop adding new product-facing runtime logic to `Trading-Algo`.
5. Add parity tests proving local Docker worker and cloud ACA Job outputs match for the same strategy, data, and parameters.

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
| Metadata DB | Azure Database for PostgreSQL Flexible Server | Relational ownership/versioning/audit model with JSONB escape hatch. |
| Artifacts | Azure Blob Storage | Cheap durable storage for Parquet, JSON, source bundles, and backtest outputs. |
| Secrets | Azure Key Vault | OAuth secrets, API keys, storage credentials if not fully using managed identity. |
| Observability | Azure Monitor/Application Insights | API/worker logs, job failures, latency, cost and health telemetry. |
| Infrastructure as Code | Azure Bicep | Native Azure IaC for Azure resources. |
| CI/CD | GitHub Actions | Build, test, provision, and deploy from repo workflows using OIDC federation. |

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
| `Trading-Algo` | Legacy migration source only. Do not add new QuantFoundry product workflows here. |

A separate `QuantFoundry-Infra` repo is not part of the MVP.

Each repo should use exactly two primary GitHub Actions workflows: one pull request pipeline and one official pipeline.

```text
.github/workflows/
  QuantFoundry-<Repo>-PullRequest.yml
  QuantFoundry-<Repo>-Official.yml
```

Pipeline responsibilities:

| Pipeline | Responsibility |
|---|---|
| `QuantFoundry-Core-PullRequest.yml` | Build package, run unit tests, run type/lint checks when configured. |
| `QuantFoundry-Core-Official.yml` | Publish/version the Core package after merge to `main`. |
| `QuantFoundry-API-PullRequest.yml` | Build API/worker images, run unit tests, run migration checks. |
| `QuantFoundry-API-Official.yml` | After merge to `main`: deploy dev automatically, then deploy prod behind GitHub Environment manual approval. |
| `QuantFoundry-Web-PullRequest.yml` | Typecheck/build frontend and create Vercel preview. |
| `QuantFoundry-Web-Official.yml` | After merge to `main`: deploy dev/preview automatically, then prod behind approval or Vercel production gate. |

Authentication should use GitHub Actions OIDC federation into Azure, not long-lived Azure credentials stored as GitHub secrets. Vercel is connected directly to the `QuantFoundry-Web` repository for preview/prod deploys.

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

Local research should use the same QuantFoundry product runtime rather than a separate `Trading-Algo` execution path. The local stack should mirror cloud architecture with local substitutions for cloud services.

```text
QuantFoundry-Web locally
  -> QuantFoundry-API locally
  -> local Postgres
  -> local Blob emulator or filesystem artifact store
  -> local Docker engine starts QuantFoundry-Worker containers
  -> local data files are mounted/read by worker
```

Local and cloud execution should share:

- `CandleData`.
- `StrategyParams`.
- Strategy metadata.
- Output validation.
- Artifact schemas.
- Worker image/runtime.
- API request/response contracts.

They should differ only in execution backend:

| Concern | Local/internal | Cloud/user-facing |
|---|---|---|
| Web | Local Vite/React dev server | Vercel. |
| API | Local FastAPI process | Azure Container Apps. |
| DB | Local Postgres | Azure Database for PostgreSQL. |
| Artifacts | Local Blob emulator or filesystem adapter | Azure Blob Storage. |
| Worker execution | Docker starts local worker containers | API starts ACA Job executions. |
| Data | Local mounted daily/quarterly files or local IB/TWS connection | Blob quarterly research data and IB/TWS live candle fetch for deployments. |

Local-only research features are acceptable if feature-flagged off by default in deployed environments. Example flags:

```text
ENABLE_LOCAL_RESEARCH_TOOLS=true
ENABLE_LOCAL_DATA_BROWSER=true
ENABLE_UNSAFE_DEV_SHORTCUTS=false
```

Production deployments should fail closed: local-only features must require explicit opt-in and should not be enabled by missing environment variables.

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

The backend should support arbitrary non-overlapping zones. The UI presents a guided Train / Validation / Test layout by default; researchers may add multiple zones of the same type, including multiple test zones.

```text
Zone:
  id
  project_id
  name
  zone_type        # Train | Validation | Test
  start_at_utc     # timestamptz; inclusive lower bound per zone_manager.md
  end_at_utc       # timestamptz; inclusive upper bound per zone_manager.md
  created_at
```

Rules:

- UTC range must be valid, non-empty, and non-overlapping with any other zone in the project.
- No enforced count or ordering in the backend.
- API accepts user-friendly date inputs where appropriate but persists normalized UTC instants consistent with `docs/SaaS/zone_manager.md`.

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

Portfolio versions are immutable snapshots.

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
  deployment_id
  name
  key_prefix          # safe display prefix, e.g. qf_sig_live_abc123
  key_hash
  scopes              # e.g. read:signals
  last_used_at
  expires_at
  revoked_at
  created_at
```

MVP deployment means daily signal generation and retrieval.

Signal API keys are deployment-scoped, not global per user. MVP allows one active key per deployment for simplicity. Rotation creates a replacement key and automatically revokes the prior active key for that deployment.

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

MVP starts with a simple built-in plan table. Stripe/billing maps onto these plans when payments are implemented.

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
  -> reserve usage or start job
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
- Artifact URIs.
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
  research/timeframe=D/ticker=ES/as_of=2026Q2/part.parquet
  live-snapshots/deployment_run=<run_id>/ticker=ES/candles.parquet

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

## 8. Market Data Operations

MVP data should be daily futures data only. Separate the research/backtest dataset story from the hosted deployment signal story.

### 8.1 Research and Backtest Data

For research and backtesting, QuantFoundry reads curated daily futures Parquet files from Blob Storage. These files are updated in place as part of platform data operations. The platform does not expose data-selection controls to users in the MVP.

```text
candles/research/timeframe=D/ticker=ES/part.parquet
```

Backtests use whatever curated research data is active in the environment at run time. Backtest results should store the run timestamp and input date range, but no dataset-versioning system is included in the MVP.

### 8.2 Quarterly Backadjustment Flow

Quarterly futures backadjustment updates the curated research Parquet files.

```text
Quarterly rollover review
  -> local raw data landing folder
  -> import latest TWS history as needed
  -> run backadjustment script locally
  -> produce adjustment report by ticker/contract/roll date
  -> write updated backadjusted Parquet files
  -> publish to dev Blob path
  -> rerun smoke backtests and data quality checks
  -> copy the same files to prod Blob path
```

Suggested scripts:

```text
scripts/data/import_tws_history.py
scripts/data/build_continuous_futures.py
scripts/data/validate_candle_dataset.py
scripts/data/publish_dataset.py --target dev
scripts/data/promote_dataset.py --from dev --to prod
```

This keeps data operations simple: the platform has one active research dataset per environment.

### 8.3 Hosted Deployment Live Data

Hosted deployments need the most recent candles at signal time. Instead of publishing a new full research dataset every day, scheduled deployment workers should fetch the required lookback window from Interactive Brokers/TWS.

MVP approach:

```text
ACA scheduled signal trigger
  -> enumerate running deployments
  -> start one ACA Job per deployment, or per small deployment batch
  -> each worker connects to the platform IB/TWS gateway
  -> fetch required daily candle lookback for declared tickers
  -> run deployed portfolio strategies
  -> write signal result to Postgres and optional Blob artifact
```

This requires a reliable headless IB/TWS gateway or IB Gateway container strategy. The gateway should be treated as platform infrastructure, not user infrastructure. It should not expose broker execution for MVP; it is only a market data source.

Risks to validate early:

- IB/TWS gateway session reliability and reauthentication behavior.
- Rate limits and pacing violations when many deployments request overlapping symbols.
- Data adjustment differences between live IB lookback data and quarterly backadjusted research data.
- Timezone/session-close handling for daily bars.
- Whether one shared data fetch step should deduplicate overlapping ticker requests before fanout.

The MVP implementation should prefer a shared data fetch step if direct per-deployment IB requests would exceed pacing limits.

### 8.4 Data Validation Gates

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

## 9. API Shape

API paths are illustrative and can change during implementation. During MVP, app-internal APIs used only by `QuantFoundry-Web` do not need URL versioning. Versioning should be reserved for public/external APIs where third-party clients may depend on stable contracts.

Recommended convention:

| API surface | Path style | Reason |
|---|---|---|
| Web app API | `/api/...` | Frontend and backend ship together; avoiding `/v1` reduces churn while the product is evolving. |
| External Signal API | `/api/v1/signals/...` | Users may automate against it, so breaking changes need explicit versioning. |
| Internal worker/admin APIs | Not public or separately authenticated | Prefer job payload contracts over public endpoints. |

Breaking app API changes should be handled by deploying compatible Web and API revisions together.

### 9.1 Health

```http
GET /api/health
```

### 9.2 Projects and Zones

```http
POST /api/projects
GET  /api/projects
GET  /api/projects/{project_id}

POST /api/projects/{project_id}/zones
GET  /api/projects/{project_id}/zones
PUT  /api/projects/{project_id}/zones/{zone_id}
DELETE /api/projects/{project_id}/zones/{zone_id}
```

### 9.3 Strategies

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

### 9.4 Backtests

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
  "zone_ids": ["uuid"]
}
```

```json
{
  "scope": "portfolio",
  "portfolio_version_id": "uuid",
  "zone_ids": ["uuid"]
}
```

Single-strategy backtests are the main research loop. Portfolio backtests are required before deployment because users need to validate combined weights, correlations, and aggregate drawdown. MVP portfolio backtests run as one worker job for the whole portfolio.

### 9.5 Portfolios

```http
POST /api/projects/{project_id}/portfolios
GET  /api/projects/{project_id}/portfolios
POST /api/portfolios/{portfolio_id}/versions
GET  /api/portfolio-versions/{portfolio_version_id}
```

### 9.6 Deployments and Signals

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

### 9.7 Rate Limiting and Abuse Controls

Rate limiting must happen before expensive work starts.

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
- Keep Signal API key limits separate from web app OAuth user limits.

Abuse response should be explicit:

```http
429 Too Many Requests       # short-window API limit
402 Payment Required        # paid plan quota exceeded, if billing is active
403 Forbidden               # plan does not allow requested resource size
409 Conflict                # concurrent job limit reached
```

## 10. Worker Flow

```text
1. API receives backtest request.
2. API validates ownership, strategy version, parameters, zones, and quotas.
3. API writes BacktestRun(status='queued').
4. API starts an ACA Job execution with the run ID and job payload reference.
5. ACA Job starts worker container.
6. Worker loads job payload and strategy source.
7. Worker loads candle Parquet for tickers/timeframe/date range.
8. Worker converts data to CandleData windows and calls compute() per bar.
9. Worker validates outputs and records warnings.
10. Worker writes forecast stream and summary artifacts to Blob.
11. Worker updates BacktestRun status and artifact URIs.
12. Frontend polls API for status/results.
```

Workers should be idempotent. Retrying the same job should either overwrite a deterministic staging path safely or create a new attempt path and atomically mark the successful attempt in Postgres.

The MVP implementation uses direct ACA Job startup from the API.

## 11. Deployment and Hosted Signal Automation

MVP deployment means hosted daily signal generation for a committed `PortfolioVersion`.

### 11.1 Deployment Lifecycle

```text
User creates portfolio version
  -> user clicks Deploy
  -> API creates Deployment(status='running')
  -> scheduled ACA Job includes deployment in the next all-deployments signal run
  -> worker computes latest signals using the IB/TWS lookback window
  -> signals are stored and exposed through UI/API/notifications
```

Users should not need to press a browser button every day. Once a deployment is running, scheduled backend automation should produce signals.

### 11.2 Scheduled Signal Runs

Use an ACA scheduled job for MVP. All running deployments should be evaluated from the same scheduled trigger. The scheduler should fan out ACA Job executions across deployments so users receive signals from the same market data window.

Recommended daily flow:

```text
ACA scheduled signal trigger fires
  -> fetch all running deployments
  -> group deployments by timeframe and overlapping ticker needs
  -> optionally prefetch/deduplicate IB/TWS candle lookbacks
  -> start deployment signal jobs in parallel
  -> write deployment-signals artifacts
  -> update latest_signal records
```

Hosted signal automation depends on IB/TWS candle availability. If fresh candles are unavailable for a trading day, the scheduler should not silently produce stale signals. It should mark affected deployments as `waiting_for_data` or `data_unavailable` and expose that status in the dashboard and API.

### 11.3 Signal Delivery

The MVP supports pull-based delivery through the website and Signal API.

| Delivery | MVP recommendation |
|---|---|
| Website dashboard | Required. Show latest signal, timestamp, data source, and deployment status. |
| Signal API | Required. Users can pull latest signals with deployment-scoped API keys. |
| CSV/JSON download | Useful and cheap. |

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
  "data_source": "ibkr_tws",
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

### 11.4 Hosted Portfolio Runtime

Hosted portfolio signal generation should reuse the same worker image and Core strategy contract as backtests.

Differences from backtesting:

| Concern | Backtest | Hosted signal run |
|---|---|---|
| Date range | Historical range | Latest required rolling window only. |
| Output | Full forecast stream and metrics | Latest signal snapshot plus optional recent history. |
| Trigger | User action or API request | Scheduled automation after data update. |
| Storage | Backtest artifacts | Deployment signal artifacts and latest pointer. |
| Notifications | None in MVP | User pulls through dashboard or Signal API. |

## 12. Frontend Structure

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

## 13. OAuth and Authorization

MVP auth should be OAuth/OIDC only.

Current product decision: use Clerk for MVP. The user experience should remain Google/OAuth-first, with no QuantFoundry-managed passwords.

API requirements:

- Verify JWT issuer, audience, signature, and expiry.
- Map provider subject to internal `users.id`.
- Enforce ownership on every project, strategy, backtest, portfolio, deployment, artifact, and API key.
- Store only provider identity metadata and no passwords.

## 14. Deployment Environments

Suggested environments should be kept lean:

| Environment | Purpose |
|---|---|
| Local | Developer machine, local API, local worker runner, local/readonly data. |
| Dev | Shared cloud development environment with low quotas. |
| Prod | Customer-facing environment. |

Do not create a standing staging environment for MVP.

Local development should support:

- `QuantFoundry-Core` editable install.
- API running locally.
- Worker running locally against a sample job payload.
- Optional local Postgres.
- Local Blob emulator for integration parity, with a filesystem artifact adapter for fast Core/unit tests.

## 15. Cost Controls

MVP cost controls:

- ACA Jobs scale to zero.
- User quotas for concurrent jobs, monthly backtests, tickers, bars, memory, CPU, and timeout.
- API rate limits prevent polling/submission abuse before jobs start.
- Logs sampled and retained conservatively.
- Small Postgres tier until usage requires scaling.

Worker compute should be close to serverless economics. A 1 vCPU / 2 GiB worker running for 10 minutes is expected to cost only cents before free grants. Fixed platform costs will usually dominate early: Postgres, Vercel Pro, monitoring, registry, and baseline API availability.

## 16. Testing Strategy

| Layer | Tests |
|---|---|
| Core unit tests | Strategy contract, metadata validation, zone slicing, output validation. |
| Worker unit tests | Job payload parsing, source loading, result writing, warning behavior. |
| API tests | Auth mapping, ownership checks, CRUD, job submission, status transitions. |
| Local integration tests | Run sample strategies against local daily data and produce artifacts. |
| Cloud parity tests | Same strategy/data/params produce equivalent local and worker outputs. |
| Security tests | Banned imports/calls, suspicious AST patterns, timeout/memory/quota behavior. |
| Data tests | Quarterly release validation, TWS lookback fetching, backadjustment reports. |
| Deployment automation tests | Scheduled signal generation, latest signal API, notification formatting. |

## 17. Implementation Decisions

These items need concrete implementation choices during buildout.

1. Final `QuantFoundry-Core` strategy contract details: exact `CandleData`, `StrategyParams`, metadata schema, and validation strictness.
2. Data operations v1: quarterly research data release scripts and hosted signal IB/TWS lookback fetch design.
3. Result artifact format: Parquet-only for time series plus JSON summaries, or Arrow IPC for some paths.
4. Strategy source package shape: single `source.py` plus metadata JSON vs zipped package with controlled modules.
5. Warning thresholds: when a run becomes `completed_with_warnings` vs `failed`.
6. Signal API response format and rate limits.
7. Exact path for extracting reusable logic from `Trading-Algo` into `QuantFoundry-Core` and retiring duplicate execution paths.
8. Rate limit counters: which counters should be exact vs approximate.
9. IB/TWS gateway deployment model for hosted signal lookback data.

Closed MVP decisions:

- Auth provider: Clerk.
- Database/migrations: SQLAlchemy 2 + Alembic.
- Local artifacts: filesystem adapter first.
- Worker orchestration: API directly starts ACA Jobs for MVP.
- Quotas: editable defaults in `QuantFoundry-API/config/plans.yml`.
- Signal API keys: one active deployment-scoped key at a time.
- Portfolio backtests: support both single-strategy and portfolio backtests; MVP portfolio backtests use one whole-portfolio worker job.
- IaC: Bicep/GitHub Actions live in the owning repos, mainly `QuantFoundry-API` for backend infrastructure and `QuantFoundry-Web` for Vercel config.
- Hosted signal scheduler: ACA scheduled job starts the all-deployments run and fans out deployment jobs in parallel.

## 18. Near-Term Implementation Order

1. Update `QuantFoundry-Core` to match the `compute(candles, params)` strategy contract.
2. Add output validation, metadata models, zone slicing, and artifact schemas to Core.
3. Add local Docker-based worker execution wired to local API, Postgres, and local Blob/filesystem artifacts.
4. Add API models/endpoints for projects, zones, strategy validation, and strategy version commit.
5. Add SQLAlchemy 2 models and Alembic migrations for the core product schema.
6. Add Worker package/image scaffold using shared-image runtime-loaded source.
7. Add backtest submission/status/result APIs.
8. Update Web to match the Dashboard / Research Workspace / Strategy Library / Portfolio Builder / Deployment structure.
9. Add cloud deployment scripts for Vercel + Azure Container Apps + ACA Jobs + Blob + Postgres.
10. Add rate limit/quota admission checks before compute starts.
11. Add quarterly research data publishing and backadjustment scripts.
12. Add ACA scheduled deployment runner, IB/TWS lookback fetch, and Signal API for latest portfolio signals.

