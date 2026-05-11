# Strategy Contract Specification

---

## MVP Scope

The following sections define the MVP contract. Features deferred to future releases are documented in the **Future Additions** section at the end of this document.

---

## 1. Overview

A `Strategy` is the fundamental user-defined unit of signal generation in the SaaS platform. It is a stateless, pure function that, given a rolling window of historical candle data, produces a forecast score per ticker. These scores are bounded to `[-2, 2]`, where `2` represents maximum long exposure, `-2` maximum short exposure, and `0` flat. This common output scale allows any number of strategies to be combined downstream by a separate combining layer without requiring knowledge of each strategy's internal logic.

Continuous strategies may output any value within `[-2, 2]`. Binary strategies naturally output `{-2, 0, 2}`. Both are valid — the platform does not distinguish between them.

---

## 2. Execution Model

### 2.1 The Rolling Window

The platform executes each strategy inside an event-based for-loop. For a backtest over `N` bars:

- At bar `T`, the platform passes the full accumulated candle window `[bar_0 ... bar_T]` to the strategy.
- The strategy is called once per bar, always receiving all available data up to and including the current bar.
- The strategy is a pure function — it receives data, returns a score dict, and holds no server-side state between calls.
- The window grows by one bar each iteration.

The strategy receives exactly one timeframe's worth of data per call (multi-timeframe is a future addition).

### 2.2 Distributed Execution

Each strategy is allocated its own worker/core. Workers run independently and in parallel. Once all workers have completed their for-loops, their output forecast streams are passed to the combining layer. Workers do not communicate with each other during execution.

### 2.3 Output Collection

At each bar the platform collects the `dict[str, float]` output from the strategy, applies output validation (see Section 5.1), and appends it to the forecast stream for that strategy. The final output of a strategy run is a time-indexed forecast stream per ticker.

---

## 3. The Strategy Contract

### 3.1 Call Signature

Every strategy must implement a single entry-point function:

```
compute(
    candles: dict[str, CandleData],
    params: StrategyParams
) -> dict[str, float]
```

- `candles`: Maps each ticker symbol string to a `CandleData` struct containing the full OHLCV history accumulated up to and including the current bar.
- `params`: The resolved parameter object for this strategy run (see Section 3.4).
- Return value: Maps each declared ticker symbol to a forecast score. The platform clips any value outside `[-2, 2]`.

The signature is always multi-ticker. Single-ticker strategies must unpack the dict themselves. This ensures a uniform interface as strategies evolve to support more tickers over time.

### 3.1.1 CandleData

`CandleData` is a lightweight frozen struct containing numpy arrays, one per OHLCV field plus a timestamp array:

```
CandleData:
    open:      np.ndarray   # shape (N,), dtype float64
    high:      np.ndarray   # shape (N,), dtype float64
    low:       np.ndarray   # shape (N,), dtype float64
    close:     np.ndarray   # shape (N,), dtype float64
    volume:    np.ndarray   # shape (N,), dtype float64
    timestamp: np.ndarray   # shape (N,), dtype datetime64[ns]
```

All arrays are ordered oldest-first. `N` is the number of bars accumulated up to and including the current bar, and grows by 1 on each call as the rolling window advances.

**Rationale**: Passing numpy arrays rather than Python object lists eliminates per-bar object construction and attribute lookup overhead. Users who want a DataFrame can construct one trivially from the arrays. All scientific computing libraries (numpy, pandas, scipy, ta-lib) operate natively on numpy arrays, so the strategy's internal indicator logic runs at C speed without any conversion step. Unpacking a `dict[str, list[Candle]]` of Python objects on every bar adds unnecessary overhead that compounds over long backtests and across many tickers.

Users should write indicator logic that operates directly on these arrays using numpy/pandas vectorised operations rather than iterating over individual bars inside the strategy. The for-loop is the platform's responsibility; the strategy's job is to express its logic as array operations over the window it receives.

### 3.2 Required Metadata

Every strategy must declare the following at registration time:

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Human-readable strategy name |
| `version` | `str` | Semantic version (e.g. `"1.0.0"`). Must be incremented on any logic change. |
| `description` | `str` | What the strategy does and its intended market conditions |
| `timeframe` | `TimeFrame` | Single timeframe: `D`, `W`, or `M` |
| `tickers` | `TickerMode` | See Section 3.3 |
| `max_lookback` | `int` | Maximum bars required before the strategy produces a meaningful signal. Subject to platform cap (see Section 5.3). |
| `lookback_params` | `list[str]` | Parameter names that directly determine lookback (see Section 3.5) |
| `params_schema` | `list[ParamSpec]` | Declared parameters with types and constraints (see Section 3.4) |
| `warmup_mode` | `WarmupMode` | `strict` or `flexible` (see Section 4) |

### 3.3 Ticker Declaration

Two modes are supported. A strategy must declare exactly one.

**Static mode**: The strategy hardcodes which tickers it operates on. The ticker list is part of the strategy metadata and cannot be changed at backtest time. Use this for strategies whose logic is inherently ticker-specific (e.g. a pairs trade between ES and NQ).

**Dynamic mode**: The strategy accepts any ticker or set of tickers, supplied by the user as a parameter at backtest time. The strategy logic must be ticker-agnostic. The ticker list is declared as a parameter in the `params_schema`.

### 3.4 Parameter Declaration

Parameters are supplied by the user at backtest time. Each parameter must be declared in `params_schema`:

| Field | Description |
|---|---|
| `name` | Parameter name, used as key in `StrategyParams` |
| `type` | One of: `int`, `float`, `bool`, `str_enum` |
| `allowed_values` | For `str_enum`: exhaustive list of valid string values |
| `min` / `max` | For `int` and `float`: inclusive bounds |
| `default` | Optional default value |
| `description` | Human-readable explanation shown in the UI |

No other parameter types are supported. Strategies requiring more complex configuration must encode it as a `str_enum` or decompose it into the supported primitives.

### 3.5 Max Lookback and `lookback_params`

`max_lookback` is the single integer the platform uses to enforce the platform cap and to source sufficient historical data from the broker. The user declares this value explicitly.

`lookback_params` must list every parameter name whose value directly affects how many historical bars the strategy requires. The platform uses this to validate that no parameter combination at backtest time produces an effective lookback exceeding the platform cap, and to display accurate warmup information in the UI as the user adjusts parameters.

It is the user's responsibility to ensure that `max_lookback` and `lookback_params` are correct and consistent with the strategy logic. The platform does not verify this computationally.

---

## 4. Warmup

The platform calls the strategy from bar 0, passing whatever candle history is available at that point. There is no platform-enforced warmup gate — the strategy is always called, regardless of how many bars have accumulated. The strategy is fully responsible for handling the period before sufficient history is available.

### 4.1 Warmup Modes

**`strict`**: The strategy commits to outputting `0` (flat) for all bars before `max_lookback` bars have been accumulated. No partial signals are produced. The platform does not enforce this — it is a user commitment that affects how results are interpreted downstream.

**`flexible`**: The strategy may produce partial signals during warmup (e.g. computing a moving average over however many bars are currently available). The platform accepts and records these outputs without any special treatment.

In both modes the platform clips all outputs to `[-2, 2]` and records them. The warmup mode declaration is informational — it does not change platform behavior at execution time.

---

## 5. Platform Enforcement

### 5.1 Output Validation

At each bar the platform validates the strategy's return value:

- Values outside `[-2, 2]` are silently clipped to the nearest bound. A warning is logged against the strategy run.
- A missing declared ticker in the return dict results in `0` being substituted. A warning is logged.
- An undeclared ticker in the return dict is discarded. A warning is logged.
- `NaN` values are replaced with `0`. A warning is logged.

Repeated warnings above a threshold are surfaced to the user as a strategy health indicator in the UI.

### 5.2 Security

All strategy code executes inside an isolated sandbox container with:

- No network access
- No filesystem write access
- No access to other strategies, workers, or shared memory
- A restricted Python runtime — the following are blocked: `eval`, `exec`, `open`, `compile`, `__import__`, and all of `os`, `sys`, `subprocess`, `socket`, `requests`, `urllib`, `ctypes`, `importlib`

At registration time the platform performs static AST analysis on all submitted code. Any reference to a blocked builtin or module causes immediate rejection before the code enters a sandbox. The sandbox is the primary security boundary; AST analysis is a fast-fail layer on top.

### 5.3 Complexity Limits

| Limit | Cap | Behavior on Breach |
|---|---|---|
| `max_lookback` | 500 bars | Rejected at registration |
| Per-bar time budget | 100ms | Strategy run killed, backtest fails with error |
| Memory | 512MB | Strategy run killed, backtest fails with error |

The per-bar time budget and memory cap are enforced at the container level.

---

## 6. Caching

The platform operates two distinct cache tiers with different lifecycles, invalidation rules, and user visibility. Both tiers share the same cache key structure but serve different purposes.

### 6.1 Cache Key

```
(strategy_id, strategy_version, source_code_hash, ticker_set, timeframe, parameter_hash, date_range)
```

`source_code_hash` is a SHA-256 of the submitted source code. It is the primary invalidation guard — if a user changes their strategy logic without incrementing the version, the hash changes and all cached results for that strategy are invalidated regardless of the declared version. Version bumps alone are not sufficient; the hash is always checked.

A cache hit skips the for-loop entirely and returns the stored forecast stream. Partial cache hits are not supported — any change to the date range invalidates the full cache entry for that run.

### 6.2 Tier 1 — Portfolio Cache (Selected Strategy Cache)

When a user marks a strategy as selected for their portfolio, that strategy's backtest results are promoted to the portfolio cache. This cache is:

- **Persistent**: Not subject to general eviction. Entries remain until the strategy is deselected or the cache is explicitly invalidated.
- **Automatically invalidated on code change**: When the platform detects that the `source_code_hash` for a selected strategy has changed (because the user uploaded a new version), all portfolio cache entries for that strategy are immediately invalidated and the user is notified. The strategy must be re-run before the portfolio backtest can proceed.
- **Used by portfolio backtests**: When a user runs a portfolio-level backtest, the orchestrator checks the portfolio cache first for each selected strategy. A hit means no worker is launched for that strategy — the cached forecast stream is used directly. This makes portfolio backtests fast even across large numbers of strategies.
- **Platform-managed**: The user does not control this cache directly. It is created when a strategy is selected and destroyed when it is deselected or invalidated.

### 6.3 Tier 2 — Research Cache (User Opt-In)

Users engaged in iterative research (e.g. exploring a large parameter space) can opt in to research caching at the time of backtest submission. When enabled, the result of each individual strategy run is cached and associated with the user's research session.

Key properties:

- **User-controlled**: The user explicitly enables research caching per session. It is not on by default.
- **Survives session end**: Cached results persist beyond the current session. A user can close the browser, return the next day, and reuse previously computed results without re-running anything.
- **Parameter sweep efficiency**: For a sweep of 1,000 parameter combinations, each combination that matches an existing cache entry is skipped entirely. Only new or changed combinations trigger worker execution. This makes iterative parameter exploration cheap after the first full run.
- **User-visible**: Users can inspect what is currently cached (strategy, parameters, date range, timestamp), manually invalidate individual entries, and set an expiry on the session cache.
- **Expiry**: Research cache entries carry a configurable TTL. The default TTL should reflect a reasonable research horizon (e.g. 30 days). Entries that expire are evicted automatically. Users can extend or shorten the TTL manually.
- **Invalidated on code change**: As with Tier 1, a `source_code_hash` change invalidates all research cache entries for that strategy, even if the version number was not changed. The platform surfaces this clearly in the UI so the user knows which prior results are no longer valid.

---

## 7. Stop Losses and Take Profits

The platform does not provide built-in stop-loss or take-profit infrastructure. Users who require SL/TP behavior must implement that logic within the strategy itself and encode it in the forecast score output.

The known limitation: for strategies operating on daily or weekly timeframes, SL/TP logic evaluated at bar close will experience one-bar slippage before the position adjustment is reflected in the output. Users must account for this in their strategy design and backtest interpretation. Higher-frequency timeframe support (which would reduce this slippage) is a future addition — see Section 10.1.

---

## 8. Live Trading Compatibility

The `Strategy` interface is identical for backtesting and live trading. No code changes are required to move a strategy from backtest to live execution. The platform is responsible for sourcing the rolling candle window from the broker (Interactive Brokers) rather than from a historical dataset, and for enforcing the same `max_lookback` constraint against the broker's available history window.

The user is responsible for ensuring that `max_lookback` does not exceed the broker's available history for the declared instrument and timeframe.

---

## 9. Cloud Infrastructure

This section describes the infrastructure architecture for running backtests at scale. The design is provider-agnostic — the patterns apply equally to any major cloud provider. No specific services are named so that the decision between providers remains open.

### 9.1 Design Goals

- **Speed**: A backtest over a portfolio of strategies should complete in minutes, not hours, regardless of how many strategies are involved.
- **Cost efficiency**: Compute should scale to zero when no backtests are running. Users pay only for what they use.
- **Horizontal scale**: Many users submitting backtests simultaneously must not degrade each other's execution time. The system must grow capacity dynamically with demand.
- **Isolation**: Each strategy runs in its own container with no access to other users' strategies or data. This satisfies both the security requirements from Section 5.2 and multi-tenant correctness.

### 9.2 Data Flow

When a user submits a backtest, the following sequence occurs:

1. **Submission**: The user's backtest configuration (strategies, tickers, timeframe, date range, parameters) is validated by the platform API and placed on a job queue.
2. **Decomposition**: An orchestrator pulls the job from the queue and decomposes it into one worker job per strategy. Each worker job is independent.
3. **Dispatch**: Worker jobs are submitted to a managed batch compute service. Each job maps to one containerised strategy execution.
4. **Execution**: Workers spin up in parallel. Each worker fetches only the candle data it requires (ticker + timeframe + date range) from the central candle data store, runs the for-loop, and writes its forecast stream to the results store.
5. **Completion detection**: The orchestrator monitors all worker jobs. When every worker for a given backtest has completed, it triggers the combining layer to aggregate forecast streams.
6. **Result delivery**: The combined output is written to the results store and the user is notified.

### 9.3 Core Infrastructure Components

**Job Queue**
A durable, managed message queue receives backtest submissions and buffers them when compute capacity is temporarily saturated. The queue decouples the user-facing API from the compute layer — the API responds immediately to the user, and the queue absorbs bursts of concurrent submissions without dropping requests.

**Orchestrator**
A workflow orchestration layer (DAG-based) is responsible for:
- Decomposing a backtest job into per-strategy worker jobs
- Tracking completion of all workers belonging to a backtest
- Triggering downstream steps (combining, result storage, user notification) once all workers are done
- Retrying failed workers up to a configurable limit before marking the backtest as failed

The orchestrator does not perform any computation itself — it only coordinates. It must be lightweight and always-on (not subject to the same scale-to-zero behaviour as compute workers).

**Batch Compute Workers**
Each strategy worker is a short-lived container executing one strategy's for-loop. Workers are managed by a batch compute service that:
- Provisions containers on demand as jobs arrive in the queue
- Scales the number of concurrent workers dynamically based on queue depth
- Deallocates containers immediately when a job finishes
- Enforces per-container resource limits (CPU, memory, execution time) as specified in Section 5.3

Workers are stateless and disposable — if a worker fails, it is replaced by a fresh container running the same job. No local state is written to the worker's filesystem that affects correctness.

**Candle Data Store**
Historical candle data is stored as Parquet files, one file per ticker per timeframe. Parquet's columnar layout means reading only the OHLCV columns a worker needs is cheap, and the format compresses well for numerical time-series data. For the MVP, these files are loaded into memory on the worker at job start — there is no database to query. The worker reads the relevant Parquet file, slices the required date range, converts to `CandleData` numpy arrays, and begins the for-loop.

Parquet files are pre-validated and cleaned before storage. Workers receive clean data and perform no data quality checks themselves. All timestamp alignment across tickers is resolved at the data preparation stage, not inside the strategy.

**Results Store**
Forecast streams produced by workers are written to an object store (key-value, file-based) rather than a database, since they are written once and read infrequently (once per combining run, and once when the user views results). Object storage is cheap, durable, and scales without configuration.

**Strategy Cache**
The cache (described in Section 6) sits in front of the worker dispatch step. Before a worker job is created for a strategy, the orchestrator checks the cache. A hit means no worker is launched for that strategy — the stored forecast stream is used directly. This is the primary cost and speed optimisation for repeated backtests with unchanged strategies.

### 9.4 Dynamic Scaling

The compute layer must scale dynamically in two directions:

**Scale out (more users, more strategies)**: As the job queue depth grows, the batch compute service provisions additional worker containers automatically. The upper bound on concurrent workers should be configurable and enforced as a platform cost control, not left unbounded.

**Scale to zero**: When the queue is empty and no workers are running, the compute layer should consume no resources. This is the default state during off-peak hours. The orchestrator and API remain always-on since they are lightweight, but the actual strategy execution infrastructure should have zero idle cost.

**Preemptible / spot compute**: Strategy workers are batch workloads with no latency requirements — they are ideal candidates for preemptible or spot compute instances, which are typically 60–90% cheaper than on-demand. Workers must be designed to handle sudden termination gracefully: the orchestrator detects the failure and re-queues the job. Because the strategy for-loop is stateless (each run starts from bar 0), restarting a worker from scratch is always safe.

### 9.5 Multi-Tenant Isolation

Each user's backtest jobs run in containers that are isolated from other users' containers at both the network and filesystem level. Users cannot observe each other's running jobs, candle data access patterns, or results.

Resource quotas should be enforced per user to prevent a single user from monopolising compute capacity (e.g. a user submitting a backtest with 1000 strategy variants). The orchestrator is responsible for applying these quotas at job dispatch time.

### 9.6 Deployment and Strategy Containers

When a user uploads a strategy, the platform builds a container image containing the strategy code bundled into the sandboxed execution environment. This image is pushed to a container registry and tagged with the strategy ID and version. When a backtest job is dispatched, the worker pulls this image from the registry.

Container builds happen at upload/registration time, not at backtest submission time. This means the cost and latency of building the image is paid once per strategy version, not once per backtest run.

**Shared base image**: All strategy containers are built on top of a single platform-managed base image containing the Python runtime, numpy, pandas, scipy, and any other approved scientific computing libraries. Only the user's strategy code is layered on top. Since the base image is identical across all users and all strategies, it will already be cached on any compute node that has run at least one prior strategy. The only layer that must be pulled fresh for a new strategy is the thin user code layer. This dramatically reduces cold-start time: the large base image pull happens at most once per compute node over its lifetime, not once per strategy run.

The base image is versioned and updated by the platform independently of individual strategies. When the base image is updated (e.g. a library version bump or security patch), all strategy images must be rebuilt against the new base. The platform handles this automatically at base image release time.

### 9.7 Networking and Security

- Compute workers must have no outbound internet access. All required data (candles, strategy image) is sourced from within the private network.
- The candle data store and results store are accessible only from within the platform's private network — not from the public internet.
- The user-facing API is the only public ingress point. All internal communication between the orchestrator, queue, workers, and data stores occurs within the private network.
- Strategy container images should be scanned for known vulnerabilities at build time. Images that fail scanning are rejected before they enter the registry.

---

## 10. Future Additions

The following features are explicitly out of scope for the MVP. They are documented here to preserve design intent and to record decisions made during the initial architecture phase.

---

### 10.1 Multi-Timeframe Support

Multi-timeframe strategies are not supported in the MVP. Each strategy operates on exactly one declared timeframe. Users who require signals from multiple timeframes must run separate strategies on each timeframe and combine their outputs via the downstream combining layer.

**Why this matters**: The primary motivation for multi-timeframe support is stop-loss and take-profit evaluation. A daily strategy that wants to trail a stop intraday cannot do so reliably with only daily bars — the SL may be hit and recovered within a single day, never appearing in the daily close price. Accessing a lower timeframe within the same strategy is the only clean solution.

**Two approaches were considered for the future implementation:**

#### Option A — Platform-Sourced Multi-TF Windows

The user declares multiple timeframes in the strategy metadata. The platform sources data independently at each declared timeframe. The event loop runs at the **lowest** declared timeframe. At each bar the strategy receives all candle windows for all declared timeframes:

```
compute(
    candles: dict[Ticker, dict[TimeFrame, list[Candle]]],
    params: StrategyParams
) -> dict[Ticker, float]
```

The platform attempts to include only completed bars in higher timeframe windows (e.g. the current incomplete weekly bar is excluded). However, this approach carries an inherent **look-ahead bias risk** that is entirely dependent on timestamp precision and implementation correctness. The determination of when a higher-TF bar is "completed" relative to the current lowest-TF bar timestamp is subtle — any off-by-one in bar boundary logic, timezone handling, or session close alignment can silently introduce look-ahead bias. This risk lives in the platform implementation and is invisible to the user.

- Pros: no user aggregation logic needed, call signature is clean, higher-TF windows are pre-aligned by the platform
- Cons: look-ahead bias risk if bar completion timestamps are handled imprecisely — and this failure mode is silent and hard to detect. If the loop runs at hourly cadence, there are 24x more strategy calls per day than a pure daily strategy. Higher-TF windows don't change on every call, so user code must handle the stale-window case.

#### Option B — Lowest Timeframe + Platform Aggregation Utility

The user declares only the lowest timeframe they need. The platform provides a stateless utility function that aggregates lower-TF candles into higher-TF bars on demand:

```
aggregate(candles: list[Candle], target_tf: TimeFrame) -> list[Candle]
```

Because the user already received the lowest-TF candle window from the platform, any aggregation performed on that window can only use data that already exists in the window — it is structurally impossible to aggregate future data that hasn't been passed in yet. Look-ahead bias is eliminated by construction.

- Pros: **zero look-ahead bias risk by construction** — aggregation is bounded by the data already received, custom timeframes are possible (e.g. 4H from 1H), single data feed simplifies platform data sourcing
- Cons: aggregation must be re-executed on every backtest run since there is no caching of the intermediate aggregated series. If the utility is called unconditionally on every bar, this is significant wasted compute for long backtests. This could be partially mitigated by caching the aggregated candle series as a derived artifact keyed on the lowest-TF series, but that requires additional platform infrastructure.

**Status**: Neither approach has been selected. Option B's look-ahead safety is a strong argument in its favor, and the compute cost is the primary remaining concern. The caching question for intermediate aggregated series needs to be resolved before a final decision is made.

---

### 10.2 Indicator Contract

Indicators are stateless, pure-function building blocks that strategies can depend on. They follow the same rolling window model as strategies but have no restrictions on their output type or scale. An indicator's output is a `dict[str, Any]` — the consuming strategy is responsible for interpreting it correctly.

Indicators will have their own `max_lookback` declaration, their own `lookback_params`, and their own versioning and caching. The platform will compute the effective max lookback for a strategy as the maximum across the strategy itself and all transitive indicator dependencies.

Indicators will not declare a timeframe, tickers, or warmup mode — those concerns belong to the strategy that uses them.

This feature is deferred because it introduces the dependency graph resolution infrastructure (see Section 9.3), which is out of scope for the MVP.

---

### 10.3 Strategy and Indicator Composition

Users will be able to declare dependencies on other user-defined indicators and strategies. All composition is user-defined — the platform will not provide a built-in indicator library.

Dependencies are declared by platform ID (not by Python module path) in the strategy metadata. At registration time the platform will:

1. Resolve the full dependency graph recursively.
2. Reject circular dependencies.
3. Enforce a maximum dependency depth of 3 levels.
4. Bundle all dependency code into the strategy's sandbox under a controlled namespace.

At runtime the strategy imports dependencies using the platform-controlled namespace. No arbitrary Python import statements are permitted — the sandbox blocks all imports not pre-resolved by the platform. This means the dependency graph is fully known before any code executes.

The cache key for strategies with dependencies will include a `dependency_version_hash` — a combined hash of all transitive dependency IDs and versions. A change to any dependency requires a version bump on that dependency, which propagates cache invalidation to all strategies that depend on it.

This feature is deferred because it requires the dependency graph infrastructure and changes to both the registration pipeline and the sandbox build process. These are non-trivial additions that would block MVP delivery.
