 # Zone Manager

> **Status: forward-looking SaaS design — not yet implemented.** This document
> defines the planned zone model for the QuantFoundry SaaS platform. The
> `quantfoundry_core.zone_manager` module, the `ZoneSnapshot`/`ZoneConfig` Core
> types, and the PostgreSQL `Zone` entity referenced below **do not exist in the
> repo today** (no `quantfoundry_core` package and no `zone_manager.py` are
> present; only a `tests/unit-tests/quantfoundry_core/` test stub exists). The
> contamination/test-integrity doctrine (§8) is the design discipline this model
> is meant to enforce; treat the APIs and storage as target, not current, state.

## 1. Purpose

This document defines **QuantFoundry** data zones for research projects: the **three zone types**, **time-boundary rules**, where logic lives (**Core** vs **API**), and how configurations are **stored and snapshotted**. Product-level lifecycle and research flow are in `data_flow.md`; API entity shapes are in `technical_design.md` §6.3. **UI:** the current local workspace UI is the Flask app described in `ui_ux.md`; dedicated hosted zone-management pages are planned but not yet present in `docs/`.

---

## 2. Zone types and tiers

Zones operate at two distinct tiers. Mixing them up is the primary source of confusion when reasoning about multi-strategy portfolios.

### Tier 1 — Project-level zone (one per project)

| Zone | Meaning |
|---|---|
| **Project Test** | A single, project-wide holdout window. Fixed at project creation. All portfolio-level evaluation happens here. No strategy in the project may use this window for fitting, parameter selection, or robustness testing. See §5 for the full contract and §8 for the contamination rules that govern it. |

### Tier 2 — Strategy-level zones (per strategy, constrained to the pre-test window)

Each strategy-level zone has a stable `zone_id` (UUID), a researcher-defined **`name`** (display only), and a **`zone_type`**.

| `zone_type`   | Meaning |
|---------------|---------|
| **Train**     | Bars used for strategy development, fitting, and IS robustness tests. Must end before `project_test_start`. |
| **Validation** | OOS iteration for the individual strategy — research-grade evaluation used to check strategy behaviour before committing to the portfolio. Must fall entirely before `project_test_start`. Not a portfolio evaluation tool. |

**Important:** Strategy-level **validation** is OOS with respect to the strategy's own training window, but it is still inside the pre-test research period. It tells you whether an individual strategy generalises. It does not tell you how the portfolio performs — that is exclusively the job of the project test zone.

User-defined **names** must never replace **`zone_type`** for engine decisions.

---

## 3. Time boundaries (canonical contract)

All zone boundaries are interpreted in **UTC**.

- **Inclusive lower bound:** every bar with timestamp **`>= start_at_utc`** is eligible for membership unless excluded by overlap rules with another zone (zones must not overlap).
- **Inclusive upper bound:** every bar with timestamp **`<= end_at_utc`** is eligible.

The Web app and API may collect **calendar dates** from users; they **normalize** to UTC instants (e.g. start-of-day and end-of-day in UTC for daily bars) before persistence and before calling **QuantFoundry-Core**. Exact normalization for each timeframe belongs in Core and worker tests; this document fixes **UTC + inclusive ends** as the platform contract.

**QuantFoundry-Core** (`quantfoundry_core.zone_manager` or equivalent) is intended to implement this contract so local research, workers, and parity tests agree. This module does not exist yet — it is part of the planned Core extraction (`technical_design.md` §2).

---

## 5. Project-Level Test Zone

### 5.1 Purpose

The project test zone exists to answer one question honestly: **how does this portfolio perform on data that was never used, directly or indirectly, to make any research decision?**

It is the only zone that can give an unbiased estimate of live performance. All other zones — strategy train, strategy validation — are IS with respect to the research process. The project test zone is the sole OOS window for the portfolio as a whole.

### 5.2 Structural contract

- **Defined once at project creation.** The researcher sets `project_test_start` and `project_test_end` before any strategy research begins. Changing these boundaries after research has started is a contamination event (§8.3).
- **Applies to all strategies in the project.** There is no per-strategy test zone. Strategy-level zones may only span `[data_start, project_test_start)`.
- **Hard constraint enforced by Core:** any strategy zone with `end_at_utc > project_test_start` is rejected. The worker similarly rejects any fit or parameter selection job that requests bars inside the project test window.
- **Portfolio evaluation uses this window exclusively.** When the researcher evaluates the combined portfolio, the system runs the evaluation on `[project_test_start, project_test_end]` regardless of individual strategy zone boundaries.

### 5.3 Multi-asset alignment

Different strategies cover different assets with different data histories. Within the project test window:

- Strategies with full data coverage contribute a signal for every bar in the test window.
- Strategies whose underlying asset was listed after `project_test_start` contribute a signal from their listing date onward and hold zero position before that.
- No special zone logic is required — sparse coverage is handled naturally as a zero/null signal in the position sizer.

The project test zone definition does not need to change to accommodate assets with shorter histories. The portfolio is evaluated on whatever signal is available from each strategy during the test window.

### 5.4 Default suggestion

When a project is created, the platform suggests a default test window of the most recent 20% of available data across all selected assets. The researcher can adjust this before any research begins. Once the first strategy training run is submitted, the boundary is locked.

---

## 6. Structural rules

For zones within the same **project**:

1. Each strategy zone has a non-empty UTC `[start_at_utc, end_at_utc]` range with `end_at_utc < project_test_start`.
2. Strategy zone ranges must **not overlap** pairwise within the same strategy.
3. There is **no** required global ordering, minimum count, or maximum count of strategy zones.
4. A project has exactly **one** project test zone, defined at creation.

Discovery and UI ordering can sort by `start_at_utc` for timeline display.

---

## 7. Responsibilities by layer

### 7.1 QuantFoundry-Core (pure library)

Core owns **deterministic** operations only — no database, no HTTP:

- **Validate** a zone configuration: overlaps, empty ranges, duplicate handling, optional index alignment checks against a provided `DatetimeIndex`.
- **Materialize** slices: given a normalized `ZoneConfig` and a pandas object (conventions for single vs multi-ticker frames are defined in Core), return per-zone views plus diagnostics (bar counts, index min/max per slice, optional gap hints on the index).
- **Serialize / deserialize** a versioned **`ZoneSnapshot`** (JSON or equivalent) for embedding in job requests and immutable artifacts.

Core returns structured **`Result`**-style outcomes (success vs validation errors) as per project conventions; avoid untyped `dict[str, DataFrame]` as the only public contract.

### 7.2 QuantFoundry-API (PostgreSQL)

- **Source of truth** for the live project: CRUD on `Zone` rows (`technical_design.md` §6.3).
- Normalize user input to **`start_at_utc` / `end_at_utc`** before save.
- On **strategy version commit**, **portfolio version create**, or **job enqueue**, persist an embedded **`zone_snapshot_json`** (or equivalent) so results remain reproducible even if project zones are edited later.

### 7.3 QuantFoundry-Worker

- Load zone definitions from the job payload (prefer **embedded snapshot** for strict replay).
- Fetch candles for the union of requested ranges, then call Core to split by zone for execution and reporting.

### 7.4 Platform dataset coverage (future)

When the catalog of published candles is authoritative, the API or worker can add **coverage checks** (requested zone vs available data range). Core stays unaware of Blob/Postgres; injected **`DataCoverage`** or similar policy can attach **warnings** or **hard errors** at orchestration time. MVP may ship without this and rely on empty slices or diagnostics from Core.

---

### 7.5 ZoneConfig and snapshots

A **`ZoneSnapshot`** is an ordered, versioned structure embedded in commits and jobs. Conceptually:

```text
ZoneSnapshot:
  schema_version:      int
  project_test_start:  datetime  # UTC — the project-level boundary
  project_test_end:    datetime  # UTC
  strategy_zones:      list[ZoneSpec]

ZoneSpec:
  zone_id:          uuid
  name:             str
  zone_type:        Train | Validation
  start_at_utc:     datetime  # UTC — must be < project_test_start
  end_at_utc:       datetime  # UTC — must be < project_test_start
```

Exact encoding (Pydantic models, JSON Schema) lives in **QuantFoundry-Core** once the package exists. Application code must branch on `zone_type` (enum), not on `name`. Maintain a single registry — `allowed_robustness_tests(zone_type) -> frozenset[TestId]` — so the robustness catalog is driven by zone type, not by string matching on user-defined names.

---

## 8. Test Set Integrity and Data Contamination

### 8.1 What "unbiased" means

The project test zone produces an unbiased performance estimate when the data in it had no influence — direct or indirect — on any decision made during research. This includes parameter selection, strategy selection, weight decisions, and portfolio composition decisions.

Unbiased does not mean the result will be good. It means the result is neither artificially inflated (by optimising for it) nor artificially deflated (by construction). It is the closest available approximation to what a live account would have experienced over that period.

### 8.2 How contamination is introduced

Contamination is not introduced by *observing* test set results. It is introduced by *acting on* them to improve performance. The distinction matters:

**Viewing the test set results is not contamination.** The researcher is allowed to know how the portfolio performed during the test period. Measurement is not bias.

**Making portfolio decisions in response to test set results is contamination.** If the researcher adjusts weights, removes strategies, or adds strategies in a way that is influenced by what happened in the test period, the test set is now partially a fitness function. Its performance estimate is no longer unbiased — it reflects, at least partly, the researcher's optimisation against it.

The contamination does not need to be deliberate. A researcher who sees poor test performance, develops a new strategy, and adds it to the portfolio has contaminated the test set even if they never explicitly used test period data in the new strategy's development — because the *motivation* to add the strategy came from the test period observation.

### 8.3 Contamination taxonomy

| Action | Contaminated? | Reason |
|---|---|---|
| Viewing test set results | No | Measurement, not optimisation |
| Adding a new strategy developed entirely on IS data, where the test period was not consulted at any stage | No | Research was independent of the test set |
| Applying a pre-specified removal rule (CUSUM, rolling Sharpe threshold) defined before any test results were viewed | No | Decision rule was locked in before observation; tests structural change, not outcome |
| Adjusting strategy weights based on test set performance | Yes | Optimising against the test set |
| Removing a strategy because test set performance is poor, without a pre-specified trigger | Yes | Using the test set as a fitness function |
| Adding a strategy that was selected, even partly, because it performed well during the test period | Yes | Selection bias from the test period |
| Changing portfolio composition after viewing test results, for any reason other than pre-specified rules | Yes | The observation motivated the change |

### 8.4 The pre-specified removal rule exception

The only legitimate reason to remove a strategy after viewing test set results is if a **pre-specified statistical rule** triggers. Two such rules are supported:

**CUSUM structural break test** (preferred): The CUSUM statistic on the strategy's test-period return stream exceeds the critical threshold. This tests whether the strategy's return distribution has structurally changed — not whether the outcome was good or bad. A strategy can fail CUSUM during a profitable period; it can pass CUSUM during a losing period. The test is about the signal's statistical properties, not the researcher's satisfaction with the result. Because the decision rule is defined independently of the outcome, removing a strategy on this basis is not contamination.

**Rolling Sharpe threshold**: The rolling Sharpe over a trailing window drops below a pre-specified floor for a pre-specified number of consecutive periods. This is slightly weaker than CUSUM because the thresholds are more arbitrary, but it is still a pre-specified rule if locked in before research begins.

Both rules must be registered on the portfolio object before the researcher views any test set results. Specifying a removal rule after viewing results — even if the rule itself is statistically reasonable — is contamination, because the motivation for choosing that rule came from the observation.

### 8.5 The practical standard

The ideal is to use the test set exactly once, after all research is complete, and never modify the portfolio afterwards. In practice this is not achievable: researchers add new strategies over time, markets change, and strategies die.

The practical standard is simpler: **treat the test set as a measurement instrument, not a fitness function.** Use it to answer "is this strategy alive?" via pre-specified rules. Do not use it to answer "how can I make this portfolio look better?" Even if performance is poor, the portfolio should not be modified to improve test set numbers. Poor performance on a genuinely unbiased test set is honest information about the strategy's edge — removing or adjusting based on it converts that honest information into a biased one.

### 8.6 Platform enforcement

The platform cannot fully prevent contamination — it is ultimately a research discipline problem, not a technical one. What the platform can do:

- **Record when test results are first viewed** (timestamp per researcher per project). Flag any portfolio modification occurring after this timestamp as "potentially contaminated" in the audit trail.
- **Require pre-registration of removal rules** before test results can be viewed. The removal rules are stored on the portfolio object with a creation timestamp.
- **Display a contamination warning** when a researcher attempts to add or remove a strategy after viewing test results, unless the action is triggered by a pre-registered rule.
- **Label strategy additions post-test-view** with a badge in the portfolio history indicating the addition occurred after test results were observed. This does not prevent the addition but makes the contamination visible in the audit trail.

The goal is not to make contamination impossible but to make it visible. A researcher who contaminates their test set knowingly has made a research decision; the platform records it so the audit trail is honest.

---

## 9. Related documents

| Document | Relevance |
|----------|-----------|
| `data_flow.md` | Research phases, portfolio snapshots, discipline around test zones. |
| `technical_design.md` | Zone entity (§6.3), REST routes, worker flow (all planned). |
| `ui_ux.md` | Current local research UI (Flask); hosted zone-management pages are planned. |

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
