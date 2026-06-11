# Research report — what the agent analyzes, and how it reports

The deliverable of a research run is **not a dashboard and not a pass/fail score** — it is a
**memo written by the agent** after analyzing the raw results holistically. This doc is the
contract for that memo: the questions the agent must answer, the raw data the pipeline must emit
so it can, and the rule that keeps it honest. Pairs with [[Strategy_research/strategy_spec]] (what
to run) and [[Strategy_research/execution_architecture]] (how it fills).

> [!important]
> **The pipeline emits raw data. The agent renders judgment. No numerical gate decides anything.**
> A pure numerical pipeline of pass/fail tests is exactly what we are *not* building — it
> data-mines and it misses the holistic picture. The agent reads the full evidence and reasons
> like an analyst; the human approves or rejects. Neither step is a threshold.

---

## 1. Three layers — and the one trust rule

| Layer | Job | Produces |
|-------|-----|----------|
| **Pipeline** (deterministic) | compute every number from returns/forecasts/candles | raw-data artifacts (CSV/JSON) + a run manifest |
| **Agent** (analyst) | read the raw data, reason holistically, write the memo | the research memo (markdown) |
| **Human** | read the memo, drill on demand, decide | approve / reject |

**The trust rule: code computes, the agent judges.** The agent never estimates a Sharpe, a
correlation, or a p-value by eye — those are computed by the pipeline and read from JSON. What the
agent does that code cannot is **interpret**: is this overfit, is the rationale sound, is the
plateau wide enough, is the IS→val drop within noise. Numbers are trustworthy because they are
computed; conclusions are valuable because they are reasoned. This is the line that lets us trust
an agent-driven analysis.

Existing automated thresholds (e.g. `rank_correlation_floor=0.2`,
`pairwise_corr_flag_threshold=0.75`, `delta_sr_threshold`) are **demoted to advisory reference
lines** in the raw data — the agent sees "pairwise corr 0.78 (ref 0.75)" and weighs it; the old
boolean pass/fail no longer decides. See the no-auto-gates stance in
[[Strategy_research/strategy_spec]] § Philosophy.

---

## 2. The analytical lenses

These are the questions you already ask in feature research, made explicit. For each: the **raw
data** the pipeline emits (most already exists today), and the **holistic judgment** the agent
renders. No lens has a threshold that auto-decides.

### A. Economic rationale & simplicity (the "why")
- **Raw data:** the `hypothesis` text from the `StrategySpec`; behavioral diagnostics — when it
  trades, holding-period distribution, long/short balance, per-regime and per-instrument behavior,
  turnover.
- **Agent judges:** does the realized behavior match the *claimed mechanism*, or does the edge
  look incidental / data-mined? Is the rule simple (few params, legible logic) or a fragile
  contraption? A strong economic story with simple logic is the single best guard against overfit.

### B. Parameter robustness — plateau & sensitivity
- **Raw data:** the full param-grid → metric surface (every combo in the ≤300 grid with its
  metrics); parameter-perturbation results (`research/feature/pipelines/param_perturbation.py`,
  `ParamSensitivityConfig` / `ParamPerturbationSpec`); the exploration `param_sensitivity.csv`.
- **Agent judges:** do the **neighbors of the best combo** also perform well (a broad plateau), or
  is the winner a lone spike (overfit)? How fast does performance decay as params move? A wide,
  smooth plateau is robust; a sharp peak is not.

### C. Overfit
- **Raw data:** in-sample vs validation vs test metrics side by side; permutation null
  distributions (`PermutationResearchConfig` in-sample vector-shuffle, `RobustnessResearchConfig`
  return-shuffle nulls) with the real statistic's percentile; the plateau width from (B); the
  number of combos searched.
- **Agent judges:** is the in-sample result distinguishable from the permutation null, or could
  noise produce it? Does the edge survive out of sample? Does the breadth of the search (combos
  tried) inflate the best result? This is a *weighing*, not a p-value cutoff.

### D. In-sample → validation persistence
- **Raw data:** rank correlation between the IS metric and the validation metric across the param
  grid (the validation rank-correlation scatter; `RankCorrelationScope`, `rank_correlation_floor`
  as a reference line).
- **Agent judges:** do the combos that looked best in-sample **stay** best in validation, or does
  the ranking scramble? High persistence means the IS signal is real; a scrambled ranking means IS
  performance was luck.

### E. Degradation
- **Raw data:** train → validation → test metric deltas; rolling-Sharpe stability series; CUSUM
  and equity-band diagnostics (`ValidationRobustnessConfig`); holdout monitoring
  ([[Portfolio_research/holdout]]).
- **Agent judges:** has the edge weakened over time or across the walk-forward boundary? Is any
  decay gradual (regime drift) or a cliff (structural break)? Is the validation/test result a fair
  draw or a degraded one?

### F. Diversification / correlation to the book
- **Raw data:** candidate-vs-peers validation-window forecast correlations (per ticker then
  aggregated — `research/feature/inclusion_gates.py`); feature-vault correlation
  (`run_feature_vault_correlation`); with/without-candidate portfolio metrics.
- **Agent judges:** does the candidate **add something the book does not already have**, or is it a
  near-duplicate of an existing sleeve? A mediocre-standalone but uncorrelated strategy can be
  worth more than a strong-but-redundant one. Which sleeve does it belong to?

### Cross-cutting: the two-bounds execution check
- For overnight strategies, the c2c-vs-rollover-bounded-o2c decomposition from
  [[Strategy_research/execution_architecture]] (§ swap avoidance): how much of the edge is the
  overnight gap, and does it survive realistic CFD fills?

---

## 3. The memo — phase-structured

The agent writes the memo as the run progresses, one section per phase, then a top-level verdict.

```
# <strategy name> — research memo

## Verdict (agent's read)
<2-4 sentences: what this is, whether it looks real, the main concern, recommendation to the human>

## In-sample (exploration)
- Economic rationale & simplicity (A)
- Parameter plateau & sensitivity (B)
- Overfit — in-sample permutation (C, first pass)

## Validation
- IS → validation persistence (D)
- Degradation vs train (E)
- Diversification / correlation to the book (F)

## Portfolio addition
- With/without book metrics, sleeve fit (F)

## Holdout (test — scored once, never used for selection)
- Final degradation check, test vs validation (C, E)

## Concerns
<ranked list of everything that gives the agent pause, each tied to its evidence>

## Artifacts
<links to the ≤2 worth eyeballing: the combined tearsheet, one sensitivity heatmap>
```

The agent links **only the one or two artifacts that matter** — not a catalog. Everything else is
on disk for drill-down if the human asks. Files are files: VS Code renders the PNG/HTML directly,
no server.

---

## 4. Raw data the pipeline must emit (per phase)

Most of this already exists; the work is consolidating it into one agent-readable bundle + the run
manifest (today's `research/feature/ui/workspace_manifest.py`), not new computation.

| Phase | Artifacts (CSV/JSON) |
|-------|----------------------|
| In-sample | param-grid metric surface; `param_sensitivity.csv`; perturbation results; in-sample permutation null |
| Validation | per-window metrics (train/val); IS↔val rank-correlation; rolling-Sharpe / CUSUM / band diagnostics; behavioral diagnostics |
| Portfolio addition | with/without-candidate metrics; candidate↔peer correlation table; assigned weight |
| Holdout | test-fold metrics; test-vs-validation deltas; holdout monitoring report |
| Always | equity/returns series; a **manifest** indexing every artifact with metadata |

The agent reads the **manifest** to know what exists and where, then reads the specific
CSV/JSON it needs per lens. The deterministic `analyze` step that assembles this bundle is the
agent's only computational dependency — it must not require the Flask app.

---

## 5. What changed

- **The auto accept/reject gates** — `PortfolioAdditionGateConfig` and friends still *compute*
  their diagnostic numbers (delta-SR, ulcer, stress DD) as raw data, but their boolean verdict is
  demoted to an advisory flag. The human decides via the frontend results views or via a memo.
- **The old Flask dashboard** is superseded. The current browser is the React/Vite frontend app
  (`frontend/web/`) which serves Plotly-based results views (plateau / equity / grid / headline).
  The artifact + manifest **generators** (`research/feature/ui/`) are retained.
- **The `/research` memo loop** (agent orchestrator) is superseded by the frontend app as the
  primary workflow. The skill remains available for agent-driven analysis.

## Open items

- **Standalone `analyze` bundle** — a deterministic step that consolidates phase artifacts +
  manifest into one agent-readable `bundle.json`, decoupled from any web server, would still be
  useful for fully agent-driven analysis sessions. Not yet built.
- **Memo template** — a structured memo template + analyst subagent that fills it from the bundle.
  Part of the `/research` skill design; not a dependency of the frontend workflow.

> _Authored 2026-06-06. Updated 2026-06-07 to reflect frontend app replacing the Flask dashboard._

> _Verified against current code via CodeGraph on 2026-06-07._
