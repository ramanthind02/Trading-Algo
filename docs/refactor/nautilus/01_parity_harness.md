# WP-1 — Research Parity Harness (the gate for everything)

> **Build this FIRST and capture baselines BEFORE touching any other file.**
> Every later work package is gated on this harness showing no material drift in
> `feature_research` and `portfolio_research` outputs.

## Objective

Create a reproducible harness that:

1. Runs representative `feature_research` and `portfolio_research` pipelines on a
   pinned dataset/date range/config.
2. Serializes their numeric outputs to **golden snapshots** committed to the repo.
3. Re-runs after any change and **diffs against the golden snapshots** with explicit
   tolerances, failing loudly on material drift.

This is what lets us "cull aggressively" without breaking research (invariants
I1/I2 in [00_overview.md](00_overview.md)).

## Why this is first

The data-layer migration (WP-2) swaps how candles are produced. The only credible
proof that "candles are identical and therefore results are identical" is a snapshot
captured *before* the swap and re-checked *after*. No harness ⇒ no safe cull.

## Scope

**In scope:** a test/CLI harness, golden snapshot files, a diff/report utility, and
documentation of the pinned configs.

**Out of scope:** changing any pipeline behaviour. This WP must be behaviour-neutral.

## Current-state map (confirm with codegraph before coding)

Research entrypoints to snapshot:

- `feature_research`
  - In-sample: `feature_research/in_sample/run_is.py` (`python -m feature_research.in_sample.run_is`)
  - OOS pipeline: `feature_research/pipelines/oos.py::run_oos_pipeline` / `run_oos_pipeline_with_bundle`
  - Config: `feature_research/config.py`
  - Table exports: `feature_research/research_table_exports.py` (uses
    `calculate_strategy_returns_from_positions`, `compute_rolling_sharpe`)
  - Robustness: `feature_research/validation/robustness_runner.py`,
    `feature_research/pipelines/{robustness,param_perturbation,permutation}.py`
  - Portfolio-addition gate: `feature_research/portfolio_addition/gate_runner.py`
- `portfolio_research`
  - `portfolio_research/run_portfolio_test.py::run_portfolio_test`
    → `portfolio_research/pipelines/portfolio_test.py`
  - Config: `portfolio_research/config.py::load_config` /
    `load_prop_firm_portfolio_research_config`
  - Weight-layer CV: `portfolio_research/weight_layer_cv.py`
  - Vault correlation: `portfolio_research/run_feature_vault_correlation.py`
  - Holdout robustness: `portfolio_research/holdout/`

Return convention under test (must be preserved):
`ensemble/portfolio_impl/portfolio_tester.py::calculate_strategy_returns_from_positions`
— default `instrument_return_kind='log_intraday'` (enter `open[t+1]`, exit `close[t+1]`,
positions shifted forward one bar). Also exercise `log` and `simple` where pipelines use them.

Metrics layer (the comparison surface): `quantfoundry_core.metrics` /
`.robustness` via `feature_selection/validation/objective_metrics.py` and
`feature_research/research_table_exports.py`.

## Target design

```
tests/parity/                      # NEW (integration-style, cache-backed)
  conftest.py                      # pins dataset, date range, tickers, configs
  snapshots/                       # golden artifacts (committed)
    feature_research/<case>.parquet|json
    portfolio_research/<case>.parquet|json
  test_feature_research_parity.py
  test_portfolio_research_parity.py
  _diff.py                         # numeric diff + tolerance + human report
  README.md                        # how to (re)generate and interpret
```

Harness principles:

- **Deterministic inputs.** Pin tickers, `start`/`end`, vault ensemble set, seeds
  (`random_seed`, `n_bootstrap`), and `USE_CACHE=True` against repo-backed data
  (`data/ohlc_data`). If cache is missing, populate via
  `CacheManager.populate_cache(...)` or skip with an explicit message (per repo
  testing-boundary rules in `CLAUDE.md`).
- **Snapshot the numbers that matter**, not formatting: per-(ticker,datetime)
  `position_fraction`, the strategy **returns series**, and the headline
  `quantfoundry_core` metrics (Sharpe, Sortino, max DD, Calmar, total return,
  rolling-Sharpe summary). For robustness pipelines, snapshot the report scalars.
- **Tolerances:** default `rtol=1e-8, atol=1e-10` for the data-layer phase (WP-2 is
  expected to be *exactly* identical because only the candle source changes). Provide
  a separate, looser profile only if a deliberate, documented change is introduced.
- **One command to regenerate**, one to verify:
  - `.\.venv\Scripts\python.exe -m pytest tests/parity -m regen` (writes snapshots)
  - `.\.venv\Scripts\python.exe -m pytest tests/parity` (verifies)

### What to snapshot per pipeline (minimum)

| Pipeline | Artifact(s) | Key columns / scalars |
|----------|-------------|-----------------------|
| feature_research IS/OOS single-feature | positions, returns, metrics | `position_fraction`; daily `strategy_return`; Sharpe/Sortino/maxDD/Calmar |
| feature_research robustness/perturbation | report scalars | bootstrap CIs, p-values, stability flags |
| feature_research portfolio-addition gate | gate report | hurdle, IDM improvement, weight assessment, pass/fail |
| portfolio_research portfolio_test | per-phase returns + combined | train/val/test `strategy_return` series; per-phase metrics; baseline returns |
| portfolio_research weight_layer_cv | fold scalars | per-fold SR/Calmar; selected weights |

## Tasks

1. Inventory the exact configs the team considers representative (ask if unclear);
   encode them as fixtures. Cover at least: one single-feature IS+OOS case, one
   robustness case, the default `portfolio_research.load_config()` test run, and the
   prop-firm profile config.
2. Implement `_diff.py` (DataFrame + scalar diff with tolerances; emits a readable
   report of the worst offenders).
3. Implement the two parity test modules with a `regen` marker path.
4. Generate and **commit** golden snapshots from the current `main` behaviour.
5. Write `tests/parity/README.md` documenting regeneration, tolerances, and the rule
   that snapshots are only regenerated for *intentional, reviewed* behaviour changes.
6. Wire a single convenience entry (e.g. `scripts/run_parity.py` or a `pytest` alias)
   so other WPs can run it in one line.

## Acceptance criteria (gate definition)

- `pytest tests/parity` passes on unmodified `main` (snapshots reproduce).
- Re-running twice is bit-stable (no nondeterminism leaks; seeds pinned).
- The harness runs in CI-feasible time on the pinned subset (document the runtime).
- README clearly states: **a parity failure blocks the change**; snapshots are not
  silently regenerated.

## Risks / notes

- **Nondeterminism**: bootstrap/seed leakage, dict ordering, multi-thread reductions.
  Pin every seed; sort before snapshotting.
- **Cache dependence**: ensure the harness either uses committed cache or rebuilds
  deterministically; document which.
- **Scope creep**: do not "improve" any pipeline here. Behaviour-neutral only.

## Subagent instructions

- Start with `codegraph_explore "run_is run_oos_pipeline run_portfolio_test load_config calculate_strategy_returns_from_positions"` to confirm signatures/paths.
- Do **not** modify any file under the Frozen list in [00_overview.md](00_overview.md) §3.
- Hand back: the harness, committed snapshots, runtime numbers, and the exact commands
  to run it. WP-2 cannot start until this gate is green on `main`.
