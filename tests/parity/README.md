# Research Parity Harness (WP-1)

The **gate** that proves `feature_research` and `portfolio_research` numeric
outputs are unchanged across the repo cull / folder restructure (WP-8) and the
later NautilusTrader migration (WP-2+).

> **A parity failure BLOCKS the change.** Snapshots are golden. They are *only*
> regenerated for an **intentional, reviewed** behaviour change — never to "make
> the test green". If a refactor is supposed to be behaviour-neutral and parity
> fails, the refactor is wrong, not the snapshot.

## What it does

1. Runs two representative, **pinned** research pipelines on repo-backed data
   (`data/ohlc_data`, `USE_CACHE=True`).
2. Serializes their numeric outputs to **golden snapshots** committed under
   `snapshots/`.
3. Re-runs after any change and **diffs against the snapshots** with tight
   tolerances, failing loudly (with worst-offender report) on material drift.

## Pinned cases

| Case | Entry point | Config (pinned) | Snapshotted numbers |
|------|-------------|-----------------|---------------------|
| `feature_research/oos_si_sma_regime` | `feature_research.pipelines.oos.run_oos_pipeline` | `feature_research.config.load_config()` — SI daily **SMA(252)** regime signal, signed-signal LONG_SHORT, train 2000–2018 / val 2019–2022, seed=42 | `WalkforwardRunReport` frames (`folds_df`, `fold_scores_df`, `selection_summary_df`, `portfolio_results_df`, `fold_signal_metrics_df`), `aggregate_oos_returns` series, headline metrics |
| `portfolio_research/portfolio_test_default` | `portfolio_research.pipelines.portfolio_test.run_single_phase_for_prop_firm` (phase=`test`, `emit_tearsheets=False`, `run_purpose='metrics_only'`) | `portfolio_research.config.load_config()` — ES/NQ/GC/CL, prop-vault D/W/M ensembles, train/val/test windows, τ=0.07, `hierarchy_equal` | per-`(ticker, datetime)` `position_fraction`, combined strategy returns series, combined baseline returns series, headline metrics (strategy + baseline) |

The feature_research case is the single, simple, deterministic single-feature
IS+OOS run the spec asks for (the IS exploration sweep is only 2 param combos;
the OOS evaluation is fully deterministic). The portfolio_research case is the
default `load_config()` test run.

Both runs redirect every artifact write to a temp `output_root`, so the repo
working tree is never polluted. Bias-node caches are populated on demand by the
pipeline's own cache preflight (first run is slower while caches build).

### Headline metrics

Computed from the relevant returns series via the **project metrics layer**
(`feature_selection.validation.objective_metrics` → `quantfoundry_core.metrics`,
and `metrics.risk.drawdown`) — not a re-implementation — at 252 trading-day
annualization:

`sharpe`, `sortino`, `max_drawdown`, `calmar`, `total_return`, plus
`n_obs`, `mean_return`, `std_return` for context.

## Commands (Windows, repo root)

Use the repo venv interpreter directly (per `CLAUDE.md`); never create a venv. If
your venv folder is `venv` rather than `.venv`, swap the prefix.

```powershell
# VERIFY (the gate) — diff current outputs against golden snapshots:
.\.venv\Scripts\python.exe -m pytest tests/parity

# REGEN — (re)write golden snapshots (intentional, reviewed changes only):
.\.venv\Scripts\python.exe -m pytest tests/parity -m regen

# Restrict to one case:
.\.venv\Scripts\python.exe -m pytest tests/parity -k feature_research
.\.venv\Scripts\python.exe -m pytest tests/parity -k portfolio_research

# One-line convenience wrapper (same thing):
.\.venv\Scripts\python.exe scripts\run_parity.py            # verify
.\.venv\Scripts\python.exe scripts\run_parity.py --regen    # regenerate
.\.venv\Scripts\python.exe scripts\run_parity.py -k feature_research
```

A bare `pytest tests/parity` is a **pure verify run**: the snapshot-writing tests
are marked `regen` and auto-skipped unless you pass `-m regen`, so a normal run
can never silently overwrite the golden files.

## Tolerances

Default: **`rtol=1e-8`, `atol=1e-10`** (`tests/parity/_diff.py::Tolerance`).

These are deliberately tight. The data-layer migration (WP-2) only swaps the
*source* of candles, not any math, so results are expected to be **exactly**
identical. The cull/restructure (WP-8) just moves files — also exactly identical.
A looser tolerance profile may be passed explicitly to the diff helpers for a
deliberate, documented behaviour change, but the default gate stays tight.

Comparison rules (see `_diff.py`):
- Values are compared, not formatting: frames/series are sorted and aligned
  before diffing, so column/row ordering is never load-bearing.
- `NaN == NaN` (both-NaN is not a diff); a NaN appearing on only one side is.
- Index/column structure mismatches are reported as structural failures.
- Non-numeric columns are compared for exact equality.
- Failures print the **worst offenders** (largest absolute diff first).

## Determinism

- Every reachable seed is pinned: the configs pin `random_seed=42`
  (permutation/robustness); the session fixture pins `numpy` global state and
  `PYTHONHASHSEED=0`.
- Snapshots are written sorted/aligned so dict ordering and row order can't flap.
- Parquet round-trips are byte-stable and lossless (verified).
- Re-running verify twice is stable.

## Snapshot files

Written under `snapshots/`:

```
feature_research/oos_si_sma_regime__folds_df.parquet
feature_research/oos_si_sma_regime__fold_scores_df.parquet
feature_research/oos_si_sma_regime__selection_summary_df.parquet
feature_research/oos_si_sma_regime__portfolio_results_df.parquet
feature_research/oos_si_sma_regime__fold_signal_metrics_df.parquet
feature_research/oos_si_sma_regime__aggregate_oos_returns.parquet
feature_research/oos_si_sma_regime__metrics.json
portfolio_research/portfolio_test_default__test__positions.parquet
portfolio_research/portfolio_test_default__test__strategy_returns.parquet
portfolio_research/portfolio_test_default__test__baseline_returns.parquet
portfolio_research/portfolio_test_default__test__metrics.json
```

## Skips (cache / data / environment)

The tests **skip with an explicit message** (never fail) when they cannot run:

- **Missing data** — `data/ohlc_data` absent/empty.
- **Missing snapshot** — verify has no golden file yet (regen first on a
  known-good build).
- **Pipeline import failure** — the research entrypoints can't be imported. On
  this branch the pipelines require a `quantfoundry_core` that exports
  `ParamPerturbationSpec` from `quantfoundry_core.robustness`; an older installed
  package makes `feature_research.config` (and transitively
  `portfolio_research.config`) fail to import. The skip message includes the
  exact `ImportError`.

> ⚠️ **Baselines were NOT captured in the build environment** because of the
> `quantfoundry_core` import blocker above — the pipelines could not be executed,
> so no golden numbers were fabricated. The harness, the diff/report utility, the
> regen path, the runner, and this doc are complete and self-tested. To capture
> baselines: fix the environment (upgrade/point the venv at a `quantfoundry_core`
> that exports `ParamPerturbationSpec` from `quantfoundry_core.robustness`), then
> run the **REGEN** command above and commit the `snapshots/` files. WP-2 cannot
> start until verify is green on `main`.

## Acceptance (gate definition)

- `pytest tests/parity` passes on unmodified `main` once snapshots exist.
- Re-running twice is stable (seeds pinned, sorted snapshots).
- Runs in CI-feasible time on the pinned subset (record the runtime when you
  first regen).
- A parity failure blocks the change; snapshots are not silently regenerated.
