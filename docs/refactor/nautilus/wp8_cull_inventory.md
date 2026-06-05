# WP-8 Phase 1 — Cull Inventory (non-destructive discovery)

> **Scope:** Evidence-backed delete-list for WP-8 Phase 1. **Nothing was deleted or
> edited.** Every SAFE-DELETE row has BOTH a clean `codegraph_callers`/`codegraph_impact`
> result AND a clean repo-wide grep for dynamic/string/`python -m`/config/JSON refs.
> Branch: `nautilus-refactor`. Discovery date: 2026-06-04.

## Verdict summary

| Candidate | Classification | Evidence (callers / grep) | Files to remove |
|---|---|---|---|
| `feature_selection/base_models/base_model.py` legacy stubs (`BinningModelBase`, `ContinuousBinningModel`, `RuleBasedModel`) | **SAFE-DELETE** | codegraph: 0 callers each. Grep: only refs are the file itself, the `__getattr__` shim in `base_models/__init__.py`, and the test `test_model_naming_cutover.py` (which asserts they raise). No JSON control file / vault feature JSON references `legacy_binning`/`ContinuousBinning`/`RuleBased`/`n_bins`. | `feature_selection/base_models/base_model.py`; the legacy `__getattr__` block in `feature_selection/base_models/__init__.py`; the `ContinuousBinningModel`/`RuleBasedModel` import + `test_legacy_models_are_exported_and_raise_on_init` test in `tests/unit-tests/feature_selection/base_models/test_model_naming_cutover.py` |
| `feature_selection/base_models/feature_base_model.py` (`BaseModel`) | **KEEP** | Live signed-signal path. Imported by `ensemble/vault/manager.py`, `ensemble/ensemble_utils.py` (`create_base_model_from_config`, 2 callers), `feature_research/{ui/vault_save,save_feature_to_vault,inclusion_gates}.py`. | — |
| `create_base_model_from_config` (`ensemble/ensemble_utils.py`) | **KEEP** | 2 live callers: `diversified_ensemble._initialize_from_control_file`, `vault/manager.load_feature_base_models`. The active replacement path. | — |
| `metrics/__init__.py` barrel re-exports | **NEEDS-HUMAN (partial)** | **Zero** `from metrics import ...` callers anywhere (grep for `from metrics\b(?!\.)` / `import metrics$` = no matches). The barrel (lines 10–26 re-exporting `metric_sharpe`, `metric_sortino`, equity/risk helpers) is dead. BUT the `metrics` package itself must stay (live submodules below). Recommend gutting the re-exports, not deleting the file. | Re-export body of `metrics/__init__.py` (keep a minimal package docstring); do **not** delete the file |
| `metrics/plotting/` (`graphing/quantstats_reports.py` + the two `__init__` shells) | **KEEP** | `generate_tearsheet` / `compute_performance_report` imported directly via `metrics.plotting.graphing.quantstats_reports` by 8+ live modules (`ensemble/portfolio_impl/portfolio_tester.py`, `feature_research/inclusion_gates.py`, `portfolio_research/{pipelines/portfolio_test,holdout/rolling_eval,holdout/monitoring_tearsheets,futures_sim}.py`, `utils/evaluation/walkforward/runner.py`) + tests. | — |
| `metrics/risk/drawdown.py` — `max_drawdown`, `drawdown_series` | **KEEP** | `max_drawdown` imported live by `portfolio_research/weight_layer_cv.py:23`. `drawdown_series` re-exported + used internally. | — |
| `metrics/risk/drawdown.py` — `check_drawdown_breach`, `trailing_drawdown_threshold`, `daily_drawdown` | **NEEDS-HUMAN (tied to prop_firm_simulator)** | Only external consumer is `utils/simulation/prop_firm_simulator/challenge_engine.py` (itself SAFE-DELETE) + `test_prop_firm_simulator.py`. Become dead once the simulator is culled. | These 3 functions (after the simulator is removed) — leave the rest of `drawdown.py` |
| `metrics/equity/` (`cumulative.py`, `tracking.py`, `__init__.py`) — `cumulative_returns`, `equity_curve`, `equity_peak` | **NEEDS-HUMAN** | No external importers (only used inside `metrics/risk/drawdown.py` and re-exported by the dead barrel). `metrics.risk` depends on `cumulative_returns`/`equity_peak`, so they cannot be deleted while `drawdown.py` lives. `equity_curve` (the alias) has zero users. | Optionally `metrics/equity/tracking.py`'s `equity_curve` alias is unused, but `cumulative_returns`+`equity_peak` are load-bearing for `drawdown.py` — keep the package |
| `metrics/performance/` | **N/A (does not exist)** | Glob `metrics/**/*.py` shows no `performance/` dir. Ledger assumption is stale. | — |
| `utils/simulation/prop_firm_simulator/` (whole package) | **SAFE-DELETE** | codegraph `PropFirmChallengeSimulator` real callers = its own package + `scripts/demo_prop_firm_simulator.py` + `tests/unit-tests/utils/test_prop_firm_simulator.py` only (the long impact list is file-level `:1` noise). Grep: no live app/research import. Docs (`docs/library/Testing/prop_firms.md:16`) call it "legacy", superseded by `prop_firms/`. `prop_firms/` defines its own `SimulationConfig`/`ChallengeRules`/`ChallengeCosts` (distinct classes). WP-5 ledger confirms `prop_firms/` stays. | `utils/simulation/prop_firm_simulator/{__init__,challenge_engine,data_structures,path_generators,statistics}.py`; `utils/simulation/__init__.py` (1-line, becomes empty dir); `scripts/demo_prop_firm_simulator.py`; `tests/unit-tests/utils/test_prop_firm_simulator.py` |
| `research/` (top-level) | **SAFE-DELETE (with note)** | `research/__init__.py` is a 1-line comment. **Zero** Python importers (`import research` / `from research import` = no matches; the lone grep hit is unrelated prose in `docs/nautilustrader/concepts/overview.md`). Contains only `ib_test.ipynb` + stale `__pycache__`. **NOTE:** Phase 2 plans to *repurpose* `research/` as the research umbrella — so "delete" may mean "empty it" rather than remove the dir. | `research/__init__.py`, `research/__pycache__/` (and decide on `research/ib_test.ipynb` — a notebook, NEEDS-HUMAN) |
| `utils/evaluation/holdout_robustness.py` | **KEEP** | 6 codegraph callers (live: `feature_research/validation/robustness_runner.py`, `portfolio_research/holdout/{portfolio_holdout_runner,strategy_monitoring}.py`; + `feature_research/shared/holdout_robustness_config.py` uses `HoldoutRobustnessConfig`). It is **repo-specific glue** (config dataclass + full pipeline orchestration over `quantfoundry_core.robustness`), not a thin pass-through. | — |
| Other `utils/evaluation/*` (`walkforward/*`, `permutation_test/*`) | **KEEP** | Live: `walkforward/runner.py` imports tearsheet; `walkforward` + `permutation_test` are used across research pipelines. Not investigated for sub-symbol dead code (out of ledger scope) — none flagged. | — |
| `scripts/_bootstrap.py` | **KEEP (not a probe)** | Despite the `_` prefix, it is a **shared bootstrap helper** (`ensure_project_root_on_path`, `.env` loader). Imported by `data_platform/providers/mt5/scraper.py:56` and ~15 scripts (`enigma_*_forecast.py`, `demo_ib_data_fetch.py`, `build_nyse_holiday_events.py`, `scrape_fomc_dates.py`, `diagnose_calendar_ensemble.py`, etc.). Referenced in `.env.example`. | — |
| `scripts/_check_catalog.py` | **SAFE-DELETE** | Pure top-level `__main__` diagnostic (reads `data/instruments/catalog.parquet`, prints). No functions; zero importers; no doc/config/`python -m` references. | `scripts/_check_catalog.py` |
| `scripts/_check_ndx_data.py` | **NEEDS-HUMAN** | Standalone NDX/catalog sanity probe, no importers. **BUT** cited as a starting point in active plan `docs/refactor/nautilus/03_backtest_validation_lane.md:124`. Confirm with the WP-3 owner before deleting. | (`scripts/_check_ndx_data.py` — pending human OK) |
| `scripts/_etf_position_sizing.py` | **SAFE-DELETE** | Standalone MT5 ETF sizing analysis (`__main__`, live `mt5.initialize()`). No importers; no doc/config refs. | `scripts/_etf_position_sizing.py` |
| `scripts/_mt5_m1_sizing.py` | **NEEDS-HUMAN** | Standalone storage-sizing calculator, no importers. **BUT** referenced as the documented methodology source in `docs/library/Data/darwinex_universe.md:206`. Either keep (→ `scripts/dev/`) or delete + drop the doc line. | (`scripts/_mt5_m1_sizing.py` — pending human OK) |
| `scripts/_tick_download_schedule.py` | **SAFE-DELETE** | Standalone tick-download time estimate (`__main__`). No importers; no doc/config refs. | `scripts/_tick_download_schedule.py` |
| `scripts/_mt5_*` (other than `_mt5_m1_sizing`) | **N/A (do not exist)** | Only `scripts/_mt5_m1_sizing.py` matches `scripts/_mt5*.py`. The real MT5 probes live in `data_platform/providers/mt5/probes/` (kept — see below). `.env.example:39` mentions a `scripts/_mt5_discovery.py` that **does not exist** (stale comment). | — |
| `data_platform/providers/mt5/probes/*` | **KEEP** | Documented dev/diagnostic tooling, runnable via `python -m data_platform.providers.mt5.probes.<name>` (README + `docs/library/Data/{mt5_data_scraper,darwinex_universe}.md`). Namespaced, not in the top-level `scripts/_*` cull scope. | — |
| QF-duplicated metric/robustness/prop_firm code | **None found beyond the above** | `holdout_robustness.py`, `objective_metrics.py`, `prop_firms/`, `portfolio_research/holdout/*` all already delegate to `quantfoundry_core` (53 files import it). The only local reimplementation that duplicates QF's prop-firm sim is `utils/simulation/prop_firm_simulator/` (already SAFE-DELETE). | — |

---

## SAFE-TO-DELETE FILE LIST (ready for the destructive round)

### Whole files (zero live callers, clean dynamic-ref grep)

```
utils/simulation/prop_firm_simulator/__init__.py
utils/simulation/prop_firm_simulator/challenge_engine.py
utils/simulation/prop_firm_simulator/data_structures.py
utils/simulation/prop_firm_simulator/path_generators.py
utils/simulation/prop_firm_simulator/statistics.py
utils/simulation/__init__.py                              # becomes an empty package after the above
scripts/demo_prop_firm_simulator.py                       # only consumer of the simulator
tests/unit-tests/utils/test_prop_firm_simulator.py        # only test of the simulator (and the 3 dead drawdown fns)

feature_selection/base_models/base_model.py               # BinningModelBase/ContinuousBinningModel/RuleBasedModel stubs

scripts/_check_catalog.py
scripts/_etf_position_sizing.py
scripts/_tick_download_schedule.py

research/__init__.py                                       # see Phase-2 repurpose note; remove or empty
```

### Partial edits (dead code inside a file that must stay)

```
feature_selection/base_models/__init__.py
    - remove the `import importlib` + legacy `__getattr__` block (lines ~5,12-17);
      keep `from feature_selection.base_models.feature_base_model import BaseModel` and `__all__ = ["BaseModel"]`.

tests/unit-tests/feature_selection/base_models/test_model_naming_cutover.py
    - remove `from feature_selection.base_models import ContinuousBinningModel, RuleBasedModel`
      and `test_legacy_models_are_exported_and_raise_on_init`;
      keep `build_member_model_name` + its two tests (independent of the stubs).

metrics/__init__.py
    - the barrel re-exports (lines 10-26) have ZERO importers; gut them, keep the package
      + docstring. (Do NOT delete the file — `metrics.plotting`/`metrics.risk` are live.)

metrics/risk/drawdown.py  (ONLY after the prop_firm_simulator is removed)
    - `check_drawdown_breach`, `trailing_drawdown_threshold`, `daily_drawdown` become dead
      once `challenge_engine.py` + `test_prop_firm_simulator.py` are gone.
      `max_drawdown` (used by portfolio_research/weight_layer_cv.py) and `drawdown_series` STAY.

metrics/risk/__init__.py
    - drop `trailing_drawdown_threshold`, `daily_drawdown`, `check_drawdown_breach` from the
      imports/__all__ in the same batch as the drawdown.py edit above.
```

### Documentation/config follow-ups (not code, but must be repointed/removed)

```
.env.example:39                       # mentions non-existent scripts/_mt5_discovery.py — stale, fix/remove
docs/library/Data/darwinex_universe.md:206   # cites scripts/_mt5_m1_sizing.py — update if that script is culled
docs/refactor/nautilus/03_backtest_validation_lane.md:124  # cites scripts/_check_ndx_data.py — update if culled
CLAUDE.md:106                         # "ABC is BinningModelBase" — stale once stubs deleted (Phase 3 docs fix)
docs/library/* base-model & prop-firm pages reference the stubs/legacy sim as "retired/legacy" — already accurate
```

---

## NEEDS-HUMAN (do NOT auto-cull)

1. **`scripts/_check_ndx_data.py`** — no importers, but referenced by active plan `03_backtest_validation_lane.md`. Confirm WP-3 doesn't still want it.
2. **`scripts/_mt5_m1_sizing.py`** — no importers, but it is the documented sizing-methodology source in `darwinex_universe.md`. Keep (→ `scripts/dev/`) or delete + drop the doc line.
3. **`research/ib_test.ipynb`** — a notebook (not graphed by codegraph). The `research/` dir is slated for Phase-2 *repurpose*, so don't blindly `rm -rf research/`.
4. **`metrics/risk` drawdown trio + `metrics/equity` reach** — only dead *after* the prop_firm_simulator cull lands; sequence the edits (simulator first, then the drawdown.py trim) and re-run parity between.
5. **`metrics/__init__.py` barrel** — dead as a barrel but the package is live; gut vs. delete is a judgment call (recommend gut).

---

## NEW DEAD CODE FOUND (beyond the ledger)

- **`scripts/demo_prop_firm_simulator.py`** — sole runnable consumer of the legacy simulator; dies with it (ledger only listed the package).
- **`metrics/risk/drawdown.py` trio** (`check_drawdown_breach`, `trailing_drawdown_threshold`, `daily_drawdown`) — their ONLY consumer is the legacy simulator; becomes dead in the same batch. (Ledger framed metrics as "shell re-exports" only.)
- **`metrics/equity/tracking.py` `equity_curve` alias** — zero callers (the live path uses `cumulative_returns`/`equity_peak`).
- **`metrics/__init__.py` barrel re-exports** — zero `from metrics import` consumers repo-wide; the package is consumed only via concrete submodules.
- **`.env.example` reference to `scripts/_mt5_discovery.py`** — that script does not exist (stale comment).
- **`metrics/performance/`** does not exist — the ledger's "dead `metrics/performance`" assumption is stale; no action.

---

## Items the ledger flagged that are NOT dead (corrections)

- **`scripts/_bootstrap.py`** — the ledger's `_*.py` wildcard would sweep it, but it is a load-bearing shared bootstrap helper imported by `data_platform/providers/mt5/scraper.py` and ~15 scripts. **KEEP.**
- **`utils/evaluation/holdout_robustness.py`** — repo-specific orchestration glue with 6 live callers, not a thin QF wrapper. **KEEP.**
- **`metrics/plotting` + `metrics/risk.max_drawdown`** — live, heavily used. **KEEP.**
- **`data_platform/providers/mt5/probes/*`** — documented, namespaced dev tooling run via `python -m`. **KEEP** (out of `scripts/_*` scope).

---

## Method notes (how each verdict was reached)

- codegraph `codegraph_callers` / `codegraph_impact` run on: `BinningModelBase`, `ContinuousBinningModel`, `RuleBasedModel`, `create_base_model_from_config`, `PropFirmChallengeSimulator`, `run_holdout_robustness_pipeline`.
- Repo-wide greps for dynamic refs: `from metrics import` barrel, `from scripts._bootstrap`, `import scripts.`, simulator symbol set, drawdown function names, JSON control files (`**/*.json` for `legacy_binning`/`n_bins`/binning names — none), `.bat` files (deploy targets `data_platform.providers.mt5.scraper`, not `_*` scripts), `.env.example`, and `docs/**`.
- `nodes/_taxonomy.py`-style string registries: no candidate symbol appears in any JSON or registry.
