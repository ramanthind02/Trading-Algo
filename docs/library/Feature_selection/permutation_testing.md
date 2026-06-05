# Permutation Testing

> [!important]
> Canonical theory and gate policy: [[SaaS/robustness_tests/in_sample]] §4 (failure modes, gates vs diagnostics).
> Broader robustness context: [[SaaS/robustness_tests/index]].

## Two failure modes (read this first)

In-sample overfitting is not one problem — it is two:

| Mode | Question | Local test |
|---|---|---|
| **1 — Temporal overfitting** | Did this combo work only because signal values aligned with *this* return sequence? | **Vector shuffle** (per combo) |
| **2 — Parameter mining** | Did grid search find a peak a naive $N$-trial search could hit on noise? | **DSR** (gate) + **full-grid return-shuffle** (diagnostic) |

Vector shuffle and full-grid answer **different** questions. Neither substitutes for the other. DSR addresses Mode 2 analytically; full-grid addresses it empirically.

**Do not** run full-grid only on combos that passed vector shuffle on the same IS window — that conditions the grid on data used in the null and invalidates the p-value ([[SaaS/robustness_tests/in_sample]] §4.2).

## Recommended exploration gates vs diagnostics

**Gates (required before parameter lock):**

| Check | Config / artifact | Pass |
|---|---|---|
| Vector shuffle | `PermutationResearchConfig` | Per-combo pass at $\alpha$ |
| DSR | `robustness_report.json` → `dsr` | $\geq 0.95$ |
| NW t-stat | `robustness_report.json` → `newey_west` | $\geq 2.0$ |
| Rolling positive fraction | `robustness_report.json` → `rolling_is` | $\geq 70\%$ (configurable) |
| CUSUM | `robustness_report.json` → `rolling_is` / stability chart | Within 5% bounds |

**Diagnostics (report; do not hard-gate structured grids):**

| Check | Artifact | Action |
|---|---|---|
| Full-grid return-shuffle | `full_grid_p_value`, `robustness_report.json` | Flag null percentile $< 20\%$ for manual review |
| Sharpe CI | `sharpe_ci` | Lower bound $> 0$ sanity check |

For focused indicator families (same module, parameter ranges only), **DSR is the Mode 2 gate**. Full-grid can read conservative when strategy returns are autocorrelated and the null uses plain Sharpe on IID shuffled returns — see [[SaaS/robustness_tests/in_sample]] §4.1.

## Role in the workflow

Permutation testing is part of **exploration**, not a standalone phase.

Canonical order:

1. parameter sweep
2. automatic robustness (DSR, NW, rolling, CUSUM)
3. vector shuffle (Mode 1)
4. parameter sensitivity
5. parameter lock → validation

## Local implementation

| Test | Config | Runs when | Failure mode |
|---|---|---|---|
| **Full-grid search-bias** | `RobustnessResearchConfig.run_full_grid_permutation` | Robustness step | Mode 2 (diagnostic) |
| **Vector shuffle** | `PermutationResearchConfig.enabled` + `run_vector_shuffle` | After robustness | Mode 1 (gate) |

### Full-grid (return-shuffle)

Quant Foundry Core (`run_grid_permutation_test`): each null iteration shuffles the **target** return series and re-scores the entire parameter grid. Signals and inter-combo correlations are preserved.

**Metric split:**

- **Real-data combo selection:** `selection_metric` → typically NW-adjusted t-stat.
- **Full-grid observed + null:** `full_grid_permutation_metric` → default plain annualized Sharpe (both sides use the same plain metric).

NW t-stat on return-shuffle nulls is miscalibrated (HAC collapses under IID shuffled returns while real-data scores stay deflated). Plain Sharpe fixes that asymmetry but does not encode autocorrelation in observed strategy returns — expect conservative full-grid p-values on persistent strategies; trust DSR + vector shuffle gates.

### Vector shuffle (signal timing)

**Null:** random temporal assignment of the same signal values is as good as real alignment for the objective metric.

**Procedure:**

1. align feature and target on the exploration window
2. compute observed metric per parameter combination
3. permute the **feature** vector in time
4. repeat for `nreps` null draws
5. pass/fail from empirical $(1 - \alpha)$ null quantile

**P-value:** `p_value = (1 + null_ge_count) / (n_reps + 1)` (one-sided, +1 pseudo-count).

This is the local Mode 1 test. SaaS §3.2 individual **return-shuffle** applies only to genuinely pre-specified combos, not grid winners.

## Local exports

| Location | Contents |
|---|---|
| `reports_dir/robustness_report.json` | DSR, NW, rolling, `full_grid_permutation` |
| `reports_dir/robustness_summary.csv` | `full_grid_p_value`, DSR, $N_\text{eff}$ |
| `reports_dir/visualization/permutation_vector_shuffle.csv` | Per-combo vector-shuffle table |
| `reports_dir/permutation_summary.csv` / `.md` | Combined summary (full-grid section + vector shuffle) |

## Configuration

**Full-grid (diagnostic)** — `RobustnessResearchConfig`:

- `run_full_grid_permutation`
- `n_permutations` (local preset often 100; SaaS default 1000)
- `selection_metric` — NW-adjusted metric for best-combo selection on real data
- `full_grid_permutation_metric` — plain metric for null (default `None` → annualized Sharpe)

**Vector shuffle (gate)** — `PermutationResearchConfig`:

- `objective_metric` (typically NW t-stat or same family as selection)
- `enabled`, `run_vector_shuffle`, `nreps`, `alpha`, `n_jobs_reps`

## Interpretation

- Passing vector shuffle + DSR + NW + rolling is necessary but not sufficient for validation — read parameter sensitivity and lock deliberately.
- Full-grid failing while DSR passes on a structured grid: investigate (metric mismatch, $N_\text{eff}$, autocorrelation) — not automatic rejection.
- Full-grid passing while DSR fails: treat DSR as authoritative for Mode 2 on structured grids.
- When local docs and SaaS docs differ, prefer the SaaS interpretation.

## Related

- [[Feature_selection/pipeline]]
- [[Feature_selection/exploration]]
- [[Feature_selection/parameter_sensitivity]]
- [[SaaS/robustness_tests/in_sample]]
- [[SaaS/robustness_tests/index]]

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
