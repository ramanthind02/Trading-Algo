# In-Sample Permutation Testing

> [!note] Status: Library reference — **Phase 2** of [[Feature_selection/pipeline]] (in-sample screen)  
> Last updated: 2026-04-05

> [!important] **Current `feature_research` practice:** In-sample permutation is **vector shuffle** only. `PermutationResearchConfig` no longer exposes a second stage; `to_permutation_test_config()` always sets pipeline Stage 2 off. Optional Stage 2 machinery remains in `feature_selection.validation` for advanced use. Results are exported for **Power BI** alongside CSV/MD summaries (see below).

Relates to: [[Feature_selection/pipeline]] | [[Feature_selection/Phase_1_IS/eda]] | [[Feature_selection/Phase_1_IS/candle_permutation]] (candle null — reference only; not run by default)

---

## Purpose

**Question:** Does the fitted feature vector carry timing information that improves the objective metric vs. random reorderings of the same signal values?

**Role:** Cheap computational filter on the full in-sample window before walk-forward and strict OOS. Failing here usually means walk-forward is not worth the cost.

**Design notes:**

- **Feature-level lens:** Per-combo pass/fail is diagnostic; the usual research question is whether the **feature class** shows any evidence under the chosen null, not to pre-filter the param grid for walk-forward (see [[Feature_selection/Phase_2_WF/param_stability]] for selection rules).
- **Full IS period** is valid for this null: the test compares the observed metric to a distribution from permutations on the same window.

---

## Vector shuffle (Stage 1)

**Null:** Random temporal assignment of the same signal values is as good as the real alignment for the objective metric (e.g. Sharpe, Sortino).

**Procedure:**

1. Align feature and target on the research window (cache-backed loads, same as EDA).
2. For each parameter combination, build the strategy metric from **signal × target** (or the configured objective on aligned returns).
3. **Permute** the feature vector in time (same marginals, broken timing vs. target).
4. Repeat for `nreps` (typical 500–1000) to form a null distribution.
5. **Pass** if the observed metric exceeds the `(1 − α)` quantile of the null (default `α = 0.10` in research config unless you change it).

**Scope:** One report row per `param_combo`. Implementation: `feature_selection.validation.orchestration.run_permutation_test_suite` (Stage 1); entry from `feature_research.pipelines.permutation.run_permutation_pipeline`.

---

## Exports (Power BI + summaries)

Runs from `feature_research/in_sample/run_is.py` when `permutation.enabled=True`. **`reports_dir`** comes from `feature_research.config.load_config()`.

| Location | Contents |
|----------|----------|
| `reports_dir/powerbi/permutation_vector_shuffle.csv` | Flat table: `feature_name`, `feature_type`, `objective_metric`, `param_combo`, `observed_metric`, `p_value`, `passed`, `alpha`, `n_reps`, `critical_value` |
| `reports_dir/powerbi/permutation_vector_shuffle.parquet` | Same as CSV |
| `reports_dir/powerbi/permutation_powerbi_manifest.json` | UTC timestamp, objective label, artifact paths |
| `reports_dir/permutation_summary.csv` | Same columns as the vector-shuffle export (no second-stage columns in `feature_research`) |
| `reports_dir/permutation_summary.md` | Human-readable table; Stage 2 section appears only if Stage 2 ran |

**Power BI:** Load the Parquet or CSV under `powerbi/` and relate `param_combo` to in-sample **param sensitivity** tables (`param_combo_label` / `param_combo`) from [[Feature_selection/Phase_1_IS/eda]].

**Configuration:** `PermutationResearchConfig` in `feature_research/config.py` — `objective_metric`, `enabled`, `nreps`, `alpha`, `n_jobs_reps`, `run_vector_shuffle`, `candidate_source`, etc.

---

## Optional Stage 2 (not in `feature_research` config)

Pipeline feature shuffle and related Stage 2 machinery remain in `feature_selection.validation` for advanced or custom entrypoints. **`PermutationResearchConfig` does not turn Stage 2 on**; wiring it requires constructing `PermutationTestConfig` / `InSamplePermutationConfig` yourself.

For a **stronger bar-structure null** (OHLC reconstruction, gap pools, etc.), see the retained spec at [[Feature_selection/Phase_1_IS/candle_permutation]] — not wired through `feature_research` today.

---

## Related docs

- [[Feature_selection/pipeline]] — where IS permutation sits in the tiered flow  
- [[Feature_selection/Phase_1_IS/eda]] — EDA and param-sensitivity exports  
- [[Feature_selection/Phase_2_WF/walkforward]] — walk-forward robustness after IS  
