# Exploration

> [!important]
> The SaaS robustness docs are the source of truth for this stage:
> [[SaaS/robustness_tests/in_sample]],
> [[SaaS/robustness_tests/parameter_sensitivity]],
> and [[SaaS/robustness_tests/parameter_selection]].

## Purpose

Exploration is the first stage of `feature_research/`.

Its job is to answer one question:

> Is there enough evidence in the research window to lock one production definition for validation?

## Automatic filter gates (exploration only)

When ``ResearchConfig.exploration_filter_gates.enabled`` is true (default in
``load_config()``), the in-sample ``bias_spec`` should be **signal-only**. Exploration
expands it at runtime into four catalog branches:

1. **A** — ungated signal (baseline)
2. **B/C trend** — ``filter_gate`` + ``filter_gate_entry_only`` with ``sma_above_filter`` period 252
3. **B/C vol low** — ``atr_percentile_filter`` (period 32, lookback 252, low tail, ±30% rank grid)
4. **B/C vol high** — same with high tail

Default ``scope`` is ``winning_signal_only``: robustness runs on the **signal grid only**
(e.g. 42 Donchian combos), then filter A/B/C branches (~15 combos) run on the **winning**
signal parameters. Pass 1 uses an ATR% decile chart on the ungated eval combo.

Set ``scope='full_signal_grid'`` only if you need the legacy ``N_signal × 15`` full Cartesian
product (expensive). Set ``enabled=False`` to skip filter follow-up entirely.

### Where filter results appear (local UI)

Filter follow-up writes under the **shared** visualization folder (not under
``reports_dir/donchian_long_only_gc``):

- ``feature_research/in_sample/results/visualization/filter_exploration_summary.csv`` — A/B/C table (Sharpe, gate type, trade reduction)
- ``feature_research/in_sample/results/visualization/filter_exploration_long.csv`` — long pivot rows
- ``feature_research/in_sample/results/visualization/matplotlib/filter_gate_comparison.png`` — bar chart (after exploration finishes / report refresh)

In the research workspace (**Exploration** phase):

1. **Matplotlib Reports** — ``filter_gate_comparison.png`` (featured panel when present)
2. **Visualization Data** — the two ``filter_exploration_*.csv`` files
3. **Parameter sensitivity** (top of the Parameter Sensitivity section) — interactive pivot heatmap when sensitivity CSVs exist; dataset **“Gate + signal (filter exploration)”** when ``filter_exploration_summary.csv`` exists

Pass 1 only produces ``atr_pct_decile_chart.png`` (vol-regime hint); filter charts require a completed run **including** the post-robustness filter follow-up step.

## What exploration includes

Exploration bundles sweep, robustness, permutation, sensitivity, and parameter lock:

1. parameter sweep
2. automatic in-sample robustness (DSR, NW t-stat, rolling/CUSUM) — see gate table in [[Feature_selection/permutation_testing]]
3. vector-shuffle permutation (Mode 1 — temporal overfitting)
4. full-grid return-shuffle when enabled (Mode 2 **diagnostic**; DSR is the Mode 2 **gate**)
5. parameter sensitivity review
6. parameter selection and parameter lock

Validation starts only after one combination is intentionally selected and frozen.

> [!note]
> Two overfitting failure modes (timing vs search) and the gates-vs-diagnostics policy are documented in [[SaaS/robustness_tests/in_sample]] §4.

## Main local artifacts

Exploration exports shared visualization CSVs under `reports_dir / visualization /`, including:

- `param_sensitivity.csv`
- `param_sensitivity_by_ticker.csv`
- `param_combo_long.csv`
- `equity_curve.csv`

These artifacts help answer:

- does the parameter region look broad enough to trust?
- is the behavior shared across tickers?
- does the signal path look stable enough to keep investigating?

## Matplotlib review path (non-sensitivity charts)

Matplotlib still renders equity curves, filter-gate comparison, permutation summaries, and other CSV-backed charts:

```bash
python -m feature_research.visualization.matplotlib_reports \
  --input-dir <reports_dir>/visualization
```

Parameter sensitivity surfaces are reviewed in the research workspace **Parameter Sensitivity** section via the interactive pivot explorer (not Matplotlib PNGs).

## Continuous vs discrete ideas

Continuous bias nodes can still be explored for:

- distribution shape
- normalization checks
- sensitivity analysis

But they are not production-ready artifacts on their own.

For native signed signals `-1/0/+1`, focus on:

- per-level returns
- transition behavior
- turnover and persistence
- whether the selected region is a plateau rather than a spike

## Running exploration locally

The canonical entrypoint runs sweep, robustness, and permutation in order:

```bash
python -m feature_research exploration
```

Full-grid search-bias permutation runs inside robustness when
``RobustnessResearchConfig.run_full_grid_permutation`` is true (Core API; see
[[SaaS/robustness_tests/in_sample]]). Vector-shuffle runs afterward when
``PermutationResearchConfig.enabled`` and ``run_vector_shuffle`` are both true.
See [[Feature_selection/permutation_testing]] for artifact paths.

## Hand-off

Exploration ends when the researcher locks one production definition for validation.

If the source idea began as a continuous node, the graduation target should still be a native signed-signal node rather than a runtime wrapper.

## Related

- [[Feature_selection/pipeline]]
- [[Feature_selection/permutation_testing]]
- [[Feature_selection/parameter_sensitivity]]
- [[Vault/user_guide]]
