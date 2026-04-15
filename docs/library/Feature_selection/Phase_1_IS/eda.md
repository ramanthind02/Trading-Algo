# EDA - Phase 1 Exploratory Data Analysis

> [!note]
> Informational phase only. Researchers review outputs before later gates.

## Purpose

Use Phase 1 EDA to understand:

- signal behavior
- parameter landscapes
- turnover and persistence
- cross-ticker consistency

## Power BI outputs

In-sample runs still export shared Power BI tables under `reports_dir / powerbi /`, including:

- `param_sensitivity.csv`
- `param_sensitivity_by_ticker.csv`
- `equity_curve.csv`
- `param_combo_long.csv`
- `in_sample_powerbi_manifest.json`

## Continuous research

Continuous bias nodes can still be studied here for:

- distribution shape
- normalization checks
- parameter sensitivity

But they are not production-ready artifacts on their own.

## Discrete research

For native signed signals `-1/0/+1`, focus on:

- per-level returns
- transition matrices
- bootstrap summaries
- path behavior in `in_sample_cumsum.csv`

## Hand-off

After EDA, the researcher selects a production definition and pins it for later phases. If the source idea started continuous, the production definition should be a native signed-signal node, not a runtime wrapper.

## Related

- [[Feature_selection/pipeline]]
- [[Feature_selection/Phase_1_IS/permutation_testing]]
- [[Feature_selection/Phase_2_WF/param_stability]]
- [[Vault/user_guide]]
