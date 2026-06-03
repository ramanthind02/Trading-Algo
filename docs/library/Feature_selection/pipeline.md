# Feature Research Pipeline

> [!note]
> Status: current library reference for the `feature_research/` migration.

> [!important]
> The SaaS robustness docs are the source of truth for the research workflow:
> [[SaaS/robustness_tests/index|robustness index]],
> [[SaaS/robustness_tests/in_sample|in-sample]],
> [[SaaS/robustness_tests/parameter_sensitivity|parameter sensitivity]],
> [[SaaS/robustness_tests/parameter_selection|parameter selection]],
> [[SaaS/robustness_tests/validation|validation]],
> [[SaaS/robustness_tests/portfolio_addition|portfolio addition]],
> [[SaaS/robustness_tests/portfolio_holdout|portfolio holdout]],
> and [[SaaS/robustness_tests/monitoring|monitoring]].

## Canonical phase model

The current mental model is:

```text
exploration -> validation -> portfolio_addition -> portfolio_holdout -> monitoring
```

Only the first three stages belong to the local `feature_research/` package. The final two are downstream portfolio and live-operation workflows, not additional feature-research phases.

## What each stage answers

| Stage | Core question | Source-of-truth doc | Local library page |
|---|---|---|---|
| Exploration | Is there enough evidence in the research window to lock one parameter definition? | [[SaaS/robustness_tests/in_sample]], [[SaaS/robustness_tests/parameter_sensitivity]], [[SaaS/robustness_tests/parameter_selection]] | [[Feature_selection/exploration]], [[Feature_selection/permutation_testing]], [[Feature_selection/parameter_sensitivity]] |
| Validation | Does the locked definition survive genuinely unseen strategy-level data? | [[SaaS/robustness_tests/validation]] | [[Feature_selection/validation]] |
| Portfolio addition | Does the validated strategy improve the portfolio before any project holdout is opened? | [[SaaS/robustness_tests/portfolio_addition]] | [[Feature_selection/portfolio_addition]] |
| Portfolio holdout | Did the portfolio-level assumptions hold on the project test zone? | [[SaaS/robustness_tests/portfolio_holdout]] | Use the SaaS doc directly |
| Monitoring | Is the live strategy or portfolio drifting, degrading, or structurally broken? | [[SaaS/robustness_tests/monitoring]] | Use the SaaS doc directly |

## Exploration is broader than old "IS"

The old `Phase_1_IS` folder name survives in the library tree, but the workflow should now be read as **exploration**, not as a standalone "IS phase."

Exploration now includes:

1. parameter sweep
2. automatic in-sample robustness checks (Sharpe CI, DSR, NW t-stat, rolling/CUSUM)
3. vector-shuffle permutation (Mode 1 gate) and optional full-grid diagnostic (Mode 2)
4. parameter sensitivity / perturbation review
5. parameter selection and parameter lock

See [[Feature_selection/permutation_testing]] for the exploration gate table and [[SaaS/robustness_tests/in_sample]] §4 for failure-mode theory.

Validation starts only after one combination is intentionally selected and frozen.

## Local package surfaces

Today the codebase already exposes the preferred stage names, even though some compatibility aliases remain:

| Canonical stage | Preferred surface | Compatibility alias still present | Notes |
|---|---|---|---|
| Exploration | `feature_research.exploration`, `python -m feature_research exploration` | `in_sample` | Preferred local entrypoint for sweep, visualization, and robustness work. |
| Validation | `feature_research.validation`, `python -m feature_research validation` | none worth preferring | Uses the locked `eval_bias_spec`. |
| Portfolio addition | `feature_research.portfolio_addition`, `python -m feature_research portfolio_addition` | `oos` | Some configs, scripts, and artifact paths still say `oos`; treat that as legacy naming only. |

## Migration rules for reading older docs

- Treat `in_sample` as a compatibility alias for **exploration**.
- Treat `oos` as a compatibility alias for **portfolio addition**, not as the preferred stage name.
- Do not treat project holdout or live monitoring as part of `feature_research/`; those belong to the downstream SaaS portfolio workflow.

## Production contract

- Production features are native signed-signal bias nodes.
- Continuous bias nodes can be researched, but they are not saved to production unchanged.
- If a continuous idea graduates, it should be re-expressed as a native node that emits `-1/0/+1`.

## Portfolio holdout (local)

- Command: `python -m portfolio_research holdout`
- Library page: [[Portfolio_research/holdout]]
- SaaS spec: [[SaaS/robustness_tests/portfolio_holdout]]

## Related

- [[Feature_selection/exploration]]
- [[Portfolio_research/holdout]]
- [[Feature_selection/permutation_testing]]
- [[Feature_selection/parameter_sensitivity]]
- [[Feature_selection/validation]]
- [[Feature_selection/portfolio_addition]]
- [[Feature_selection/Features/base_feature]]
- [[Vault/user_guide]]
