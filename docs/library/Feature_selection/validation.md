# Validation

> [!important]
> The source of truth for this stage is [[SaaS/robustness_tests/validation]].

## Role in the workflow

Validation is the first post-exploration stage.

It answers:

> Does the locked strategy survive genuinely unseen strategy-level data?

The important contract is simple:

- exploration searches
- parameter selection locks one definition
- validation evaluates that locked definition only

## Locked handoff: `eval_bias_spec`

The local handoff surface is `eval_bias_spec` in `feature_research/config.py`.

Conceptually this means:

- do not carry the full exploration grid into validation
- do not re-open parameter search in validation
- evaluate one chosen definition on the validation window

## Read validation through the SaaS lens

The SaaS validation doc is the canonical reference for:

- IS vs validation Sharpe comparison
- bootstrap CI on validation Sharpe
- CUSUM with IS parameters
- neighborhood performance on validation
- rank correlation of the parameter grid

This library page exists only to explain the local command and artifact surface around that same stage.

## Local command surface

```bash
python -m feature_research validation
```

Vector-shuffle permutation is a **separate** command (not chained into the main validation run):

```bash
python -m feature_research validation_permutation
```

## Iterative vs release profiles

The default `load_config()` runs the **release** profile: walkforward on the locked combo, validation robustness (including rank correlation over the exploration grid), and the portfolio addition gate with HTML tearsheets.

For faster iteration while tuning `eval_bias_spec` or windows, use `apply_fast_validation_profile()` in code:

```python
from dataclasses import replace
from feature_research.config import load_config, apply_fast_validation_profile

config = apply_fast_validation_profile(load_config())
# python -m feature_research validation  # via UI or run_validation with this config
```

| Setting | Release (default) | Fast (`apply_fast_validation_profile`) |
|--------|-------------------|----------------------------------------|
| `portfolio_addition_gate.enabled` | `True` | `False` |
| `portfolio_addition_gate.emit_tearsheets` | `True` | `False` (gate off) |
| `validation_robustness.n_bootstrap` | `1000` | `200` |

Additional knobs (manual `dataclasses.replace`):

- `portfolio_addition_gate.emit_tearsheets=False` — keep gate metrics, skip QuantStats HTML.
- `portfolio_addition_gate.n_jobs` — parallel portfolio phase backtests (defaults to `ResearchConfig.n_jobs`).
- `cache_population_mode=CachePopulationMode.ANALYSIS_PLUS_LOOKBACK` — narrower bias-cache warmup (opt-in; enum defined in `feature_research/config.py`, default `FULL_HISTORY`; consumed by `populate_cache_if_needed` in `feature_research/in_sample/data_loader.py`).
- Run `validation_permutation` only after core validation passes.

## Local artifact surface

Validation artifacts currently write under:

```text
output_root/visualization/validation/
```

This area is the local validation-stage view of the locked combo.

The built-in plotting surface can consume the visualization tree directly:

```bash
python -m feature_research.visualization.matplotlib_reports --input-dir <output_root>/visualization
```

## Relationship to exploration artifacts

Exploration artifacts are for parameter screening and robustness review.

Validation artifacts are for realistic evaluation of the **locked** combo on later data.

Do not read exploration outputs as if they were already validation evidence.

## What validation is not

Validation is not:

- a second sweep
- permission to retune parameters
- the same thing as portfolio addition
- the project holdout

The next stage after validation is [[Feature_selection/portfolio_addition]], not a generic legacy "OOS phase."

## Related

- [[Feature_selection/pipeline]]
- [[Feature_selection/exploration]]
- [[Feature_selection/parameter_sensitivity]]
- [[Feature_selection/portfolio_addition]]
- [[SaaS/robustness_tests/validation]]

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
