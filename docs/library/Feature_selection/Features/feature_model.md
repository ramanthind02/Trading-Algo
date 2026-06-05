# Feature Model

> [!summary]
> A feature is a bias node or small composition of bias nodes that produces a column used in research and, when production-safe, in ensembles.

## Two practical cases

| Case | Output | What happens next |
|---|---|---|
| Native signed-signal node | `-1`, `0`, `+1` | Run the research pipeline, then save to the vault if it passes |
| Continuous node | Float | Research it as-is (`python -m feature_research binning`); if it graduates, reimplement the production version as a native signed-signal node |

Continuous-node research lives in the standalone `feature_research.binning` package
(`feature_research/binning/config.py`, `feature_research/binning/pipeline.py`). It produces
quantile/decile research tables only — it never emits a production artifact. The main
`feature_research.config.load_config()` config describes the signed-signal trading pipeline.

## Deprecated production ideas

Do not add new features that rely on:

- runtime quantile/bin fitting
- wrapper nodes that translate continuous outputs into production signals
- production-time learned threshold geometry

These runtimes are gone, not just discouraged: the legacy classes
`BinningModelBase`, `ContinuousBinningModel`, and `RuleBasedModel` in
`feature_selection/base_models/base_model.py` raise `RuntimeError` on construction.

## Related

- [[Feature_selection/pipeline]]
- [[Feature_selection/Features/base_feature]]
- [[bias_nodes/creating_nodes]]
- [[Vault/architecture]]

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
