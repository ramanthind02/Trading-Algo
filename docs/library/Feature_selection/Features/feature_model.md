# Feature Model

> [!summary]
> A feature is a bias node or small composition of bias nodes that produces a column used in research and, when production-safe, in ensembles.

## Two practical cases

| Case | Output | What happens next |
|---|---|---|
| Native signed-signal node | `-1`, `0`, `+1` | Run the research pipeline, then save to the vault if it passes |
| Continuous node | Float | Research it as-is; if it graduates, reimplement the production version as a native signed-signal node |

## Deprecated production ideas

Do not add new features that rely on:

- runtime quantile/bin fitting
- wrapper nodes that translate continuous outputs into production signals
- production-time learned threshold geometry

## Related

- [[Feature_selection/pipeline]]
- [[Feature_selection/Features/base_feature]]
- [[bias_nodes/creating_nodes]]
- [[Vault/architecture]]
