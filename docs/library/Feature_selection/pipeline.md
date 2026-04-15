# Feature Selection Pipeline

> [!note]
> Status: library reference

## Current contract

- Production features are native signed-signal bias nodes.
- Continuous bias nodes can be researched, but they are not saved to production unchanged.
- If a continuous idea graduates, it should be re-expressed as a native node that emits `-1/0/+1`.

## Research flow

1. Build a hypothesis and implement a bias node.
2. Run in-sample EDA and parameter review.
3. Run permutation or other robustness checks when needed.
4. Select a single production definition.
5. Validate on the holdout window.
6. Run final OOS confirmation.
7. Save the native signed-signal feature to the vault.

## Feature types

| Source | Output | Production status |
|---|---|---|
| Native discrete bias node | `-1`, `0`, `+1` | Allowed |
| Continuous bias node | Float | Research only until reimplemented as native discrete |

## Entrypoints

- `python -m feature_research in_sample`
- `python -m feature_research validation`
- `python -m feature_research oos`
- `python -m feature_research validation-permutation`
- `python -m feature_research oos-permutation`

## Related

- [[Feature_selection/Phase_1_IS/eda]]
- [[Feature_selection/Phase_1_IS/permutation_testing]]
- [[Feature_selection/Phase_2_WF/walkforward]]
- [[Feature_selection/Phase_3_OOS/oos_validation]]
- [[Feature_selection/Features/base_feature]]
- [[Vault/user_guide]]
