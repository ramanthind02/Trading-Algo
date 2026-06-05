# Rule-Based / Native Discrete Signals

> [!summary]
> Native discrete bias nodes emit `-1`, `0`, or `+1` directly and are the canonical production feature type.

## EDA

For native discrete features, focus on:

- per-level returns
- transition behavior
- turnover and persistence
- parameter stability

See [[Feature_selection/exploration]].

## Contract

The production contract is simple:

- implement a native discrete node (see [[bias_nodes/creating_nodes]])
- validate it in the research pipeline (`research.feature`, signed-signal mode)
- save it to the vault as a signed-signal feature (`python -m research.feature.save_feature_to_vault`)

> [!note]
> "Rule-based" no longer means a runtime rule-binning model. The old
> `RuleBasedModel` has been **deleted** (the `feature_selection/base_models/base_model.py`
> stub module no longer exists). A rule-based idea must be implemented as a native
> signed-signal bias node, not a runtime translation layer.

## Related

- [[bias_nodes/creating_nodes]]
- [[Feature_selection/pipeline]]
- [[Feature_selection/Features/feature_model]]

> _Verified against commit a07b6bf->197221e on 2026-06-04 (docs Phase A; WP-8 restructure repoint)._
