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
- validate it in the research pipeline (`feature_research`, signed-signal mode)
- save it to the vault as a signed-signal feature (`python -m feature_research.save_feature_to_vault`)

> [!note]
> "Rule-based" no longer means a runtime rule-binning model. The old
> `RuleBasedModel` in `feature_selection/base_models/base_model.py` is a retired stub that
> raises `RuntimeError` if instantiated. A rule-based idea must be implemented as a native
> signed-signal bias node, not a runtime translation layer.

## Related

- [[bias_nodes/creating_nodes]]
- [[Feature_selection/pipeline]]
- [[Feature_selection/Features/feature_model]]

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
