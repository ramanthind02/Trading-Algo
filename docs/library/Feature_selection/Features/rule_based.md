# Rule-Based / Native Discrete Signals

> [!summary]
> Native discrete bias nodes emit `-1`, `0`, or `+1` directly and are the canonical production feature type.

## EDA

For native discrete features, focus on:

- per-level returns
- transition behavior
- turnover and persistence
- parameter stability

See [[Feature_selection/Phase_1_IS/eda]].

## Contract

The production contract is simple:

- implement a native discrete node
- validate it in the research pipeline
- save it to the vault as a signed-signal feature

## Related

- [[bias_nodes/creating_nodes]]
- [[Feature_selection/pipeline]]
- [[Feature_selection/Features/feature_model]]
