# Base Feature

> [!summary] Definition
> A **base feature** is one frozen bias-node contract plus one feature column.
> In the domain-discrete workflow, the persisted artifact names the source node, ticker scope, cutpoints, selected bins, and spec version.

## Canonical Type

| Type | Node output | Persisted contract |
|------|-------------|--------------------|
| **Domain discrete** | Signed signal `-1 / 0 / +1` | Frozen `DomainDiscreteSpec` |

---

## Contract Shape

- `source_bias_node_spec` identifies the underlying indicator node and its parameters.
- `ticker_scope` declares which ticker or ticker group the feature applies to.
- `edges` are absolute cutpoints chosen in research.
- `long_bins` and `short_bins` encode the frozen sign policy.
- `spec_version` changes whenever the source recipe, scope, or cutpoints change.

## Runtime

- Input: OHLCV candles plus the frozen spec.
- Output: a signed signal column in `-1 / 0 / +1`.
- No runtime step learns bin geometry.

## Naming

- Column names should encode the source recipe and spec version.
- Changing cutpoints or scope should create a new column name and a new vault entry.

---

> [!info] See also
> - [[domain_discrete_signals]] — frozen signed-signal contract
> - [[creating_nodes]] — bias node design and output format specs
