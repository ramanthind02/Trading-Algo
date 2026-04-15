# Base Model

> [!summary]
> In the current architecture, a base model is a thin adapter around one native signed-signal bias-node definition.

## Current implementation

- one production feature maps to one saved base-model row
- the row points directly to a native `bias_node_spec`
- `model_type` is `signed_signal`

## Historical note

Older docs used “base model” to describe ensembles of fitted binning members. That is no longer the production contract and should be treated as legacy context only.

## Purpose

The base-model layer gives ensembles a stable way to:

- reconstruct bias nodes
- extract signals from candles
- align those signals with strategy direction
- persist feature identity through the vault

## Related

- [[Feature_selection/Features/base_feature]]
- [[Vault/architecture]]
- [[weight_layer]]
