# Base Model

> [!summary]
> In the current architecture a base model is a thin, node-backed adapter around one native **signed-signal** bias-node definition. There is no fitted binning geometry — the model reconstructs a bias node, extracts its signed signal, and aligns it with the configured strategy direction.

## Current implementation

- One production feature maps to one base-model entry in an ensemble control file.
- Each entry points directly at a native `bias_node_spec` (module + params + timeframe).
- `model_type` is always `"signed_signal"`. Any other value is rejected as a legacy artifact.
- Base models are constructed by `create_base_model_from_config` in [`ensemble/ensemble_utils.py`](../../../ensemble/ensemble_utils.py), which builds a thin node-backed `BaseModel` (no fitted feature geometry, no `constructor_params`).

### Required config keys

`_validate_signed_signal_model_config` (in `ensemble/ensemble_utils.py`) enforces these keys on every base-model entry:

| Key | Meaning |
|---|---|
| `name` | Unique base-model name within the ensemble |
| `model_type` | Must be the literal `"signed_signal"` |
| `feature_column` | Feature column name (e.g. `rsi_signal_D_lookback_14`) |
| `strategy` | A `Direction` value (e.g. `long` / `short`); aligns the signed signal |
| `bias_node_spec` | Native node spec (validated by `validate_signed_signal_bias_node_spec`) |

`SIGNED_SIGNAL_FEATURE_TYPE = "signed_signal"` is the shared constant used by the
feature-research pipelines and the vault feature-file validators
(`ensemble/vault/feature_files.py`).

## Legacy binning is removed

The legacy binning classes `BinningModelBase`, `ContinuousBinningModel`, and `RuleBasedModel`
have been **deleted** (the old `feature_selection/base_models/base_model.py` stub module no
longer exists; only `features/models/feature_base_model.py` survives). Quantile / decision-tree /
rule-based binning is no longer part of the production contract — the active path is
`create_base_model_from_config` (signed-signal), which builds node-backed `BaseModel` instances.

> Older docs used "base model" to describe ensembles of *fitted binning members* with
> `n_bins` / `constructor_params`. That is legacy context only — the production contract is
> the node-backed signed-signal adapter described above.

## Purpose

The base-model layer gives [`DiversifiedEnsemble`](../../../ensemble/diversified_ensemble.py) a stable way to:

- reconstruct a bias node from its saved spec,
- extract the node's signed signal from candles,
- align that signal with the strategy direction,
- persist feature identity through the vault feature files.

## Related

- [Feature base feature](../Feature_selection/Features/base_feature.md)
- [Vault architecture](../Vault/architecture.md)
- [Weight layer](weight_layer.md)
- [Portfolio pipeline](portfolio.md)

> _Verified against commit a07b6bf->197221e on 2026-06-04 (docs Phase A; WP-8 restructure repoint)._
