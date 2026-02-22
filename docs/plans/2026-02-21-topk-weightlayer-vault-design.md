# Top-K Selection, Weight Layer, and Vault Refactor Design

## Context

This design covers two related changes:

1. Make walkforward ensemble methods (`top_k`, `enhanced`, `stable_region`) selector-only.
2. Move forecast combination responsibility to the weight layer while introducing hard-cutover multi-member base models and vault metadata upgrades.

The user decisions driving this design are:

- Hard cutover (no backward-compatibility shims for legacy schema/behavior)
- Multi-member signals are flattened using deterministic `model_name` member identifiers
- Researcher controls are enum-backed for selection method and weight-layer algorithm

## Goals

- Base models hold multiple binning/rule members and emit multiple member-level signals.
- Walkforward selection chooses member sets and records selection metadata, but does not average selected signals.
- Weight layer becomes the only forecast combiner for selected member signals.
- Vault persists selection method and hyperparameters (including top-k hyperparameters) and member metadata required for reproducibility.
- Research configs support explicit, validated experiment method/algorithm selection.

## Non-Goals

- No migration support for old vault/control-file payloads.
- No plugin-based method registry in this phase.
- No redesign of existing weight algorithms beyond contract compliance for larger member cardinality.

## Target Architecture

### 1) Base-Model Contract (Multi-Member)

- A base model is a container of member models.
- Each member has deterministic identity and parameters.
- Prediction output is flattened into independent member rows with deterministic `model_name` values.

### 2) Selection Layer Contract (Walkforward)

- Selection methods output selected member identities/params only.
- Selection layers do not produce averaged ensemble signals.
- Fold artifacts store selection method and hyperparameters used for that fold.

### 3) Combination Layer Contract (Weight Layer)

- Weight layer combines selected member-level forecasts.
- Portfolio paths use weight-layer combination as first-class behavior in this mode.
- Missing required weight-layer configuration is a hard error.

### 4) Persistence Contract (Vault)

- Strategy payloads persist:
  - selection method identity
  - selection hyperparameters (including top-k)
  - member definitions/fitted metadata
- Legacy schema payloads are rejected under hard cutover.

## File-Level Ownership

- `feature_selection/base_models/feature_base_model.py`: multi-member core model contract.
- `feature_selection/base_models/base_model.py`: shared base-model mechanics/type surfaces.
- `feature_selection/base_models/continuous_binning.py`: member construction for binning variants.
- `feature_selection/base_models/rule_based.py`: member construction for rule variants.
- `ensemble/diversified_ensemble.py`: member-level signal propagation.
- `ensemble/portfolio.py`: strict weight-layer-centric aggregation path.
- `ensemble/weight_layer.py`: combination behavior under flattened member naming.
- `feature_research/walkforward/config.py`: enum-backed method/algorithm selection.
- `feature_research/in_sample/continuous_binning/config.py`: researcher-facing configuration defaults.
- `feature_research/walkforward/runner.py`: selection-only fold outputs + metadata propagation.
- `feature_research/walkforward/portfolio_evaluator.py`: fold evaluation using weight-layer combiner path.
- `ensemble/vault_manager.py`: member-aware persistence and loading.
- `ensemble/ensemble_utils.py`: strict schema validation.
- `docs/library/Feature_selection/Parameter Sensitivity/top_k_ensemble_selection.md`: updated selector-only semantics.
- `docs/library/Ensemble/weight_layer.md`: updated combiner semantics.
- `docs/library/Vault/vault_user_guide.md`: updated persisted schema and examples.

## Validation Strategy

- Unit tests for:
  - multi-member base-model fit/predict and deterministic naming
  - walkforward config enum validation
  - selection-only runner behavior
  - diversified ensemble and portfolio member-level flow
  - vault schema persistence and strict loading
- Integration tests for cross-layer behavior where selection metadata and weight-layer combination interact.
- Documentation tests/checks for updated behavior statements and schema examples.

## Hard-Cutover Invariants

- Legacy single-member or legacy vault/control payloads are not accepted.
- Selection stage never computes final ensemble averages.
- Weight layer is required for selected member combination in this architecture path.
- Strategy artifacts must include selection hyperparameters used for generation.

## Risks

- Deterministic naming collisions if identifiers are underspecified.
- Contract mismatches between validator and vault writer/reader implementations.
- Hidden fallback averaging paths in portfolio code if not removed/guarded.
- Documentation drift causing invalid researcher inputs.

## Success Criteria

- End-to-end flow supports multi-member base models with weight-layer combination.
- Walkforward artifacts fully describe selection method/hyperparameters per run.
- Vault artifacts are reproducible and include top-k hyperparameter provenance.
- Docs and tests enforce the new contract consistently.
