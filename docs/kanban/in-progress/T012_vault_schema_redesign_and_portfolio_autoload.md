# T012 — Vault Schema Redesign and Portfolio Auto-Load

## Goal
Redesign vault feature persistence to canonical per-feature files with per-model params/member metadata, and allow `Portfolio` to auto-load ensembles directly from `vault/` when `ensembles=None`.

## Context / References
- `ensemble/vault_manager.py`
- `ensemble/ensemble_utils.py`
- `ensemble/diversified_ensemble.py`
- `ensemble/portfolio.py`
- `feature_selection/base_models/feature_base_model.py`
- `docs/api/ensemble.md`
- `docs/api/base_models.md`
- `docs/library/Vault/vault.md`

## Scope
In scope:
- New vault writer schema:
  - one file per canonical feature `{module}_{feature}_{tf}.json`
  - top-level `bias_node_spec` without params
  - per-model `bias_node_params`, `requires_fit`, `members`, and bin-index bounds
- Backward-compatible readers for old vault payloads and model type aliases.
- Consolidation utility to migrate fragmented feature files into canonical files.
- Member fit/predict persistence flow (save/load/update member fitted params).
- Portfolio vault auto-load path with optional ensemble-name filtering.

Out of scope:
- Automatic migration during read.
- Changes to unrelated research workflow tasks.

## Interfaces (must match)
- Modify: `ensemble/vault_manager.py`
  - `generate_model_id(..., bias_node_params=None)`
  - `add_feature_to_ensemble(...)` with canonical `feature_name` + backward-compatible legacy call style
  - `load_feature_base_models(feature_name, ...)`
  - `update_base_model_fitted_params(..., member_name=None)`
  - `consolidate_feature_files(ensemble_dir) -> List[str]`
  - `save_member_fitted_params(...)`
- Modify: `ensemble/ensemble_utils.py`
  - `create_base_model_from_config(...)` merge `bias_node_params` and attach members
  - validation accepts optional/empty `members` and both legacy/new member key shapes
- Modify: `feature_selection/base_models/feature_base_model.py`
  - fit members during `fit(...)` with `requires_fit` skip semantics
  - add `predict_members_from_candles(...)`
  - add `get_fitted_params_for_members(...)`
- Modify: `ensemble/diversified_ensemble.py`
  - use `predict_members_from_candles(...)` in active member prediction path
  - fit skip and member exposure handling aligned to `requires_fit`
- Modify: `ensemble/portfolio.py`
  - ctor args: `ensemble_names`, `vault_root`
  - add `_load_ensembles_from_vault(...)`
- Modify: `feature_selection/base_models/continuous_binning.py`
  - include `bin_index_min`/`bin_index_max` in `get_params()`

## Data Contracts
- New feature file top-level:
  - `feature_name`, `bias_node_spec`, `tickers`, `base_models`, timestamps
- `feature_column` is not written in new files.
- Per-model entries include:
  - `model_id`, `model_name`, `bias_node_params`, `binning_model_type`, `strategy`
  - `binning_model_params`, `requires_fit`, `is_fitted`, `fitted_params`, `members`
- Per-member entries include:
  - `member_name`, `binning_model_type`, `binning_model_params`
  - `requires_fit`, `is_fitted`, `fitted_params`

## Invariants / Constraints
- Deterministic model IDs and model naming.
- Reader compatibility with legacy schema, forward-only new writes.
- If old `fitted_params` are not `binning_v2`, they are treated as unfitted during consolidation.
- `members` can be absent or empty.

## Acceptance tests
1. `source venv/bin/activate && PYTHONPATH=. pytest tests/unit-tests/vault -q`
2. `source venv/bin/activate && PYTHONPATH=. pytest tests/unit-tests/ensemble -q`
3. `source venv/bin/activate && PYTHONPATH=. pytest tests/unit-tests/feature_selection/base_models -q`
4. `source venv/bin/activate && PYTHONPATH=. pytest tests/integration/test_vault_portfolio_autoload_integration.py -q`
5. `source venv/bin/activate && PYTHONPATH=. pytest tests/feature_research/ -q`
6. `source venv/bin/activate && PYTHONPATH=. pytest tests/ -q`

## Definition of done
- [ ] Vault write path emits new schema only.
- [ ] Vault read path supports legacy and new schemas.
- [ ] Consolidation utility merges fragmented files and removes originals.
- [ ] Portfolio can auto-load vault ensembles when `ensembles=None`.
- [ ] Unit/integration tests added or updated and passing.
- [ ] API + vault docs updated.

## Notes
- Default selection metadata for temporary control payloads is `selection_method="manual"` when `is_fit=True` and metadata is absent.
