# T020 — Vault Integration for Validated Features

## Goal
Provide seamless integration between the FeatureValidator pipeline and the Vault system, enabling researchers to save validated features, form ensembles, and persist fitted base models for production deployment.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — Vault Integration section (lines 377-457)
- `docs/library/Vault/vault_user_guide.md` — Vault structure and operations
- `ensemble/vault_manager.py` — Existing vault management utilities
- `feature_selection/base_models/base_model.py` — BaseModel save_to_vault() interface
- `docs/library/Feature_selection/permutation_testing/in-sample_pt.md` — Ensemble formation section (§4)
- `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md` — unit vs integration test standards

## Scope
In scope:
- Vault save workflow from ValidationReport
- Ensemble directory creation and management
- Base model fitting and persistence for validated parameters
- Validation of strategy/direction matching
- Feature column naming validation
- Model ID generation and collision detection

Out of scope:
- FeatureValidator API implementation (T018)
- Configuration management (T019)
- Individual phase implementations (EDA, Binning, ParamSens, PermTest tasks)
- Vault loading/deployment (separate deployment task)
- DiversifiedEnsemble construction (separate ensemble task)

## Interfaces (must match)

### Extend: `feature_selection/feature_validator.py`

**Add vault integration methods to FeatureValidator class:**
```python
from pathlib import Path
from typing import List, Optional, Dict
from ensemble.vault_manager import create_ensemble_directory
from utils.enums import Direction, TimeFrame, Ticker

class FeatureValidator:
    # ... existing methods from T018 ...

    def save_validated_features_to_vault(
        self,
        vault_root: Path,
        ensemble_name: str,
        direction: Direction,
        candles: pd.DataFrame,
        target: pd.Series,
        param_filter: Optional[List[tuple]] = None,
        force_refit: bool = False
    ) -> Dict[str, List[str]]:
        """Save validated features to vault for ensemble deployment.

        Process:
        1. Get validated parameters (passed all validation stages)
        2. Filter to param_filter subset if provided (for ensemble formation)
        3. Create ensemble directory in vault
        4. For each validated parameter:
           a. Fit base model on full in-sample data
           b. Save to vault using base_model.save_to_vault()
           c. Track model IDs
        5. Return summary of saved features

        Args:
            vault_root: Root vault directory (e.g., Path('vault'))
            ensemble_name: Name for ensemble (e.g., 'momentum_strategy')
            direction: Long or Short (must match binning strategy)
            candles: Full in-sample OHLCV data for final fitting
            target: Full in-sample target data aligned with candles
            param_filter: Optional subset of validated params to save (for ensemble selection)
            force_refit: If True, overwrite existing models (default: False, raises error on collision)

        Returns:
            Dict mapping feature_column to list of saved model_ids:
            {
                'rsi_signal_D_lookback_14': ['quantile_binning_15'],
                'rsi_signal_D_lookback_5': ['quantile_binning_15'],
                ...
            }

        Raises:
            ValueError: If direction doesn't match binning strategy
            ValueError: If feature column timeframe doesn't match validator timeframe
            FileExistsError: If model ID already exists and force_refit=False
        """
        ...

    def create_ensemble_from_validated_params(
        self,
        vault_root: Path,
        ensemble_name: str,
        direction: Direction,
        candles: pd.DataFrame,
        target: pd.Series,
        stable_neighborhood: List[tuple],
        binning_config: Optional[Dict] = None
    ) -> Path:
        """Create ensemble from stable parameter neighborhood.

        Convenience method for typical workflow:
        1. Researcher reviews validation reports
        2. Identifies stable neighborhood (e.g., RSI lookback [3, 4, 5, 10])
        3. Saves all params in neighborhood to vault as ensemble

        Args:
            vault_root: Root vault directory
            ensemble_name: Name for ensemble
            direction: Long or Short
            candles: Full in-sample OHLCV data
            target: Full in-sample target data
            stable_neighborhood: List of param combos to include in ensemble
            binning_config: Optional override for binning hyperparameters
                          (default: use config from ValidationConfig)

        Returns:
            Path to created ensemble directory

        Example:
            # After running validation pipeline
            validator = FeatureValidator(...)
            reports = validator.run_full_pipeline(candles, target, date_range)

            # Review parameter sensitivity report, identify stable region
            # stable_params = [(lookback=3,), (lookback=4,), (lookback=5,), (lookback=10,)]

            ensemble_dir = validator.create_ensemble_from_validated_params(
                vault_root=Path('vault'),
                ensemble_name='rsi_momentum_long',
                direction=Direction.LONG,
                candles=candles,
                target=target,
                stable_neighborhood=stable_params
            )
        """
        ...
```

### Extend: `ensemble/vault_manager.py`

**Add helper functions for feature validation workflow:**
```python
from pathlib import Path
from typing import Dict, List
from utils.enums import Direction, TimeFrame

def create_ensemble_directory(
    vault_root: Path,
    timeframe: TimeFrame,
    ensemble_name: str,
    direction: Direction
) -> Path:
    """Create ensemble directory structure in vault.

    Structure:
        vault/
        └── {timeframe}/                    # e.g., D, W, M
            └── {ensemble_name}_{direction}/  # e.g., rsi_momentum_long
                └── features/                # base model JSON files go here

    Args:
        vault_root: Root vault directory (e.g., Path('vault'))
        timeframe: TimeFrame enum (D, W, M)
        ensemble_name: Ensemble name (e.g., 'momentum_strategy')
        direction: Direction enum (LONG, SHORT)

    Returns:
        Path to ensemble features directory

    Raises:
        FileExistsError: If ensemble directory already exists
    """
    ...

def validate_feature_column_timeframe(
    feature_column: str,
    expected_timeframe: TimeFrame
) -> None:
    """Validate that feature column timeframe matches expected timeframe.

    Feature column format: {module}_{feature}_{timeframe}_{param}_{value}
    Example: rsi_signal_D_lookback_14

    Args:
        feature_column: Feature column name
        expected_timeframe: Expected TimeFrame

    Raises:
        ValueError: If timeframe in feature column doesn't match expected
    """
    ...

def validate_strategy_direction_match(
    binning_strategy: str,  # 'long' or 'short'
    ensemble_direction: Direction
) -> None:
    """Validate that binning strategy matches ensemble direction.

    Args:
        binning_strategy: Strategy from binning model ('long' or 'short')
        ensemble_direction: Direction enum (LONG or SHORT)

    Raises:
        ValueError: If strategy and direction don't match
    """
    ...

def get_ensemble_summary(ensemble_dir: Path) -> Dict[str, any]:
    """Get summary of ensemble directory contents.

    Args:
        ensemble_dir: Path to ensemble features directory

    Returns:
        Dict with summary:
        {
            'ensemble_name': str,
            'direction': str,
            'timeframe': str,
            'num_feature_columns': int,
            'num_models': int,
            'feature_columns': List[str],
            'model_ids_per_feature': Dict[str, List[str]]
        }
    """
    ...
```

## Data Contracts

**Vault directory structure:**
```
vault/
└── D/                              # TimeFrame.D
    └── rsi_momentum_long/          # ensemble_name_direction
        └── features/
            ├── rsi_signal_D_lookback_3.json
            ├── rsi_signal_D_lookback_4.json
            ├── rsi_signal_D_lookback_5.json
            └── rsi_signal_D_lookback_10.json
```

**Base model JSON format (reference):**
```json
{
  "feature_column": "rsi_signal_D_lookback_14",
  "model_id": "quantile_binning_15",
  "binning_config": {
    "n_bins": 15,
    "selection_metric": "sharpe",
    "strategy": "long"
  },
  "fitted_state": {
    "bin_edges": [...],
    "position_multipliers": [...],
    "regions": [...]
  },
  "metadata": {
    "ticker": "ES",
    "timeframe": "D",
    "fit_date": "2026-02-15",
    "sample_size": 6000
  }
}
```

**Strategy/direction validation:**
- `binning_strategy='long'` matches `Direction.LONG`
- `binning_strategy='short'` matches `Direction.SHORT`
- Mismatch raises ValueError

**Feature column timeframe validation:**
- Feature column `rsi_signal_D_lookback_14` matches `TimeFrame.D`
- Feature column `rsi_signal_W_lookback_14` matches `TimeFrame.W`
- Mismatch raises ValueError

## Dependencies
- `ensemble/vault_manager.py` — vault operations
- `feature_selection/base_models/base_model.py` — BaseModel interface
- `feature_selection/base_models/quantile_binning.py` — QuantileBinningModel
- `utils/enums.py` — Direction, TimeFrame, Ticker
- `pathlib` — Path operations
- `json` — serialization

## Invariants / Constraints

**Feature persistence:**
- One JSON file per feature column (e.g., `rsi_signal_D_lookback_14.json`)
- Multiple model variants per feature (different binning configs) stored in single file
- Model IDs are deterministic (same binning config => same model ID)
- No duplicate model IDs within a feature column (unless force_refit=True)

**Validation:**
- Strategy/direction matching enforced before save
- Timeframe matching enforced before save
- No cross-timeframe contamination (can't save Daily feature to Weekly ensemble)
- No cross-direction contamination (can't save short model to long ensemble)

**Ensemble formation:**
- Ensemble contains only validated parameters (passed all stages)
- All features in ensemble share same ticker and timeframe
- Researcher manually selects stable neighborhood (validator doesn't auto-select)

**Determinism:**
- Same validated params + same binning config => same vault contents
- Model IDs are deterministic (no timestamps or random components)

## Acceptance tests

**Unit tests:**

Validation helpers (`validate_feature_column_timeframe`, `validate_strategy_direction_match`) and directory-path construction are pure logic and must be covered by unit tests. Location: `tests/validators/test_vault_integration.py`

- `test_validate_feature_column_timeframe_match()` — pass `rsi_signal_D_lookback_14` and `TimeFrame.D`; assert no error raised
- `test_validate_feature_column_timeframe_mismatch()` — pass `rsi_signal_D_lookback_14` and `TimeFrame.W`; assert `ValueError`
- `test_validate_strategy_direction_match_long()` — strategy `'long'` and `Direction.LONG`; assert no error
- `test_validate_strategy_direction_mismatch()` — strategy `'long'` and `Direction.SHORT`; assert `ValueError` with clear message
- `test_ensemble_directory_path_construction()` — call `create_ensemble_directory` with a temp root; verify returned path equals `{root}/{timeframe}/{ensemble_name}_{direction}/features/`
- `test_model_id_collision_raises()` — write a stub JSON file at the expected model path; attempt save without `force_refit`; assert `FileExistsError`
- `test_force_refit_overwrites()` — write a stub JSON file; save again with `force_refit=True`; assert file content updated
- `test_get_ensemble_summary_empty_dir()` — call `get_ensemble_summary` on an empty features directory; assert zero counts

**Integration tests:**

Location: `tests/integration/feature_validator/test_vault_integration.py`

These tests use real feature data extracted from `data/ohlc_data/` and verify actual file I/O to a temporary vault directory.

Default integration config:
```python
DEFAULT_CONFIG = {
    'bias_module': 'rsi',
    'param_name': 'lookback',
    'param_value': 5,
    'ticker': Ticker.ES,
    'timeframe': TimeFrame.D,
    'date_range': ('2020-01-01', '2023-12-31'),
    'direction': Direction.LONG
}
```

- `test_save_single_feature_to_vault()` — extract real RSI lookback 5 features from `data/ohlc_data/`; fit model; call `save_validated_features_to_vault()`; assert JSON file created at `{vault_root}/D/rsi_long/features/rsi_signal_D_lookback_5.json`; assert valid JSON with expected keys (`feature_column`, `model_id`, `fitted_state`, `metadata`)
- `test_save_multiple_features_to_vault()` — RSI lookback [3, 4, 5]; save all to vault; assert 3 JSON files created in same ensemble directory
- `test_vault_directory_structure()` — create ensemble directory; assert path matches `{vault_root}/{timeframe}/{ensemble_name}_{direction}/features/`
- `test_vault_load_reload_roundtrip()` — save a fitted model to vault; reload the JSON; assert `fitted_state.bin_edges` and `fitted_state.position_multipliers` match original
- `test_create_ensemble_from_validated_params()` — run minimal validation on RSI lookback 5, ES daily 2020-2023; call `create_ensemble_from_validated_params()` with stable neighborhood `[(lookback=5,)]`; assert ensemble directory created; assert `get_ensemble_summary()` returns `num_feature_columns=1`
- `test_full_workflow_rsi_example()` — end-to-end: validate RSI lookback 5, save to vault, reload via `get_ensemble_summary()`; print summary to terminal for manual inspection
- Also covered by `tests/integration/feature_validator/test_feature_validator_e2e.py::test_end_to_end_validation_workflow()` (vault save step at end of full pipeline)

**Cache policy:**
- Use existing cache: `USE_CACHE=True`
- If cache missing: skip with message "Run `CacheManager.populate_cache()` first"
- Cache spec: RSI lookback [5], ES, D, 2020-2023
- Vault directory: use `tmp_path` fixture (auto-cleaned); do not write to repo vault

**Researcher manual verification:**
- After running vault integration tests, inspect printed summary: verify `num_feature_columns`, `feature_columns` list, and `model_ids_per_feature`
- Open generated JSON file in `tmp_path` and confirm `fitted_state.bin_edges` are monotonically increasing and `position_multipliers` are 0 or 1
- Verify directory structure matches `vault/{timeframe}/{ensemble_name}_{direction}/features/*.json`

## Definition of done
- [ ] Tests added under `tests/integration/feature_validator/test_vault_integration.py`
- [ ] Vault integration methods added to `FeatureValidator` class
- [ ] Helper functions added to `ensemble/vault_manager.py`
- [ ] Strategy/direction validation implemented
- [ ] Timeframe validation implemented
- [ ] Model ID collision detection implemented
- [ ] Ensemble summary function implemented
- [ ] Docs updated in `docs/api/feature_selection.md` and `docs/library/Vault/vault_user_guide.md`
- [ ] `pytest tests/integration/feature_validator/test_vault_integration.py -q` passes

## Notes

**Typical workflow:**
```python
# 1. Run validation pipeline
validator = FeatureValidator(
    feature_type='continuous',
    bias_node_spec={
        'module_name': 'rsi',
        'timeframes': [TimeFrame.D],
        'params': {'lookback': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    },
    ticker=Ticker.ES,
    config=ValidationConfig.production()
)

reports = validator.run_full_pipeline(
    candles=candles_df,
    target=target_series,
    date_range=('2000-01-01', '2024-12-31')
)

# 2. Review reports (manual researcher step)
# - Check parameter sensitivity report for stable regions
# - Review permutation test results
# - Review walkforward stability

# 3. Select stable neighborhood (example)
stable_params = [
    (lookback=3,),
    (lookback=4,),
    (lookback=5,),
    (lookback=10,)
]

# 4. Create ensemble in vault
ensemble_dir = validator.create_ensemble_from_validated_params(
    vault_root=Path('vault'),
    ensemble_name='rsi_momentum_long',
    direction=Direction.LONG,
    candles=candles_df,
    target=target_series,
    stable_neighborhood=stable_params
)

# 5. Verify ensemble
summary = get_ensemble_summary(ensemble_dir)
print(f"Created ensemble with {summary['num_feature_columns']} features")
```

**Model ID generation:**
- Model ID should be deterministic hash of binning hyperparameters
- Example: `quantile_binning_15` for n_bins=15, quantile strategy
- Format: `{binning_type}_{n_bins}` (simplified)
- Future: Support multiple binning strategies (decision tree, rule-based)

**Ensemble naming conventions:**
- Use descriptive names: `rsi_momentum_long`, `breakout_commodity_short`
- Include direction in name for clarity (even though directory structure has it)
- Avoid generic names: `ensemble1`, `test_ensemble`

**Vault maintenance:**
- Ensemble directories should have associated metadata file (future enhancement)
- Track creation date, researcher notes, validation config used
- Consider versioning for ensemble updates (e.g., `rsi_momentum_long_v2`)

**Error handling:**
- Provide clear error messages for validation failures
- Include suggestions for fixes (e.g., "Use Direction.SHORT or change binning strategy to 'long'")
- Log all vault operations for debugging

**Future extensions:**
- Automatic ensemble versioning (detect changes, create new version)
- Ensemble comparison tools (diff two ensembles)
- Vault cleanup utilities (remove old/unused ensembles)
- Integration with DiversifiedEnsemble loader (automatic base model loading)
