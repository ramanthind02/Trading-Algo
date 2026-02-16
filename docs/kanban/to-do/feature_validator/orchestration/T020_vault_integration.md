# T020 — Vault Integration for Validated Features

## Goal
Provide seamless integration between the FeatureValidator pipeline and the Vault system, enabling researchers to save validated features, form ensembles, and persist fitted base models for production deployment.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — Vault Integration section (lines 377-457)
- `docs/library/Vault/vault_user_guide.md` — Vault structure and operations
- `ensemble/vault_manager.py` — Existing vault management utilities
- `feature_selection/base_models/base_model.py` — BaseModel save_to_vault() interface
- `docs/library/Feature_selection/permutation_testing/in-sample_pt.md` — Ensemble formation section (§4)

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

1. `pytest tests/integration/feature_validator/test_vault_integration.py::test_save_single_feature_to_vault -v`
   - Setup: Validate RSI lookback 14, save to vault
   - Validates: Feature JSON file created in correct location
   - Checks: File exists, correct directory structure, valid JSON

2. `pytest tests/integration/feature_validator/test_vault_integration.py::test_save_multiple_features_to_vault -v`
   - Setup: Validate RSI lookback [3, 4, 5, 10], save all to vault
   - Validates: All features saved correctly
   - Checks: 4 JSON files created, all in same ensemble directory

3. `pytest tests/integration/feature_validator/test_vault_integration.py::test_ensemble_directory_creation -v`
   - Setup: Create new ensemble directory
   - Validates: Directory structure matches vault spec
   - Checks: {vault_root}/{timeframe}/{ensemble_name}_{direction}/features/

4. `pytest tests/integration/feature_validator/test_vault_integration.py::test_strategy_direction_validation -v`
   - Setup: Try to save long binning model to short ensemble
   - Validates: ValueError raised with clear message
   - Checks: No files created, error message mentions strategy/direction mismatch

5. `pytest tests/integration/feature_validator/test_vault_integration.py::test_timeframe_validation -v`
   - Setup: Try to save Daily feature to Weekly ensemble
   - Validates: ValueError raised
   - Checks: Feature column timeframe doesn't match ensemble timeframe

6. `pytest tests/integration/feature_validator/test_vault_integration.py::test_model_id_collision -v`
   - Setup: Save feature, then try to save again with same binning config
   - Validates: FileExistsError raised (force_refit=False)
   - Checks: Error message mentions duplicate model ID

7. `pytest tests/integration/feature_validator/test_vault_integration.py::test_force_refit -v`
   - Setup: Save feature, then save again with force_refit=True
   - Validates: File overwritten successfully
   - Checks: New fitted state replaces old

8. `pytest tests/integration/feature_validator/test_vault_integration.py::test_create_ensemble_from_validated_params -v`
   - Setup: Run validation, identify stable neighborhood, create ensemble
   - Validates: Convenience method works end-to-end
   - Checks: Ensemble directory created, all params saved, summary returned

9. `pytest tests/integration/feature_validator/test_vault_integration.py::test_get_ensemble_summary -v`
   - Setup: Create ensemble with multiple features
   - Validates: Summary returns correct counts and metadata
   - Checks: num_feature_columns, num_models, feature_columns list

10. `pytest tests/integration/feature_validator/test_vault_integration.py::test_full_workflow_rsi_example -v`
    - Setup: RSI lookback [2,3,4,5,6,7,8,9,10], TimeFrame.D, validate, save stable subset [3,4,5,10]
    - Validates: Full workflow from validation to vault persistence
    - Checks: Validated params identified, stable neighborhood saved, ensemble ready for deployment

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
