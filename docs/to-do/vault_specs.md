# Vault System Specifications - Base Model Features

## Overview

The Vault is a centralized, organized storage system for validated trading features and their associated base models. This specification focuses **exclusively on base model feature management**. Ensemble-level configuration will be handled in a separate specification.

The vault serves as the single source of truth for all production-ready base model features and enables clean, scalable management of hundreds of features across multiple ensembles.

---

## 1. Design Goals

### Functional Requirements
- **Single Source of Truth**: Both fitted and unfitted models exist in the same control file
- **Flexible Configuration**: Support multiple base model types per feature (e.g., QuantileBinning with bins=3, bins=5, DecisionTreeBinning with max_depth=3)
- **Easy Persistence**: Simple API to save/load base models via JSON control files
- **Production Ready**: Load pre-fitted models ready for immediate prediction
- **Scalable Organization**: Clean structure supporting hundreds of features
- **Incremental Updates**: Add new base model variants to existing features without replacing the entire file
- **Auto-Generated IDs**: Model IDs are automatically generated based on model type and hyperparameters
- **Bias Node Reconstruction**: Control files contain all information needed to reconstruct bias nodes for production deployment
- **Base Model Reconstruction**: Control files contain all information needed to instantiate base models

### Non-Functional Requirements
- **Type Safety**: Use enums for sector, timeframe, and direction
- **Validation**: Comprehensive validation of control file structure and strategy matching
- **Clean Migration**: Complete transition from old control file format (no backward compatibility)
- **Documentation**: Clear naming conventions and file organization
- **DRY Principle**: Avoid duplication between fitted/unfitted states

---

## 2. Directory Structure

```
vault/
├── STOCK_INDICES_D_LONG/           # Ensemble: Stock indices, daily, long-only
│   └── features/                    # Feature control files
│       ├── rsi_signal_D_lookback_2.json
│       ├── rsi_signal_D_lookback_3.json
│       ├── momentum_signal_D_lookback_20.json
│       └── ewmac_signal_D_fast_8_slow_32.json
│
├── STOCK_INDICES_D_SHORT/          # Ensemble: Stock indices, daily, short-only
│   └── features/
│       ├── rsi_signal_D_lookback_2.json
│       └── momentum_signal_D_lookback_20.json
│
├── STOCK_INDICES_W_LONG/           # Ensemble: Stock indices, weekly, long-only
│   └── features/
│       └── momentum_signal_W_lookback_10.json
│
├── CURRENCIES_D_LONG/              # Ensemble: Currency futures, daily, long-only
│   └── features/
│       ├── rsi_signal_D_lookback_2.json
│       └── ma_diff_signal_D_fast_10_slow_50.json
│
├── ENERGIES_D_LONG/                # Ensemble: Energy futures, daily, long-only
│   └── features/
│       └── momentum_signal_D_lookback_20.json
│
└── README.md                        # Vault documentation
```

**Note**: Ensemble-level metadata (if needed) will be managed separately and is not part of this specification.

### Naming Convention
- **Ensemble directories**: `{SECTOR}{TIMEFRAME}_{DIRECTION}`
  - *Sector must use all-uppercase with no underscores or separators*
  - Example: `STOCKINDICES_D_LONG`, `CURRENCIES_W_SHORT`
- **Feature files**: `{feature_column_name}.json`
  - Example: `rsi_signal_D_lookback_2.json`
  - Must match the standardized feature column name format
  - Parsed by `utils.helpers.parse_feature_column_name()`

---

## 3. Enums (Add to utils/enums.py)

### 3.1 Sector Enum

```python
from enum import Enum
from typing import List
from utils.enums import Ticker

class Sector(Enum):
    """
    Market sectors with their associated ticker symbols.
    
    Each sector groups related instruments for ensemble construction.
    Sectors are used to organize ensembles and diversify across
    different market areas.
    """
    
    # Equity Indices
    STOCKINDICES = [Ticker.ES, Ticker.NQ, Ticker.YM, Ticker.RTY]
    
    # Currencies (FX)
    CURRENCIES = [Ticker.EU, Ticker.JY, Ticker.BP, Ticker.CD, Ticker.SF]
    
    # Energy
    ENERGIES = [Ticker.CL, Ticker.HO]
    
    # Metals
    METALS = [Ticker.GC, Ticker.HG, Ticker.SI, Ticker.PL]
    
    # Fixed Income (Bonds)
    FIXEDINCOME = [Ticker.TY, Ticker.FV, Ticker.US, Ticker.TU]
    
    # Agricultural (Grains)
    GRAINS = [Ticker.C, Ticker.S, Ticker.W]
    
    # Livestock (Meat)
    LIVESTOCK = [Ticker.GF]
    
    def get_tickers(self) -> List[Ticker]:
        """Get the list of tickers in this sector."""
        return self.value
    
    def __lt__(self, other):
        """Enable sorting by enum definition order."""
        return tuple(self.__class__).index(self) < tuple(self.__class__).index(other)
    
    @classmethod
    def from_tickers(cls, tickers: List[Ticker]) -> 'Sector':
        """
        Find the sector that matches a list of tickers.
        
        Parameters
        ----------
        tickers : List[Ticker]
            List of ticker symbols
            
        Returns
        -------
        Sector
            The matching sector
            
        Raises
        ------
        ValueError
            If no matching sector found
        """
        ticker_set = set(tickers)
        for sector in cls:
            if set(sector.get_tickers()) == ticker_set:
                return sector
        raise ValueError(f"No sector found matching tickers: {tickers}")
    
    @classmethod
    def from_name(cls, name: str) -> 'Sector':
        """
        Get sector from string name (case-insensitive).
        
        Parameters
        ----------
        name : str
            Sector name (e.g., 'STOCK_INDICES', 'stock_indices')
            
        Returns
        -------
        Sector
            The matching sector
            
        Raises
        ------
        ValueError
            If sector name not found
        """
        try:
            return cls[name.upper()]
        except KeyError:
            valid_names = [s.name for s in cls]
            raise ValueError(
                f"Invalid sector name: '{name}'. "
                f"Valid sectors: {valid_names}"
            )
```

### 3.2 Direction Enum

```python
from enum import Enum

class Direction(Enum):
    """
    Trading direction for ensemble strategies.
    
    Ensembles are separated by direction to allow portfolio-level
    allocation between long and short strategies (e.g., 60% long, 40% short).
    """
    
    LONG = 'long'
    SHORT = 'short'
    
    def __str__(self) -> str:
        """String representation returns the value."""
        return self.value
    
    def __lt__(self, other):
        """Enable sorting (LONG before SHORT)."""
        return tuple(self.__class__).index(self) < tuple(self.__class__).index(other)
    
    @classmethod
    def from_string(cls, direction_str: str) -> 'Direction':
        """
        Convert string to Direction enum (case-insensitive).
        
        Parameters
        ----------
        direction_str : str
            Direction string ('long' or 'short')
            
        Returns
        -------
        Direction
            The matching direction
            
        Raises
        ------
        ValueError
            If direction string is invalid
        """
        direction_lower = direction_str.lower()
        for direction in cls:
            if direction.value == direction_lower:
                return direction
        raise ValueError(
            f"Invalid direction: '{direction_str}'. "
            f"Must be 'long' or 'short'"
        )
```

---

## 4. Feature Control File Structure

### 4.1 Feature Control File (features/{feature_name}.json)

Each feature has its own control file containing **all base model variants** for that feature.

**Key Design Points:**
1. **Bias Node Reconstruction**: `bias_node_spec` contains everything needed to recreate the feature via MLManager
2. **Model ID Auto-Generation**: `model_id` is auto-generated from model type and hyperparameters
3. **Model Name Format**: `{feature_column}::{model_id}` ensures uniqueness
4. **Easy Base Model Reconstruction**: All constructor params are stored for instantiation
5. **Single Source of Truth**: Fitted and unfitted params coexist in same file

```json
{
  "feature_name": "rsi_signal_D_lookback_2",
  "feature_column": "rsi_signal_D_lookback_2",
  "created_at": "2025-01-07T10:30:00Z",
  "updated_at": "2025-01-07T18:20:00Z",
  "bias_node_spec": {
    "module_name": "rsi",
    "timeframes": ["D"],
    "params": {
      "lookback": 2
    }
  },
  "base_models": [
    {
      "model_id": "quantile_binning_3",
      "model_name": "rsi_signal_D_lookback_2::quantile_binning_3",
      "model_type": "QuantileBinningModel",
      "strategy": "long",
      "constructor_params": {
        "n_bins": 3,
        "selection_metric": "sortino",
        "normalize_by": null,
        "strategy": "long"
      },
      "is_fitted": false,
      "fitted_params": null
    },
    {
      "model_id": "quantile_binning_5",
      "model_name": "rsi_signal_D_lookback_2::quantile_binning_5",
      "model_type": "QuantileBinningModel",
      "strategy": "long",
      "constructor_params": {
        "n_bins": 5,
        "selection_metric": "sortino",
        "normalize_by": null,
        "strategy": "long"
      },
      "is_fitted": false,
      "fitted_params": null
    },
    {
      "model_id": "decision_tree_binning_3",
      "model_name": "rsi_signal_D_lookback_2::decision_tree_binning_3",
      "model_type": "DecisionTreeBinningModel",
      "strategy": "long",
      "constructor_params": {
        "n_bins": 3,
        "selection_metric": "sortino",
        "normalize_by": null,
        "max_depth": 3,
        "strategy": "long"
      },
      "is_fitted": false,
      "fitted_params": null
    }
  ]
}
```

### 4.2 Model ID Auto-Generation Rules

Model IDs are automatically generated based on the model type and distinguishing hyperparameters:

**Pattern**: `{model_type_snake_case}_{key_hyperparam_values}`

**Examples:**
- `QuantileBinningModel(n_bins=3)` → `quantile_binning_3`
- `QuantileBinningModel(n_bins=10)` → `quantile_binning_10`
- `DecisionTreeBinningModel(n_bins=3, max_depth=5)` → `decision_tree_binning_3_depth5`
- `DecisionTreeBinningModel(n_bins=5, max_depth=3)` → `decision_tree_binning_5_depth3`

**Rules:**
1. Convert model class name from CamelCase to snake_case
2. Remove "Model" suffix
3. Append key hyperparameters that distinguish variants
4. Use underscores to separate components

**Implementation** (add to `ensemble/vault_manager.py`):

```python
def generate_model_id(model_type: str, constructor_params: Dict[str, Any]) -> str:
    """
    Auto-generate model ID from model type and hyperparameters.
    
    Parameters
    ----------
    model_type : str
        Model class name (e.g., 'QuantileBinningModel')
    constructor_params : Dict[str, Any]
        Constructor parameters for the model
        
    Returns
    -------
    str
        Auto-generated model ID
        
    Examples
    --------
    >>> generate_model_id('QuantileBinningModel', {'n_bins': 3, 'selection_metric': 'sortino'})
    'quantile_binning_3'
    >>> generate_model_id('DecisionTreeBinningModel', {'n_bins': 5, 'max_depth': 3})
    'decision_tree_binning_5_depth3'
    """
    import re
    
    # Convert CamelCase to snake_case and remove 'Model' suffix
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', model_type)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    name = name.replace('_model', '')
    
    # Append key hyperparameters
    if 'n_bins' in constructor_params:
        name += f"_{constructor_params['n_bins']}"
    
    if 'max_depth' in constructor_params:
        name += f"_depth{constructor_params['max_depth']}"
    
    return name
```

### 4.3 Fitted Base Models Example

**When fitted**, individual base models get their fitted parameters updated:

```json
{
  "feature_name": "rsi_signal_D_lookback_2",
  "feature_column": "rsi_signal_D_lookback_2",
  "created_at": "2025-01-07T10:30:00Z",
  "updated_at": "2025-01-07T18:20:00Z",
  "bias_node_spec": {
    "module_name": "rsi",
    "timeframes": ["D"],
    "params": {
      "lookback": 2
    }
  },
  "base_models": [
    {
      "model_id": "quantile_binning_3",
      "model_name": "rsi_signal_D_lookback_2::quantile_binning_3",
      "model_type": "QuantileBinningModel",
      "strategy": "long",
      "constructor_params": {
        "n_bins": 3,
        "selection_metric": "sortino",
        "normalize_by": null,
        "strategy": "long"
      },
      "is_fitted": true,
      "fitted_at": "2025-01-07T18:20:00Z",
      "train_start": "2020-01-01",
      "train_end": "2024-12-31",
      "fitted_params": {
        "thresholds": [25.5, 75.3],
        "best_long_bin": 2,
        "best_short_bin": 0,
        "bin_stats": {
          "0": {
            "mean_return": -0.0002,
            "std_return": 0.012,
            "downside_std": 0.009,
            "sortino_metric": -0.35,
            "sortino_metric_short": 1.15,
            "count": 320,
            "feature_min": 0.0,
            "feature_max": 25.5
          },
          "1": {
            "mean_return": 0.0005,
            "std_return": 0.010,
            "downside_std": 0.007,
            "sortino_metric": 1.13,
            "sortino_metric_short": -0.98,
            "count": 310,
            "feature_min": 25.6,
            "feature_max": 75.3
          },
          "2": {
            "mean_return": 0.0012,
            "std_return": 0.011,
            "downside_std": 0.008,
            "sortino_metric": 2.38,
            "sortino_metric_short": -2.15,
            "count": 310,
            "feature_min": 75.4,
            "feature_max": 100.0
          }
        }
      }
    },
    {
      "model_id": "quantile_binning_5",
      "model_name": "rsi_signal_D_lookback_2::quantile_binning_5",
      "model_type": "QuantileBinningModel",
      "strategy": "long",
      "constructor_params": {
        "n_bins": 5,
        "selection_metric": "sortino",
        "normalize_by": null,
        "strategy": "long"
      },
      "is_fitted": false,
      "fitted_params": null
    }
  ]
}
```

### 4.4 Bias Node Reconstruction

The `bias_node_spec` in each control file enables easy reconstruction of features for production:

```python
# Read control file
with open('vault/STOCK_INDICES_D_LONG/features/rsi_signal_D_lookback_2.json') as f:
    feature_config = json.load(f)

# Extract bias node spec
bias_node_spec = feature_config['bias_node_spec']
# {
#     'module_name': 'rsi',
#     'timeframes': ['D'],
#     'params': {'lookback': 2}
# }

# Use with MLManager to regenerate features
from feature_extraction.ml_manager import MLManager
ml_manager = MLManager(ticker=Ticker.ES, base_tf=TimeFrame.D)
features_df = ml_manager.get_features(bias_node_specs=[bias_node_spec])

# The feature column will be: 'rsi_signal_D_lookback_2'
```

This makes production deployment trivial - just read control files and pass bias node specs to MLManager.

---

## 5. Helper Functions (Add to ensemble/vault_manager.py)

Create a new module `ensemble/vault_manager.py` with these functions:

### 5.1 Ensemble Directory Management

```python
def create_ensemble_directory(
    vault_root: str,
    sector: Sector,
    timeframe: TimeFrame,
    direction: Direction
) -> str:
    """
    Create a new ensemble directory in the vault.
    
    Parameters
    ----------
    vault_root : str
        Root directory of the vault
    sector : Sector
        Market sector enum
    timeframe : TimeFrame
        Trading timeframe enum
    direction : Direction
        Trading direction enum
        
    Returns
    -------
    str
        Path to the created ensemble directory
        
    Raises
    ------
    ValueError
        If ensemble directory already exists
    """
    pass


def get_ensemble_path(
    vault_root: str,
    sector: Sector,
    timeframe: TimeFrame,
    direction: Direction
) -> str:
    """
    Get the path to an ensemble directory.
    
    Parameters
    ----------
    vault_root : str
        Root directory of the vault
    sector : Sector
        Market sector enum
    timeframe : TimeFrame
        Trading timeframe enum
    direction : Direction
        Trading direction enum
        
    Returns
    -------
    str
        Path to the ensemble directory
    """
    pass


def list_ensembles(vault_root: str) -> pd.DataFrame:
    """
    List all ensembles in the vault.
    
    Parameters
    ----------
    vault_root : str
        Root directory of the vault
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - ensemble_name: str
        - sector: str
        - timeframe: str
        - direction: str
        - n_features: int
    """
    pass
```

### 5.2 Feature Management

```python
def generate_model_id(model_type: str, constructor_params: Dict[str, Any]) -> str:
    """
    Auto-generate model ID from model type and hyperparameters.
    
    Pattern: {model_type_snake_case}_{key_hyperparam_values}
    
    Parameters
    ----------
    model_type : str
        Model class name (e.g., 'QuantileBinningModel')
    constructor_params : Dict[str, Any]
        Constructor parameters for the model
        
    Returns
    -------
    str
        Auto-generated model ID
        
    Examples
    --------
    >>> generate_model_id('QuantileBinningModel', {'n_bins': 3, 'selection_metric': 'sortino'})
    'quantile_binning_3'
    >>> generate_model_id('DecisionTreeBinningModel', {'n_bins': 5, 'max_depth': 3})
    'decision_tree_binning_5_depth3'
    """
    pass


def add_feature_to_ensemble(
    ensemble_dir: str,
    feature_column: str,
    bias_node_spec: Dict[str, Any],
    base_model: BaseModel
) -> str:
    """
    Add a base model variant to a feature control file.
    
    If the feature control file doesn't exist, creates it.
    If it exists, adds the new base model variant.
    Model ID is auto-generated based on model type and hyperparameters.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
    feature_column : str
        Feature column name (e.g., 'rsi_signal_D_lookback_2')
    bias_node_spec : Dict[str, Any]
        Bias node specification for feature reconstruction
        Format: {'module_name': str, 'timeframes': [TimeFrame], 'params': dict}
    base_model : BaseModel
        Fitted or unfitted base model instance
        
    Returns
    -------
    str
        The auto-generated model_id
        
    Raises
    ------
    ValueError
        If model_id already exists for this feature, or if base model strategy
        doesn't match ensemble direction
        
    Examples
    --------
    >>> from feature_selection.base_models import QuantileBinningModel
    >>> model = QuantileBinningModel(n_bins=3, selection_metric='sortino', strategy='long')
    >>> model.fit(X_train['rsi_signal_D_lookback_2'], y_train)
    >>> 
    >>> bias_spec = {'module_name': 'rsi', 'timeframes': [TimeFrame.D], 'params': {'lookback': 2}}
    >>> model_id = add_feature_to_ensemble('vault/STOCK_INDICES_D_LONG', 'rsi_signal_D_lookback_2', bias_spec, model)
    >>> print(model_id)  # 'quantile_binning_3'
    """
    pass


def load_feature_base_models(
    ensemble_dir: str,
    feature_column: str,
    fitted_only: bool = False
) -> Dict[str, BaseModel]:
    """
    Load all base model variants for a feature.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
    feature_column : str
        Feature column name
    fitted_only : bool, default=False
        If True, only return fitted models
        
    Returns
    -------
    Dict[str, BaseModel]
        Dictionary mapping model_id to BaseModel instance
        Keys are model IDs (e.g., 'quantile_bins3')
        Values are reconstructed BaseModel instances
        
    Examples
    --------
    >>> models = load_feature_base_models('vault/STOCK_INDICES_D_LONG', 'rsi_signal_D_lookback_2')
    >>> models
    {
        'quantile_bins3': <QuantileBinningModel fitted>,
        'quantile_bins5': <QuantileBinningModel unfitted>,
        'tree_depth3': <DecisionTreeBinningModel unfitted>
    }
    """
    pass


def update_base_model_fitted_params(
    ensemble_dir: str,
    feature_column: str,
    model_id: str,
    fitted_params: Dict[str, Any],
    train_start: str,
    train_end: str
) -> None:
    """
    Update fitted parameters for a specific base model variant.
    
    Called after BaseModel.fit() to save the fitted state.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
    feature_column : str
        Feature column name
    model_id : str
        Model ID to update
    fitted_params : Dict[str, Any]
        Fitted parameters from the base model
    train_start : str
        Training start date (YYYY-MM-DD)
    train_end : str
        Training end date (YYYY-MM-DD)
    """
    pass


def remove_base_model_variant(
    ensemble_dir: str,
    feature_column: str,
    model_id: str
) -> None:
    """
    Remove a base model variant from a feature.
    
    Useful if a model variant fails validation or is deprecated.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
    feature_column : str
        Feature column name
    model_id : str
        Model ID to remove
    """
    pass


def list_features(ensemble_dir: str) -> pd.DataFrame:
    """
    List all features in an ensemble.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - feature_name: str
        - feature_column: str
        - n_base_models: int
        - n_fitted: int
        - created_at: str
        - updated_at: str
    """
    pass


def get_bias_node_specs(ensemble_dir: str) -> List[Dict[str, Any]]:
    """
    Get all bias node specifications from an ensemble.
    
    This is useful for production deployment - read all bias node specs
    and pass to MLManager to regenerate all features.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
        
    Returns
    -------
    List[Dict[str, Any]]
        List of bias node specifications, one per feature
        
    Examples
    --------
    >>> bias_specs = get_bias_node_specs('vault/STOCK_INDICES_D_LONG')
    >>> # Use with MLManager
    >>> ml_manager = MLManager(ticker=Ticker.ES, base_tf=TimeFrame.D)
    >>> features_df = ml_manager.get_features(bias_node_specs=bias_specs)
    """
    pass


def get_all_base_model_names(ensemble_dir: str) -> List[str]:
    """
    Get all base model names (feature::model_id) in an ensemble.
    
    These are the keys used in ensemble weights.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
        
    Returns
    -------
    List[str]
        List of model names in format 'feature_column::model_id'
        
    Examples
    --------
    >>> get_all_base_model_names('vault/STOCK_INDICES_D_LONG')
    [
        'rsi_signal_D_lookback_2::quantile_bins3',
        'rsi_signal_D_lookback_2::quantile_bins5',
        'rsi_signal_D_lookback_2::tree_depth3',
        'momentum_signal_D_lookback_20::quantile_bins3'
    ]
    """
    pass
```

### 5.3 Vault-Level Operations

```python
def initialize_vault(vault_root: str) -> None:
    """
    Initialize a new vault directory structure.
    
    Creates the root directory and README.md.
    
    Parameters
    ----------
    vault_root : str
        Path to vault root directory
    """
    pass


def validate_ensemble_directory(ensemble_dir: str) -> None:
    """
    Validate an ensemble directory structure and all feature control files.
    
    Checks:
    - Directory exists and has features/ subdirectory
    - All feature control files are valid JSON
    - All base model strategies match ensemble direction
    - All bias node specs are valid and parseable
    - All model IDs are unique within each feature
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
        
    Raises
    ------
    ValueError
        If validation fails
    """
    pass
```

---

## 6. BaseModel Integration

### 6.1 Add BaseModel.save_to_vault()

Update `feature_selection/base_models/base_model.py`:

```python
def save_to_vault(
    self,
    ensemble_dir: str,
    bias_node_spec: Dict[str, Any]
) -> str:
    """
    Save base model to vault.
    
    This is the primary API for researchers to save validated features.
    Model ID is auto-generated based on model type and hyperparameters.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory in vault
    bias_node_spec : Dict[str, Any]
        Bias node specification for feature reconstruction.
        Format: {'module_name': str, 'timeframes': [TimeFrame], 'params': dict}
        
    Returns
    -------
    str
        The auto-generated model_id
        
    Raises
    ------
    ValueError
        If feature_column not set, or if strategy doesn't match ensemble direction
        
    Examples
    --------
    >>> from feature_selection.base_models import QuantileBinningModel
    >>> from ensemble.vault_manager import get_ensemble_path
    >>> from utils.enums import Sector, TimeFrame, Direction
    >>> 
    >>> # Create and fit model
    >>> model = QuantileBinningModel(n_bins=3, selection_metric='sortino', strategy='long')
    >>> model.fit(X_train['rsi_signal_D_lookback_2'], y_train)
    >>> 
    >>> # Save to vault
    >>> ensemble_dir = get_ensemble_path('vault', Sector.STOCK_INDICES, TimeFrame.D, Direction.LONG)
    >>> bias_spec = {
    ...     'module_name': 'rsi',
    ...     'timeframes': [TimeFrame.D],
    ...     'params': {'lookback': 2}
    ... }
    >>> model_id = model.save_to_vault(ensemble_dir, bias_spec)
    >>> print(model_id)  # 'quantile_binning_3'
    """
    from ensemble.vault_manager import add_feature_to_ensemble
    
    if self.feature_column is None:
        raise ValueError(
            "feature_column not set. Call fit() with a named pd.Series first."
        )
    
    model_id = add_feature_to_ensemble(
        ensemble_dir=ensemble_dir,
        feature_column=self.feature_column,
        bias_node_spec=bias_node_spec,
        base_model=self
    )
    
    return model_id
```

### 6.2 Add BaseModel.update_fitted_params_in_vault()

```python
def update_fitted_params_in_vault(
    self,
    ensemble_dir: str,
    model_id: str,
    train_start: str,
    train_end: str
) -> None:
    """
    Update fitted parameters in vault after fitting.
    
    Call this after fitting a model that was loaded from the vault.
    Model ID must be provided (should match the ID from initial save).
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory
    model_id : str
        Model ID to update (e.g., 'quantile_binning_3')
    train_start : str
        Training start date (YYYY-MM-DD)
    train_end : str
        Training end date (YYYY-MM-DD)
        
    Examples
    --------
    >>> # Load unfitted model from vault
    >>> models = load_feature_base_models(ensemble_dir, 'rsi_signal_D_lookback_2')
    >>> model = models['quantile_binning_3']
    >>> 
    >>> # Fit on new data
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Update vault
    >>> model.update_fitted_params_in_vault(
    ...     ensemble_dir=ensemble_dir,
    ...     model_id='quantile_binning_3',
    ...     train_start='2020-01-01',
    ...     train_end='2024-12-31'
    ... )
    """
    from ensemble.vault_manager import update_base_model_fitted_params
    
    if not self.is_fitted_:
        raise ValueError("Model must be fitted before updating vault")
    
    if self.feature_column is None:
        raise ValueError("feature_column not set")
    
    fitted_params = {
        'thresholds': self.thresholds_.tolist() if self.thresholds_ is not None else None,
        'best_long_bin': self.best_long_bin_,
        'best_short_bin': self.best_short_bin_,
        'bin_stats': self.bin_stats_
    }
    
    update_base_model_fitted_params(
        ensemble_dir=ensemble_dir,
        feature_column=self.feature_column,
        model_id=model_id,
        fitted_params=fitted_params,
        train_start=train_start,
        train_end=train_end
    )
```

### 6.3 Deprecate save_to_feature_list()

Mark the old method as deprecated:

```python
def save_to_feature_list(
    self,
    filepath: str,
    tickers: Optional[List[str]] = None
) -> None:
    """
    DEPRECATED: Use save_to_vault() instead.
    
    This method will be removed in a future version.
    """
    import warnings
    warnings.warn(
        "save_to_feature_list() is deprecated and will be removed. "
        "Use save_to_vault() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Original implementation stays for now...
```

---

## 7. Workflow Examples

### 7.1 Create New Ensemble and Add First Feature

```python
from utils.enums import Sector, TimeFrame, Direction
from ensemble.vault_manager import create_ensemble_directory, get_ensemble_path
from feature_selection.base_models import QuantileBinningModel

# Step 1: Create ensemble directory
ensemble_dir = create_ensemble_directory(
    vault_root='vault',
    sector=Sector.STOCK_INDICES,
    timeframe=TimeFrame.D,
    direction=Direction.LONG
)

# Step 2: Validate feature (using OSFeatureSelector, etc.)
# ... feature validation code ...

# Step 3: Create and fit base model
model = QuantileBinningModel(n_bins=3, selection_metric='sortino', strategy='long')
model.fit(X_train['rsi_signal_D_lookback_2'], y_train)

# Step 4: Define bias node spec (for feature reconstruction)
bias_spec = {
    'module_name': 'rsi',
    'timeframes': [TimeFrame.D],
    'params': {'lookback': 2}
}

# Step 5: Save to vault (model_id is auto-generated)
model_id = model.save_to_vault(ensemble_dir, bias_spec)
print(f"Saved model with ID: {model_id}")  # quantile_binning_3
```

### 7.2 Add More Base Model Variants to Existing Feature

```python
# Add 5-bin variant
model_5bin = QuantileBinningModel(n_bins=5, selection_metric='sortino', strategy='long')
model_5bin.fit(X_train['rsi_signal_D_lookback_2'], y_train)

bias_spec = {
    'module_name': 'rsi',
    'timeframes': [TimeFrame.D],
    'params': {'lookback': 2}
}

model_id_5 = model_5bin.save_to_vault(ensemble_dir, bias_spec)
print(f"Saved model with ID: {model_id_5}")  # quantile_binning_5

# Add tree-based variant
from feature_selection.base_models import DecisionTreeBinningModel
model_tree = DecisionTreeBinningModel(n_bins=3, selection_metric='sortino', strategy='long', max_depth=3)
model_tree.fit(X_train['rsi_signal_D_lookback_2'], y_train)

model_id_tree = model_tree.save_to_vault(ensemble_dir, bias_spec)
print(f"Saved model with ID: {model_id_tree}")  # decision_tree_binning_3_depth3
```

### 7.3 Load Base Models from Vault

```python
from ensemble.vault_manager import load_feature_base_models

# Load all base model variants for a feature
models = load_feature_base_models('vault/STOCK_INDICES_D_LONG', 'rsi_signal_D_lookback_2')

# Access specific model
model = models['quantile_binning_3']
print(f"Model fitted: {model.is_fitted_}")

# If unfitted, fit it
if not model.is_fitted_:
    model.fit(X_train['rsi_signal_D_lookback_2'], y_train)
    model.update_fitted_params_in_vault(
        ensemble_dir='vault/STOCK_INDICES_D_LONG',
        model_id='quantile_binning_3',
        train_start='2020-01-01',
        train_end='2024-12-31'
    )
```

### 7.4 Production Deployment: Reconstruct Features from Vault

```python
from ensemble.vault_manager import get_bias_node_specs
from feature_extraction.ml_manager import MLManager
from utils.enums import Ticker, TimeFrame

# Get all bias node specs from ensemble
ensemble_dir = 'vault/STOCK_INDICES_D_LONG'
bias_specs = get_bias_node_specs(ensemble_dir)

# Reconstruct features using MLManager (no base_tf needed)
ml_manager = MLManager(ticker=Ticker.ES, bias_node_specs=bias_specs)
# Features are generated incrementally via add_candle() in streaming mode

# Load fitted base models
from ensemble.vault_manager import get_all_base_model_names, load_feature_base_models

all_model_names = get_all_base_model_names(ensemble_dir)
fitted_models = {}

for model_name in all_model_names:
    feature_column, model_id = model_name.split('::')
    models = load_feature_base_models(ensemble_dir, feature_column, fitted_only=True)
    if model_id in models:
        fitted_models[model_name] = models[model_id]

# Generate predictions
for model_name, model in fitted_models.items():
    feature_column = model_name.split('::')[0]
    predictions = model.predict(features_df[feature_column], strategy='long')
    print(f"{model_name}: {predictions.sum()} signals")
```

---

## 8. Refactoring Tasks

### 8.1 Add Enums
- [ ] Add `Sector` enum to `utils/enums.py`
- [ ] Add `Direction` enum to `utils/enums.py`
- [ ] Add tests for enum methods

### 8.2 Create Vault Manager
- [ ] Create `ensemble/vault_manager.py`
- [ ] Implement `generate_model_id()` function
- [ ] Implement ensemble directory management functions
- [ ] Implement feature management functions (add/load/update/remove)
- [ ] Implement `get_bias_node_specs()` function
- [ ] Implement vault-level operations (initialize, validate)
- [ ] Add comprehensive docstrings
- [ ] Add input validation with strategy matching

### 8.3 Update BaseModel
- [ ] Add `save_to_vault()` method with bias_node_spec parameter
- [ ] Add `update_fitted_params_in_vault()` method
- [ ] Deprecate `save_to_feature_list()` with warning
- [ ] Update docstrings and examples

### 8.4 Delete Old Control Files
- [ ] Remove `deployment/config/` directory and all old control files
- [ ] Remove any references to old control file format in code
- [ ] Update all imports and references

### 8.5 Testing
- [ ] Add unit tests for vault_manager functions
- [ ] Add unit tests for model_id auto-generation
- [ ] Add integration tests for vault workflow
- [ ] Add tests for strategy validation
- [ ] Add tests for bias node reconstruction
- [ ] Test base model reconstruction from control files

### 8.6 Documentation
- [ ] Create vault usage guide in `docs/`
- [ ] Update README with vault examples
- [ ] Add examples for production deployment workflow
- [ ] Document model_id auto-generation rules

---

## 9. Validation Rules

### 9.1 Ensemble Validation
- Sector must be valid Sector enum
- Timeframe must be valid TimeFrame enum
- Direction must be valid Direction enum
- ensemble_name must match pattern: `{SECTOR}_{TIMEFRAME}_{DIRECTION}`

### 9.2 Feature Validation
- feature_column must be parseable by `parse_feature_column_name()`
- feature_column timeframe must match ensemble timeframe
- All model_ids must be unique within a feature
- model_name must match pattern: `{feature_column}::{model_id}`
- bias_node_spec must be valid and contain: module_name, timeframes, params

### 9.3 Base Model Validation
- **Strategy must match ensemble direction** (long model in long ensemble, short model in short ensemble)
  - **Validation timing**: On save via `save_to_vault()` - raise ValueError if mismatch
- model_type must be valid BaseModel class name
- constructor_params must be valid for model_type
- If is_fitted=True, fitted_params must be present and valid
- model_id must be auto-generated correctly from model type and params

---

## 10. Benefits of This Design

### Scalability
- Clean separation: each feature gets its own file
- No single monolithic control file
- Easy to manage hundreds of features

### Flexibility
- Multiple base model variants per feature
- Independent fitting of base models
- Easy to add/remove variants
- Auto-generated model IDs prevent naming conflicts

### Single Source of Truth
- Fitted and unfitted params in same file
- Clear versioning with timestamps
- Bias node specs enable feature reconstruction

### Production Ready
- Load fitted models instantly
- Reconstruct features from bias node specs
- Easy base model instantiation from control files
- No manual configuration needed

### Developer Experience
- Intuitive API (`save_to_vault`, auto-generated IDs)
- Type-safe enums
- Comprehensive validation at save time
- Clear error messages for strategy mismatches

---

## 11. Next Steps

1. **Review and approve specifications**
2. **Implement enums** (Sector, Direction in utils/enums.py)
3. **Create vault_manager.py** with all helper functions
4. **Update BaseModel** with vault methods
5. **Test with MVP** (RSI-2 on stock indices)
6. **Delete old control files** (deployment/config/)
7. **Write comprehensive tests**
8. **Update documentation**

---

**End of Specifications - Base Model Features**

**Note**: Ensemble-level vault management (ensemble metadata, ensemble fitting, etc.) will be specified in a separate document.
