# T019 — Configuration Management for Feature Validation

## Goal
Provide a frozen configuration dataclass (`ValidationConfig`) with pre-specified thresholds, user-customizable parameters, and sensible defaults to prevent data snooping and ensure consistent validation across features.

## Context / References
- `docs/library/Feature_selection/feature_validator.md` — Configuration and Defaults section (lines 486-505)
- `docs/library/Feature_selection/permutation_testing/in-sample_pt.md` — permutation test parameters
- `docs/library/Feature_selection/stability/grid_search_parameter_stability.md` — stability ratio thresholds
- `utils/enums.py` — TimeFrame enum for window size defaults
- `docs/kanban/to-do/feature_validator/INTEGRATION_TESTING_SPEC.md` — unit vs integration test standards

## Scope
In scope:
- `ValidationConfig` frozen dataclass with all configurable parameters
- Factory functions for common presets (exploratory, production, conservative)
- Validation logic to ensure pre-specification (no data-dependent defaults)
- Serialization/deserialization (JSON) for reproducibility
- Documentation of parameter meanings and recommended values

Out of scope:
- FeatureValidator API implementation (T018)
- Individual phase implementations (separate EDA, Binning, ParamSens, PermTest tasks)
- Runtime parameter adjustment (config is immutable after creation)

## Interfaces (must match)

### Add: `feature_selection/validation_config.py`

**Main configuration class:**
```python
from dataclasses import dataclass
from typing import Optional, Literal
from pathlib import Path
import json

from utils.enums import TimeFrame

@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for feature validation pipeline.

    All thresholds and parameters must be pre-specified before seeing data
    to prevent data snooping and overfitting.

    Example usage (exploratory preset):
        config = ValidationConfig.exploratory()

    Example usage (production preset):
        config = ValidationConfig.production()

    Example usage (custom):
        config = ValidationConfig(
            n_bins=20,
            selection_metric='sortino',
            metric_threshold=0.6,
            t_stat_threshold=2.5,
            significance_level=0.05,
            permutation_replicates=1000
        )
    """

    # Binning parameters (continuous features only)
    n_bins: int = 15
    """Number of quantile bins for continuous features (default: 15)."""

    selection_metric: Literal['sharpe', 'sortino', 'mean_return', 'calmar'] = 'sharpe'
    """Objective metric for binning and parameter selection (default: 'sharpe')."""

    metric_threshold: float = 0.5
    """Pre-specified threshold for objective metric (e.g., Sharpe > 0.5).
    Must be set before seeing data. Default: 0.5 for exploratory."""

    t_stat_threshold: float = 2.0
    """Minimum t-statistic for bin/region acceptance (default: 2.0)."""

    min_region_width: int = 2
    """Minimum number of consecutive bins for a tradeable region (default: 2).
    Rejects isolated single-bin spikes."""

    # EDA parameters
    rolling_window_days: Optional[int] = None
    """Rolling window size for EDA metrics (days).
    If None, auto-set based on timeframe: Daily=252, Weekly=52, Monthly=24."""

    eda_decile_bins: int = 15
    """Number of bins for decile analysis in EDA (default: 15, same as n_bins)."""

    # Parameter sensitivity parameters
    stability_ratio_threshold: float = 0.8
    """Minimum stability ratio for parameter to be in stable region (default: 0.8).
    stability_ratio = smoothed_objective / raw_objective."""

    neighbor_smoothing_steps: int = 1
    """Number of grid steps for neighbor smoothing (default: 1).
    1-step = immediate neighbors only."""

    # Permutation testing parameters
    permutation_replicates: int = 500
    """Number of permutation replicates for null distribution (default: 500).
    Increase to 1000 for production."""

    significance_level: float = 0.10
    """Significance level (alpha) for permutation tests (default: 0.10).
    0.10 = 90th percentile threshold (exploratory).
    0.05 = 95th percentile threshold (production)."""

    permutation_seed: int = 42
    """Random seed for permutation tests (determinism)."""

    pipeline_permutation_mode: Literal['feature_shuffle', 'candle_shuffle'] = 'candle_shuffle'
    """Stage 2 permutation mode for continuous features:
    - 'feature_shuffle': shuffle raw feature vector (quick screen)
    - 'candle_shuffle': shuffle bars, recompute feature (stronger null, recommended)
    Rule-based features always use 'candle_shuffle'."""

    # Walkforward stability parameters
    walkforward_fold_years: int = 2
    """Size of each walkforward fold in years (default: 2)."""

    walkforward_top_k: int = 3
    """Number of top parameters to track per fold (default: 3)."""

    walkforward_min_folds: int = 3
    """Minimum number of folds a parameter must appear in top-K (default: 3)."""

    # Output parameters
    save_plots: bool = True
    """Whether to save diagnostic plots to disk (default: True)."""

    plot_format: Literal['png', 'svg', 'pdf'] = 'png'
    """File format for saved plots (default: 'png')."""

    plot_dpi: int = 150
    """DPI for rasterized plots (default: 150)."""

    verbose: bool = True
    """Whether to log detailed progress messages (default: True)."""

    def get_rolling_window(self, timeframe: TimeFrame) -> int:
        """Get rolling window size (days) based on timeframe.

        If rolling_window_days is explicitly set, use that.
        Otherwise, use timeframe-appropriate defaults:
        - Daily: 252 days (~1 year)
        - Weekly: 52 weeks (~1 year)
        - Monthly: 24 months (~2 years)

        Args:
            timeframe: TimeFrame enum (D, W, M)

        Returns:
            Rolling window size in days
        """
        if self.rolling_window_days is not None:
            return self.rolling_window_days

        defaults = {
            TimeFrame.D: 252,
            TimeFrame.W: 52,
            TimeFrame.M: 24
        }
        return defaults.get(timeframe, 252)

    def to_json(self, path: Path) -> None:
        """Serialize configuration to JSON file.

        Args:
            path: Output file path
        """
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> 'ValidationConfig':
        """Load configuration from JSON file.

        Args:
            path: Input file path

        Returns:
            ValidationConfig instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def exploratory(cls) -> 'ValidationConfig':
        """Preset for exploratory research (relaxed thresholds).

        Characteristics:
        - Lower metric threshold (Sharpe > 0.4)
        - Higher significance level (α = 0.10)
        - Fewer permutation replicates (500)
        - Suitable for initial feature screening
        """
        return cls(
            metric_threshold=0.4,
            significance_level=0.10,
            permutation_replicates=500,
            t_stat_threshold=1.5,
            min_region_width=2
        )

    @classmethod
    def production(cls) -> 'ValidationConfig':
        """Preset for production deployment (strict thresholds).

        Characteristics:
        - Higher metric threshold (Sharpe > 0.6)
        - Lower significance level (α = 0.05)
        - More permutation replicates (1000)
        - Stricter t-stat threshold (2.5)
        - Suitable for final validation before deployment
        """
        return cls(
            metric_threshold=0.6,
            significance_level=0.05,
            permutation_replicates=1000,
            t_stat_threshold=2.5,
            min_region_width=3
        )

    @classmethod
    def conservative(cls) -> 'ValidationConfig':
        """Preset for conservative validation (very strict).

        Characteristics:
        - High metric threshold (Sharpe > 0.8)
        - Very low significance level (α = 0.01)
        - Many permutation replicates (2000)
        - Suitable for high-stakes deployment or skeptical review
        """
        return cls(
            metric_threshold=0.8,
            significance_level=0.01,
            permutation_replicates=2000,
            t_stat_threshold=3.0,
            min_region_width=3,
            stability_ratio_threshold=0.9,
            walkforward_min_folds=4
        )

    def validate_pre_specification(self) -> None:
        """Validate that all thresholds are properly pre-specified.

        Raises:
            ValueError: If any threshold is invalid or missing
        """
        if self.metric_threshold <= 0:
            raise ValueError(f"metric_threshold must be positive, got {self.metric_threshold}")

        if not 0 < self.significance_level < 1:
            raise ValueError(f"significance_level must be in (0, 1), got {self.significance_level}")

        if self.n_bins < 3:
            raise ValueError(f"n_bins must be >= 3, got {self.n_bins}")

        if self.min_region_width < 1:
            raise ValueError(f"min_region_width must be >= 1, got {self.min_region_width}")

        if self.permutation_replicates < 100:
            raise ValueError(f"permutation_replicates must be >= 100, got {self.permutation_replicates}")

        if not 0 < self.stability_ratio_threshold <= 1:
            raise ValueError(f"stability_ratio_threshold must be in (0, 1], got {self.stability_ratio_threshold}")

        if self.walkforward_fold_years < 1:
            raise ValueError(f"walkforward_fold_years must be >= 1, got {self.walkforward_fold_years}")

        if self.walkforward_top_k < 1:
            raise ValueError(f"walkforward_top_k must be >= 1, got {self.walkforward_top_k}")

    def __post_init__(self) -> None:
        """Validate configuration on initialization."""
        self.validate_pre_specification()
```

## Data Contracts

**Configuration file format (JSON):**
```json
{
  "n_bins": 15,
  "selection_metric": "sharpe",
  "metric_threshold": 0.5,
  "t_stat_threshold": 2.0,
  "min_region_width": 2,
  "rolling_window_days": null,
  "eda_decile_bins": 15,
  "stability_ratio_threshold": 0.8,
  "neighbor_smoothing_steps": 1,
  "permutation_replicates": 500,
  "significance_level": 0.1,
  "permutation_seed": 42,
  "pipeline_permutation_mode": "candle_shuffle",
  "walkforward_fold_years": 2,
  "walkforward_top_k": 3,
  "walkforward_min_folds": 3,
  "save_plots": true,
  "plot_format": "png",
  "plot_dpi": 150,
  "verbose": true
}
```

**Constraints:**
- All numeric thresholds > 0
- significance_level in (0, 1)
- permutation_replicates >= 100
- Configuration is immutable (frozen dataclass)

## Dependencies
- `utils.enums` — TimeFrame enum
- `dataclasses` — frozen dataclass
- `typing` — type hints for Literal
- `pathlib` — Path for file I/O
- `json` — serialization

## Invariants / Constraints

**Pre-specification requirement:**
- All thresholds must be set before pipeline runs
- No data-dependent defaults (e.g., can't set threshold based on mean observed metric)
- Same config used for original data and permuted data
- Thresholds cannot be modified mid-pipeline (frozen dataclass enforces this)

**Consistency across features:**
- Same config should be used for all features in a validation session
- Don't tune thresholds per feature (data snooping)
- Document any config changes between sessions

**Determinism:**
- Same config + same seed => identical results
- permutation_seed controls all random operations

**Validation on creation:**
- __post_init__ validates all thresholds
- Raises ValueError immediately if any parameter is invalid
- Prevents creation of invalid configurations

## Acceptance tests

**Unit tests:**

All config logic is pure (no I/O, no external dependencies) — unit tests are the primary test vehicle. Location: `tests/validators/test_validation_config.py`

- `test_default_config()` — construct with defaults; assert `n_bins=15`, `metric_threshold=0.5`, `significance_level=0.10`, `permutation_replicates=500`
- `test_exploratory_preset()` — `ValidationConfig.exploratory()`; assert `metric_threshold=0.4`, `significance_level=0.10`, `permutation_replicates=500`, `t_stat_threshold=1.5`
- `test_production_preset()` — `ValidationConfig.production()`; assert `metric_threshold=0.6`, `significance_level=0.05`, `permutation_replicates=1000`, `t_stat_threshold=2.5`
- `test_conservative_preset()` — `ValidationConfig.conservative()`; assert `metric_threshold=0.8`, `significance_level=0.01`, `permutation_replicates=2000`, `walkforward_min_folds=4`
- `test_rolling_window_auto_defaults()` — config with `rolling_window_days=None`; call `get_rolling_window()` for each `TimeFrame`; assert `D→252`, `W→52`, `M→24`
- `test_rolling_window_explicit()` — config with `rolling_window_days=100`; assert returns `100` for all timeframes regardless
- `test_json_round_trip()` — create config, call `to_json()` to a temp file, call `from_json()`; assert `loaded_config == original_config` for all fields
- `test_invalid_metric_threshold()` — `metric_threshold=0` raises `ValueError`
- `test_invalid_significance_level()` — `significance_level=1.5` raises `ValueError`
- `test_invalid_n_bins()` — `n_bins=2` raises `ValueError`
- `test_invalid_permutation_replicates()` — `permutation_replicates=50` raises `ValueError`
- `test_immutability()` — attempt to assign to any field on frozen instance raises `FrozenInstanceError`
- `test_custom_config()` — create config with all fields set to non-default values; assert all values preserved exactly

**Integration tests:**

`ValidationConfig` has no pipeline dependencies — integration tests are not required for config parsing and field validation. However, config loading from a real JSON file stored on disk is verified as follows:

- `test_config_load_from_real_file()` — save `ValidationConfig.production()` to a temp file path, reload with `ValidationConfig.from_json()`, assert equality. Confirms JSON round-trip works with the filesystem.
- Covered by `tests/integration/feature_validator/test_feature_validator_e2e.py`: the end-to-end test constructs `ValidationConfig.exploratory()` and passes it into the full pipeline, verifying that config values (thresholds, seed, replicates) actually govern pipeline behavior.

Location: `tests/integration/feature_validator/test_feature_validator_e2e.py` (config exercised as part of the full workflow; no standalone integration test file required).

**Cache policy:** Not applicable — config tests use no external data.

**Researcher manual verification:**
- Inspect JSON file written by `to_json()` and confirm all field values match the preset documentation
- Confirm that using `exploratory` vs `production` preset in the end-to-end test changes how many params survive validation (exploratory should pass more params than production)

## Definition of done
- [ ] Tests added under `tests/unit/feature_validator/test_validation_config.py`
- [ ] `ValidationConfig` class implemented in `feature_selection/validation_config.py`
- [ ] Factory methods (exploratory, production, conservative) implemented
- [ ] JSON serialization/deserialization implemented
- [ ] Validation logic in __post_init__ implemented
- [ ] Immutability verified (frozen=True)
- [ ] Docs updated in `docs/api/feature_selection.md`
- [ ] `pytest tests/unit/feature_validator/test_validation_config.py -q` passes

## Notes

**Recommended usage workflow:**
1. Start with exploratory preset for initial screening
2. Use production preset for final validation before deployment
3. Document any custom configs in research notes
4. Save config alongside validation reports for reproducibility

**Common customizations:**
- **High-frequency features**: Increase rolling_window_days (e.g., 500+ days)
- **Noisy markets**: Increase t_stat_threshold (e.g., 2.5 or 3.0)
- **Conservative validation**: Use conservative preset or increase permutation_replicates to 2000+
- **Quick iteration**: Reduce permutation_replicates to 250-300 (exploratory only)

**Anti-patterns to avoid:**
- Setting thresholds after seeing preliminary results (data snooping)
- Tuning thresholds per feature (overfitting)
- Modifying config mid-pipeline (breaks reproducibility)
- Using different configs for different parameter combinations of same feature

**Future extensions:**
- Bayesian parameter inference for threshold selection
- Multi-objective optimization (Pareto frontier)
- Adaptive threshold selection based on prior research (pre-committed)
