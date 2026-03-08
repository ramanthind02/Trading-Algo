"""Shared feature_research configuration. Edit here once; all phases reuse it.

Permutation uses a single shared config for in-sample and OOS phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from feature_selection.validation.config import OOSCandidateSource, PermutationModeStage2
from feature_selection.validation.objective_metrics import ObjectiveMetricSpec
from utils.core.enums import Direction, DirectionInput, Ticker, TimeFrame, coerce_direction

_FEATURE_RESEARCH_DIR = Path(__file__).resolve().parent

# Single place to change default timeframe for research (in_sample, OOS, presets).
DEFAULT_TIMEFRAME: TimeFrame = TimeFrame.D

RAW_TARGET_COLS: frozenset[str] = frozenset({"log_return", "raw_return"})


def build_objective_metric_presets(tf: TimeFrame = TimeFrame.D) -> dict[str, ObjectiveMetricSpec]:
    """Reusable objective metric presets for research/permutation configs.

    Keep this catalog researcher-friendly so in-sample / OOS stages can
    switch objectives by key without editing resolver internals.
    """
    return {
        "sortino": ObjectiveMetricSpec(builtin="sortino"),
        "sharpe": ObjectiveMetricSpec(builtin="sharpe"),
        "calmar": ObjectiveMetricSpec(builtin="calmar"),
        "t_stat": ObjectiveMetricSpec(builtin="t_stat"),
        "profit_factor": ObjectiveMetricSpec(builtin="profit_factor"),
        # Timeframe-aware annualized variants (useful when comparing to tearsheet ratios).
        "sortino_annualized": ObjectiveMetricSpec(
            builtin="sortino",
            kwargs={"annualization_factor": float(tf.bars_per_year)},
        ),
        "sharpe_annualized": ObjectiveMetricSpec(
            builtin="sharpe",
            kwargs={"annualization_factor": float(tf.bars_per_year)},
        ),
        "calmar_annualized": ObjectiveMetricSpec(
            builtin="calmar",
            kwargs={"annualization_factor": float(tf.bars_per_year)},
        ),
    }


@dataclass(frozen=True)
class OOSWindowConfig:
    """Explicit train/test window for a single out-of-sample fold.

    Set in load_config(). When None on BaseResearchConfig, OOS can be disabled.
    """

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    def __post_init__(self) -> None:
        if self.train_end >= self.test_start:
            raise ValueError(
                "OOSWindowConfig: train_end must be before test_start "
                f"(got train_end={self.train_end!s}, test_start={self.test_start!s})."
            )
        if self.test_start >= self.test_end:
            raise ValueError(
                "OOSWindowConfig: test_start must be before test_end "
                f"(got test_start={self.test_start!s}, test_end={self.test_end!s})."
            )


class FeatureType(str, Enum):
    """Feature type determines validation path."""

    CONTINUOUS = "continuous"
    RULE_BASED = "rule_based"


@dataclass(frozen=True)
class PermutationResearchConfig:
    """Shared permutation settings for in-sample and OOS phases (Stage 1 + 2 only; no walkforward Stage 3)."""

    # Required fields (set in load_config())
    objective_metric: ObjectiveMetricSpec
    # Permutation-only
    enabled: bool = False
    nreps_stage1: int = 1000   # vector shuffle (cheaper per rep; 1000 default)
    nreps_stage2: int = 100   # candle shuffle (expensive)
    alpha: float = 0.1
    metric_threshold: float = 0.0
    random_seed: int | None = 42
    permutation_mode_stage2: PermutationModeStage2 = "candle_shuffle"
    n_jobs_stage1_reps: int = 8
    n_jobs_stage2_reps: int = 8
    run_stage1: bool = True
    run_stage2: bool = False
    # OOS
    candidate_source: OOSCandidateSource = "stage2_passers"


@dataclass(frozen=True)
class ParamSensitivityConfig:
    """Settings for EDA parameter sensitivity plots and smoothing helpers."""

    stability_threshold: float = 0.5  # for plot shading threshold when stable_regions are provided
    plot_3d_mode: str = "surface_slices"  # "heatmap_slices" | "surface_slices"
    max_eda_output_combos: int = 25  # max EDA folders saved for rule-based; 0 = no limit
    # Center param weight for neighbor smoothing.
    smoothing_self_weight: float = 3.0
    # Visualization-only floor for parameter sensitivity metrics.
    # When not None, parameter sensitivity plots clip metric values from below
    # at this threshold (e.g. t-stat ≈ 2) so sub-tradable regions don't
    # dominate the color scale.
    metric_floor: float | None = 2.0


@dataclass(frozen=True)
class BinningAnalysisConfig:
    """Shared continuous binning model parameter config.

    Continuous binning outputs binary signals (0 / 1 for long, 0 / -1 for short).
    When bin_index_max is set, only bin indices in [bin_index_min, bin_index_max]
    are considered (e.g. mean reversion tail 0–3). Set after EDA, before OOS,
    to avoid lookahead bias.
    """

    bin_counts: list[int] = field(default_factory=lambda: [10, 8, 5, 3])
    strategy: DirectionInput = Direction.LONG
    bin_index_min: int = 0
    bin_index_max: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "strategy",
            coerce_direction(self.strategy, field_name="binning_params.strategy"),
        )
        if self.bin_index_min < 0:
            raise ValueError("bin_index_min must be >= 0")
        if self.bin_index_max is not None and self.bin_index_min > self.bin_index_max:
            raise ValueError("bin_index_max must be >= bin_index_min when set")


@dataclass(frozen=True)
class InSamplePhaseDefaultsConfig:
    """Researcher-editable in-sample defaults for a single feature type."""

    bias_spec: dict[str, Any]
    target_col: str
    strategy: DirectionInput
    reports_dir: Path
    binning_params_overrides: dict[str, Any] = field(default_factory=dict)
    eval_bias_spec: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "strategy",
            coerce_direction(self.strategy, field_name="in_sample_defaults.strategy"),
        )


def _default_continuous_in_sample_defaults(
    tf: TimeFrame = TimeFrame.D,
) -> InSamplePhaseDefaultsConfig:
    """Default continuous in-sample research settings.

    After running IS EDA, you can optionally set eval_bias_spec (via
    InSampleDefaultsCatalog in load_config()) to pin a specific param combo
    for validation+OOS. For example:

        eval_bias_spec={
            "module_name": "cyclical_rsi",
            "timeframes": [tf],
            "params": {"short_period": [4], "long_period": [120], "rsi_period": [2]},
        }
    """
    return InSamplePhaseDefaultsConfig(
        bias_spec={
            "module_name": "cyclical_rsi",
            "timeframes": [tf],
            "params": {
                "short_period": [4],
                "long_period": [120],
                "rsi_period": [2]
            },
        },
        target_col="log_return_atr",
        strategy=Direction.LONG,
        reports_dir=_FEATURE_RESEARCH_DIR / "in_sample" / "results" / "continuous" / "rsi",
        binning_params_overrides={"bin_counts": [10]},
    )


def _default_rule_based_in_sample_defaults(
    tf: TimeFrame = TimeFrame.D,
) -> InSamplePhaseDefaultsConfig:
    """Rule-based preset: buy_hold bias node (no params). Uses DEFAULT_TIMEFRAME."""
    return InSamplePhaseDefaultsConfig(
        bias_spec={
            "module_name": "eoy_sp500",
            "timeframes": [DEFAULT_TIMEFRAME],
            "params": {},
        },
        target_col="log_return_atr",
        strategy=Direction.LONG,
        reports_dir=_FEATURE_RESEARCH_DIR / "in_sample" / "results" / "rule_based",
    )


@dataclass(frozen=True)
class InSampleDefaultsCatalog:
    """Single source of truth for in-sample phase presets by feature type."""

    continuous: InSamplePhaseDefaultsConfig
    rule_based: InSamplePhaseDefaultsConfig

    @classmethod
    def default_for(cls, tf: TimeFrame = TimeFrame.D) -> "InSampleDefaultsCatalog":
        return cls(
            continuous=_default_continuous_in_sample_defaults(tf),
            rule_based=_default_rule_based_in_sample_defaults(tf),
        )

    def for_feature_type(self, feature_type: FeatureType) -> InSamplePhaseDefaultsConfig:
        match feature_type:
            case FeatureType.CONTINUOUS:
                return self.continuous
            case FeatureType.RULE_BASED:
                return self.rule_based
        raise ValueError(f"Unknown feature_type: {feature_type}")


@dataclass(frozen=True)
class VaultSaveConfig:
    """Optional vault save target. When set, save_to_vault script persists the research model here."""

    ensemble_name: str
    direction: DirectionInput
    params_to_save: dict[str, Any] | None = None  # single param combo; None = first from bias_spec

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "direction",
            coerce_direction(self.direction, field_name="vault_save.direction"),
        )


@dataclass(frozen=True)
class ResearchConfig:
    """Unified research configuration for all phases (in-sample, OOS, validation).

    Edit load_config() below to customize for your research.
    """

    tickers: list[Ticker]
    start: datetime
    end: datetime
    use_cache: bool
    populate_cache: bool
    permutation: PermutationResearchConfig
    objective_metric_presets: dict[str, ObjectiveMetricSpec]
    binning_params: BinningAnalysisConfig
    timeframe: TimeFrame = TimeFrame.D
    feature_type: FeatureType = FeatureType.CONTINUOUS
    in_sample_defaults: InSampleDefaultsCatalog = field(
        default_factory=InSampleDefaultsCatalog.default_for
    )
    param_sensitivity: ParamSensitivityConfig = field(default_factory=ParamSensitivityConfig)
    validation_window: OOSWindowConfig | None = None
    oos_window: OOSWindowConfig | None = None
    # Flat evaluation fields
    n_jobs: int = 8
    output_root: Path = field(default_factory=lambda: Path("feature_research/shared_results"))
    generate_ticker_tearsheets: bool = False
    vault_save: VaultSaveConfig | None = None
    sector_allocation_config_path: str | None = None

    @property
    def _phase_defaults(self) -> InSamplePhaseDefaultsConfig:
        return self.in_sample_defaults.for_feature_type(self.feature_type)

    @property
    def bias_spec(self) -> dict[str, Any]:
        """Full EDA grid — used by in-sample pipeline."""
        return self._phase_defaults.bias_spec

    @property
    def eval_bias_spec(self) -> dict[str, Any]:
        """Narrowed spec for validation+OOS. Falls back to bias_spec when not set."""
        phase_defaults = self._phase_defaults
        return phase_defaults.eval_bias_spec or phase_defaults.bias_spec

    @property
    def target_col(self) -> str:
        return self._phase_defaults.target_col

    @property
    def strategy(self) -> Direction:
        return self._phase_defaults.strategy

    @property
    def reports_dir(self) -> Path:
        return self._phase_defaults.reports_dir


# Backwards-compatible alias for older code/tests.
BaseResearchConfig = ResearchConfig


def load_config() -> ResearchConfig:
    """Single source of truth for tickers, date range, cache, and phase defaults.

    Edit here; all phases (in-sample, OOS, validation) use this config.
    """
    # ==========================================================================
    # EDIT BELOW
    # ==========================================================================
    tickers = [

        Ticker.ES

    ]
    sector_allocation_config_path = str(
        _FEATURE_RESEARCH_DIR / "config" / "sector_buy_hold_60_20_20.json"
    )
    start = datetime(2000, 1, 1)
    end = datetime(2025, 9, 18)
    timeframe = DEFAULT_TIMEFRAME
    use_cache = True
    populate_cache = True
    # Build metric presets with correct timeframe (not hardcoded daily)
    objective_metric_presets = build_objective_metric_presets(timeframe)

    validation_window = OOSWindowConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2017, 12, 31),
        test_start=datetime(2018, 1, 1),
        test_end=datetime(2022, 12, 31),
    )

    # OOS: single fold with explicit train/test dates. test_end must be within
    # available OHLC data (data/ohlc_data); otherwise no tickers pass coverage.
    # Example with data through 2025: train 2000–2022, test 2023–2025.
    # Set to None to disable OOS.
    oos_window = OOSWindowConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2022, 12, 31),
        test_start=datetime(2023, 1, 1),
        test_end=datetime(2025, 9, 18),
    )

    permutation = PermutationResearchConfig(
        objective_metric=objective_metric_presets["t_stat"],
        enabled=True,
    )

    feature_type = FeatureType.RULE_BASED
    in_sample_defaults = InSampleDefaultsCatalog.default_for(timeframe)
    # Example: after IS EDA, set eval_bias_spec to pin specific param combos
    # for validation+OOS:
    #
    # in_sample_defaults = InSampleDefaultsCatalog(
    #     continuous=InSamplePhaseDefaultsConfig(
    #         bias_spec={
    #             "module_name": "cyclical_rsi",
    #             "timeframes": [timeframe],
    #             "params": {
    #                 "short_period": [3, 4, 5, 6, 7],   # wide EDA grid
    #                 "long_period": [100, 120, 140],
    #                 "rsi_period": [2, 3, 4],
    #             },
    #         },
    #         eval_bias_spec={  # narrowed after IS analysis
    #             "module_name": "cyclical_rsi",
    #             "timeframes": [timeframe],
    #             "params": {"short_period": [4], "long_period": [120], "rsi_period": [2]},
    #         },
    #         target_col="log_return_atr",
    #         strategy="long",
    #         reports_dir=_FEATURE_RESEARCH_DIR / "in_sample" / "results" / "continuous" / "rsi",
    #     ),
    #     rule_based=_default_rule_based_in_sample_defaults(timeframe),
    # )
    param_sensitivity = ParamSensitivityConfig()
    # Optional: set to save research model to vault after satisfying results.
    # vault_save = VaultSaveConfig(
    #     ensemble_name="mean_reversion_indices",
    #     direction=Direction.LONG,
    #     params_to_save={"short_period": 4, "long_period": 120, "rsi_period": 2},
    # )
    vault_save = VaultSaveConfig(
        ensemble_name="mr_indices",
        direction=Direction.LONG,
        params_to_save={},
    )
    # ==========================================================================
    # EDIT ABOVE
    # ==========================================================================

    # Align binning_params.strategy with phase default so portfolio build and reporting use same strategy.
    binning_params = BinningAnalysisConfig(
        strategy=in_sample_defaults.for_feature_type(feature_type).strategy,
    )

    return ResearchConfig(
        tickers=tickers,
        start=start,
        end=end,
        use_cache=use_cache,
        populate_cache=populate_cache,
        permutation=permutation,
        objective_metric_presets=objective_metric_presets,
        binning_params=binning_params,
        timeframe=timeframe,
        feature_type=feature_type,
        in_sample_defaults=in_sample_defaults,
        param_sensitivity=param_sensitivity,
        validation_window=validation_window,
        oos_window=oos_window,
        n_jobs=8,
        output_root=Path("feature_research/shared_results"),
        generate_ticker_tearsheets=True,
        vault_save=vault_save,
        sector_allocation_config_path=sector_allocation_config_path,
    )
