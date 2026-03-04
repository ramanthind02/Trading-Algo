"""Shared feature_research configuration. Edit here once; all phases reuse it.

Permutation uses a single shared config for in-sample and OOS phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from feature_selection.validation.config import OOSCandidateSource, PermutationModeStage2
from feature_selection.validation.objective_metrics import ObjectiveMetricSpec
from utils.core.enums import Ticker, TimeFrame

_FEATURE_RESEARCH_DIR = Path(__file__).resolve().parent

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


OBJECTIVE_METRIC_PRESETS: Mapping[str, ObjectiveMetricSpec] = build_objective_metric_presets()


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
    """Shared permutation settings for in-sample and OOS phases."""

    # Required fields (set in load_config())
    objective_metric: ObjectiveMetricSpec
    top_k: int
    min_folds_stable: int = 1
    fold_years: int = 1
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

    This is the single schema/default source used by in-sample research.

    When bin_index_max is set, only bin indices in [bin_index_min, bin_index_max]
    are considered (e.g. mean reversion tail 0–3). Set after EDA, before OOS,
    to avoid lookahead bias.
    """

    bin_counts: list[int] = field(default_factory=lambda: [10, 8, 5, 3])
    selection_metric: str = "t_stat"
    strategy: str = "long"  # For continuous long-only research, keep "long".
    metric_threshold: float = 0.0
    t_threshold: float = 2.0
    min_region_width: int = 2  # Legacy, ignored by new approach
    use_coverage_bonus: bool = False
    coverage_bonus_per_10pct: float = 0.02
    max_coverage_bonus: float = 0.2
    shrinkage_k: float = 20.0
    long_clip_min: float = 0.5
    long_clip_max: float = 2.0
    short_clip_min: float = 0.5
    short_clip_max: float = 2.0
    bin_index_min: int = 0
    bin_index_max: int | None = 0

    def __post_init__(self) -> None:
        if self.bin_index_min < 0:
            raise ValueError("bin_index_min must be >= 0")
        if self.bin_index_max is not None and self.bin_index_min > self.bin_index_max:
            raise ValueError("bin_index_max must be >= bin_index_min when set")


@dataclass(frozen=True)
class InSamplePhaseDefaultsConfig:
    """Researcher-editable in-sample defaults for a single feature type."""

    bias_spec: dict[str, Any]
    target_col: str
    strategy: str
    reports_dir: Path
    binning_params_overrides: Mapping[str, Any] = field(default_factory=dict)


def _default_continuous_in_sample_defaults(
    tf: TimeFrame = TimeFrame.D,
) -> InSamplePhaseDefaultsConfig:
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
        strategy="long",
        reports_dir=_FEATURE_RESEARCH_DIR / "in_sample" / "results" / "continuous" / "rsi",
        binning_params_overrides={
            "bin_counts": [10],
            "t_threshold": 1,
            "use_coverage_bonus": False,
        },
    )


def _default_rule_based_in_sample_defaults(
    tf: TimeFrame = TimeFrame.D,
) -> InSamplePhaseDefaultsConfig:
    return InSamplePhaseDefaultsConfig(
        bias_spec={
            "module_name": "rsi_signal",
            "timeframes": [tf],
            "params": {
                "rsi_period": [2],
                "oversold": list(range(5, 31, 5)),
                "overbought": list(range(95, 64, -5)),
                "strategy_mode": "long",
                "exit_policy": "threshold_or_bars",
                "exit_bars": 5,
            },
        },
        target_col="log_return_atr",
        strategy="long",
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
class BaseResearchConfig:
    """Shared researcher-editable settings. Phases add reports_dir and phase-specific fields."""

    tickers: list[Ticker]
    start: datetime
    end: datetime
    use_cache: bool
    populate_cache: bool
    permutation: PermutationResearchConfig
    timeframe: TimeFrame = TimeFrame.D
    feature_type: FeatureType = FeatureType.CONTINUOUS
    in_sample_defaults: InSampleDefaultsCatalog = field(
        default_factory=InSampleDefaultsCatalog.default_for
    )
    param_sensitivity: ParamSensitivityConfig = field(default_factory=ParamSensitivityConfig)
    validation_window: OOSWindowConfig | None = None
    oos_window: OOSWindowConfig | None = None
    # Flat evaluation fields
    top_k: int = 1
    objective_metric_key: str = "t_stat"
    smoothing_self_weight: float = 3.0
    n_jobs: int = 8
    output_root: Path = field(default_factory=lambda: Path("feature_research/shared_results"))


def load_config() -> BaseResearchConfig:
    """Single source of truth for tickers, date range, cache, and phase defaults.

    Edit here; in_sample and OOS phases import and extend this.
    """
    # ==========================================================================
    # EDIT BELOW
    # ==========================================================================
    tickers = [
        Ticker.ES,
        Ticker.NQ
    ]
    start = datetime(2000, 1, 1)
    end = datetime(2017, 12, 31)
    timeframe = TimeFrame.D
    use_cache = True
    populate_cache = True
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
        top_k=1,
        min_folds_stable=1,
        fold_years=1,
        enabled=True,
    )

    feature_type = FeatureType.CONTINUOUS
    in_sample_defaults = InSampleDefaultsCatalog.default_for(timeframe)
    param_sensitivity = ParamSensitivityConfig()
    # ==========================================================================
    # EDIT ABOVE
    # ==========================================================================

    return BaseResearchConfig(
        tickers=tickers,
        start=start,
        end=end,
        use_cache=use_cache,
        populate_cache=populate_cache,
        permutation=permutation,
        timeframe=timeframe,
        feature_type=feature_type,
        in_sample_defaults=in_sample_defaults,
        param_sensitivity=param_sensitivity,
        validation_window=validation_window,
        oos_window=oos_window,
        top_k=1,
        objective_metric_key="t_stat",
        smoothing_self_weight=3.0,
        n_jobs=8,
        output_root=Path("feature_research/shared_results"),
    )
