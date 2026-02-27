"""Shared feature_research configuration. Edit here once; all phases reuse it.

Permutation uses a single shared config for in-sample, walkforward, and OOS phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from feature_research.walkforward.config import (
    MemberPredictionMode,
    WalkforwardResearchConfig,
    WalkforwardSelectionMethod,
    WeightLayerAlgorithm,
)
from feature_research.walkforward.marginal_peak_selection import MarginalPeakConfig
from feature_selection.validation.config import OOSCandidateSource, PermutationModeStage2
from feature_selection.validation.objective_metrics import ObjectiveMetricSpec
from utils.core.enums import Ticker, TimeFrame

_FEATURE_RESEARCH_DIR = Path(__file__).resolve().parent

RAW_TARGET_COLS: frozenset[str] = frozenset({"log_return", "raw_return"})


def build_objective_metric_presets() -> dict[str, ObjectiveMetricSpec]:
    """Reusable objective metric presets for research/permutation configs.

    Keep this catalog researcher-friendly so in-sample / walkforward / OOS stages can
    switch objectives by key without editing resolver internals.
    """
    return {
        "sortino": ObjectiveMetricSpec(builtin="sortino"),
        "sharpe": ObjectiveMetricSpec(builtin="sharpe"),
        "calmar": ObjectiveMetricSpec(builtin="calmar"),
        "t_stat": ObjectiveMetricSpec(builtin="t_stat"),
        "profit_factor": ObjectiveMetricSpec(builtin="profit_factor"),
        # Daily-bar annualized variants (useful when comparing to tearsheet ratios).
        "sortino_252": ObjectiveMetricSpec(
            builtin="sortino",
            kwargs={"annualization_factor": 252.0},
        ),
        "sharpe_252": ObjectiveMetricSpec(
            builtin="sharpe",
            kwargs={"annualization_factor": 252.0},
        ),
        "calmar_252": ObjectiveMetricSpec(
            builtin="calmar",
            kwargs={"annualization_factor": 252.0},
        ),
    }


OBJECTIVE_METRIC_PRESETS: Mapping[str, ObjectiveMetricSpec] = build_objective_metric_presets()


@dataclass(frozen=True)
class GlobalResearchDefaults:
    """Single source of truth for shared research knobs across walkforward and permutation.

    Set once in load_config(); WalkforwardDefaultsConfig and PermutationResearchConfig
    are built from this so top_k, objective, fold_years, etc. stay aligned and DRY.
    Window-based: train_window_years and test_window_years define fold boundaries;
    test_step (days) is derived as int(test_window_years * 365) when building WF config.
    """

    objective_metric_key: str = "t_stat"  # key in OBJECTIVE_METRIC_PRESETS
    top_k: int = 1  # load_config() uses this; keep aligned so effective default is 1
    min_folds_stable: int = 1
    fold_years: int = 1
    train_window_years: float = 15.0
    test_window_years: float = 2.0
    num_steps: int = 4
    selection_method: WalkforwardSelectionMethod = WalkforwardSelectionMethod.MARGINAL_PEAK


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


def compute_first_fold_bounds(
    start: datetime,
    end: datetime,
    train_window_years: float,
    test_window_years: float,
    num_steps: int,
) -> tuple[datetime, datetime, datetime, datetime]:
    """Compute the first fold's train/test boundaries (walkforward scheme).

    Uses the same formula as build_walkforward: first fold's train_end = end
    - num_steps * test_step_days; train_start = train_end - train_window;
    test period = (train_end + 1 day) through (train_end + test_step_days).
    Keeps portfolio in-sample period aligned with feature_research.

    Returns
    -------
    tuple[datetime, datetime, datetime, datetime]
        (train_start, train_end, test_start, test_end) for the first fold.
    """
    test_step_days = int(test_window_years * 365)
    train_end_first = end - timedelta(days=num_steps * test_step_days)
    if train_end_first <= start:
        train_end_first = end - timedelta(days=test_step_days)
    train_start_first = train_end_first - timedelta(days=int(train_window_years * 365))
    if train_start_first < start:
        train_start_first = start
    test_start_first = train_end_first + timedelta(days=1)
    test_end_first = train_end_first + timedelta(days=test_step_days)
    return train_start_first, train_end_first, test_start_first, test_end_first


class FeatureType(str, Enum):
    """Feature type determines validation path."""

    CONTINUOUS = "continuous"
    RULE_BASED = "rule_based"


@dataclass(frozen=True)
class PermutationResearchConfig:
    """Shared permutation settings for in-sample, walkforward, and OOS phases.

    Global-sourced fields (first four) must be passed from GlobalResearchDefaults
    in load_config(); no defaults so global remains the single source of truth.
    """

    # From GlobalResearchDefaults (set in load_config() only)
    objective_metric: ObjectiveMetricSpec
    top_k: int
    min_folds_stable: int
    fold_years: int
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
class WalkforwardDefaultsConfig:
    """Walkforward options; global-sourced fields are required and set from GlobalResearchDefaults in load_config().

    build_walkforward() derives train_start/train_end/test_step from train_window_years,
    test_window_years, and num_steps. No defaults for shared knobs so global remains
    the single source of truth.
    """

    # From GlobalResearchDefaults (set in load_config() only)
    top_k: int
    train_window_years: float
    test_window_years: float
    num_steps: int
    selection_method: WalkforwardSelectionMethod
    min_folds_stable: int
    fold_years: int
    objective_metric_name: str
    # Walkforward-only
    enabled: bool = False
    output_root: Path = field(default_factory=lambda: Path("feature_research/shared_results"))
    output_per_fold_tearsheets: bool = False
    weight_layer_algorithm: WeightLayerAlgorithm = WeightLayerAlgorithm.INVERSE_CORRELATION
    member_prediction_mode: MemberPredictionMode = MemberPredictionMode.BINARY
    run_oracle_baseline: bool = True
    n_jobs: int = 8  # parallel jobs for scoring param combos within each fold; -1 = all CPUs
    # Center param weight relative to each 1-step neighbor in smoothing.
    # 1.0 = equal weight (most aggressive); 2.0–3.0 reduces boundary-param dilution.
    # Should match ParamSensitivityConfig.smoothing_self_weight so EDA and WF use the same landscape.
    smoothing_self_weight: float = 3.0


@dataclass(frozen=True)
class ParamSensitivityConfig:
    """Settings for parameter sensitivity and Marginal Peak Selection (MPS).

    Used by the param_sensitivity notebook and anywhere that calls
    generate_parameter_sensitivity_report(). Selection is done via MPS only.
    See docs/library/Feature_selection/Phase_2_WF/param_stability.md.
    """

    stability_threshold: float = 0.5  # for plot shading threshold when stable_regions are provided
    plot_3d_mode: str = "surface_slices"  # "heatmap_slices" | "surface_slices"
    max_eda_output_combos: int = 25  # max EDA folders saved for rule-based; 0 = no limit
    # Center param weight for neighbor smoothing.
    # Must match WalkforwardDefaultsConfig.smoothing_self_weight so EDA and WF see the same landscape.
    smoothing_self_weight: float = 3.0
    # Marginal Peak Selection (MPS): min gap to declare dominant regime; fallback when gap < min_gap.
    marginal_min_gap: float = 0.10
    marginal_min_cell_size: int = 2
    marginal_fallback_k: int = 5
    marginal_dim: int = 2  # 2 = C(D,2) pairwise tables, 3 = C(D,3) 3D tables (e.g. for 5D+ grids)


@dataclass(frozen=True)
class BinningAnalysisConfig:
    """Shared continuous binning model parameter config.

    This is the single schema/default source used by in-sample research and
    walkforward helpers that depend on binning settings.

    When bin_index_max is set, only bin indices in [bin_index_min, bin_index_max]
    are considered (e.g. mean reversion tail 0–3). Set after EDA, before OOS,
    to avoid lookahead bias.
    """

    bin_counts: list[int] = field(default_factory=lambda: [10, 8, 5, 3])
    selection_metric: str = "t_stat"
    strategy: str = "long"  # For continuous long-only research, keep "long"; walkforward pipeline does not override.
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
    walkforward_train_window_years: float | None = None
    walkforward_test_window_years: float | None = None
    walkforward_num_steps: int | None = None
    walkforward_enabled: bool | None = None


def _default_continuous_in_sample_defaults() -> InSamplePhaseDefaultsConfig:
    return InSamplePhaseDefaultsConfig(
        bias_spec={
            "module_name": "cumulative_rsi",
            "timeframes": [TimeFrame.D],
            "params": {"lookback": [2,3,4],
            "avg_period": [2,3]},
        },
        target_col="log_return_atr",
        strategy="long",
        reports_dir=_FEATURE_RESEARCH_DIR / "in_sample" / "results" / "continuous" / "rsi",
        binning_params_overrides={
            "bin_counts": [12,11,10,9,8,7],
            "t_threshold": 1,
            "use_coverage_bonus": False,
        },
    )


def _default_rule_based_in_sample_defaults() -> InSamplePhaseDefaultsConfig:
    return InSamplePhaseDefaultsConfig(
        bias_spec={
            "module_name": "rsi_signal",
            "timeframes": [TimeFrame.D],
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

    continuous: InSamplePhaseDefaultsConfig = field(
        default_factory=_default_continuous_in_sample_defaults
    )
    rule_based: InSamplePhaseDefaultsConfig = field(
        default_factory=_default_rule_based_in_sample_defaults
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
    """Shared researcher-editable settings. Phases add reports_dir and phase-specific fields.

    permutation and walkforward_defaults are required; load_config() builds them from
    GlobalResearchDefaults so global is the single source of truth.
    """

    tickers: list[Ticker]
    start: datetime
    end: datetime
    use_cache: bool
    populate_cache: bool
    permutation: PermutationResearchConfig
    walkforward_defaults: WalkforwardDefaultsConfig
    feature_type: FeatureType = FeatureType.CONTINUOUS
    in_sample_defaults: InSampleDefaultsCatalog = field(default_factory=InSampleDefaultsCatalog)
    param_sensitivity: ParamSensitivityConfig = field(default_factory=ParamSensitivityConfig)
    oos_window: OOSWindowConfig | None = None

    def build_walkforward(
        self,
        *,
        train_window_years: float | None = None,
        test_window_years: float | None = None,
        num_steps: int | None = None,
        enabled: bool | None = None,
    ) -> WalkforwardResearchConfig:
        """Build walkforward config from window-based spec.

        First fold's train_end = end - num_steps * test_step_days; train_start = train_end - train_window.
        ``top_k`` is sourced exclusively from ``walkforward_defaults.top_k``.
        """
        wf_defaults = self.walkforward_defaults
        tw_years = train_window_years if train_window_years is not None else wf_defaults.train_window_years
        test_years = test_window_years if test_window_years is not None else wf_defaults.test_window_years
        steps = num_steps if num_steps is not None else wf_defaults.num_steps
        train_start_first, train_end_first, _, _ = compute_first_fold_bounds(
            self.start, self.end, tw_years, test_years, steps
        )
        test_step_days = int(test_years * 365)
        top_k = wf_defaults.top_k  # single source of truth
        if wf_defaults.selection_method == WalkforwardSelectionMethod.MARGINAL_PEAK:
            ps = self.param_sensitivity
            marginal_peak: object = MarginalPeakConfig(
                k_max=top_k,
                min_gap=ps.marginal_min_gap,
                min_cell_size=ps.marginal_min_cell_size,
                fallback_k=ps.marginal_fallback_k,
                marginal_dim=ps.marginal_dim,
            )
        else:
            marginal_peak = None
        return WalkforwardResearchConfig(
            train_start=train_start_first,
            train_end=train_end_first,
            enabled=enabled if enabled is not None else wf_defaults.enabled,
            test_step=test_step_days,
            num_steps=steps,
            top_k=top_k,
            objective_metric_name=wf_defaults.objective_metric_name,
            selection_method=wf_defaults.selection_method,
            marginal_peak=marginal_peak,
            weight_layer_algorithm=wf_defaults.weight_layer_algorithm,
            member_prediction_mode=wf_defaults.member_prediction_mode,
            output_root=wf_defaults.output_root,
            run_oracle_baseline=wf_defaults.run_oracle_baseline,
            output_per_fold_tearsheets=wf_defaults.output_per_fold_tearsheets,
            n_jobs=wf_defaults.n_jobs,
            smoothing_self_weight=wf_defaults.smoothing_self_weight,
        )



def load_config() -> BaseResearchConfig:
    """Single source of truth for tickers, date range, cache, and phase defaults.

    Edit here; in_sample and walkforward phases import and extend this.
    """
    # ==========================================================================
    # EDIT BELOW
    # ==========================================================================
    tickers = [
        Ticker.ES,
        Ticker.NQ

    ]
    start = datetime(2000, 1, 1)
    end = datetime(2023, 12, 30)
    use_cache = True
    populate_cache = True

    # Global shared knobs: one place for top_k, objective, fold_years, windows, etc.
    # Used by both walkforward and permutation so they stay aligned.
    global_defaults = GlobalResearchDefaults(
        objective_metric_key="t_stat",
        top_k=1,
        min_folds_stable=1,
        fold_years=1,
        train_window_years=17.0,
        test_window_years=1.0,
        num_steps=8,
        selection_method=WalkforwardSelectionMethod.MARGINAL_PEAK,
    )

    walkforward_defaults = WalkforwardDefaultsConfig(
        global_defaults.top_k,
        global_defaults.train_window_years,
        global_defaults.test_window_years,
        global_defaults.num_steps,
        global_defaults.selection_method,
        global_defaults.min_folds_stable,
        global_defaults.fold_years,
        global_defaults.objective_metric_key,
        enabled=True,
        output_root=Path("feature_research/shared_results"),
        output_per_fold_tearsheets=False,
        weight_layer_algorithm=WeightLayerAlgorithm.INVERSE_CORRELATION,
        member_prediction_mode=MemberPredictionMode.BINARY,
        run_oracle_baseline=False,
    )

    # OOS: single fold with explicit train/test dates (e.g. train 2007–2023, test 2024–2025).
    # Set to None to disable OOS.
    oos_window = OOSWindowConfig(
        train_start=datetime(2009, 1, 1),
        train_end=datetime(2023, 12, 30),
        test_start=datetime(2024, 1, 1),
        test_end=datetime(2025, 12, 31),
    )

    permutation = PermutationResearchConfig(
        OBJECTIVE_METRIC_PRESETS["t_stat"],
        global_defaults.top_k,
        global_defaults.min_folds_stable,
        global_defaults.fold_years,
        enabled=True,
    )

    feature_type = FeatureType.RULE_BASED
    in_sample_defaults = InSampleDefaultsCatalog()
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
        feature_type=feature_type,
        in_sample_defaults=in_sample_defaults,
        walkforward_defaults=walkforward_defaults,
        param_sensitivity=param_sensitivity,
        oos_window=oos_window,
    )
