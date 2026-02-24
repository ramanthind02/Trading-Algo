"""Shared feature_research configuration. Edit here once; all phases reuse it."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from feature_research.walkforward.config import (
    MemberPredictionMode,
    WalkforwardResearchConfig,
    WalkforwardSelectionMethod,
    WeightLayerAlgorithm,
)
from feature_research.walkforward.stable_region_selection import StableRegionConfig
from feature_selection.validation.config import PermutationModeStage2
from feature_selection.validation.objective_metrics import ObjectiveMetricSpec
from utils.core.enums import Ticker, TimeFrame

_FEATURE_RESEARCH_DIR = Path(__file__).resolve().parent

RAW_TARGET_COLS: frozenset[str] = frozenset({"log_return", "raw_return"})


class FeatureType(str, Enum):
    """Feature type determines validation path."""

    CONTINUOUS = "continuous"
    RULE_BASED = "rule_based"


@dataclass(frozen=True)
class PermutationSuiteConfig:
    """Settings for the in-sample permutation test suite (Stage 1/2 only)."""

    enabled: bool = False
    nreps: int = 100
    alpha: float = 0.05
    metric_threshold: float = 0.0
    top_k: int = 3
    min_folds_stable: int = 1
    random_seed: int | None = 42
    permutation_mode_stage2: PermutationModeStage2 = "candle_shuffle"
    n_jobs_stage2_reps: int = 8  # Rep-level multiprocessing (joblib/loky); >1 gives real parallelism
    run_stage1: bool = True
    run_stage2: bool = True
    fold_years: int = 1
    objective_metric: ObjectiveMetricSpec = field(
        default_factory=lambda: ObjectiveMetricSpec(builtin="sortino")
    )


@dataclass(frozen=True)
class ParamSensitivityConfig:
    """Settings for parameter sensitivity / stable region selection (notebook and reports).

    Used by the param_sensitivity notebook and anywhere that calls
    generate_parameter_sensitivity_report(). Tune fields below to match
    docs/library/Feature_selection/Phase_2_WF/param_stability.md.
    """

    use_floor_based_selection: bool = True
    stability_threshold: float = 0.8  # used only when use_floor_based_selection is False
    top_k: int = 5
    plot_3d_mode: str = "surface_slices"  # "heatmap_slices" | "surface_slices"
    # Stable region algo (forwarded to StableRegionConfig)
    floor_method: str = "adaptive"  # "adaptive" (σ-based) or "relative" (fixed δ)
    delta: float = 0.20  # for floor_method="relative": floor = best × (1 - delta)
    adaptive_sigma_multiplier: float = 0.7 # for floor_method="adaptive": floor = best - this × σ
    k_per_region: int = 5
    k_max: int = 5
    min_region_size: int = 2
    bin_count_min: int = 0  # 0 = no filter (in-sample); use 5+ for walkforward continuous

    @property
    def stable_region_config(self) -> StableRegionConfig:
        """Build StableRegionConfig from this config’s fields."""
        return StableRegionConfig(
            floor_method=self.floor_method,
            delta=self.delta,
            adaptive_sigma_multiplier=self.adaptive_sigma_multiplier,
            k_per_region=self.k_per_region,
            k_max=self.k_max,
            min_region_size=self.min_region_size,
            bin_count_min=self.bin_count_min,
        )


@dataclass(frozen=True)
class BaseResearchConfig:
    """Shared researcher-editable settings. Phases add reports_dir and phase-specific fields."""

    tickers: list[Ticker]
    start: datetime
    end: datetime
    use_cache: bool
    populate_cache: bool
    permutation_suite: PermutationSuiteConfig
    feature_type: FeatureType = FeatureType.CONTINUOUS
    walkforward_test_step: int = 365
    walkforward_num_steps: int = 8
    walkforward_output_root: Path = Path("feature_research/shared_results")
    walkforward_selection_method: WalkforwardSelectionMethod = WalkforwardSelectionMethod.STABLE_REGION
    weight_layer_algorithm: WeightLayerAlgorithm = WeightLayerAlgorithm.INVERSE_CORRELATION
    walkforward_member_prediction_mode: MemberPredictionMode = MemberPredictionMode.SHARPE_WEIGHTED
    param_sensitivity: ParamSensitivityConfig = field(default_factory=ParamSensitivityConfig)

    def build_walkforward(
        self,
        *,
        test_step: int | None = None,
        num_steps: int | None = None,
        train_window_years: float | None = None,
        enabled: bool = False,
    ) -> WalkforwardResearchConfig:
        """Build walkforward config. Use train_window_years for fixed first-fold train length (e.g. 15, 10, 5).

        When walkforward_selection_method is STABLE_REGION, stable_region and top_k are taken
        from param_sensitivity (stable_region_config and top_k). For TOP_K/ENHANCED only top_k is passed.
        """
        step = test_step if test_step is not None else self.walkforward_test_step
        steps = num_steps if num_steps is not None else self.walkforward_num_steps
        train_end = self.end - timedelta(days=step * steps)
        if train_end <= self.start:
            train_end = self.end - timedelta(days=step)
        if train_window_years is not None:
            train_start = train_end - timedelta(days=int(train_window_years * 365))
            if train_start < self.start:
                train_start = self.start
        else:
            train_start = self.start
        stable_region = (
            self.param_sensitivity.stable_region_config
            if self.walkforward_selection_method == WalkforwardSelectionMethod.STABLE_REGION
            else None
        )
        top_k = self.param_sensitivity.top_k
        return WalkforwardResearchConfig(
            train_start=train_start,
            train_end=train_end,
            enabled=enabled,
            test_step=step,
            num_steps=steps,
            top_k=top_k,
            selection_method=self.walkforward_selection_method,
            stable_region=stable_region,
            weight_layer_algorithm=self.weight_layer_algorithm,
            member_prediction_mode=self.walkforward_member_prediction_mode,
            output_root=self.walkforward_output_root,
        )


def load_config() -> BaseResearchConfig:
    """Single source of truth for tickers, date range, cache, and walkforward defaults.

    Edit here; in_sample and walkforward phases import and extend this.
    """
    # ==========================================================================
    # EDIT BELOW
    # ==========================================================================
    tickers = [
        Ticker.ES,
        Ticker.NQ,
        Ticker.YM,
        Ticker.RTY,
    ]
    start = datetime(2000, 1, 1)
    end = datetime(2023, 12, 31)
    use_cache = True
    populate_cache = True
    permutation_suite = PermutationSuiteConfig(enabled=True)
    walkforward_test_step = 365
    walkforward_num_steps = 8
    walkforward_selection_method = WalkforwardSelectionMethod.STABLE_REGION
    weight_layer_algorithm = WeightLayerAlgorithm.INVERSE_CORRELATION
    walkforward_member_prediction_mode = MemberPredictionMode.SHARPE_WEIGHTED
    feature_type = FeatureType.CONTINUOUS  # or FeatureType.RULE_BASED for rule-based param sensitivity
    # Param sensitivity: edit ParamSensitivityConfig defaults above, or override here e.g.:
    # param_sensitivity = ParamSensitivityConfig(adaptive_sigma_multiplier=1.5, top_k=5)
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
        permutation_suite=permutation_suite,
        feature_type=feature_type,
        param_sensitivity=param_sensitivity,
        walkforward_test_step=walkforward_test_step,
        walkforward_num_steps=walkforward_num_steps,
        walkforward_output_root=Path("feature_research/shared_results"),
        walkforward_selection_method=walkforward_selection_method,
        weight_layer_algorithm=weight_layer_algorithm,
        walkforward_member_prediction_mode=walkforward_member_prediction_mode,
    )
