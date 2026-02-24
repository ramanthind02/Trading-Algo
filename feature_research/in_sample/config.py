"""Unified in-sample research config. Consolidates continuous_binning and rule_based paths.

This module provides a single ResearchConfig that handles both feature types.
The feature_type field determines validation behavior:
  - CONTINUOUS: uses BinningAnalysisConfig, supports Phase 2 binning analysis
  - RULE_BASED: uses fixed 3-level binning, skips Phase 2 analysis
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from feature_research.config import (
    RAW_TARGET_COLS,
    BaseResearchConfig,
    FeatureType,
    PermutationSuiteConfig,
    load_config as load_base_config,
)
from feature_research.walkforward.config import (
    WalkforwardResearchConfig,
    WalkforwardSelectionMethod,
    WeightLayerAlgorithm,
)
from utils.core.enums import Ticker, TimeFrame

_FEATURE_RESEARCH_DIR = Path(__file__).resolve().parents[1]
EnumT = TypeVar("EnumT", bound=Enum)


def _coerce_enum_or_raise(value: object, enum_cls: type[EnumT], field_name: str) -> EnumT:
    """Coerce a string or enum value to the specified enum type."""
    valid_values = tuple(item.value for item in enum_cls)
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must be one of {valid_values}, got '{value}'"
            ) from exc
    raise ValueError(f"{field_name} must be one of {valid_values}, got '{value}'")


@dataclass(frozen=True)
class BinningAnalysisConfig:
    """Settings for continuous binning model parameters.

    Only used when feature_type == FeatureType.CONTINUOUS.
    Rule-based features use fixed 3 levels and ignore these settings.

    Attributes
    ----------
    bin_counts : list[int]
        List of bin counts to test in grid search.
    selection_metric : str
        Metric used for bin selection ('sharpe', 'mean', 't_stat', 'sortino').
    strategy : str
        Trading strategy ('long', 'short', 'long_short').
    metric_threshold : float
        Minimum metric value to consider a bin active.
    t_threshold : float
        Minimum |t-stat| for significance.
    min_region_width : int
        Legacy parameter (ignored by new grid search approach).
    use_coverage_bonus : bool
        Whether to add coverage bonus to Sharpe ratio.
    coverage_bonus_per_10pct : float
        Bonus added per 10% coverage above 10% floor.
    max_coverage_bonus : float
        Maximum coverage bonus cap.
    shrinkage_k : float
        James-Stein shrinkage constant.
    long_clip_min : float
        Minimum long position multiplier.
    long_clip_max : float
        Maximum long position multiplier.
    short_clip_min : float
        Minimum short position multiplier.
    short_clip_max : float
        Maximum short position multiplier.
    """

    bin_counts: list[int] = field(default_factory=lambda: [10, 8, 5, 3])
    selection_metric: str = "sortino"
    strategy: str = "long"
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


@dataclass(frozen=True)
class ResearchConfig:
    """Unified in-sample research config for both continuous and rule-based features.

    Attributes
    ----------
    feature_type : FeatureType
        CONTINUOUS or RULE_BASED. Determines validation behavior and binning approach.
    tickers : list[Ticker]
        Instruments to include. Data is concatenated across tickers.
    start : datetime
        In-sample period start (inclusive).
    end : datetime
        In-sample period end (inclusive).
    bias_spec : dict[str, Any]
        Bias-node specification. ``params`` values may be lists for grid search.
        For CONTINUOUS: typically RSI with lookback params.
        For RULE_BASED: typically RSI_SIGNAL with rsi_period, oversold, overbought, etc.
    target_col : str
        Column from ``targets_df`` to use as the prediction target.
        Options: ``"log_return"``, ``"log_return_atr"``, ``"log_return_ewsd"``.
    strategy : str
        Passed to the EDA summary label only.
        Options: ``"long"``, ``"short"``, ``"long-short"``.
    use_cache : bool
        Whether to use pre-computed caches for feature extraction.
    populate_cache : bool
        If True, run ``CacheManager.populate_cache()`` before extraction.
    reports_dir : Path
        Root output directory for EDA reports.
    permutation_suite : PermutationSuiteConfig
        Settings for permutation test suite (if enabled).
    binning_params : BinningAnalysisConfig
        Binning hyperparameters. Only used for CONTINUOUS; rule-based uses fixed 3 levels.
    walkforward_selection_method : WalkforwardSelectionMethod | str
        Method for selecting top features in walkforward splits.
    weight_layer_algorithm : WeightLayerAlgorithm | str
        Algorithm for combining forecasts.
    walkforward : WalkforwardResearchConfig
        Walkforward configuration (enabled/disabled, parameters).
    """

    feature_type: FeatureType
    tickers: list[Ticker]
    start: datetime
    end: datetime
    bias_spec: dict[str, Any]
    target_col: str
    strategy: str
    use_cache: bool
    populate_cache: bool
    reports_dir: Path
    permutation_suite: PermutationSuiteConfig = field(default_factory=PermutationSuiteConfig)
    binning_params: BinningAnalysisConfig = field(default_factory=BinningAnalysisConfig)
    walkforward_selection_method: WalkforwardSelectionMethod | str = (
        WalkforwardSelectionMethod.TOP_K
    )
    weight_layer_algorithm: WeightLayerAlgorithm | str = (
        WeightLayerAlgorithm.INVERSE_CORRELATION
    )
    walkforward: WalkforwardResearchConfig = field(
        default_factory=lambda: WalkforwardResearchConfig(
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2024, 12, 31),
            enabled=False,
        )
    )

    def __post_init__(self) -> None:
        if self.target_col in RAW_TARGET_COLS and len(self.tickers) > 1:
            raise ValueError(
                f"target_col='{self.target_col}' uses raw (unnormalized) returns with "
                f"multiple tickers ({len(self.tickers)} configured). "
                "Raw returns cannot be compared across tickers with different volatility. "
                "Use 'log_return_ewsd' or 'log_return_atr' for multi-ticker research."
            )

        normalized_selection_method = _coerce_enum_or_raise(
            self.walkforward_selection_method,
            WalkforwardSelectionMethod,
            "walkforward_selection_method",
        )
        object.__setattr__(
            self,
            "walkforward_selection_method",
            normalized_selection_method,
        )

        normalized_weight_layer_algorithm = _coerce_enum_or_raise(
            self.weight_layer_algorithm,
            WeightLayerAlgorithm,
            "weight_layer_algorithm",
        )
        object.__setattr__(
            self,
            "weight_layer_algorithm",
            normalized_weight_layer_algorithm,
        )

        nested_selection_method = _coerce_enum_or_raise(
            self.walkforward.selection_method,
            WalkforwardSelectionMethod,
            "walkforward.selection_method",
        )
        if nested_selection_method != normalized_selection_method:
            raise ValueError(
                "walkforward_selection_method must match walkforward.selection_method; "
                f"got top-level={normalized_selection_method.value!r}, "
                f"nested={nested_selection_method.value!r}"
            )

        nested_weight_layer_algorithm = _coerce_enum_or_raise(
            self.walkforward.weight_layer_algorithm,
            WeightLayerAlgorithm,
            "walkforward.weight_layer_algorithm",
        )
        if nested_weight_layer_algorithm != normalized_weight_layer_algorithm:
            raise ValueError(
                "weight_layer_algorithm must match walkforward.weight_layer_algorithm; "
                f"got top-level={normalized_weight_layer_algorithm.value!r}, "
                f"nested={nested_weight_layer_algorithm.value!r}"
            )


def load_config() -> ResearchConfig:
    """Return unified research config with feature_type dispatched from base config.

    This is the single entrypoint for loading in-sample research config.
    It checks base.feature_type to determine which phase-specific defaults to apply.
    """
    base = load_base_config()

    # Dispatch on feature_type to set phase-specific parameters
    if base.feature_type == FeatureType.CONTINUOUS:
        # CONTINUOUS BINNING DEFAULTS
        bias_spec = {
            "module_name": "rsi",
            "timeframes": [TimeFrame.D],
            "params": {"lookback": [2, 3, 4, 5, 6, 7, 8, 9, 10]},
        }
        target_col = "log_return_atr"
        strategy = "long"
        binning_params = BinningAnalysisConfig(
            bin_counts=[10, 9, 8, 7, 6, 5, 4, 3],
            strategy="long",
            t_threshold=2.0,
            use_coverage_bonus=False,
        )
        reports_dir = _FEATURE_RESEARCH_DIR / "in_sample" / "results" / "continuous"
        walkforward = base.build_walkforward(enabled=False)

    elif base.feature_type == FeatureType.RULE_BASED:
        # RULE-BASED DEFAULTS
        bias_spec = {
            "module_name": "rsi_signal",
            "timeframes": [TimeFrame.D],
            "params": {
                "rsi_period": [2, 3, 5, 7],
                "oversold": list(range(5, 26, 5)),
                "overbought": list(range(95, 64, -5)),
                "strategy_mode": "long",
                "exit_policy": "threshold_or_bars",
                "exit_bars": 5,
            },
        }
        target_col = "log_return"
        strategy = "long"
        binning_params = BinningAnalysisConfig()  # Defaults; rule-based ignores these
        reports_dir = _FEATURE_RESEARCH_DIR / "in_sample" / "results" / "rule_based"
        walkforward = base.build_walkforward(test_step=252, num_steps=8, enabled=False)

    else:
        raise ValueError(f"Unknown feature_type: {base.feature_type}")

    return ResearchConfig(
        feature_type=base.feature_type,
        tickers=base.tickers,
        start=base.start,
        end=base.end,
        bias_spec=bias_spec,
        target_col=target_col,
        strategy=strategy,
        use_cache=base.use_cache,
        populate_cache=base.populate_cache,
        reports_dir=reports_dir,
        permutation_suite=base.permutation_suite,
        binning_params=binning_params,
        walkforward_selection_method=base.walkforward_selection_method,
        weight_layer_algorithm=base.weight_layer_algorithm,
        walkforward=walkforward,
    )
