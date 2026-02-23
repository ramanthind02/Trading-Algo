"""Continuous binning EDA config. Shared settings from feature_research.config; phase-specific here."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from feature_research.config import (
    RAW_TARGET_COLS,
    BaseResearchConfig,
    PermutationSuiteConfig,
    load_config as load_base_config,
)
from feature_research.walkforward.config import (
    WalkforwardResearchConfig,
    WalkforwardSelectionMethod,
    WeightLayerAlgorithm,
)
from utils.core.enums import Ticker, TimeFrame

_FEATURE_RESEARCH_DIR = Path(__file__).resolve().parents[2]
_CB_DIR = _FEATURE_RESEARCH_DIR / "in_sample" / "continuous_binning"
EnumT = TypeVar("EnumT", bound=Enum)


def _coerce_enum_or_raise(value: object, enum_cls: type[EnumT], field_name: str) -> EnumT:
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
    """All researcher-editable settings for continuous-binning EDA.

    Attributes
    ----------
    tickers : list[Ticker]
        Instruments to include. Data is concatenated across tickers.
    start : datetime
        In-sample period start (inclusive).
    end : datetime
        In-sample period end (inclusive).
    bias_spec : dict[str, Any]
        Bias-node specification.  ``params`` values may be lists for grid search.
        Example::

            {
                "module_name": "rsi",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": [2, 3, 4, 5, 6, 7, 8, 9, 10]},
            }
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
        Default: ``feature_research/in_sample/continuous_binning/results/{module_name}/``
    binning_params : BinningAnalysisConfig
        Binning model hyperparameters (bin_counts, thresholds, coverage bonus).
    """

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
    """Return continuous-binning research config. Shared settings from feature_research.config."""
    base = load_base_config()
    # Phase-specific: bias spec, target, strategy, binning params
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
    reports_dir = _CB_DIR / "results" / bias_spec["module_name"]
    walkforward = base.build_walkforward(enabled=False)
    return ResearchConfig(
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
