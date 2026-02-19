"""Researcher-editable configuration for the continuous binning EDA pipeline.

Edit the values in load_config() to customise tickers, date range, and bias node specs.
All other scripts import from here — change once, apply everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from feature_selection.validation.config import PermutationModeStage2
from feature_selection.validation.objective_metrics import ObjectiveMetricSpec
from utils.enums import Ticker, TimeFrame

# ---------------------------------------------------------------------------
# Root of the feature_research tree — resolved at import time so scripts
# work regardless of the working directory they are launched from.
# ---------------------------------------------------------------------------
_FEATURE_RESEARCH_DIR = Path(__file__).resolve().parents[1]
_CB_DIR = _FEATURE_RESEARCH_DIR / "continuous_binning"


@dataclass(frozen=True)
class PermutationSuiteConfig:
    """Settings for running the shared permutation test suite."""

    enabled: bool = False
    nreps: int = 100
    alpha: float = 0.10
    metric_threshold: float = 0.0
    top_k: int = 3
    min_folds_stable: int = 1
    random_seed: int | None = 42
    permutation_mode_stage2: PermutationModeStage2 = "candle_shuffle"
    fold_years: int = 2
    objective_metric: ObjectiveMetricSpec = field(
        default_factory=lambda: ObjectiveMetricSpec(builtin="sharpe")
    )


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
    selection_metric: str = "sharpe"
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
        Default: ``feature_research/continuous_binning/results/{module_name}/``
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


def load_config() -> ResearchConfig:
    """Return the default research configuration.

    **Edit this function** to customise tickers, dates, and bias node specs.
    All pipeline scripts import from here.
    """
    # ==========================================================================
    # EDIT BELOW
    # ==========================================================================
    tickers = [
        Ticker.ES,   # E-Mini S&P 500
        Ticker.NQ,   # E-Mini Nasdaq-100
        Ticker.YM,   # E-Mini Dow Jones
        Ticker.RTY,  # E-Mini Russell 2000
    ]

    start = datetime(2000, 1, 1)
    end = datetime(2024, 12, 31)

    bias_spec = {
        "module_name": "rsi",
        "timeframes": [TimeFrame.D],
        "params": {"lookback": [2, 3, 4, 5, 6, 7, 8, 9, 10]},
    }

    target_col = "log_return"
    strategy = "long-short"

    # Caching
    use_cache = True
    populate_cache = True

    permutation_suite = PermutationSuiteConfig(enabled=False)

    binning_params = BinningAnalysisConfig(
        bin_counts=[10, 8, 5, 3],
        strategy="long_short",
        t_threshold=2.0,
        use_coverage_bonus=False,
    )
    # ==========================================================================
    # EDIT ABOVE
    # ==========================================================================

    module_name = bias_spec["module_name"]
    reports_dir = _CB_DIR / "results" / module_name

    return ResearchConfig(
        tickers=tickers,
        start=start,
        end=end,
        bias_spec=bias_spec,
        target_col=target_col,
        strategy=strategy,
        use_cache=use_cache,
        populate_cache=populate_cache,
        reports_dir=reports_dir,
        permutation_suite=permutation_suite,
        binning_params=binning_params,
    )
