"""Researcher-editable configuration for the rule-based EDA pipeline.

Edit the values in load_config() to customise tickers, date range, and bias node specs.
All other scripts import from here — change once, apply everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from feature_research.walkforward.config import WalkforwardResearchConfig
from feature_selection.validation.config import PermutationModeStage2
from feature_selection.validation.objective_metrics import ObjectiveMetricSpec
from utils.enums import Ticker, TimeFrame

_FEATURE_RESEARCH_DIR = Path(__file__).resolve().parents[1]
_RB_DIR = _FEATURE_RESEARCH_DIR / "rule_based"


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
class RuleBasedResearchConfig:
    """All researcher-editable settings for rule-based feature EDA.

    Attributes
    ----------
    tickers : list[Ticker]
        Instruments to include. Data is concatenated across tickers.
    start : datetime
        In-sample period start (inclusive).
    end : datetime
        In-sample period end (inclusive).
    bias_spec : dict[str, Any]
        Bias-node specification. ``params`` values may be lists for grid search.
        Example::

            {
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": [2, 3, 5, 7],
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            }
    target_col : str
        Column from ``targets_df`` to use as the prediction target.
    strategy : str
        Passed to the EDA summary label only.
    use_cache : bool
        Whether to use pre-computed caches for feature extraction.
    populate_cache : bool
        If True, run ``CacheManager.populate_cache()`` before extraction.
    reports_dir : Path
        Root output directory for EDA reports.
        Default: ``feature_research/rule_based/results/{module_name}/``
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
    walkforward: WalkforwardResearchConfig = field(
        default_factory=lambda: WalkforwardResearchConfig(
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2024, 12, 31),
            enabled=False,
        )
    )


def load_config() -> RuleBasedResearchConfig:
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

    use_cache = True
    populate_cache = True

    permutation_suite = PermutationSuiteConfig(enabled=False)
    # ==========================================================================
    # EDIT ABOVE
    # ==========================================================================

    module_name = bias_spec["module_name"]
    reports_dir = _RB_DIR / "results" / module_name

    walkforward = WalkforwardResearchConfig(
        train_start=start,
        train_end=end,
        enabled=False,
        output_root=reports_dir,
    )

    return RuleBasedResearchConfig(
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
        walkforward=walkforward,
    )
