"""Rule-based EDA config. Shared settings from feature_research.config; phase-specific here."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from feature_research.config import PermutationSuiteConfig, load_config as load_base_config
from feature_research.walkforward.config import WalkforwardResearchConfig
from utils.core.enums import Ticker, TimeFrame

_FEATURE_RESEARCH_DIR = Path(__file__).resolve().parents[2]
_RB_DIR = _FEATURE_RESEARCH_DIR / "in_sample" / "rule_based"


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
        Default: ``feature_research/in_sample/rule_based/results/{module_name}/``
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
    """Return rule-based research config. Shared settings from feature_research.config."""
    base = load_base_config()
    # Phase-specific: bias spec, target, strategy; rule_based uses test_step=252
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
    reports_dir = _RB_DIR / "results" / bias_spec["module_name"]
    walkforward = base.build_walkforward(test_step=252, num_steps=8, enabled=False)
    return RuleBasedResearchConfig(
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
        walkforward=walkforward,
    )
