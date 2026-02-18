"""Researcher-editable configuration for the rule-based EDA pipeline.

Edit the values in load_config() to customise tickers, date range, and bias node specs.
All other scripts import from here — change once, apply everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.enums import Ticker, TimeFrame

_FEATURE_RESEARCH_DIR = Path(__file__).resolve().parents[1]
_RB_DIR = _FEATURE_RESEARCH_DIR / "rule_based"


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


def load_config() -> RuleBasedResearchConfig:
    """Return the default research configuration.

    **Edit this function** to customise tickers, dates, and bias node specs.
    All pipeline scripts import from here.
    """
    # ==========================================================================
    # EDIT BELOW
    # ==========================================================================
    tickers = [
        Ticker.ES,  # E-Mini S&P 500
        Ticker.NQ,  # E-Mini Nasdaq-100
    ]

    start = datetime(2020, 1, 1)
    end = datetime(2024, 12, 31)

    bias_spec = {
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

    target_col = "log_return"
    strategy = "long"

    use_cache = True
    populate_cache = True
    # ==========================================================================
    # EDIT ABOVE
    # ==========================================================================

    module_name = bias_spec["module_name"]
    reports_dir = _RB_DIR / "results" / module_name

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
    )
