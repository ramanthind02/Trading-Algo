"""Portfolio research configuration.

Edit here once; run_portfolio_test.py and tests use load_config().
Uses the same walkforward window formula as feature_research (compute_first_fold_bounds)
and OOSWindowConfig so portfolio in-sample and OOS periods stay aligned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from feature_research.config import (
    OOSWindowConfig,
    compute_first_fold_bounds,
)
from feature_research.walkforward.stable_region_selection import StableRegionConfig
from utils.core.enums import Ticker, TimeFrame

_PORTFOLIO_RESEARCH_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PortfolioResearchConfig:
    """Configuration for portfolio test script (vault ensembles, two datasets).

    Two datasets: (1) in-sample walkforward — bounds from walkforward_train_test_bounds();
    (2) OOS — explicit oos_window when set.

    Attributes
    ----------
    tickers : list[Ticker]
        Instruments to load candles for.
    timeframe : TimeFrame
        Bar timeframe (e.g. D for daily).
    start : datetime
        Start of walkforward period (inclusive).
    end : datetime
        End of walkforward period (inclusive).
    use_cache : bool
        Whether ensembles/portfolio use cache for feature extraction.
    populate_cache : bool
        If True, populate feature cache before run (default False; portfolio uses vault + candles only).
    train_window_years : float
        Training window length in years (first fold); align with feature_research.
    test_window_years : float
        Test window length in years (first fold); align with feature_research.
    num_steps : int
        Number of walkforward steps (used only to compute first fold bounds).
    oos_window : OOSWindowConfig | None
        Explicit OOS train/test window; None to skip OOS.
    ensemble_dirs : Mapping[str, str]
        Name -> vault path for each ensemble to load.
    target_volatility : float
        Target annual volatility for ensembles/portfolio.
    weight_layer_method : str
        WeightLayer method (e.g. 'inverse_correlation').
    weight_layer_kwargs : Mapping[str, Any]
        Extra kwargs for WeightLayer (e.g. fdm_max).
    max_position_pct : float
        Max position as fraction of capital (e.g. 3.5).
    baseline_mode : str
        'equal_weight' or 'buy_hold' for PortfolioTester.
    output_root : Path
        Root directory for tearsheets and artifacts.
    stable_region_config : StableRegionConfig | None
        Reserved for future per-group parameter selection; unused by script.
    """

    tickers: list[Ticker]
    timeframe: TimeFrame
    start: datetime
    end: datetime
    use_cache: bool
    populate_cache: bool = False
    train_window_years: float = 15.0
    test_window_years: float = 2.0
    num_steps: int = 4
    oos_window: OOSWindowConfig | None = None
    ensemble_dirs: Mapping[str, str] = field(default_factory=dict)
    target_volatility: float = 0.15
    weight_layer_method: str = "inverse_correlation"
    weight_layer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    max_position_pct: float = 3.5
    baseline_mode: str = "equal_weight"
    output_root: Path = field(default_factory=lambda: _PORTFOLIO_RESEARCH_DIR / "results")
    stable_region_config: StableRegionConfig | None = None

    def __post_init__(self) -> None:
        if self.baseline_mode not in ("equal_weight", "buy_hold"):
            raise ValueError(
                f"baseline_mode must be 'equal_weight' or 'buy_hold', got '{self.baseline_mode}'"
            )
        if not self.ensemble_dirs:
            raise ValueError("ensemble_dirs must be non-empty")

    def walkforward_train_test_bounds(
        self,
    ) -> tuple[datetime, datetime, datetime, datetime]:
        """Return (train_start, train_end, test_start, test_end) for the first fold.

        Uses the same formula as feature_research so portfolio in-sample period
        matches walkforward.
        """
        return compute_first_fold_bounds(
            self.start,
            self.end,
            self.train_window_years,
            self.test_window_years,
            self.num_steps,
        )


def load_config() -> PortfolioResearchConfig:
    """Single source of truth for portfolio research. Edit the block below."""
    # ==========================================================================
    # EDIT BELOW
    # ==========================================================================
    tickers = [
        Ticker.ES,
        Ticker.NQ,
        Ticker.RTY,
    ]
    timeframe = TimeFrame.D
    start = datetime(2005, 1, 1)
    end = datetime(2023, 12, 30)
    use_cache = True
    populate_cache = False

    train_window_years = 15.0
    test_window_years = 2.0
    num_steps = 8

    oos_window = OOSWindowConfig(
        train_start=datetime(2007, 1, 1),
        train_end=datetime(2023, 12, 30),
        test_start=datetime(2024, 1, 1),
        test_end=datetime(2025, 12, 31),
    )

    ensemble_dirs = {
        "buy_hold": "vault/D/buy-hold_indices_long",
        "mean_reversion_long": "vault/D/mean-reversion_indices_long",
        "seasonal_indices_eom": "vault/D/seasonal_indices_eom_long",
        "seasonal_bonds_month": "vault/D/seasonal_bonds_month_long_short",
    }

    target_volatility = 0.15
    weight_layer_method = "inverse_correlation"
    weight_layer_kwargs = {"fdm_max": 2.5}
    max_position_pct = 3.5
    baseline_mode = "equal_weight"
    output_root = _PORTFOLIO_RESEARCH_DIR / "results"
    stable_region_config = None
    # ==========================================================================
    # EDIT ABOVE
    # ==========================================================================

    return PortfolioResearchConfig(
        tickers=tickers,
        timeframe=timeframe,
        start=start,
        end=end,
        use_cache=use_cache,
        populate_cache=populate_cache,
        train_window_years=train_window_years,
        test_window_years=test_window_years,
        num_steps=num_steps,
        oos_window=oos_window,
        ensemble_dirs=ensemble_dirs,
        target_volatility=target_volatility,
        weight_layer_method=weight_layer_method,
        weight_layer_kwargs=weight_layer_kwargs,
        max_position_pct=max_position_pct,
        baseline_mode=baseline_mode,
        output_root=output_root,
        stable_region_config=stable_region_config,
    )
