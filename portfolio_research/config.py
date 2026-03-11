"""Portfolio research configuration.

Edit here once; run_portfolio_test.py and tests use load_config().
Uses the same walkforward window formula as feature_research (compute_first_fold_bounds)
and OOSWindowConfig so portfolio in-sample and OOS periods stay aligned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from feature_research.config import OOSWindowConfig
from utils.core.enums import Ticker, TimeFrame

_PORTFOLIO_RESEARCH_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ResearchWindow:
    """Explicit start/end window for portfolio research phases."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError(
                "ResearchWindow: start must be before end "
                f"(got start={self.start!s}, end={self.end!s})."
            )


@dataclass(frozen=True)
class PortfolioResearchConfig:
    """Configuration for portfolio test script (vault ensembles, multi-window datasets).

    Primary windows: explicit train, validation, and test windows. Additional
    composite tearsheets are derived from these (e.g. validation+test, all).

    Attributes
    ----------
    tickers : list[Ticker]
        Instruments to load candles for.
    timeframe : TimeFrame
        Research-level candle timeframe setting. Ensemble trading timeframes are auto-detected
        from vault metadata/path (`ensemble.base_tf`), and combined tearsheet aggregation is
        always daily-aligned.
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
        WeightLayer method (``'cluster_equal'`` or ``'cluster_corr_ulcer'``).
    weight_layer_kwargs : Mapping[str, Any]
        Extra kwargs for WeightLayer (e.g. fdm_max).
    max_position_pct : float
        Max position as fraction of capital (e.g. 3.5).
    baseline_mode : str
        'equal_weight' or 'buy_hold' for PortfolioTester.
    output_root : Path
        Root directory for tearsheets and artifacts.
    sector_allocation_config_path : str | None
        Optional path to sector allocation JSON; when set, passed to Portfolio.
    """

    tickers: list[Ticker]
    timeframe: TimeFrame
    start: datetime
    end: datetime
    use_cache: bool
    train_window: ResearchWindow
    validation_window: ResearchWindow
    test_window: ResearchWindow
    populate_cache: bool = False
    train_window_years: float = 15.0
    test_window_years: float = 2.0
    num_steps: int = 4
    oos_window: OOSWindowConfig | None = None
    ensemble_dirs: Mapping[str, str] = field(default_factory=dict)
    target_volatility: float = 0.15
    weight_layer_method: str = "cluster_equal"
    weight_layer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    max_position_pct: float = 3.5
    baseline_mode: str = "equal_weight"
    output_root: Path = field(default_factory=lambda: _PORTFOLIO_RESEARCH_DIR / "results")
    sector_allocation_config_path: str | None = None

    def __post_init__(self) -> None:
        if self.baseline_mode not in ("equal_weight", "buy_hold"):
            raise ValueError(
                f"baseline_mode must be 'equal_weight' or 'buy_hold', got '{self.baseline_mode}'"
            )
        if not self.ensemble_dirs:
            raise ValueError("ensemble_dirs must be non-empty")
        if self.train_window.end >= self.validation_window.start:
            raise ValueError(
                "PortfolioResearchConfig: train_window.end must be before "
                f"validation_window.start (got train_end={self.train_window.end!s}, "
                f"validation_start={self.validation_window.start!s})."
            )
        if self.validation_window.end >= self.test_window.start:
            raise ValueError(
                "PortfolioResearchConfig: validation_window.end must be before "
                f"test_window.start (got validation_end={self.validation_window.end!s}, "
                f"test_start={self.test_window.start!s})."
            )


def _discover_ensemble_dirs() -> Mapping[str, str]:
    """Discover all ensemble directories under vault/ for use as defaults.

    An ensemble folder is any directory under vault/<TF>/ whose leaf directory
    contains a 'features' subdirectory with at least one *.json file. The name
    is the leaf directory name; the path is repository-relative.

    Buy-and-hold ensembles are excluded from auto-discovery by default to avoid
    overwhelming strategy-driven portfolios in multi-timeframe blends. They can
    still be included explicitly by setting ``ensemble_dirs`` in ``load_config``.
    """
    vault_root = _PORTFOLIO_RESEARCH_DIR.parent / "vault"
    if not vault_root.exists():
        return {}

    def _has_feature_json(features_dir: Path) -> bool:
        return features_dir.is_dir() and any(
            child.suffix == ".json" for child in features_dir.iterdir()
        )

    def _is_buy_hold_ensemble(ensemble_dir: Path) -> bool:
        if ensemble_dir.name.startswith("buy_hold"):
            return True

        features_dir = ensemble_dir / "features"
        feature_files = sorted(features_dir.glob("*.json"))
        if not feature_files:
            return False

        # Treat ensemble as buy-hold only if every readable feature config is buy_hold.
        has_readable_feature = False
        for feature_file in feature_files:
            try:
                with feature_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                return False

            has_readable_feature = True
            module_name = (
                payload.get("bias_node_spec", {}).get("module_name")
                if isinstance(payload, dict)
                else None
            )
            if module_name != "buy_hold":
                return False

        return has_readable_feature

    timeframe_dirs = [
        d for d in vault_root.iterdir() if d.is_dir()
    ]

    ensembles = {
        ensemble_dir.name: str(
            ensemble_dir.relative_to(_PORTFOLIO_RESEARCH_DIR.parent)
        )
        for tf_dir in timeframe_dirs
        for ensemble_dir in tf_dir.iterdir()
        if (
            ensemble_dir.is_dir()
            and _has_feature_json(ensemble_dir / "features")
            and not _is_buy_hold_ensemble(ensemble_dir)
        )
    }

    return ensembles


def load_config() -> PortfolioResearchConfig:
    """Single source of truth for portfolio research. Edit the block below."""
    # ==========================================================================
    # EDIT BELOW
    # ==========================================================================
    tickers = [
        Ticker.ES,
        Ticker.NQ,
        Ticker.RTY,
        Ticker.TLT,
        Ticker.GC,
    ]
    timeframe = TimeFrame.D
    start = datetime(2006, 1, 1)
    end = datetime(2025, 9, 18)
    use_cache = True
    populate_cache = False

    # Explicit research windows driving portfolio phases.
    train_window = ResearchWindow(
        start=start,
        end=datetime(2017, 12, 31),
    )
    validation_window = ResearchWindow(
        start=datetime(2018, 1, 1),
        end=datetime(2022, 12, 31),
    )
    test_window = ResearchWindow(
        start=datetime(2023, 1, 1),
        end=end,
    )

    train_window_years = 15.0
    test_window_years = 2.0
    num_steps = 8

    oos_window = OOSWindowConfig(
        train_start=datetime(2007, 1, 1),
        train_end=datetime(2023, 12, 30),
        test_start=datetime(2024, 1, 1),
        test_end=datetime(2025, 12, 31),
    )

    # By default, use all ensembles discoverable in the vault. Researchers can
    # override this by assigning a custom mapping here.
    ensemble_dirs = _discover_ensemble_dirs()

    target_volatility = 0.15
    weight_layer_method = "cluster_equal"
    weight_layer_kwargs = {"fdm_max": 2.0}
    max_position_pct = 3.5
    baseline_mode = "equal_weight"
    output_root = _PORTFOLIO_RESEARCH_DIR / "results"
    sector_allocation_config_path = str(
        _PORTFOLIO_RESEARCH_DIR.parent / "feature_research" / "config" / "sector_buy_hold_60_20_20.json"
    )
    # ==========================================================================
    # EDIT ABOVE
    # ==========================================================================

    return PortfolioResearchConfig(
        tickers=tickers,
        timeframe=timeframe,
        start=start,
        end=end,
        use_cache=use_cache,
        train_window=train_window,
        validation_window=validation_window,
        test_window=test_window,
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
        sector_allocation_config_path=sector_allocation_config_path,
    )
