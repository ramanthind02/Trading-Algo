"""Portfolio research configuration.

Edit here once; run_portfolio_test.py and tests use load_config().
Uses the same walkforward window formula as feature_research (compute_first_fold_bounds)
and OOSWindowConfig so portfolio in-sample and OOS periods stay aligned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

from ensemble.vault import feature_files as _vault_feature_files
from feature_research.config import OOSWindowConfig
from utils.cache import extract_cross_ticker_names
from utils.core.enums import Ticker, TimeFrame

logger = logging.getLogger(__name__)

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
class FeatureVaultCorrelationConfig:
    """Gate and paths for ``portfolio_research.run_feature_vault_correlation`` exports."""

    enabled: bool = True
    vault_root: Path | None = None
    output_subdir: str = "powerbi/feature_vault_correlation"


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
        WeightLayer method (e.g. ``equal_signal``, ``inverse_avg_pairwise_corr``, ``hrp_classic``).
    weight_layer_kwargs : Mapping[str, Any]
        Extra kwargs for WeightLayer: ``fdm_max``, ``group_weight_cap``, ``rho_cut``.
    max_position_pct : float
        Max position as fraction of capital (e.g. 3.5).
    baseline_mode : str
        'equal_weight' or 'buy_hold' for PortfolioTester.
    output_root : Path
        Root directory for tearsheets and artifacts.
    export_per_timeframe_tearsheets : bool
        If True, write ``{daily|weekly|monthly}_Portfolio_<phase>_window_tearsheet.html`` when
        multiple trading timeframes are present. Set False to skip (faster runs).
    export_per_ensemble_tearsheets : bool
        If True, write per-ensemble and per-base-model tearsheets under each timeframe
        subfolder. Set False to skip (faster runs). Combined portfolio tearsheets for each
        phase and under ``combined/`` are always written.
    feature_vault_correlation : FeatureVaultCorrelationConfig
        Optional export: after running the feature-research OOS pipeline, correlate selected
        research returns with every vault feature JSON on the same timeframe. Writes CSV under
        ``output_root / feature_vault_correlation.output_subdir`` when enabled.
    strict_cache_preflight : bool
        If True, abort the portfolio test when vault bias/EWSD cache refresh reports any
        failure. If False (default), print failures and continue (research may still fail
        later if required data is missing).
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
    weight_layer_method: str = "equal_signal"
    weight_layer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    max_position_pct: float = 3.5
    baseline_mode: str = "equal_weight"
    output_root: Path = field(default_factory=lambda: _PORTFOLIO_RESEARCH_DIR / "results")
    export_per_timeframe_tearsheets: bool = True
    export_per_ensemble_tearsheets: bool = True
    feature_vault_correlation: FeatureVaultCorrelationConfig = field(
        default_factory=FeatureVaultCorrelationConfig
    )
    strict_cache_preflight: bool = False

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


def _discover_ensemble_dirs(
    allowed_timeframes: Iterable[TimeFrame] | None = None,
) -> Mapping[str, str]:
    """Discover ensemble directories under vault/ for use as defaults.

    An ensemble folder is any directory under vault/<TF>/ whose leaf directory
    contains a 'features' subdirectory with at least one *.json file. The name
    is the leaf directory name; the path is repository-relative.

    When ``allowed_timeframes`` is provided, only those vault timeframe folders
    are considered.
    """
    vault_root = _PORTFOLIO_RESEARCH_DIR.parent / "vault"
    if not vault_root.exists():
        return {}

    allowed_tf_names = (
        {timeframe.name for timeframe in allowed_timeframes}
        if allowed_timeframes is not None
        else None
    )

    def _has_feature_json(features_dir: Path) -> bool:
        return features_dir.is_dir() and any(
            child.suffix == ".json" for child in features_dir.iterdir()
        )

    timeframe_dirs = [
        d for d in vault_root.iterdir() if d.is_dir()
    ]
    if allowed_tf_names is not None:
        timeframe_dirs = [
            timeframe_dir
            for timeframe_dir in timeframe_dirs
            if timeframe_dir.name in allowed_tf_names
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
        )
    }

    return ensembles


def _single_feature_instrument_tickers(feature_config: Mapping[str, object]) -> list[str]:
    """Normalize feature ``tickers`` to non-empty uppercase symbols."""
    raw = feature_config.get("tickers", [])
    if not isinstance(raw, list):
        return []
    return [
        item.strip().upper()
        for item in raw
        if isinstance(item, str) and item.strip()
    ]


def required_tickers_for_ensemble_dir(ensemble_repo_relative_path: str) -> frozenset[Ticker]:
    """Return tickers this ensemble *must* have in the portfolio to run correctly.

    - ``cross_tickers`` in each feature's ``bias_node_spec.params`` (e.g. ES vs TLT legs).
    - If that feature's ``tickers`` has **exactly one** symbol, that instrument is
      required (single-instrument features: ``seasonal_bonds_month`` on TLT,
      ``seasonal_indices_eof`` on ES, primary leg of rebalancing with one ticker row).
    - If ``tickers`` lists multiple symbols, it is treated as an authoring universe
      (e.g. buy/hold, multi-index); the portfolio subset can omit some of them.
    """
    features_dir = _PORTFOLIO_RESEARCH_DIR.parent / ensemble_repo_relative_path / "features"
    if not features_dir.is_dir():
        return frozenset()

    required: set[Ticker] = set()
    for _path, feature_config in _vault_feature_files.iter_validated_feature_configs(
        features_dir
    ):
        spec = feature_config.get("bias_node_spec")
        if not isinstance(spec, Mapping):
            continue
        params = spec.get("params")
        if isinstance(params, Mapping):
            for name in extract_cross_ticker_names(params):
                required.add(Ticker[name])
        symbols = _single_feature_instrument_tickers(feature_config)
        if len(symbols) == 1:
            required.add(Ticker[symbols[0]])
    return frozenset(required)


def filter_ensemble_dirs_for_portfolio_tickers(
    ensemble_dirs: Mapping[str, str],
    portfolio_tickers: Iterable[Ticker],
) -> dict[str, str]:
    """Drop vault ensembles that need symbols outside the portfolio (see ``required_tickers_for_ensemble_dir``).

    Preflight may bootstrap extra symbols, but ``PortfolioCacheQuery`` only loads
    candles for ``config.tickers``; incompatible ensembles cause ``fit_from_candles``
    failures and unfitted ensemble slots.
    """
    allowed = frozenset(portfolio_tickers)
    kept: dict[str, str] = {}
    skipped: list[str] = []
    for name, path in ensemble_dirs.items():
        need = required_tickers_for_ensemble_dir(path)
        if need <= allowed:
            kept[name] = path
        else:
            missing = sorted(t.name for t in (need - allowed))
            skipped.append(f"{name} (needs {', '.join(missing)} not in portfolio tickers)")
    if skipped:
        logger.warning(
            "Skipping %d vault ensemble(s) incompatible with portfolio tickers: %s",
            len(skipped),
            "; ".join(skipped),
        )
    if not kept:
        raise ValueError(
            "No ensembles remain after filtering to portfolio tickers. "
            "Add the missing symbols to config.tickers or set ensemble_dirs explicitly."
        )
    return kept


def load_config() -> PortfolioResearchConfig:
    """Single source of truth for portfolio research. Edit the block below."""
    # ==========================================================================
    # EDIT BELOW
    # ==========================================================================
    tickers = [
        Ticker.ES,
        Ticker.NQ,
        Ticker.GC
    ]
    timeframe = TimeFrame.D
    start = datetime(2000, 1, 1)
    end = datetime(2026, 2, 20)
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
        test_end=datetime(2026, 2, 20),
    )

    # By default, use daily + monthly ensembles that only need configured tickers.
    ensemble_dirs = filter_ensemble_dirs_for_portfolio_tickers(
        _discover_ensemble_dirs(allowed_timeframes=(TimeFrame.D, TimeFrame.M)),
        tickers,
    )

    target_volatility = 0.15
    weight_layer_method = "hrp_classic"
    weight_layer_kwargs = {
        "fdm_max": 2.0,
        "rho_cut": 0.05
    }
    max_position_pct = 3.5
    baseline_mode = "equal_weight"
    output_root = _PORTFOLIO_RESEARCH_DIR / "results"
    export_per_timeframe_tearsheets = False
    export_per_ensemble_tearsheets = False
    feature_vault_correlation = FeatureVaultCorrelationConfig(enabled=False)
    strict_cache_preflight = False
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
        export_per_timeframe_tearsheets=export_per_timeframe_tearsheets,
        export_per_ensemble_tearsheets=export_per_ensemble_tearsheets,
        feature_vault_correlation=feature_vault_correlation,
        strict_cache_preflight=strict_cache_preflight,
    )
