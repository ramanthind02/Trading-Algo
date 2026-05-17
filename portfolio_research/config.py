"""Portfolio research configuration.

Edit here once; run_portfolio_test.py and tests use load_config().
Uses the same walkforward window formula as feature_research (compute_first_fold_bounds)
and OOSWindowConfig so portfolio in-sample and OOS periods stay aligned.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime
from enum import Enum
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

from ensemble.vault import feature_files as _vault_feature_files
from feature_research.config import OOSWindowConfig
from utils.cache import extract_cross_ticker_names
from utils.core.enums import Ticker, TimeFrame
from utils.futures_micro_specs import canonical_listed_micro_futures
from utils.vault_paths import default_vault_discovery_dirnames

logger = logging.getLogger(__name__)

_PORTFOLIO_RESEARCH_DIR = Path(__file__).resolve().parent


class LeverageMode(Enum):
    """Controls how margin/leverage is applied in the futures contract simulation.

    FINITE
        Leverage is tied to the starting account size.  Each bar the engine checks
        that total initial margin ≤ ``account_capital`` and flags any breach.

    INFINITE
        No margin / leverage constraint is enforced.  Contracts are still rounded
        to integers; the account is treated as having unlimited buying power.
    """

    FINITE = "finite"
    INFINITE = "infinite"


@dataclass(frozen=True)
class FuturesInstrumentSpec:
    """Per-instrument futures metadata for the contract simulation layer.

    Parameters
    ----------
    multiplier : float
        Dollar value per index point (e.g. 2.0 for MNQ, 5.0 for MES, 10.0 for MGC).
    margin_long : float
        Maintenance margin per contract for a long position (USD).
    margin_short : float
        Maintenance margin per contract for a short position (USD).
    product_code : str
        Exchange product code (e.g. ``"MNQ"``, ``"MES"``, ``"MGC"``).
    """

    multiplier: float
    margin_long: float
    margin_short: float
    product_code: str = ""

    def __post_init__(self) -> None:
        if self.multiplier <= 0:
            raise ValueError(f"FuturesInstrumentSpec.multiplier must be positive, got {self.multiplier}")
        if self.margin_long < 0 or self.margin_short < 0:
            raise ValueError("Margin values must be non-negative")


@dataclass(frozen=True)
class FuturesSimConfig:
    """Optional futures-contract simulation layer for portfolio research.

    When attached to ``PortfolioResearchConfig``, the pipeline runs a parallel
    contract-discrete PnL path alongside the standard fractional-return path and
    writes diagnostic CSVs to ``output_root / "futures_sim/"``.

    Parameters
    ----------
    enabled : bool
        Set False to skip the entire simulation (zero overhead).
    account_capital : float
        Starting account size in USD used to convert ``position_fraction`` to
        contract counts: ``contracts = round(position_fraction * capital /
        contract_value)``.
    instrument_specs : Mapping[str, FuturesInstrumentSpec]
        Research ticker → futures spec.  Keys must match ``config.tickers`` names
        (e.g. ``"ES"``, ``"NQ"``, ``"GC"``).  Tickers not in this map are skipped
        in the contract path (fractional returns are still computed for them).
    leverage_mode : LeverageMode
        ``FINITE`` enforces margin checks; ``INFINITE`` skips them.
    emit_tracking_error_csv : bool
        Write ``futures_sim/{phase}_tracking_error.csv`` with per-bar discrete vs
        fractional return diff, cumulative tracking error, and summary stats.
    emit_diagnostics_csv : bool
        Write ``futures_sim/{phase}_diagnostics.csv`` with per-bar contract counts,
        notional values, margin usage, and leverage-breach flags.
    """

    enabled: bool = False
    account_capital: float = 100_000.0
    instrument_specs: Mapping[str, FuturesInstrumentSpec] = field(default_factory=dict)
    leverage_mode: LeverageMode = LeverageMode.FINITE
    emit_tracking_error_csv: bool = True
    emit_diagnostics_csv: bool = True

    def __post_init__(self) -> None:
        if self.account_capital <= 0:
            raise ValueError(f"FuturesSimConfig.account_capital must be positive, got {self.account_capital}")


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
    """Gate and paths for ``portfolio_research.run_feature_vault_correlation`` exports.

    When ``vault_root`` is ``None``, the export uses the **prop** vault
    (:func:`~portfolio_research.vault_correlation.resolve_default_vault_root`). Set a path to
    scan ``vault_personal`` or another root.
    """

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
        WeightLayer method (e.g. ``equal_signal``, ``ledoit_wolf_min_corr``, ``hierarchy_equal``).
        ``load_config()`` uses ``ledoit_wolf_min_corr``; use ``hierarchy_equal`` with
        ``build_hierarchy_spec_for_ensemble_dirs`` when you want vault group buckets.
    weight_layer_kwargs : Mapping[str, Any]
        Extra kwargs for WeightLayer: ``fdm_max``, ``hierarchy_spec``, ``hierarchy_path``.
        For ``hierarchy_equal``, pass ``hierarchy_spec`` (nested dict) and/or ``hierarchy_path``.
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
    exclude_feature_stems_by_ensemble : Mapping[str, frozenset[str]] | None
        Repo-relative ensemble directory (posix) → feature JSON stems to omit.
        Used for leave-one-feature-out; must stay aligned with ``weight_layer_kwargs``
        hierarchy built via ``build_hierarchy_spec_for_ensemble_dirs`` with the same map.
    ensemble_vault_refit : bool
        When True (default research), vault ensembles load with ``refit=True`` and are
        **re-fit on cache** inside each phase (walk-forward research). When False (prop-firm
        profile), matches live ``enigma_live_forecast``: ``refit=False`` so frozen vault
        fitted payloads drive forecasts — discrete micro exposure then aligns with live
        sizing for the same τ / weight layer / ticker set.
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
    exclude_feature_stems_by_ensemble: Mapping[str, frozenset[str]] | None = None
    ensemble_vault_refit: bool = True
    futures_sim: FuturesSimConfig = field(default_factory=FuturesSimConfig)

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
    """Discover ensemble directories under default vault roots (prop + personal) for defaults.

    Scans each top-level directory from :func:`utils.vault_paths.default_vault_discovery_dirnames`
    that exists (typically ``vault/`` and ``vault_personal/``).

    Supports:

    - **Nested:** ``<vault_top>/<TF>/<weight_hierarchy_group>/<ensemble>/features/*.json``
    - **Legacy flat:** ``<vault_top>/<TF>/<ensemble>/features/*.json``

    The map key is the ensemble leaf directory name; the value is the
    repository-relative path. If the same leaf name exists in more than one vault,
    the first vault in discovery order wins and later duplicates are skipped.

    When ``allowed_timeframes`` is provided, only those vault timeframe folders
    are considered.
    """
    from ensemble.vault.constants import VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES

    repo_root = _PORTFOLIO_RESEARCH_DIR.parent

    allowed_tf_names = (
        {timeframe.name for timeframe in allowed_timeframes}
        if allowed_timeframes is not None
        else None
    )

    def _has_feature_json(features_dir: Path) -> bool:
        return features_dir.is_dir() and any(
            child.suffix == ".json" for child in features_dir.iterdir()
        )

    ensembles: dict[str, str] = {}
    for vault_top in default_vault_discovery_dirnames():
        vault_root = repo_root / vault_top
        if not vault_root.is_dir():
            continue

        timeframe_dirs = [d for d in vault_root.iterdir() if d.is_dir()]
        if allowed_tf_names is not None:
            timeframe_dirs = [
                timeframe_dir
                for timeframe_dir in timeframe_dirs
                if timeframe_dir.name in allowed_tf_names
            ]

        for tf_dir in timeframe_dirs:
            for child in sorted(tf_dir.iterdir()):
                if not child.is_dir():
                    continue
                candidates: list[Path] = []
                if child.name in VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES:
                    candidates = sorted(p for p in child.iterdir() if p.is_dir())
                else:
                    candidates = [child]
                for ensemble_dir in candidates:
                    if _has_feature_json(ensemble_dir / "features"):
                        key = ensemble_dir.name
                        if key not in ensembles:
                            ensembles[key] = str(ensemble_dir.relative_to(repo_root))

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

    # Explicit research windows: portfolio_test uses these phase splits
    # (Train: fit+score on train; Validation: fit on train, score on val; Test: fit on train+val, score on test).
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

    target_volatility = 0.07
    # Ledoit–Wolf–shrinkage correlation → inverse column-sum weights (see WeightLayer).
    weight_layer_method = "ledoit_wolf_min_corr"
    weight_layer_kwargs = {
        "fdm_max": 2.0,
    }
    max_position_pct = 3.5
    baseline_mode = "equal_weight"
    output_root = _PORTFOLIO_RESEARCH_DIR / "results"
    export_per_timeframe_tearsheets = False
    export_per_ensemble_tearsheets = False
    feature_vault_correlation = FeatureVaultCorrelationConfig(enabled=True)
    strict_cache_preflight = False

    # ==========================================================================
    # EDIT ABOVE
    # ==========================================================================

    # ------------------------------------------------------------------
    # Futures contract simulation (optional; set enabled=True to run)
    # Maps research ticker name → micro-futures spec (canonical table in
    # ``utils.futures_micro_specs``). Margins are illustrative.
    # ------------------------------------------------------------------
    _micro = canonical_listed_micro_futures()
    futures_sim = FuturesSimConfig(
        enabled=True,
        account_capital=100_000.0,
        instrument_specs={
            k: FuturesInstrumentSpec(
                multiplier=_micro[k].micro_dollars_per_point,
                margin_long=_micro[k].illustrative_margin_long_usd,
                margin_short=_micro[k].illustrative_margin_short_usd,
                product_code=_micro[k].micro_symbol,
            )
            for k in ("NQ", "ES", "GC")
        },
        leverage_mode=LeverageMode.FINITE,
        emit_tracking_error_csv=True,
        emit_diagnostics_csv=True,
    )

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
        exclude_feature_stems_by_ensemble=None,
        ensemble_vault_refit=True,
        futures_sim=futures_sim,
    )


def load_prop_firm_portfolio_research_config(
    *,
    research_end: datetime | None = None,
) -> PortfolioResearchConfig:
    """Portfolio research config aligned with **live prop** ``live_forecast_config_prop.json``.

    - **Universe:** ``ES``, ``NQ``, ``GC`` only (prop ``tradeable_tickers``). No ``TLT`` /
      ``RTY`` in the research ticker list, so vault ensembles that require those symbols
      are dropped by :func:`filter_ensemble_dirs_for_portfolio_tickers` (same rule as
      narrowing the book to micros you actually trade).
    - **Risk / combine:** ``target_volatility=0.20``, ``max_position_pct=2.5``,
      ``WeightLayer`` ``equal_signal`` with ``fdm_max=2.0`` (matches live
      ``GlobalPortfolio`` defaults).
    - **Discrete sim:** ``futures_sim.account_capital=50_000`` and micro specs for
      ES/NQ/GC only.

    - **Vault fit mode:** ``ensemble_vault_refit=False`` matches live Enigma
      (``load_ensemble_from_vault(..., refit=False)`` frozen JSON fits). Default research
      uses ``refit=True`` walk-forward refits on cache.

    ``research_end`` defaults to **today** (naive midnight) and updates ``end``,
    ``test_window.end``, and ``oos_window.test_end`` so the pipeline runs through the
    latest requested calendar day.

    Artifacts go under ``results/prop_firm_profile`` to avoid clobbering the default
    research ``results/`` tree.
    """
    base = load_config()
    end_dt = research_end or datetime.combine(date.today(), datetime.min.time())
    tickers = [Ticker.ES, Ticker.NQ, Ticker.GC]
    ensemble_dirs = base.ensemble_dirs
    _micro = canonical_listed_micro_futures()
    futures_sim = FuturesSimConfig(
        enabled=True,
        account_capital=50_000.0,
        instrument_specs={
            k: FuturesInstrumentSpec(
                multiplier=_micro[k].micro_dollars_per_point,
                margin_long=_micro[k].illustrative_margin_long_usd,
                margin_short=_micro[k].illustrative_margin_short_usd,
                product_code=_micro[k].micro_symbol,
            )
            for k in ("NQ", "ES", "GC")
        },
        leverage_mode=LeverageMode.FINITE,
        emit_tracking_error_csv=True,
        emit_diagnostics_csv=True,
    )
    oos = base.oos_window
    oos_new = (
        replace(oos, test_end=end_dt)
        if oos is not None
        else None
    )
    return replace(
        base,
        tickers=tickers,
        ensemble_dirs=ensemble_dirs,
        end=end_dt,
        test_window=ResearchWindow(start=base.test_window.start, end=end_dt),
        oos_window=oos_new,
        target_volatility=0.20,
        max_position_pct=2.5,
        weight_layer_method="equal_signal",
        weight_layer_kwargs={"fdm_max": 2.0},
        futures_sim=futures_sim,
        feature_vault_correlation=replace(base.feature_vault_correlation, enabled=False),
        output_root=_PORTFOLIO_RESEARCH_DIR / "results" / "prop_firm_profile",
        export_per_timeframe_tearsheets=False,
        export_per_ensemble_tearsheets=False,
        ensemble_vault_refit=False,
    )
