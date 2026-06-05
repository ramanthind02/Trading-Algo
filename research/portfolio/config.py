"""Portfolio research configuration.

Edit here once; run_portfolio_test.py and tests use load_config().
Uses the same walkforward window formula as feature_research (compute_first_fold_bounds)
and ResearchWindowConfig so portfolio in-sample and validation periods stay aligned.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from enum import Enum, auto
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import pandas as pd

from ensemble.vault import feature_files as _vault_feature_files
from research.feature.config import OOSWindowConfig, ResearchWindowConfig
from lib.cache import extract_cross_ticker_names
from lib.core.enums import Ticker, TimeFrame
from lib.core.futures_micro_specs import canonical_listed_micro_futures
from lib.core.vault_paths import (
    VaultProfile,
    default_vault_discovery_dirnames,
    vault_discovery_dirnames_for_profile,
)

logger = logging.getLogger(__name__)

_PORTFOLIO_RESEARCH_DIR = Path(__file__).resolve().parent
# Repo root is two levels up from research/portfolio/ (was one level up when this
# module lived at portfolio_research/). Use this for vault/ensemble path anchoring.
_REPO_ROOT = _PORTFOLIO_RESEARCH_DIR.parents[1]


class EnsembleDirsPolicy(Enum):
    """Whether portfolio research config may use an empty ``ensemble_dirs`` map."""

    REQUIRE_NON_EMPTY = auto()
    ALLOW_EMPTY = auto()


class PortfolioFitMode(Enum):
    """How the portfolio is fit when scoring the project test (holdout) zone."""

    ROLLING_HOLDOUT = auto()
    SINGLE_FIT = auto()


@dataclass(frozen=True)
class PortfolioHoldoutRobustnessConfig:
    """Per-strategy and portfolio monitoring settings on the test window."""

    enabled: bool = True
    sharpe_confidence: float = 0.95
    n_bootstrap: int = 1000
    random_seed: int | None = 42
    cusum_alpha: float = 0.05
    equity_band_fraction_limit: float = 0.20
    rolling_window: int = 60
    rolling_z_threshold: float = -1.5
    rolling_fraction_limit: float = 0.30
    periods_per_year: int = 252
    evaluation_trailing_months: int = 12
    generate_monthly_history: bool = True
    min_evaluation_bars: int = 60
    monitoring_weight_overrides: dict[str, float] | None = None
    split_reference_calibration: bool = True
    reference_vol_regime_shift_threshold: float = 0.30
    reference_vol_recent_weight_on_shift: float = 0.70
    export_strategy_monitoring_tearsheets: bool = True


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
    emit_discrete_tearsheet : bool
        When True, write ``futures_sim/{phase}_discrete_tearsheet.html`` comparing
        rounded-contract vs fractional PnL. Default False (CFD / fractional research).
    """

    enabled: bool = False
    account_capital: float = 100_000.0
    instrument_specs: Mapping[str, FuturesInstrumentSpec] = field(default_factory=dict)
    leverage_mode: LeverageMode = LeverageMode.FINITE
    emit_tracking_error_csv: bool = True
    emit_diagnostics_csv: bool = True
    emit_discrete_tearsheet: bool = False

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


# QF Core PurchasePolicyConfig: explicit 10**9 lifts the funded-account cap (organic max
# still bounded by challenge_account_cap). None defers to the FundedNext preset (typically 6).
UNLIMITED_FUNDED_ACCOUNT_CAP = 10**9


@dataclass(frozen=True)
class PropFirmReportConfig:
    """FundedNext CFD prop-firm reports integrated into the portfolio test pipeline.

    Simulation uses ``quantfoundry_core.prop_firm`` (default preset ``fundednext``).
    Reports are written under ``{output_root}/{phase}/prop_firm/{firm_id}/``.
    """

    enabled: bool = True
    firm_id: str = "fundednext"
    phases: tuple[str, ...] = ("train", "validation", "test")
    output_subdir: str = "prop_firm"
    account_code: str = "50000"
    report_stem: str = "fundednext_portfolio_report"
    save_csvs: bool = True
    funded_account_cap: int | None = UNLIMITED_FUNDED_ACCOUNT_CAP
    challenge_account_cap: int = 6
    challenges_per_purchase_window: int = 1
    payout_buffer_amount: float = 2500.0
    payout_withdrawal_fraction: float = 1.0
    challenge_vol_multiplier: float = 2.0
    funded_vol_multiplier: float = 0.5
    return_target_annual_volatility: float | None = 0.10
    return_target_sharpe: float = 2.0
    return_annualization_factor: float = 252.0
    return_random_seed: int = 44
    rolling_enabled: bool = True
    rolling_window_months: int = 12

    def __post_init__(self) -> None:
        if not self.firm_id.strip():
            raise ValueError("PropFirmReportConfig.firm_id must be non-empty")
        if not self.phases:
            raise ValueError("PropFirmReportConfig.phases must be non-empty")
        allowed = frozenset({"train", "validation", "test"})
        invalid = tuple(phase for phase in self.phases if phase not in allowed)
        if invalid:
            raise ValueError(
                f"PropFirmReportConfig.phases entries must be train/validation/test, got {invalid}"
            )
        if self.rolling_window_months < 1:
            raise ValueError("PropFirmReportConfig.rolling_window_months must be >= 1")
        if self.funded_account_cap is not None and self.funded_account_cap < 1:
            raise ValueError("PropFirmReportConfig.funded_account_cap must be >= 1 when set")


@dataclass(frozen=True)
class FeatureVaultCorrelationConfig:
    """Gate and paths for ``portfolio_research.run_feature_vault_correlation`` exports.

    When ``vault_root`` is ``None``, the export uses the **prop** vault
    (:func:`~portfolio_research.vault_correlation.resolve_default_vault_root`). Set a path to
    scan ``vault_personal`` or another root.
    """

    enabled: bool = True
    vault_root: Path | None = None
    output_subdir: str = "visualization/feature_vault_correlation"


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
        'equal_weight' or 'buy_hold' for PortfolioTester when ``benchmark_ticker`` is None.
    benchmark_ticker : Ticker | None
        When set (default ES), tearsheets and combined baselines use that ticker's buy-and-hold.
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
        research returns with every vault feature JSON on the same timeframe. Writes
        Matplotlib-ready CSVs under ``output_root / feature_vault_correlation.output_subdir``
        when enabled.
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
    ensemble_dirs_policy: EnsembleDirsPolicy = EnsembleDirsPolicy.REQUIRE_NON_EMPTY
    target_volatility: float = 0.15
    weight_layer_method: str = "equal_signal"
    weight_layer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    max_position_pct: float = 3.5
    baseline_mode: str = "buy_hold"
    benchmark_ticker: Ticker | None = Ticker.ES
    output_root: Path = field(default_factory=lambda: _PORTFOLIO_RESEARCH_DIR / "results")
    export_per_timeframe_tearsheets: bool = True
    export_per_ensemble_tearsheets: bool = True
    feature_vault_correlation: FeatureVaultCorrelationConfig = field(
        default_factory=FeatureVaultCorrelationConfig
    )
    strict_cache_preflight: bool = False
    exclude_feature_stems_by_ensemble: Mapping[str, frozenset[str]] | None = None
    ensemble_vault_refit: bool = True
    # P&L lane selector (WP-3). "vectorized" (default) preserves today's frozen
    # baseline path; "nautilus" routes positions into the realistic BacktestEngine
    # lane (WP-3 Unit 2, not yet implemented).
    pnl_engine: Literal["vectorized", "nautilus"] = "vectorized"
    futures_sim: FuturesSimConfig = field(default_factory=FuturesSimConfig)
    prop_firm_report: PropFirmReportConfig = field(default_factory=PropFirmReportConfig)
    portfolio_fit_mode: PortfolioFitMode = PortfolioFitMode.ROLLING_HOLDOUT
    holdout_robustness: PortfolioHoldoutRobustnessConfig = field(
        default_factory=PortfolioHoldoutRobustnessConfig
    )
    min_holdout_fold_days: int = 60  # legacy; holdout uses exactly two folds (see build_holdout_fold_specs)

    def __post_init__(self) -> None:
        if self.baseline_mode not in ("equal_weight", "buy_hold"):
            raise ValueError(
                f"baseline_mode must be 'equal_weight' or 'buy_hold', got '{self.baseline_mode}'"
            )
        if (
            not self.ensemble_dirs
            and self.ensemble_dirs_policy is EnsembleDirsPolicy.REQUIRE_NON_EMPTY
        ):
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
    *,
    vault_discovery_dirnames: tuple[str, ...] | None = None,
) -> Mapping[str, str]:
    """Discover ensemble directories under configured vault roots.

    Scans each top-level directory in ``vault_discovery_dirnames`` when provided,
    otherwise :func:`utils.vault_paths.default_vault_discovery_dirnames` (prop + personal).

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

    repo_root = _REPO_ROOT

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
    vault_tops = (
        vault_discovery_dirnames
        if vault_discovery_dirnames is not None
        else default_vault_discovery_dirnames()
    )
    for vault_top in vault_tops:
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


def discover_ensemble_dirs(
    allowed_timeframes: Iterable[TimeFrame] | None = None,
    *,
    vault_discovery_dirnames: tuple[str, ...] | None = None,
) -> Mapping[str, str]:
    """Public wrapper for vault ensemble discovery used by research and portfolio scripts."""

    return _discover_ensemble_dirs(
        allowed_timeframes,
        vault_discovery_dirnames=vault_discovery_dirnames,
    )


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


def _ensemble_config_ticker_symbols(ensemble_repo_relative_path: str) -> frozenset[str]:
    """Uppercase symbols from ``ensemble_config.json`` when present."""
    config_path = (
        _REPO_ROOT / ensemble_repo_relative_path / "ensemble_config.json"
    )
    if not config_path.is_file():
        return frozenset()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw = payload.get("tickers", [])
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(
        item.strip().upper()
        for item in raw
        if isinstance(item, str) and item.strip()
    )


def cross_tickers_for_ensemble_dir(ensemble_repo_relative_path: str) -> frozenset[Ticker]:
    """Peer symbols referenced in ``cross_tickers`` params (OHLC for signal construction only)."""
    features_dir = _REPO_ROOT / ensemble_repo_relative_path / "features"
    if not features_dir.is_dir():
        return frozenset()

    cross: set[Ticker] = set()
    for _path, feature_config in _vault_feature_files.iter_validated_feature_configs(
        features_dir
    ):
        spec = feature_config.get("bias_node_spec")
        if not isinstance(spec, Mapping):
            continue
        params = spec.get("params")
        if isinstance(params, Mapping):
            cross.update(Ticker[name] for name in extract_cross_ticker_names(params))
    return frozenset(cross)


def traded_tickers_for_ensemble_dir(ensemble_repo_relative_path: str) -> frozenset[Ticker]:
    """Return instruments this ensemble trades (portfolio membership filter).

    Does **not** include ``cross_tickers`` peers (e.g. TLT for ES ``rebalancing_flow``).
    Those are loaded via :func:`cross_tickers_for_ensemble_dir` / preflight bootstrap.
    """
    features_dir = _REPO_ROOT / ensemble_repo_relative_path / "features"
    if not features_dir.is_dir():
        return frozenset()

    required: set[Ticker] = set()
    feature_ticker_sets: list[frozenset[str]] = []
    for _path, feature_config in _vault_feature_files.iter_validated_feature_configs(
        features_dir
    ):
        symbols = _single_feature_instrument_tickers(feature_config)
        if len(symbols) == 1:
            required.add(Ticker[symbols[0]])
        if symbols:
            feature_ticker_sets.append(frozenset(symbols))

    ensemble_symbols = _ensemble_config_ticker_symbols(ensemble_repo_relative_path)
    if (
        ensemble_symbols
        and feature_ticker_sets
        and all(ticker_set == ensemble_symbols for ticker_set in feature_ticker_sets)
        and len(ensemble_symbols) <= 4
    ):
        required.update(Ticker[symbol] for symbol in ensemble_symbols)
    return frozenset(required)


def required_tickers_for_ensemble_dir(ensemble_repo_relative_path: str) -> frozenset[Ticker]:
    """Return all tickers needed to fit/predict this ensemble (traded + cross peers)."""
    return traded_tickers_for_ensemble_dir(
        ensemble_repo_relative_path
    ) | cross_tickers_for_ensemble_dir(ensemble_repo_relative_path)


def scoped_tickers_for_ensemble_dirs(
    portfolio_tickers: Iterable[Ticker],
    ensemble_dirs: Mapping[str, str],
) -> tuple[Ticker, ...]:
    """Narrow portfolio candle loads to instruments required by active ensembles.

    Prevents single-instrument candidates (e.g. GC mean reversion) from loading the
    full prop-book universe and pairing forecasts with the wrong instrument returns
    (often ES when it remains the default benchmark ticker).
    """
    from ensemble.vault.manager import get_ensemble_tickers

    allowed = frozenset(portfolio_tickers)
    required: set[Ticker] = set()
    for path in ensemble_dirs.values():
        traded = traded_tickers_for_ensemble_dir(path)
        if traded:
            required.update(ticker for ticker in traded if ticker in allowed)
            continue
        for ticker in get_ensemble_tickers(path):
            if ticker in allowed:
                required.add(ticker)
    if not required:
        return tuple(portfolio_tickers)
    return tuple(sorted(required, key=lambda ticker: ticker.name))


def filter_ensemble_dirs_for_portfolio_tickers(
    ensemble_dirs: Mapping[str, str],
    portfolio_tickers: Iterable[Ticker],
    *,
    raise_if_empty: bool = True,
) -> dict[str, str]:
    """Drop vault ensembles whose *traded* tickers are outside the portfolio book.

    ``cross_tickers`` peers (e.g. TLT for ES ``rebalancing_flow``) are not portfolio legs;
    they are bootstrapped for cache when the ensemble is included. See
    :func:`traded_tickers_for_ensemble_dir` vs :func:`cross_tickers_for_ensemble_dir``.
    """
    allowed = frozenset(portfolio_tickers)
    kept: dict[str, str] = {}
    skipped: list[str] = []
    for name, path in ensemble_dirs.items():
        need = traded_tickers_for_ensemble_dir(path)
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
    if not kept and raise_if_empty:
        raise ValueError(
            "No ensembles remain after filtering to portfolio tickers. "
            "Add the missing symbols to config.tickers or set ensemble_dirs explicitly."
        )
    return kept


class HoldoutFoldRole(str, Enum):
    """Semantic role for one of the two portfolio holdout folds."""

    VALIDATION = "validation"
    HOLDOUT_TEST = "holdout_test"


@dataclass(frozen=True)
class HoldoutFoldSpec:
    """One portfolio fit/score fold (validation or project test holdout)."""

    fold_id: int
    role: HoldoutFoldRole
    fit_start: datetime
    fit_end: datetime
    test_start: datetime
    test_end: datetime


def describe_weight_layer_policy(weight_layer_kwargs: Mapping[str, Any]) -> str:
    """Human-readable label for the active hierarchy + SR / inv-corr policy."""
    if not weight_layer_kwargs.get("sr_adjustment"):
        return "hierarchy_equal (equal split, no SR tilt)"
    depth = weight_layer_kwargs.get("sr_tilt_max_depth")
    depth_label = "all levels" if depth is None else f"L1–L{int(depth)}"
    within = str(weight_layer_kwargs.get("within_group_method", "equal"))
    if within == "inverse_avg_pairwise_corr":
        inv_label = (
            "inv-corr at L3 (instruments)"
            if depth == 2
            else f"inv-corr below SR depth (min L{(depth or 0) + 1})"
        )
    else:
        inv_label = "equal within groups below SR depth"
    return f"SR tilt {depth_label}; {inv_label}"


def rebuild_weight_layer_kwargs(
    config: PortfolioResearchConfig,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Return weight-layer kwargs with ``hierarchy_spec`` built from ``ensemble_dirs`` when needed."""
    wl_kw = dict(config.weight_layer_kwargs)
    if config.weight_layer_method != "hierarchy_equal":
        return wl_kw
    if wl_kw.get("hierarchy_spec"):
        return wl_kw
    if not config.ensemble_dirs:
        return wl_kw
    from ensemble.vault.hierarchy_spec import build_asset_first_hierarchy_spec_for_ensemble_dirs

    root = repo_root if repo_root is not None else _REPO_ROOT
    wl_kw["hierarchy_spec"] = build_asset_first_hierarchy_spec_for_ensemble_dirs(
        root,
        dict(config.ensemble_dirs),
        strict_group=True,
        portfolio_ticker_names=frozenset(t.name for t in config.tickers),
    )
    return wl_kw


def with_rebuilt_weight_layer(config: PortfolioResearchConfig) -> PortfolioResearchConfig:
    """Clone config with vault-derived ``hierarchy_spec`` when using ``hierarchy_equal``."""
    return replace(config, weight_layer_kwargs=rebuild_weight_layer_kwargs(config))


def build_holdout_fold_specs(config: PortfolioResearchConfig) -> tuple[HoldoutFoldSpec, ...]:
    """Build exactly two folds: train→validation, then rolled train→test.

    Fold 0 fits on the train window and scores validation (portfolio validation phase).

    Fold 1 rolls the training window forward by the validation span: fit uses the
    same calendar length as train, ending at validation end (tail of train+val),
    then scores the project test window.
    """

    train_start = pd.Timestamp(config.train_window.start)
    train_end = pd.Timestamp(config.train_window.end)
    val_start = pd.Timestamp(config.validation_window.start)
    val_end = pd.Timestamp(config.validation_window.end)
    test_start = pd.Timestamp(config.test_window.start)
    test_end = pd.Timestamp(config.test_window.end)

    if val_start <= train_end:
        raise ValueError("validation_window must start after train_window ends")
    if test_start <= val_end:
        raise ValueError("test_window must start after validation_window ends")

    train_span = train_end - train_start
    if train_span <= pd.Timedelta(0):
        raise ValueError("train_window must span at least one day")

    rolled_fit_start = val_end - train_span

    return (
        HoldoutFoldSpec(
            fold_id=0,
            role=HoldoutFoldRole.VALIDATION,
            fit_start=train_start.to_pydatetime(),
            fit_end=train_end.to_pydatetime(),
            test_start=val_start.to_pydatetime(),
            test_end=val_end.to_pydatetime(),
        ),
        HoldoutFoldSpec(
            fold_id=1,
            role=HoldoutFoldRole.HOLDOUT_TEST,
            fit_start=rolled_fit_start.to_pydatetime(),
            fit_end=val_end.to_pydatetime(),
            test_start=test_start.to_pydatetime(),
            test_end=test_end.to_pydatetime(),
        ),
    )


def load_config() -> PortfolioResearchConfig:
    """Single source of truth for portfolio research. Edit the block below."""
    # ==========================================================================
    # EDIT BELOW
    # ==========================================================================
    tickers = [
        Ticker.ES,
        Ticker.NQ,
        Ticker.GC,
        Ticker.CL,
    ]
    timeframe = TimeFrame.D
    start = datetime(2000, 1, 1)
    # Latest daily OHLC in data/ohlc_data for ES, NQ, GC, CL (D_* parquet).
    end = datetime(2026, 5, 13)
    use_cache = True
    populate_cache = True

    # Explicit research windows: portfolio_test uses these phase splits
    # (Train: fit+score on train; Validation: fit on train, score on val; Test: fit on train+val, score on test).
    train_window = ResearchWindow(
        start=start,
        end=datetime(2018, 12, 31),
    )
    validation_window = ResearchWindow(
        start=datetime(2019, 1, 1),
        end=datetime(2022, 12, 31),
    )
    test_window = ResearchWindow(
        start=datetime(2023, 1, 1),  # full post-validation live data (~840 bars)
        end=end,
    )

    train_window_years = 15.0
    test_window_years = 2.0
    num_steps = 8  # walkforward window alignment only; holdout is always two folds

    oos_window = OOSWindowConfig(
        train_start=datetime(2007, 1, 1),
        train_end=datetime(2023, 12, 30),
        val_start=datetime(2024, 1, 1),
        val_end=datetime(2026, 5, 13),
    )

    # Daily + monthly + weekly (Williams %R) ensembles from the prop vault.
    ensemble_dirs = filter_ensemble_dirs_for_portfolio_tickers(
        _discover_ensemble_dirs(
            allowed_timeframes=(TimeFrame.D, TimeFrame.M, TimeFrame.W),
            vault_discovery_dirnames=vault_discovery_dirnames_for_profile("prop"),
        ),
        tickers,
    )

    target_volatility = 0.07
    # Ledoit–Wolf–shrinkage correlation → inverse column-sum weights (see WeightLayer).
    weight_layer_method = "hierarchy_equal"
    weight_layer_kwargs = {
        "fdm_max": 2.0,
        "sr_adjustment": True,
        "sr_avg": 0.5,
        "sr_p_step": 0.01,
        "sr_min_years": 5.0,
        # SR tilt at L1 (asset class) and L2 (style group). Inverse-correlation only at
        # L3 (instrument streams within a style) — avoids crushing passive buy_hold vs
        # tactical sleeves at L2 while keeping ES/NQ diversification inside buy_hold.
        # CV: train→validation + 4 expanding folds on IS (2011–2022); see
        # ``python -m portfolio_research.weight_layer_cv`` and results/weight_layer_cv/.
        # Production validation (fit 2000–2018, score 2019–2022):
        #   SR-L2 + inv-L3 → SR 0.796, Calmar 0.819, buy_hold 20.8%
        #   SR-L1 + inv-L2+L3 → SR 0.692, Calmar 0.647, buy_hold 15.0%
        #   SR-L1 + equal L2/L3 → SR 0.622, buy_hold 21.9%
        "sr_tilt_max_depth": 2,
        "within_group_method": "inverse_avg_pairwise_corr",
    }
    max_position_pct = 3.5
    baseline_mode = "buy_hold"
    benchmark_ticker = Ticker.ES
    output_root = _PORTFOLIO_RESEARCH_DIR / "results"
    export_per_timeframe_tearsheets = False
    export_per_ensemble_tearsheets = False
    feature_vault_correlation = FeatureVaultCorrelationConfig(enabled=True)
    strict_cache_preflight = False

    # ==========================================================================
    # EDIT ABOVE
    # ==========================================================================

    # CFD / fractional research: skip micro-futures discrete-contract simulation.
    # Set ``enabled=True`` only when studying integer contract rounding vs fractional PnL.
    futures_sim = FuturesSimConfig(enabled=False)

    prop_firm_report = PropFirmReportConfig(enabled=True)

    return with_rebuilt_weight_layer(
        PortfolioResearchConfig(
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
            benchmark_ticker=benchmark_ticker,
            output_root=output_root,
            export_per_timeframe_tearsheets=export_per_timeframe_tearsheets,
            export_per_ensemble_tearsheets=export_per_ensemble_tearsheets,
            feature_vault_correlation=feature_vault_correlation,
            strict_cache_preflight=strict_cache_preflight,
            exclude_feature_stems_by_ensemble=None,
            ensemble_vault_refit=True,
            futures_sim=futures_sim,
            prop_firm_report=prop_firm_report,
        )
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
        prop_firm_report=replace(base.prop_firm_report, enabled=True),
    )
