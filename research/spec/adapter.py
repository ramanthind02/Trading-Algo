"""Adapter — translate a :class:`~research.spec.strategy_spec.StrategySpec` into the existing
pipeline configs and the Nautilus execution engine.

This is a **pure translation layer**. It never mutates the canonical configs
(``research/feature/config.py`` / ``research/portfolio/config.py``); it builds **fresh** config
objects from the spec, composing the spec-driven fields with sensible research defaults and
reusing the canonical sub-config dataclasses + helpers (so we fill the existing pipeline, never
reimplement it).

Mapping (``strategy_spec.md`` §8):

| Spec field                          | Maps onto                                                   |
|-------------------------------------|-------------------------------------------------------------|
| ``tickers`` / ``timeframe``         | ``ResearchConfig`` / ``PortfolioResearchConfig`` tickers/tf |
| ``data_feed``                       | ``data_feed`` literal (``"cfd"`` / ``"futures"``)           |
| ``windows``                         | ``ResearchWindowConfig`` (feature) + 3x ``ResearchWindow``  |
| ``signal.module_name`` + param_grid | ``bias_spec`` dict with list-valued params (grid expansion) |
| ``direction``                       | ``in_sample_defaults.strategy`` + ``vault_save.direction``  |
| ``risk.target_vol`` / max_position  | ``target_volatility`` / ``max_position_pct``                |
| ``execution.*``                     | ``NautilusPnLEngine`` window/execution/cross-after policies |
| ``vault.*``                         | ``VaultSaveConfig`` group + ensemble_name                   |

The Nautilus engine import is **lazy** (inside :func:`to_pnl_engine`) so importing this module —
and building the feature/portfolio configs — does not require ``nautilus_trader``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from research.feature.config import (
    BinningAnalysisConfig,
    EvaluationDefaultsCatalog,
    EvaluationPhaseDefaultsConfig,
    FeatureType,
    InSampleDefaultsCatalog,
    InSamplePhaseDefaultsConfig,
    PermutationResearchConfig,
    ResearchConfig,
    ResearchWindowConfig,
    VaultSaveConfig,
    build_objective_metric_presets,
)
from research.portfolio.config import (
    EnsembleDirsPolicy,
    FuturesSimConfig,
    PortfolioResearchConfig,
    ResearchWindow,
)
from research.spec.strategy_spec import (
    MAX_GRID_COMBOS,
    DataFeed,
    Holding,
    ResearchWindows,
    StrategyMode,
    StrategySpec,
    VolScaling,
)

if TYPE_CHECKING:  # avoid importing nautilus_trader at module import time
    from research.portfolio.pnl.nautilus_engine import NautilusPnLEngine

# Feature research's target return column (matches the canonical ``load_config``).
_TARGET_COL = "log_return"
_FEATURE_RESULTS_ROOT = Path("feature_research/shared_results")


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def default_windows(mode: StrategyMode) -> ResearchWindows:
    """Per-mode window defaults (README "Windows" constraint / ``strategy_spec.md`` §3).

    The test window's end is ``now`` ("→ latest"); it is locked out of selection regardless.
    """

    latest = datetime.now()
    if mode is StrategyMode.DAILY:
        # Split exactly at the 2018-01-01 intraday cutover so each window maps to ONE feed:
        # train (pre-2018) = faithful RATIO futures for vectorized research; validation/test
        # (post-2018) = de-staled CFD for realistic fills/spreads/costs. No splicing, no mixing.
        return ResearchWindows(
            train=(datetime(2000, 1, 1), datetime(2017, 12, 31)),
            validation=(datetime(2018, 1, 1), datetime(2022, 12, 31)),
            test=(datetime(2023, 1, 1), latest),
        )
    return ResearchWindows(
        train=(datetime(2018, 1, 1), datetime(2022, 12, 31)),
        validation=(datetime(2023, 1, 1), datetime(2024, 12, 31)),
        test=(datetime(2025, 1, 1), latest),
    )


def resolve_windows(spec: StrategySpec) -> ResearchWindows:
    """The spec's windows, or the per-mode defaults when unset."""

    return spec.windows if spec.windows is not None else default_windows(spec.mode)


# ---------------------------------------------------------------------------
# Small mappers
# ---------------------------------------------------------------------------


def feed_literal(feed: DataFeed) -> str:
    """``DataFeed`` → the ``data_feed`` literal the pipeline configs consume."""

    return "cfd" if feed is DataFeed.DARWINEX_CFD else "futures"


def exploration_feed_for(spec: StrategySpec) -> str:
    """The vectorized-exploration research feed for a spec, by asset class + mode.

    DAILY research uses the faithful RATIO futures feed (``futures_ratio``) when EVERY ticker has a
    ratio series (back-adjusted futures: ES/NQ/GC/CL/SI/…) — that is where the additive _CCB series
    sign-inverts deep history. Pure-CFD instruments (forex: AUDNZD/EURUSD — no contract roll, no
    ratio parquet) and intraday research use the CFD feed directly. The realistic validation lane is
    separate (always post-2018 CFD; see :func:`to_feature_config`'s ``data_feed``).
    """

    from data_platform.providers.mt5.cfd_candles import has_futures_ratio

    if spec.mode is StrategyMode.DAILY and all(has_futures_ratio(t) for t in spec.tickers):
        return "futures_ratio"
    return "cfd"


# vol_scaling → EWSD (short, long) blend weights. BLENDED is Carver's 70/30 (the pipeline
# default); LONG_ONLY drops the 32-day short term entirely (long-run sigma only). OFF means
# "no tau/sigma scaling at all" — a separate DiversifiedEnsemble toggle, not a blend, so it is
# not expressed here (apply_vol_scaling raises for it rather than silently mis-scaling).
_EWSD_BLEND: dict[VolScaling, tuple[float, float]] = {
    VolScaling.BLENDED: (0.7, 0.3),
    VolScaling.LONG_ONLY: (0.0, 1.0),
}


def ewsd_blend_for(vol_scaling: VolScaling) -> tuple[float, float] | None:
    """The (short, long) EWSD blend weights for a vol-scaling mode (``None`` for OFF)."""

    return _EWSD_BLEND.get(vol_scaling)


def apply_vol_scaling(spec: StrategySpec) -> None:
    """Set the process-global EWSD blend from ``spec.vol_scaling``.

    Call this once at pipeline entry (mirrors ``set_research_feed``) BEFORE running the
    feature/portfolio pipeline, so the EWSD volatility that feeds ``DiversifiedEnsemble``'s
    ``F = tau/sigma`` honours the spec. We do not mutate the canonical configs — this is a
    process-global set-at-entry, exactly like the research feed selector.

    ``OFF`` (no tau/sigma scaling) is a distinct ensemble toggle and is not yet wired; it raises
    here so it is never silently treated as a blend.
    """

    # Lazy import keeps this module importable without the compute layer at definition time.
    from lib.compute.daily_ewsd_volatility import set_ewsd_blend_weights, set_vol_scaling_off

    if spec.vol_scaling is VolScaling.OFF:
        set_vol_scaling_off(True)
        return

    # Ensure the bypass flag is cleared when switching back to a scaling mode.
    set_vol_scaling_off(False)
    weights = ewsd_blend_for(spec.vol_scaling)
    if weights is None:
        raise NotImplementedError(
            f"vol_scaling={spec.vol_scaling.value!r} is not yet wired."
        )
    set_ewsd_blend_weights(*weights)


def build_bias_spec(spec: StrategySpec) -> dict[str, Any]:
    """The ``bias_spec`` dict the pipeline consumes (list-valued params → cartesian grid).

    Direction is carried on ``in_sample_defaults.strategy`` (see :func:`to_feature_config`),
    not injected into params — node parameter names vary, so the author includes a
    ``strategy_mode`` entry in ``param_grid`` when their node needs one.
    """

    return {
        "module_name": spec.signal.module_name,
        "timeframes": [spec.timeframe],
        "params": {name: list(values) for name, values in spec.signal.param_grid.items()},
    }


def build_eval_bias_spec(spec: StrategySpec) -> dict[str, Any]:
    """A single representative combo (the **center** of each grid axis), scalar params.

    Mirrors the canonical ``load_config`` pairing of a grid ``bias_spec`` with a scalar
    ``eval_bias_spec``. Used by the Pass-1 ATR% decile analysis, the saved-feature eval combo,
    and anything that requires exactly one combo.
    """

    center = {
        name: list(values)[len(list(values)) // 2]
        for name, values in spec.signal.param_grid.items()
    }
    return {
        "module_name": spec.signal.module_name,
        "timeframes": [spec.timeframe],
        "params": center,
    }


# ---------------------------------------------------------------------------
# Feature research config
# ---------------------------------------------------------------------------


def to_feature_config(spec: StrategySpec) -> ResearchConfig:
    """Build a fresh :class:`ResearchConfig` (in-sample + validation) from the spec.

    Feature research has **no project test holdout** by design (the locked test window only
    runs in portfolio research / monitoring), so only train + validation map here.
    """

    validate(spec)
    windows = resolve_windows(spec)
    research_window = ResearchWindowConfig(
        train_start=windows.train[0],
        train_end=windows.train[1],
        val_start=windows.validation[0],
        val_end=windows.validation[1],
    )

    bias_spec = build_bias_spec(spec)
    reports_dir = _FEATURE_RESULTS_ROOT / "signed_signal" / spec.name
    phase_defaults = InSamplePhaseDefaultsConfig(
        bias_spec=bias_spec,
        target_col=_TARGET_COL,
        strategy=spec.direction,
        reports_dir=reports_dir,
    )
    in_sample_defaults = InSampleDefaultsCatalog(
        continuous=phase_defaults,
        signed_signal=phase_defaults,
    )

    # Scalar eval combo (center of the grid) — required by Pass-1 decile analysis + vault save.
    eval_spec = build_eval_bias_spec(spec)
    evaluation_defaults = EvaluationDefaultsCatalog(
        continuous=EvaluationPhaseDefaultsConfig(bias_spec=eval_spec),
        signed_signal=EvaluationPhaseDefaultsConfig(bias_spec=eval_spec),
    )

    presets = build_objective_metric_presets(spec.timeframe)
    vault_save = VaultSaveConfig(
        direction=spec.direction,
        ensemble_name=spec.vault.ensemble_name,
        weight_hierarchy_group=spec.vault.weight_hierarchy_group,
        tickers=tuple(spec.tickers),
        vault_profile="prop",
        dry_run=True,  # promotion is a separate, explicit, human-gated step
    )

    return ResearchConfig(
        tickers=list(spec.tickers),
        start=research_window.train_start,
        end=research_window.val_end,
        permutation=PermutationResearchConfig(objective_metric=presets["t_stat"]),
        objective_metric_presets=presets,
        binning_params=BinningAnalysisConfig(strategy=spec.direction),
        in_sample_defaults=in_sample_defaults,
        timeframe=spec.timeframe,
        feature_type=FeatureType.SIGNED_SIGNAL,
        evaluation_defaults=evaluation_defaults,
        research_window=research_window,
        output_root=_FEATURE_RESULTS_ROOT,
        vault_save=vault_save,
        # Feed policy: EXPLORATION (vectorized, daily) runs on the single SPLICED feed —
        # ratio futures pre-2018 + de-staled CFD post-2018 — set by the run executor via
        # set_research_feed("spliced"). It is faithful %-returns end-to-end (never the additive
        # _CCB series, which sign-inverts deep history). VALIDATION (realistic sim) reads this
        # config.data_feed = "cfd" (post-2018 CFD only, aligning with the train/validation split).
        # spec.data_feed is an ignored back-compat label.
        data_feed="cfd",
        # Spec-driven validation uses the fast vectorized lane for combo selection —
        # the Nautilus realistic lane is too slow to run across a full param grid (e.g.
        # 21 combos × BacktestEngine = 10+ min). Nautilus is reserved for the
        # portfolio-addition gate, which runs on the single selected combo only.
        realistic_phases=(),
        # Execution fields are kept so the gate (when it runs) can honour the spec.
        execution_entry_policy=spec.execution.entry_policy.value,
        execution_unfilled_limit=spec.execution.unfilled_limit.value,
        execution_holding=spec.execution.holding.value,
    )


# ---------------------------------------------------------------------------
# Portfolio research config
# ---------------------------------------------------------------------------


def to_portfolio_config(spec: StrategySpec) -> PortfolioResearchConfig:
    """Build a fresh :class:`PortfolioResearchConfig` (test/holdout scoring) from the spec.

    ``ensemble_dirs`` is left empty with :attr:`EnsembleDirsPolicy.ALLOW_EMPTY`: the candidate
    ensemble's vault path is wired by the executor at run time (it does not exist in the vault
    until promotion), so the spec's portfolio config carries only the declarative fields.
    """

    validate(spec)
    windows = resolve_windows(spec)
    return PortfolioResearchConfig(
        tickers=list(spec.tickers),
        timeframe=spec.timeframe,
        start=windows.train[0],
        end=windows.test[1],
        use_cache=True,
        train_window=ResearchWindow(start=windows.train[0], end=windows.train[1]),
        validation_window=ResearchWindow(start=windows.validation[0], end=windows.validation[1]),
        test_window=ResearchWindow(start=windows.test[0], end=windows.test[1]),
        ensemble_dirs={},
        ensemble_dirs_policy=EnsembleDirsPolicy.ALLOW_EMPTY,
        target_volatility=spec.risk.target_vol,
        max_position_pct=spec.risk.max_position_pct,
        futures_sim=FuturesSimConfig(enabled=False),
        # Portfolio-addition (realistic) scoring uses post-2018 CFD, matching the validation
        # realistic-sim feed in to_feature_config. spec.data_feed is an ignored back-compat label.
        data_feed="cfd",
    )


# ---------------------------------------------------------------------------
# Nautilus execution engine
# ---------------------------------------------------------------------------


def to_pnl_engine(spec: StrategySpec) -> "NautilusPnLEngine":
    """Build a :class:`NautilusPnLEngine` from the spec's execution policies.

    Limitations (build item #3 in the README): the engine works a **single** execution policy
    today, so ``entry_policy`` drives it and ``exit_policy`` is not yet honoured per-leg. The
    ``CARRY`` unfilled-limit option is approximated by a pure-passive ``CrossAfterPolicy``
    (``session_fraction=1.0``, no cross). The ``ROLLOVER_FLATTEN_REENTER`` window policy is a
    portfolio-level overlay, not a spec choice, so it is never selected here.
    """

    # Lazy import: keeps the config builders usable without nautilus_trader installed.
    from research.portfolio.pnl.nautilus_engine import (
        CrossAfterPolicy,
        ExecutionPolicy,
        ExecutionWindowPolicy,
        NautilusPnLEngine,
    )

    validate(spec)
    execution = spec.execution
    window_policy = (
        ExecutionWindowPolicy.CLOSE_TO_CLOSE
        if execution.holding is Holding.OVERNIGHT
        else ExecutionWindowPolicy.INTRADAY_OPEN_TO_CLOSE
    )
    execution_policy = ExecutionPolicy(execution.entry_policy.value)
    # CARRY → pure passive (never cross); CROSS_AFTER → engine default cutoff.
    cross_after = (
        CrossAfterPolicy(session_fraction=1.0)
        if execution.unfilled_limit.name == "CARRY"
        else CrossAfterPolicy()
    )
    return NautilusPnLEngine(
        window_policy=window_policy,
        execution_policy=execution_policy,
        cross_after=cross_after,
        capital=spec.account.capital,
        starting_balance=spec.account.capital,
    )


# ---------------------------------------------------------------------------
# End-to-end validation (adapter contract)
# ---------------------------------------------------------------------------


def validate(spec: StrategySpec) -> None:
    """Re-check the spec end-to-end per the adapter contract.

    The sub-objects already enforce these at construction; this affirms them as one gate so a
    spec assembled by any path (e.g. ``replace``) is caught before a run. Raises ``ValueError``.
    """

    if spec.signal.num_combos > MAX_GRID_COMBOS:
        raise ValueError(
            f"Signal grid expands to {spec.signal.num_combos} combos (cap {MAX_GRID_COMBOS})."
        )
    # Window ordering (resolve_windows constructs ResearchWindows, which validates ordering).
    resolve_windows(spec)
    # Sleeve membership + fill_feed consistency are enforced by VaultTarget / ExecutionSpec
    # __post_init__; touching them here re-affirms a valid resolution exists.
    _ = spec.vault.weight_hierarchy_group
    _ = spec.execution.resolved_fill_feed()
