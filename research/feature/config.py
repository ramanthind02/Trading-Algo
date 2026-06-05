"""Feature research configuration for the signed-signal trading pipeline.

Continuous-node binning / EDA research is configured separately in
``feature_research.binning.config``.

**Bias specs in ``load_config()``:** Prefer a plain dict literal for ``module_name``,
``timeframes``, and ``params`` so you can see and edit values in one place. Do **not** add
thin ``build_*_bias_spec`` helpers that only wrap a single module + params dict with no
extra validation or shared logic—those hide the live combo and add indirection for no
benefit. Reserve ``build_*`` helpers for composite nodes (e.g. filter gates) or other
specs where one function genuinely centralizes a non-trivial shape.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Literal, TypeVar

from features.validation.config import (
    InSamplePermutationConfig,
    OOSCandidateSource,
    OutOfSamplePermutationConfig,
    PermutationTestConfig,
)
from features.validation.objective_metrics import ObjectiveMetricSpec
from quantfoundry_core.robustness import ParamPerturbationSpec
from lib.core.enums import (
    Direction,
    DirectionInput,
    Ticker,
    TimeFrame,
    coerce_direction,
)
from lib.core.vault_paths import VaultProfile, resolve_vault_root, resolve_vault_root_for_profile

from research.feature._internal.bias_spec_catalog import first_bias_spec
from research.feature.exploration.filter_gate_catalog import (
    ExplorationFilterGatesConfig,
    resolved_exploration_bias_spec,
)

_FEATURE_RESEARCH_DIR = Path(__file__).resolve().parent
DEFAULT_TIMEFRAME: TimeFrame = TimeFrame.D

# Multi-asset EWMAC trend program: one signed-signal feature → one global stream per
# instrument. Asset-first weight layer maps streams by ``TICKER_ASSET_CLASS``:
# ES/NQ → equity_indices, CL → commodities, EU/JY/BP/CD/SF → fx (each under trend_following).
TREND_FOLLOWING_EQUITY_TICKERS: tuple[Ticker, ...] = (Ticker.ES, Ticker.NQ)
TREND_FOLLOWING_COMMODITY_TICKERS: tuple[Ticker, ...] = (Ticker.CL,)
TREND_FOLLOWING_FX_TICKERS: tuple[Ticker, ...] = (
    Ticker.EU,
    Ticker.JY,
    Ticker.BP,
    Ticker.CD,
    Ticker.SF,
)
TREND_FOLLOWING_UNIVERSE: tuple[Ticker, ...] = (
    *TREND_FOLLOWING_EQUITY_TICKERS,
    *TREND_FOLLOWING_COMMODITY_TICKERS,
    *TREND_FOLLOWING_FX_TICKERS,
)

RAW_TARGET_COLS: frozenset[str] = frozenset({"log_return", "raw_return"})


class CachePopulationMode(Enum):
    """How far back bias-cache population reaches before the analysis window."""

    #: Bootstrap and bias artifacts over full common OHLC (maximum warmup; default).
    FULL_HISTORY = "full_history"
    #: Bias coverage from ``analysis_start - lookback_buffer`` through ``analysis_end``.
    ANALYSIS_PLUS_LOOKBACK = "analysis_plus_lookback"


class RankCorrelationScope(Enum):
    """IS metrics source for validation rank-correlation scatter."""

    #: Score train metrics from loaded combo returns when exploration CSV is absent.
    FULL_GRID = "full_grid"
    #: Require exploration ``param_sensitivity.csv`` for IS metrics; still load grid for val metrics.
    FROM_EXPLORATION_ARTIFACTS = "from_exploration_artifacts"

TBranch = TypeVar("TBranch")


def build_filter_gate_bias_spec(
    timeframe: TimeFrame,
    *,
    filter_module: str,
    filter_params: dict[str, Any],
    signal_module: str,
    signal_params: dict[str, Any],
) -> dict[str, Any]:
    """Build a ``bias_spec`` for :class:`~nodes.composite.filter_gate.FilterGateNode`.

    ``filter_module`` / ``signal_module`` are taxonomy keys (e.g. ``adx_filter``,
    ``cyclical_rsi``, ``cyclical_rsi_signal``). Inner dicts may use list-valued keys for grid expansion
    (see ``expand_param_grid`` / ``expand_bias_specs``).

    This function does **not** pick a filter for you — pass whatever combination you want
    from ``load_config()`` or another orchestrator.
    """
    return {
        "module_name": "filter_gate",
        "timeframes": [timeframe],
        "params": {
            "filter_module": filter_module,
            "filter_params": dict(filter_params),
            "signal_module": signal_module,
            "signal_params": dict(signal_params),
        },
    }


def build_filter_gate_entry_only_bias_spec(
    timeframe: TimeFrame,
    *,
    filter_module: str,
    filter_params: dict[str, Any],
    signal_module: str,
    signal_params: dict[str, Any],
) -> dict[str, Any]:
    """Build a ``bias_spec`` for :class:`~nodes.composite.filter_gate_entry_only.FilterGateEntryOnlyNode`.

    Same shape as :func:`build_filter_gate_bias_spec`; filter applies only on **entry** (see node docstring).
    """
    return {
        "module_name": "filter_gate_entry_only",
        "timeframes": [timeframe],
        "params": {
            "filter_module": filter_module,
            "filter_params": dict(filter_params),
            "signal_module": signal_module,
            "signal_params": dict(signal_params),
        },
    }


def build_donchian_gc_validation_atr_filter_gate_spec(
    timeframe: TimeFrame,
    *,
    entry_lookback: int = 10,
    exit_lookback: int = 5,
    min_vol_rank: float = 0.3,
    atr_period: int = 32,
    atr_lookback: int = 252,
    entry_only: bool = True,
) -> dict[str, Any]:
    """Locked exploration winner + ATR%% rank above ``min_vol_rank`` (high tail; Pass-1 deciles 4–9).

    ``min_vol_rank=0.3`` → gate open when rolling ATR%% rank fraction is at least 0.3 (excludes low-vol deciles).
    """
    gate_builder = (
        build_filter_gate_entry_only_bias_spec
        if entry_only
        else build_filter_gate_bias_spec
    )
    return gate_builder(
        timeframe,
        filter_module="atr_percentile_filter",
        filter_params=build_atr_pct_min_rank_filter_params(
            min_rank_fraction=min_vol_rank,
            atr_period=atr_period,
            lookback=atr_lookback,
        ),
        signal_module="donchian_long_only",
        signal_params={
            "entry_lookback": entry_lookback,
            "exit_lookback": exit_lookback,
            "sma_period": 0,
        },
    )


def build_gc_atr_donchian_validation_filter_gate_spec(timeframe: TimeFrame) -> dict[str, Any]:
    """Scalar entry-only gate for validation / OOS / vault / inclusion (one combo, no grid).

    ATR% **low** tail (``percentile_tail="low"``), entry-only gating. In-sample uses the wider grid
    in :func:`load_config`.
    """
    return build_filter_gate_entry_only_bias_spec(
        timeframe,
        filter_module="atr_percentile_filter",
        filter_params={
            "atr_period": 10,
            "lookback": 126,
            "max_rank_fraction": 0.2,
            "rank_metric": "atr_pct",
            "percentile_tail": "low",
        },
        signal_module="donchian_long_only",
        signal_params={
            "channel_lookback": 20,
            "sma_period": 350,
        },
    )


def build_sma252_filter_gate_spec(
    timeframe: TimeFrame,
    *,
    signal_module: str,
    signal_params: dict[str, Any],
) -> dict[str, Any]:
    """Full :class:`~nodes.composite.filter_gate.FilterGateNode` with SMA(252) trend filter.

    When ``close <= SMA(252)`` the filter closes and the gated output is forced to zero on
    that bar (including open positions — unlike entry-only gating).
    """
    return build_filter_gate_bias_spec(
        timeframe,
        filter_module="sma_above_filter",
        filter_params={"period": 252},
        signal_module=signal_module,
        signal_params=dict(signal_params),
    )


def build_sma252_entry_filter_gate_spec(
    timeframe: TimeFrame,
    *,
    signal_module: str,
    signal_params: dict[str, Any],
) -> dict[str, Any]:
    """Entry-only SMA(252) wrapper (legacy). Prefer :func:`build_sma252_filter_gate_spec`."""
    return build_filter_gate_entry_only_bias_spec(
        timeframe,
        filter_module="sma_above_filter",
        filter_params={"period": 252},
        signal_module=signal_module,
        signal_params=dict(signal_params),
    )


def build_close_breakout_validation_filter_gate_spec(timeframe: TimeFrame) -> dict[str, Any]:
    """Locked ``close_breakout`` + SMA(252) full filter gate for validation / vault / inclusion.

    In-sample explores the parameter-free close > previous close signal without the filter;
    evaluation uses :func:`build_sma252_filter_gate_spec` so the signal is flat whenever
    ``close <= SMA(252)`` (including open positions).
    """
    return build_sma252_filter_gate_spec(
        timeframe,
        signal_module="close_breakout",
        signal_params={},
    )


def build_adx_rsi_exploration_grid_params() -> dict[str, Any]:
    """Cartesian grid for daily ``adx_rsi_mr`` on CL (9 combos).

    Sweeps adx_period × rsi_period with SMA(252) trend gate fixed.
    Filter exploration confirmed SMA(252) entry-only gate raises IS SR 0.12 → 0.30.
    Fixed: adx_threshold=25, oversold=30, overbought=70, exit_bars=2, trend_period=252.
    """
    return {
        "adx_period": [10, 14, 21],
        "rsi_period": [3, 5, 7],
    }


def build_adx_rsi_validation_bias_spec(
    timeframe: TimeFrame,
    *,
    adx_period: int = 10,
    adx_threshold: float = 25.0,
    rsi_period: int = 5,
    oversold: float = 30.0,
    overbought: float = 70.0,
    exit_bars: int = 2,
    trend_period: int = 252,
) -> dict[str, Any]:
    """Locked ``adx_rsi_mr`` for validation / vault / inclusion.

    Defaults: adx_period=10, rsi_period=5, trend_period=252 (best IS combo).
    """
    return {
        "module_name": "adx_rsi_mr",
        "timeframes": [timeframe],
        "params": {
            "adx_period": adx_period,
            "adx_threshold": adx_threshold,
            "rsi_period": rsi_period,
            "oversold": oversold,
            "overbought": overbought,
            "exit_bars": exit_bars,
            "trend_period": trend_period,
            "strategy_mode": "long_short",
        },
    }


def build_regime_lrsi_exploration_grid_params() -> dict[str, Any]:
    """Cartesian grid for daily ``regime_lrsi_signal`` on CL (24 combos).

    Sweeps regime-rank MA around the EasyLanguage default (58) × percentile-rank lookback.
    """
    return {
        "regime_ma_period": [38, 48, 52, 58, 64, 68, 78, 88],
        "mr_prank_period": [56, 66, 76],
    }


def build_regime_lrsi_validation_bias_spec(
    timeframe: TimeFrame,
    *,
    regime_ma_period: int = 78,
    hl_sum_period: int = 76,
    hl_range_period: int = 64,
    mr_avg_period: int = 29,
    mr_prank_period: int = 56,
    long_threshold: float = 0.6767,
    short_threshold: float = 0.3233,
    exit_bars: int = 1,
    stop_loss_ticks: int = 0,
    tick_size: float = 0.01,
) -> dict[str, Any]:
    """Locked ungated ``regime_lrsi_signal`` for validation / vault / inclusion (EL defaults)."""
    return {
        "module_name": "regime_lrsi_signal",
        "timeframes": [timeframe],
        "params": {
            "regime_ma_period": regime_ma_period,
            "hl_sum_period": hl_sum_period,
            "hl_range_period": hl_range_period,
            "mr_avg_period": mr_avg_period,
            "mr_prank_period": mr_prank_period,
            "long_threshold": long_threshold,
            "short_threshold": short_threshold,
            "exit_bars": exit_bars,
            "stop_loss_ticks": stop_loss_ticks,
            "tick_size": tick_size,
            "strategy_mode": "long_short",
        },
    }


def build_double7s_exploration_grid_params() -> dict[str, Any]:
    """Cartesian grid for daily ``double7s`` on ES/NQ (9 combos: short_period × ma_period)."""
    return {
        "short_period": [5, 7, 10],
        "ma_period": [150, 200, 252],
    }


def build_double7s_validation_bias_spec(
    timeframe: TimeFrame,
    *,
    short_period: int = 10,
    ma_period: int = 200,
) -> dict[str, Any]:
    """Locked ungated ``double7s`` for validation / vault / inclusion.

    Defaults match filter-exploration baseline **A** on ES/NQ: ``ma200 / sp10``.
    """
    return {
        "module_name": "double7s",
        "timeframes": [timeframe],
        "params": {
            "short_period": short_period,
            "ma_period": ma_period,
        },
    }


def build_double7s_validation_filter_gate_spec(
    timeframe: TimeFrame,
    *,
    short_period: int = 7,
    ma_period: int = 200,
) -> dict[str, Any]:
    """Locked Connors Double 7s + SMA(252) full filter gate for validation / vault / inclusion.

    In-sample explores ungated ``double7s`` (built-in ``ma_period`` trend filter). Evaluation wraps
    the signal in :func:`build_sma252_filter_gate_spec` so exposure is flat when
    ``close <= SMA(252)``.
    """
    return build_sma252_filter_gate_spec(
        timeframe,
        signal_module="double7s",
        signal_params={
            "short_period": short_period,
            "ma_period": ma_period,
        },
    )


def build_percent_b_exploration_grid_params() -> dict[str, Any]:
    """Cartesian grid for daily ``percent_b_signal`` on ES/NQ (24 combos).

    ``period`` × ``std_dev`` × ``upper_threshold``; ``std_dev=3`` dropped (often zero trades on ES/NQ).
    """
    return {
        "period": [10, 15, 20, 30],
        "std_dev": [2.0, 2.5],
        "upper_threshold": [0.5, 0.75, 1.0],
    }


def build_percent_b_validation_filter_gate_spec(
    timeframe: TimeFrame,
    *,
    period: int = 10,
    std_dev: float = 2.5,
    lower_threshold: float = 0.0,
    upper_threshold: float = 1.0,
    exit_bars: int = 5,
) -> dict[str, Any]:
    """Locked %B signal + SMA(252) full filter gate for validation / portfolio addition / vault.

    In-sample explores the ``percent_b_signal`` grid without the filter; evaluation uses this
    composite so the signal is flat whenever ``close <= SMA(252)``.

    Defaults: ``period=10``, ``std_dev=2.5``, ``upper_threshold=1.0`` (IS filter winner baseline).
    """
    return build_sma252_filter_gate_spec(
        timeframe,
        signal_module="percent_b_signal",
        signal_params={
            "period": period,
            "std_dev": std_dev,
            "lower_threshold": lower_threshold,
            "upper_threshold": upper_threshold,
            "strategy_mode": "long",
            "exit_policy": "threshold",
            "exit_bars": exit_bars,
        },
    )


def build_atr_pct_min_rank_filter_params(
    *,
    min_rank_fraction: float = 0.4,
    atr_period: int = 10,
    lookback: int = 252,
) -> dict[str, Any]:
    """ATR% gate params: open when rolling ATR% rank is above ``min_rank_fraction``.

    Uses ``atr_percentile_filter`` with ``percentile_tail="high"`` so the gate is open
    when ``rank_frac >= min_rank_fraction`` (vol regime above the Pass-1 decile cutoff).
    """
    if not 0.0 < min_rank_fraction < 1.0:
        raise ValueError("min_rank_fraction must be in (0, 1).")
    return {
        "atr_period": atr_period,
        "lookback": lookback,
        "max_rank_fraction": 1.0 - min_rank_fraction,
        "rank_metric": "atr_pct",
        "percentile_tail": "high",
    }


def build_rsi_signal_validation_filter_gate_spec(
    timeframe: TimeFrame,
    *,
    rsi_period: int = 2,
    oversold: float = 25.0,
    overbought: float = 75.0,
    exit_bars: int = 5,
    entry_only: bool = False,
    min_vol_rank: float | None = 0.4,
) -> dict[str, Any]:
    """Locked ``rsi_signal`` + SMA(252) filter gate for validation / vault / inclusion.

    When ``min_vol_rank`` is set (default ``0.4``), wraps the SMA+RSI gate with an outer
    ``atr_percentile_filter`` so trades only occur when ATR% rank is above that fraction.
    """
    gate_builder = (
        build_sma252_entry_filter_gate_spec
        if entry_only
        else build_sma252_filter_gate_spec
    )
    inner_gate = gate_builder(
        timeframe,
        signal_module="rsi_signal",
        signal_params={
            "rsi_period": rsi_period,
            "oversold": oversold,
            "overbought": overbought,
            "strategy_mode": "long",
            "exit_policy": "threshold_or_bars",
            "exit_bars": exit_bars,
        },
    )
    if min_vol_rank is None:
        return inner_gate

    outer_gate_module = "filter_gate_entry_only" if entry_only else "filter_gate"
    return build_filter_gate_bias_spec(
        timeframe,
        filter_module="atr_percentile_filter",
        filter_params=build_atr_pct_min_rank_filter_params(
            min_rank_fraction=min_vol_rank,
        ),
        signal_module=outer_gate_module,
        signal_params=dict(inner_gate["params"]),
    )


def build_objective_metric_presets(tf: TimeFrame = TimeFrame.D) -> dict[str, ObjectiveMetricSpec]:
    y = float(tf.bars_per_year)
    return {
        "sortino": ObjectiveMetricSpec(builtin="sortino"),
        "sharpe": ObjectiveMetricSpec(builtin="sharpe"),
        "calmar": ObjectiveMetricSpec(builtin="calmar"),
        "t_stat": ObjectiveMetricSpec(builtin="t_stat"),
        "mean_return": ObjectiveMetricSpec(builtin="mean_return"),
        "profit_factor": ObjectiveMetricSpec(builtin="profit_factor"),
        "sortino_annualized": ObjectiveMetricSpec(
            builtin="sortino", kwargs={"annualization_factor": y}
        ),
        "sharpe_annualized": ObjectiveMetricSpec(
            builtin="sharpe", kwargs={"annualization_factor": y}
        ),
        "calmar_annualized": ObjectiveMetricSpec(
            builtin="calmar", kwargs={"annualization_factor": y}
        ),
    }


@dataclass(frozen=True)
class ResearchWindowConfig:
    """Train + validation windows for feature research (no project test holdout).

    Project-level test evaluation lives in ``portfolio_research`` and monitoring.
    """

    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime

    def __post_init__(self) -> None:
        if self.train_end >= self.val_start:
            raise ValueError(
                "ResearchWindowConfig: train_end must be before val_start "
                f"(got train_end={self.train_end!s}, val_start={self.val_start!s})."
            )
        if self.val_start >= self.val_end:
            raise ValueError(
                "ResearchWindowConfig: val_start must be before val_end "
                f"(got val_start={self.val_start!s}, val_end={self.val_end!s})."
            )

    @property
    def test_start(self) -> datetime:
        """Walkforward holdout alias for the validation slice."""

        return self.val_start

    @property
    def test_end(self) -> datetime:
        """Walkforward holdout alias for the validation slice."""

        return self.val_end


# Backward-compatible alias for portfolio_research and legacy call sites.
OOSWindowConfig = ResearchWindowConfig


class FeatureType(str, Enum):
    CONTINUOUS = "continuous"
    SIGNED_SIGNAL = "signed_signal"


def _branch_for_feature_type(
    feature_type: FeatureType,
    continuous: TBranch,
    signed_signal: TBranch,
) -> TBranch:
    match feature_type:
        case FeatureType.CONTINUOUS:
            return continuous
        case FeatureType.SIGNED_SIGNAL:
            return signed_signal
    raise ValueError(f"Unknown feature_type: {feature_type}")


class VectorShuffleScope(str, Enum):
    """Which parameter combinations receive exploration vector-shuffle permutation."""

    FULL_GRID = "full_grid"
    SELECTED_COMBO = "selected_combo"


@dataclass(frozen=True)
class PermutationResearchConfig:
    objective_metric: ObjectiveMetricSpec
    enabled: bool = False
    nreps: int = 100
    alpha: float = 0.1
    metric_threshold: float = 0.0
    random_seed: int | None = 42
    n_jobs_combos: int = 1
    n_jobs_reps: int = 1
    run_vector_shuffle: bool = True
    #: ``selected_combo`` runs vector shuffle only on the robustness selection-metric winner.
    vector_shuffle_scope: VectorShuffleScope = VectorShuffleScope.SELECTED_COMBO
    candidate_source: OOSCandidateSource = "stage2_passers"

    def to_permutation_test_config(self) -> PermutationTestConfig:
        return PermutationTestConfig(
            in_sample=InSamplePermutationConfig(
                nreps=self.nreps,
                alpha=self.alpha,
                metric_threshold=self.metric_threshold,
                run_stage1=self.run_vector_shuffle,
                n_jobs_combos=self.n_jobs_combos,
                n_jobs_reps=self.n_jobs_reps,
            ),
            out_of_sample=OutOfSamplePermutationConfig(
                objective_metric=self.objective_metric,
                candidate_source=self.candidate_source,
                run_oos_permutation=False,
            ),
            random_seed=self.random_seed,
        )


@dataclass(frozen=True)
class ValidationRobustnessConfig:
    """Post-validation robustness suite (Sharpe, CUSUM, bands, rolling SR, rank correlation)."""

    enabled: bool = True
    n_bootstrap: int = 1000
    random_seed: int | None = 42
    cusum_alpha: float = 0.05
    equity_band_fraction_limit: float = 0.20
    rolling_window: int = 60
    rolling_z_threshold: float = -1.5
    rolling_fraction_limit: float = 0.30
    rank_correlation_floor: float = 0.20
    rank_correlation_scope: RankCorrelationScope = RankCorrelationScope.FULL_GRID
    rank_scatter_n_jobs: int | None = None


@dataclass(frozen=True)
class PortfolioAdditionGateConfig:
    """Portfolio addition gate (Quant Foundry Core), chained after validation walkforward."""

    enabled: bool = True
    pairwise_corr_flag_threshold: float = 0.75
    delta_sr_threshold: float = 0.02
    max_dd_tolerance_pp: float = 0.01
    max_ulcer_increase: float = 0.05
    stress_max_dd_tolerance_pp: float = 0.01
    weight_floor: float = 0.03
    n_bootstrap: int = 1000
    random_seed: int | None = 42
    #: Full-book with/without + candidate-standalone HTML under validation viz ``portfolio_gate_tearsheets/``.
    emit_tearsheets: bool = True
    #: Sleeve-scoped with/without HTML tearsheets (train, val, train+val) under validation viz dir.
    emit_sleeve_tearsheets: bool = True
    #: Parallel workers for portfolio phase backtests; ``None`` → ``ResearchConfig.n_jobs``.
    n_jobs: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.pairwise_corr_flag_threshold <= 1.0:
            raise ValueError("pairwise_corr_flag_threshold must be in (0, 1]")
        if self.delta_sr_threshold < 0.0:
            raise ValueError("delta_sr_threshold must be >= 0")
        if self.max_dd_tolerance_pp < 0.0:
            raise ValueError("max_dd_tolerance_pp must be >= 0")
        if self.max_ulcer_increase < 0.0:
            raise ValueError("max_ulcer_increase must be >= 0")
        if self.stress_max_dd_tolerance_pp < 0.0:
            raise ValueError("stress_max_dd_tolerance_pp must be >= 0")
        if not 0.0 < self.weight_floor < 1.0:
            raise ValueError("weight_floor must be in (0, 1)")
        if self.n_bootstrap < 1:
            raise ValueError("n_bootstrap must be >= 1")


@dataclass(frozen=True)
class RobustnessResearchConfig:
    selection_metric: ObjectiveMetricSpec
    #: Metric for full-grid / individual return-shuffle permutation only.
    #: ``None`` → plain annualized Sharpe (same annualization as ``selection_metric`` timeframe).
    #: Use a non-HAC metric here: return-shuffle nulls are IID by construction, so NW/HAC
    #: correction on null paths is minimal while real-data selection still uses ``selection_metric``.
    full_grid_permutation_metric: ObjectiveMetricSpec | None = None
    enabled: bool = True
    n_permutations: int = 100
    random_seed: int | None = 42
    run_full_grid_permutation: bool = False
    run_individual_permutation: bool = False
    sr_benchmark: float = 0.0
    n_eff_min_overlap: int = 30
    correlation_method: Literal["pearson", "spearman"] = "pearson"
    rolling_window: int = 504
    rolling_min_positive_fraction: float = 0.60
    rolling_alpha: float = 0.05
    sharpe_confidence: float = 0.95
    #: Rolling window for exploration stability chart (6 months ≈ 126 trading days at 252/year).
    stability_rolling_window_days: int = 126


@dataclass(frozen=True)
class ParamSensitivityConfig:
    stability_threshold: float = 0.5
    plot_3d_mode: str = "surface_slices"
    max_eda_output_combos: int = 25
    smoothing_self_weight: float = 3.0
    metric_floor: float | None = 2.0
    perturbation_enabled: bool = True
    perturbation_metric: str = "t_stat"
    perturbation_specs: dict[str, ParamPerturbationSpec] = field(default_factory=dict)


@dataclass(frozen=True)
class BinningAnalysisConfig:
    """Bin-index bounds for walkforward-style research (not Phase 0 quantile export)."""

    bin_counts: list[int] = field(default_factory=lambda: [10, 8, 5, 3])
    strategy: DirectionInput = Direction.LONG
    bin_index_min: int = 0
    bin_index_max: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "strategy",
            coerce_direction(self.strategy, field_name="binning_params.strategy"),
        )
        if self.bin_index_min < 0:
            raise ValueError("bin_index_min must be >= 0")
        if self.bin_index_max is not None and self.bin_index_min > self.bin_index_max:
            raise ValueError("bin_index_max must be >= bin_index_min when set")


@dataclass(frozen=True)
class InSamplePhaseDefaultsConfig:
    #: Single spec or ordered catalog (list of specs); see :func:`expand_bias_specs`.
    bias_spec: dict[str, Any] | list[dict[str, Any]]
    target_col: str
    strategy: DirectionInput
    reports_dir: Path
    binning_params_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "strategy",
            coerce_direction(self.strategy, field_name="in_sample_defaults.strategy"),
        )


@dataclass(frozen=True, init=False)
class InSampleDefaultsCatalog:
    continuous: InSamplePhaseDefaultsConfig
    signed_signal: InSamplePhaseDefaultsConfig

    def __init__(
        self,
        *,
        continuous: InSamplePhaseDefaultsConfig,
        signed_signal: InSamplePhaseDefaultsConfig,
    ) -> None:
        object.__setattr__(self, "continuous", continuous)
        object.__setattr__(self, "signed_signal", signed_signal)

    def for_feature_type(self, feature_type: FeatureType) -> InSamplePhaseDefaultsConfig:
        return _branch_for_feature_type(feature_type, self.continuous, self.signed_signal)


@dataclass(frozen=True)
class EvaluationPhaseDefaultsConfig:
    bias_spec: dict[str, Any]


@dataclass(frozen=True, init=False)
class EvaluationDefaultsCatalog:
    continuous: EvaluationPhaseDefaultsConfig
    signed_signal: EvaluationPhaseDefaultsConfig

    def __init__(
        self,
        *,
        continuous: EvaluationPhaseDefaultsConfig,
        signed_signal: EvaluationPhaseDefaultsConfig,
    ) -> None:
        object.__setattr__(self, "continuous", continuous)
        object.__setattr__(self, "signed_signal", signed_signal)

    def for_feature_type(self, feature_type: FeatureType) -> EvaluationPhaseDefaultsConfig:
        return _branch_for_feature_type(feature_type, self.continuous, self.signed_signal)


@dataclass(frozen=True)
class PortfolioSourceConfig:
    """How portfolio-admission loads the current working vault portfolio.

    When ``ensemble_dirs`` is unset, baseline members are discovered from a **single** vault
    selected by ``vault_profile`` (prop → ``vault/``, personal → ``vault_personal/``) or an
    explicit ``vault_root``. Discovery does not merge prop and personal trees.

    Default ``weight_layer_method`` is ``hierarchy_equal`` with an asset-first 3-level tree
    rebuilt whenever ``ensemble_dirs`` changes (see ``config_with_ensemble_dirs``).
    """

    tickers: tuple[Ticker, ...] | None = None
    ensemble_dirs: dict[str, str] | None = None
    vault_profile: VaultProfile | None = "prop"
    vault_root: str | None = None
    target_volatility: float = 0.15
    weight_layer_method: str = "hierarchy_equal"
    weight_layer_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "fdm_max": 2.0,
            "sr_adjustment": True,
            "sr_avg": 0.5,
            "sr_p_step": 0.01,
            "sr_min_years": 5.0,
            # SR at L1+L2; inv-corr at L3 only (matches portfolio_research CV winner).
            # See portfolio_research/results/weight_layer_cv/ and weight_layer_cv.py.
            "sr_tilt_max_depth": 2,
            "within_group_method": "inverse_avg_pairwise_corr",
        }
    )
    max_position_pct: float = 3.5
    baseline_mode: str = "equal_weight"
    #: Buy-and-hold benchmark when ``baseline_mode='buy_hold'``. ``None`` uses ES if it is
    #: in ``tickers``, otherwise the first portfolio ticker.
    benchmark_ticker: Ticker | None = None
    strict_cache_preflight: bool = False
    generate_tearsheets: bool = True
    export_per_timeframe_tearsheets: bool = False
    export_per_ensemble_tearsheets: bool = False

    def __post_init__(self) -> None:
        if self.baseline_mode not in ("equal_weight", "buy_hold"):
            raise ValueError(
                "portfolio_source.baseline_mode must be 'equal_weight' or 'buy_hold'"
            )
        if self.max_position_pct <= 0:
            raise ValueError("portfolio_source.max_position_pct must be > 0")
        if self.target_volatility <= 0:
            raise ValueError("portfolio_source.target_volatility must be > 0")
        if self.ensemble_dirs is not None and not self.ensemble_dirs:
            raise ValueError(
                "portfolio_source.ensemble_dirs must be non-empty when provided"
            )
        if self.vault_profile is not None and self.vault_profile not in ("prop", "personal"):
            raise ValueError(
                "portfolio_source.vault_profile must be 'prop', 'personal', or None, "
                f"got {self.vault_profile!r}"
            )


@dataclass(frozen=True)
class PortfolioInclusionConfig:
    """Settings for ``python -m feature_research.run_inclusion_gates``.

    Produces **validation-window forecast correlations** (candidate vs same-timeframe peers, per
    ticker then aggregated per peer), **per-ensemble standalone Sharpe/Sortino/Calmar** (each baseline
    and the candidate alone, for train / validation / train+val), and **portfolio comparison
    tearsheets** (full portfolio with vs without the candidate on those same windows).

    Baseline portfolio tickers, windows, and ``ensemble_dirs`` come from
    ``portfolio_research.config.load_config()``; this config controls candidate source, output
    layout under :attr:`ResearchConfig.output_root`, preflight, and whether HTML tearsheets are
    written. Stream combination always uses ``ledoit_wolf_min_corr`` (see
    :func:`feature_research.inclusion_gates.run_portfolio_inclusion`), regardless of
    ``PortfolioResearchConfig.weight_layer_method``.

    **Candidate source:** ``candidate_mode="eval_bias_spec"`` (default) materializes a one-feature
    ensemble from :attr:`ResearchConfig.eval_bias_spec` — the same frozen combo as evaluation /
    ``save_feature_to_vault`` (``evaluation_defaults`` when set, else in-sample defaults). Use
    ``candidate_mode="vault_path"`` with ``candidate_repo_relative_path`` or
    :func:`inferred_inclusion_candidate_path` when testing an ensemble already on disk.

    **CLI:** ``--candidate-path`` / ``--candidate-key`` / ``--candidate-mode`` override config.
    ``preflight`` applies unless ``--no-preflight``. By default ``emit_tearsheets`` is True; use
    ``--no-emit-tearsheets`` to skip HTML output.
    """

    output_subdir: str = "inclusion"
    candidate_mode: Literal["vault_path", "eval_bias_spec"] = "eval_bias_spec"
    ephemeral_ensemble_name: str = "inclusion_candidate"
    #: Bucket for ``hierarchy_equal`` when materializing under ``eval_bias_spec`` (must match
    #: ``ensemble.vault.constants.VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES``).
    ephemeral_weight_hierarchy_group: str = "mean_reversion_indices"
    #: Full multi-asset book for admission (UI research tickers may be a subset).
    candidate_tickers: tuple[Ticker, ...] | None = None
    candidate_repo_relative_path: str | None = None
    candidate_key: str | None = None
    preflight: bool = True
    #: Six HTML tearsheets (with vs without candidate × train / val / train+val). Default on.
    emit_tearsheets: bool = True

    def __post_init__(self) -> None:
        if self.candidate_mode not in ("vault_path", "eval_bias_spec"):
            raise ValueError(
                "candidate_mode must be 'vault_path' or 'eval_bias_spec', "
                f"got {self.candidate_mode!r}"
            )
        if not str(self.ephemeral_ensemble_name).strip():
            raise ValueError("ephemeral_ensemble_name must be non-empty")
        from ensemble.vault.constants import VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES

        _whg = str(self.ephemeral_weight_hierarchy_group).strip()
        if _whg not in VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES:
            raise ValueError(
                "ephemeral_weight_hierarchy_group must be one of "
                f"{sorted(VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES)}, got {_whg!r}"
            )


@dataclass(frozen=True)
class VaultSaveConfig:
    """Persistence target for ``python -m feature_research.save_feature_to_vault``.

    The saved feature always uses :attr:`ResearchConfig.eval_bias_spec` (scalar combo under
    ``evaluation_defaults``, or in-sample defaults when eval is unset).

    Set ``direction`` and exactly one of ``ensemble_name`` (create under ``vault/<tf>/``) or
    ``existing_ensemble_dir``. Optional ``tickers`` defaults to :attr:`ResearchConfig.tickers`.

    When ``vault_root`` is ``None``, ``vault_profile`` selects the default root (prop → ``vault/``,
    personal → ``vault_personal/``); see :func:`vault_save_effective_vault_root`.
    """

    direction: DirectionInput
    ensemble_name: str | None = None
    existing_ensemble_dir: str | Path | None = None
    weight_hierarchy_group: str | None = None
    tickers: tuple[Ticker, ...] | None = None
    init_vault: bool = False
    vault_root: str | None = None
    vault_profile: VaultProfile | None = None
    dry_run: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "direction",
            coerce_direction(self.direction, field_name="vault_save.direction"),
        )
        if self.vault_profile is not None and self.vault_profile not in ("prop", "personal"):
            raise ValueError(
                "vault_save.vault_profile must be 'prop', 'personal', or None, "
                f"got {self.vault_profile!r}"
            )
        has_create = self.ensemble_name is not None and str(self.ensemble_name).strip() != ""
        has_existing = self.existing_ensemble_dir is not None

        if has_create == has_existing:
            raise ValueError(
                "vault_save: set exactly one of ensemble_name (new ensemble) or "
                "existing_ensemble_dir."
            )
        if self.weight_hierarchy_group is not None and not str(
            self.weight_hierarchy_group
        ).strip():
            raise ValueError(
                "vault_save.weight_hierarchy_group must be a non-empty string when provided"
            )


def vault_save_effective_vault_root(vs: VaultSaveConfig) -> Path:
    """Filesystem root for vault operations. Explicit ``vault_root`` wins over ``vault_profile``."""
    if vs.vault_root is not None:
        return resolve_vault_root(vs.vault_root)
    return resolve_vault_root_for_profile(vs.vault_profile or "prop")


@dataclass(frozen=True)
class ResearchConfig:
    tickers: list[Ticker]
    start: datetime
    end: datetime
    permutation: PermutationResearchConfig
    objective_metric_presets: dict[str, ObjectiveMetricSpec]
    binning_params: BinningAnalysisConfig
    in_sample_defaults: InSampleDefaultsCatalog
    timeframe: TimeFrame = TimeFrame.D
    feature_type: FeatureType = FeatureType.CONTINUOUS
    evaluation_defaults: EvaluationDefaultsCatalog | None = None
    param_sensitivity: ParamSensitivityConfig = field(default_factory=ParamSensitivityConfig)
    research_window: ResearchWindowConfig | None = None
    n_jobs: int = 8
    output_root: Path = field(default_factory=lambda: Path("feature_research/shared_results"))
    generate_ticker_tearsheets: bool = False
    tearsheet_target_annual_volatility: float | None = None
    portfolio_source: PortfolioSourceConfig | None = None
    portfolio_inclusion: PortfolioInclusionConfig = field(
        default_factory=PortfolioInclusionConfig
    )
    vault_save: VaultSaveConfig | None = None
    robustness: RobustnessResearchConfig = field(
        default_factory=lambda: RobustnessResearchConfig(
            selection_metric=ObjectiveMetricSpec(builtin="t_stat")
        )
    )
    validation_robustness: ValidationRobustnessConfig = field(
        default_factory=ValidationRobustnessConfig
    )
    portfolio_addition_gate: PortfolioAdditionGateConfig = field(
        default_factory=PortfolioAdditionGateConfig
    )
    sector_allocation_config_path: str | None = None
    #: When True, :mod:`portfolio_research.run_feature_vault_correlation` may export
    #: research-vs-vault correlation tables (also requires
    #: ``PortfolioResearchConfig.feature_vault_correlation.enabled``). See
    #: ``docs/library/Ensemble/portfolio.md`` and vault correlation docs.
    portfolio_vault_correlation: bool = False
    cache_population_mode: CachePopulationMode = CachePopulationMode.FULL_HISTORY
    exploration_filter_gates: ExplorationFilterGatesConfig = field(
        default_factory=ExplorationFilterGatesConfig
    )

    def __post_init__(self) -> None:
        if self.feature_type is not FeatureType.SIGNED_SIGNAL:
            raise ValueError(
                "ResearchConfig now supports only FeatureType.SIGNED_SIGNAL. "
                "Continuous-node binning / EDA research has moved to "
                "feature_research.binning.config."
            )

    @property
    def _phase_defaults(self) -> InSamplePhaseDefaultsConfig:
        return self.in_sample_defaults.for_feature_type(self.feature_type)

    @property
    def bias_spec(self) -> dict[str, Any] | list[dict[str, Any]]:
        return self._phase_defaults.bias_spec

    @property
    def eval_bias_spec(self) -> dict[str, Any]:
        raw = (
            self.evaluation_defaults.for_feature_type(self.feature_type).bias_spec
            if self.evaluation_defaults is not None
            else self._phase_defaults.bias_spec
        )
        return first_bias_spec(raw)

    @property
    def target_col(self) -> str:
        return self._phase_defaults.target_col

    @property
    def strategy(self) -> Direction:
        return self._phase_defaults.strategy

    @property
    def reports_dir(self) -> Path:
        return self._phase_defaults.reports_dir

    @property
    def training_window_bounds(self) -> tuple[datetime, datetime]:
        if self.research_window is not None:
            return self.research_window.train_start, self.research_window.train_end
        return self.start, self.end

    @property
    def validation_window(self) -> ResearchWindowConfig | None:
        """Deprecated alias for :attr:`research_window`."""

        return self.research_window


def apply_fast_validation_profile(config: ResearchConfig) -> ResearchConfig:
    """Return a copy tuned for iterative validation (metrics without heavy I/O).

    Disables the portfolio addition gate, lowers bootstrap replication count, and
    skips QuantStats HTML tearsheets. Core walkforward + validation robustness still run.
    """

    gate = replace(
        config.portfolio_addition_gate,
        enabled=False,
        emit_tearsheets=False,
        emit_sleeve_tearsheets=False,
        n_bootstrap=200,
    )
    robustness = replace(
        config.validation_robustness,
        n_bootstrap=200,
    )
    return replace(
        config,
        portfolio_addition_gate=gate,
        validation_robustness=robustness,
    )


def resolve_portfolio_gate_n_jobs(config: ResearchConfig) -> int:
    """Effective parallel workers for portfolio addition gate phase runs."""

    gate_jobs = config.portfolio_addition_gate.n_jobs
    if gate_jobs is not None:
        return max(1, int(gate_jobs))
    return max(1, int(config.n_jobs))


def resolve_validation_rank_scatter_n_jobs(config: ResearchConfig) -> int:
    """Effective parallel workers for validation rank-scatter metric scoring."""

    scatter_jobs = config.validation_robustness.rank_scatter_n_jobs
    if scatter_jobs is not None:
        return max(1, int(scatter_jobs))
    return max(1, int(config.n_jobs))


def resolve_portfolio_benchmark_ticker(
    *,
    baseline_mode: str,
    portfolio_tickers: Iterable[Ticker],
    benchmark_ticker: Ticker | None = None,
) -> Ticker | None:
    """Pick a benchmark ticker that exists in the portfolio candle set."""

    tickers = tuple(portfolio_tickers)
    if baseline_mode == "equal_weight":
        return None
    if benchmark_ticker is not None:
        return benchmark_ticker if benchmark_ticker in tickers else (tickers[0] if tickers else None)
    preferred = Ticker.ES if Ticker.ES in tickers else None
    return preferred if preferred is not None else (tickers[0] if tickers else None)


def inclusion_candidate_tickers(config: ResearchConfig) -> tuple[Ticker, ...] | None:
    """Default portfolio-admission / vault-save tickers from ``load_config()``.

    CLI and UI runs apply ``--tickers`` via :func:`feature_research.ui.planner.resolve_ui_config`,
    which replaces these defaults for the active run.
    """
    inclusion = config.portfolio_inclusion
    if inclusion is None or inclusion.candidate_tickers is None:
        return None
    return inclusion.candidate_tickers


def ephemeral_weight_hierarchy_group_for_tickers(
    tickers: Iterable[Ticker],
    *,
    configured_group: str = "mean_reversion_indices",
) -> str:
    """Map research tickers to the vault ``weight_hierarchy_group`` for inclusion/gate.

    CL-only commodity runs default to ``crude_oil_mr`` (``commodities/crude_oil_mr``).
    Explicit commodity sleeves (``cl_breakout``, ``gc_breakout``, etc.) are preserved.
    Mixed-asset ticker sets keep ``configured_group``.
    """
    from ensemble.vault.constants import TICKER_ASSET_CLASS, VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES

    names = [
        ticker.name if hasattr(ticker, "name") else str(ticker) for ticker in tickers
    ]
    if not names:
        return configured_group
    asset_classes = {TICKER_ASSET_CLASS.get(name.upper(), "diversified") for name in names}
    if asset_classes == {"commodities"}:
        if (
            configured_group in VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES
            and configured_group != "mean_reversion_indices"
        ):
            return configured_group
        names_upper = {name.upper() for name in names}
        if names_upper <= {"CL"}:
            return "crude_oil_mr"
    if asset_classes == {"equity_indices"} and configured_group in (
        "mean_reversion_gc",
        "crude_oil_mr",
    ):
        return "mean_reversion_indices"
    return configured_group


def discover_portfolio_source_ensemble_dirs(
    source: PortfolioSourceConfig,
    *,
    portfolio_tickers: Iterable[Ticker],
    allowed_timeframes: Iterable[TimeFrame] | None = None,
) -> dict[str, str]:
    """Discover or return explicit baseline ensemble dirs for portfolio admission."""

    if source.ensemble_dirs is not None:
        return dict(source.ensemble_dirs)

    from research.portfolio.config import (
        discover_ensemble_dirs,
        filter_ensemble_dirs_for_portfolio_tickers,
    )
    from lib.core.vault_paths import resolve_portfolio_vault_discovery_dirnames

    timeframes = (
        tuple(allowed_timeframes)
        if allowed_timeframes is not None
        else (TimeFrame.D, TimeFrame.M)
    )
    discovered = discover_ensemble_dirs(
        allowed_timeframes=timeframes,
        vault_discovery_dirnames=resolve_portfolio_vault_discovery_dirnames(
            vault_profile=source.vault_profile,
            vault_root=source.vault_root,
        ),
    )
    return filter_ensemble_dirs_for_portfolio_tickers(
        discovered,
        portfolio_tickers,
        raise_if_empty=False,
    )


def inferred_inclusion_candidate_path(research: ResearchConfig) -> str | None:
    """Repo-relative vault path to the candidate ensemble implied by ``vault_save``.

    Used when ``portfolio_inclusion.candidate_repo_relative_path`` is unset: same ensemble you
    save features to via ``save_feature_to_vault`` (``existing_ensemble_dir`` or
    ``ensemble_name`` + ``weight_hierarchy_group`` + timeframe/direction).

    Only returns a path if that directory **exists** on disk. Tries, in order:

    1. ``existing_ensemble_dir`` (repo-relative or absolute)
    2. ``vault/<TF>/<group>/<ensemble_name>`` when ``weight_hierarchy_group`` is set (full leaf
       folder name; use when ``ensemble_name`` already matches the directory, e.g. ``*_long``)
    3. Canonical path from ``get_ensemble_path`` (``<ensemble_name>_<direction>`` under group)

    Returns ``None`` if ``vault_save`` is missing, no candidate exists, or inferred folders were
    never created (set ``portfolio_inclusion.candidate_repo_relative_path`` to a real ensemble).
    """
    vs = research.vault_save
    if vs is None:
        return None

    repo_root = _FEATURE_RESEARCH_DIR.parents[1]

    def _to_repo_relative(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return resolved.as_posix().replace("\\", "/")

    def _first_existing(candidates: list[Path]) -> str | None:
        seen: set[str] = set()
        for p in candidates:
            try:
                rp = p.resolve()
            except OSError:
                continue
            key = str(rp)
            if key in seen:
                continue
            seen.add(key)
            if rp.is_dir():
                return _to_repo_relative(rp)
        return None

    if vs.existing_ensemble_dir is not None:
        raw = Path(vs.existing_ensemble_dir)
        if not raw.is_absolute():
            return _first_existing([repo_root / raw])
        return _first_existing([raw])

    name = (vs.ensemble_name or "").strip()
    if not name:
        return None

    from ensemble.vault.manager import get_ensemble_path

    vault_path = vault_save_effective_vault_root(vs)
    root_arg = str(vault_path)
    tf_name = research.timeframe.name
    candidates: list[Path] = []
    if vs.weight_hierarchy_group:
        g = str(vs.weight_hierarchy_group).strip()
        if g:
            candidates.append(vault_path / tf_name / g / name)
    candidates.append(
        Path(
            get_ensemble_path(
                research.timeframe,
                name,
                vs.direction,
                vault_root=root_arg,
                weight_hierarchy_group=vs.weight_hierarchy_group,
            )
        )
    )
    if not vs.weight_hierarchy_group:
        candidates.append(vault_path / tf_name / name)

    return _first_existing(candidates)


def build_williamsr_signal_weekly_grid_params() -> dict[str, Any]:
    """Cartesian grid for weekly Williams %R mean-reversion on ES/NQ (24 combos)."""
    return {
        "lookback": [5, 7, 10, 14],
        "oversold": [-80.0, -75.0, -70.0],
        "overbought": [-20.0],
        "strategy_mode": ["long"],
        "exit_policy": ["threshold_or_bars"],
        "exit_bars": [2, 3],
    }


def build_williamsr_signal_validation_filter_gate_spec(
    timeframe: TimeFrame,
    *,
    lookback: int = 10,
    oversold: float = -75.0,
    overbought: float = -20.0,
    exit_bars: int = 2,
    filter_period: int = 40,
) -> dict[str, Any]:
    """Scalar ``williamsr_signal`` + SMA filter gate for validation / vault / inclusion."""
    return build_filter_gate_bias_spec(
        timeframe,
        filter_module="sma_above_filter",
        filter_params={"period": filter_period},
        signal_module="williamsr_signal",
        signal_params={
            "lookback": lookback,
            "oversold": oversold,
            "overbought": overbought,
            "strategy_mode": "long",
            "exit_policy": "threshold_or_bars",
            "exit_bars": exit_bars,
        },
    )


def build_robust_trend_breakout_exploration_grid_params() -> dict[str, Any]:
    """Cartesian grid for daily ``robust_trend_breakout`` on CL/GC (27 combos).

    Sweeps Donchian ``lookback`` × macro ``ema_period`` × ``atr_mult`` with
    ``atr_len=14``, vol filter and cooldown disabled (explored via filter follow-up).
    """
    return {
        "lookback": [20, 55, 80],
        "ema_period": [100, 126, 200],
        "atr_len": [14],
        "atr_mult": [1.5, 2.5, 3.5],
        "atr_vol_filter": [0],
        "cooldown_bars": [0],
    }


def build_robust_trend_breakout_validation_bias_spec(
    timeframe: TimeFrame,
    *,
    lookback: int = 55,
    ema_period: int = 126,
    atr_len: int = 14,
    atr_mult: float = 1.5,
    atr_vol_filter: int = 0,
    cooldown_bars: int = 0,
    strategy_mode: str = "long_short",
) -> dict[str, Any]:
    """Locked ungated ``robust_trend_breakout`` for validation / vault / inclusion."""
    return {
        "module_name": "robust_trend_breakout",
        "timeframes": [timeframe],
        "params": {
            "lookback": lookback,
            "ema_period": ema_period,
            "atr_len": atr_len,
            "atr_mult": atr_mult,
            "atr_vol_filter": atr_vol_filter,
            "cooldown_bars": cooldown_bars,
            "strategy_mode": strategy_mode,
        },
    }


def build_robust_trend_breakout_validation_filter_gate_spec(
    timeframe: TimeFrame,
    *,
    lookback: int = 20,
    ema_period: int = 200,
    atr_len: int = 14,
    atr_mult: float = 1.5,
    atr_vol_filter: int = 0,
    cooldown_bars: int = 0,
) -> dict[str, Any]:
    """Locked ``robust_trend_breakout`` + SMA(252) full filter gate for validation / vault / inclusion.

    Defaults match filter-exploration baseline **A** on ES/NQ:
    ``lb20 / ema200 / atr14×1.5 / no vol filter / no cooldown``.
    """
    return build_sma252_filter_gate_spec(
        timeframe,
        signal_module="robust_trend_breakout",
        signal_params={
            "lookback": lookback,
            "ema_period": ema_period,
            "atr_len": atr_len,
            "atr_mult": atr_mult,
            "atr_vol_filter": atr_vol_filter,
            "cooldown_bars": cooldown_bars,
        },
    )


def build_scaled_z_mr_exploration_bias_specs(timeframe: TimeFrame) -> list[dict[str, Any]]:
    """Ungated ``scaled_z_mr`` grid: lookback × z-threshold spacing (16 combos)."""
    lookbacks = [10, 15, 20, 30]
    threshold_sets: tuple[dict[str, float], ...] = (
        {"z1": -0.75, "z2": -1.25, "z3": -1.75},
        {"z1": -1.0, "z2": -1.5, "z3": -2.0},
        {"z1": -1.25, "z2": -1.75, "z3": -2.25},
        {"z1": -1.0, "z2": -2.0, "z3": -3.0},
    )
    return [
        {
            "module_name": "scaled_z_mr",
            "timeframes": [timeframe],
            "params": {"lookback": lookback, "exit_z": 0.0, **thresholds},
        }
        for lookback in lookbacks
        for thresholds in threshold_sets
    ]


def build_scaled_z_mr_validation_filter_gate_spec(
    timeframe: TimeFrame,
    *,
    lookback: int = 10,
    z1: float = -1.0,
    z2: float = -2.0,
    z3: float = -3.0,
    exit_z: float = 0.0,
) -> dict[str, Any]:
    """Locked ``scaled_z_mr`` + SMA(252) full filter gate for validation / vault / inclusion."""
    return build_sma252_filter_gate_spec(
        timeframe,
        signal_module="scaled_z_mr",
        signal_params={
            "lookback": lookback,
            "z1": z1,
            "z2": z2,
            "z3": z3,
            "exit_z": exit_z,
        },
    )


def build_casey_percent_c_exploration_grid_params() -> dict[str, Any]:
    """Cartesian grid for daily ``casey_percent_c_signal`` on SI (84 combos).

    Sweeps ema_lookback × atr_multiplier × smoothing to find correct band scaling.
    Fixed: atr_lookback=20, oversold=30, overbought=70 (standard thresholds).

    Hypothesis: Fixed multiplier=1.25 may be inverting signal (bands too wide/narrow for SI).
    Test multiplier range [0.5, 1.0, 1.25, 1.5, 2.0] to find correct mean reversion regime.
    """
    return {
        "ema_lookback": [3, 4, 5, 6, 7, 8, 9],
        "atr_lookback": [20],
        "multiplier": [0.5, 1.0, 1.25, 1.5, 2.0],
        "smoothing": [2, 3],
        "oversold": [30],
        "overbought": [70],
    }


def build_casey_percent_c_momentum_exploration_grid_params() -> dict[str, Any]:
    """Cartesian grid for daily ``casey_percent_c_signal`` on SI — MOMENTUM MODE (105 combos).

    Sweeps ema_lookback × multiplier × exit_bars to find best band scaling + hold time.
    Fixed: atr_lookback=20, smoothing=3, oversold=30, overbought=70, negate_signal=True.
    smoothing fixed at 3 (prior runs showed negligible difference vs 2).
    """
    return {
        "ema_lookback": [3, 4, 5, 6, 7, 8, 9],
        "atr_lookback": [20],
        "multiplier": [0.5, 1.0, 1.25, 1.5, 2.0],
        "smoothing": [3],
        "oversold": [30],
        "overbought": [70],
        "exit_bars": [3, 5, 7],
    }


def build_casey_percent_c_validation_bias_spec(
    timeframe: TimeFrame,
    *,
    ema_lookback: int = 3,
    atr_lookback: int = 20,
    multiplier: float = 1.25,
    smoothing: int = 3,
    oversold: float = 30.0,
    overbought: float = 70.0,
    exit_bars: int = 10,
    negate_signal: bool = False,
) -> dict[str, Any]:
    """Locked ``casey_percent_c_signal`` for validation / vault / inclusion.

    exit_policy='threshold': holds until the opposite band is crossed (no time exit).
    In long_short mode this means the signal bounces directly between +1 and -1.
    """
    return {
        "module_name": "casey_percent_c_signal",
        "timeframes": [timeframe],
        "params": {
            "ema_lookback": ema_lookback,
            "atr_lookback": atr_lookback,
            "multiplier": multiplier,
            "smoothing": smoothing,
            "oversold": oversold,
            "overbought": overbought,
            "strategy_mode": "long_short",
            "exit_policy": "threshold",
            "exit_bars": exit_bars,
            "negate_signal": negate_signal,
        },
    }


def load_config() -> ResearchConfig:
    """SI daily SMA(252) regime signal (D) — long above SMA, short/flat below SMA.

    In-sample explores ``sma_regime_signal`` over mode (long_only/long_short) — 2 combos.
    Fixed: period=252. Validation uses long_short. Sleeve: ``silver_mr``.
    """
    tickers = [Ticker.SI]
    portfolio_universe = (Ticker.ES, Ticker.NQ, Ticker.CL, Ticker.SI, Ticker.GC)

    timeframe = TimeFrame.D
    objective_metric_presets = build_objective_metric_presets(timeframe)

    research_window = ResearchWindowConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2018, 12, 31),
        val_start=datetime(2019, 1, 1),
        val_end=datetime(2022, 12, 31),
    )
    start = research_window.train_start
    end = research_window.val_end

    permutation = PermutationResearchConfig(
        objective_metric=objective_metric_presets["t_stat"],
        enabled=True,
        run_vector_shuffle=True,
        vector_shuffle_scope=VectorShuffleScope.SELECTED_COMBO,
        nreps=100,
        alpha=0.1,
        n_jobs_reps=8,
        n_jobs_combos=8,
        random_seed=42,
    )
    robustness = RobustnessResearchConfig(
        selection_metric=objective_metric_presets["t_stat"],
        enabled=True,
        n_permutations=100,
        random_seed=42,
        run_full_grid_permutation=False,
        run_individual_permutation=False,
    )
    feature_type = FeatureType.SIGNED_SIGNAL
    research_strategy = Direction.LONG_SHORT

    is_spec = {
        "module_name": "sma_regime_signal",
        "timeframes": [timeframe],
        "params": {
            "period": [252],
            "mode": ["long_only", "long_short"],
        },
    }
    eval_bias_spec = {
        "module_name": "sma_regime_signal",
        "timeframes": [timeframe],
        "params": {
            "period": 252,
            "mode": "long_short",
        },
    }

    target_col = "log_return"
    signed_reports_dir = (
        _FEATURE_RESEARCH_DIR
        / "in_sample"
        / "results"
        / "signed_signal"
        / "daily_sma_regime_si"
    )
    binning_overrides = {"bin_counts": [10], "strategy": research_strategy}
    vault_group = "silver_trend"

    in_sample_defaults = InSampleDefaultsCatalog(
        continuous=InSamplePhaseDefaultsConfig(
            bias_spec=copy.deepcopy(is_spec),
            target_col=target_col,
            strategy=research_strategy,
            reports_dir=signed_reports_dir,
            binning_params_overrides=binning_overrides,
        ),
        signed_signal=InSamplePhaseDefaultsConfig(
            bias_spec=copy.deepcopy(is_spec),
            target_col=target_col,
            strategy=research_strategy,
            reports_dir=signed_reports_dir,
            binning_params_overrides=binning_overrides,
        ),
    )

    evaluation_defaults = EvaluationDefaultsCatalog(
        continuous=EvaluationPhaseDefaultsConfig(
            bias_spec=copy.deepcopy(eval_bias_spec),
        ),
        signed_signal=EvaluationPhaseDefaultsConfig(
            bias_spec=copy.deepcopy(eval_bias_spec),
        ),
    )

    param_sensitivity = ParamSensitivityConfig(
        perturbation_specs={
            "period": ParamPerturbationSpec(min_step=25, valid_min=25),
        },
    )
    vault_save = VaultSaveConfig(
        direction=Direction.LONG_SHORT,
        ensemble_name="sma_regime_si",
        weight_hierarchy_group=vault_group,
        tickers=tuple(tickers),
        vault_profile="prop",
        dry_run=True,
    )
    portfolio_source = PortfolioSourceConfig(
        tickers=portfolio_universe,
        vault_profile="prop",
    )
    portfolio_inclusion = PortfolioInclusionConfig(
        ephemeral_weight_hierarchy_group=vault_group,
        candidate_tickers=tuple(tickers),
        emit_tearsheets=True,
    )
    portfolio_addition_gate = PortfolioAdditionGateConfig(enabled=True)

    binning_params = BinningAnalysisConfig(
        bin_counts=[10],
        strategy=research_strategy,
    )

    return ResearchConfig(
        tickers=tickers,
        start=start,
        end=end,
        permutation=permutation,
        objective_metric_presets=objective_metric_presets,
        binning_params=binning_params,
        in_sample_defaults=in_sample_defaults,
        timeframe=timeframe,
        feature_type=feature_type,
        evaluation_defaults=evaluation_defaults,
        param_sensitivity=param_sensitivity,
        research_window=research_window,
        n_jobs=8,
        output_root=Path("feature_research/shared_results"),
        generate_ticker_tearsheets=False,
        tearsheet_target_annual_volatility=0.15,
        portfolio_source=portfolio_source,
        portfolio_inclusion=portfolio_inclusion,
        vault_save=vault_save,
        robustness=robustness,
        portfolio_addition_gate=portfolio_addition_gate,
        portfolio_vault_correlation=False,
        exploration_filter_gates=ExplorationFilterGatesConfig(enabled=True),
    )
