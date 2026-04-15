"""Feature research configuration for the signed-signal trading pipeline.

Continuous-node binning / EDA research is configured separately in
``feature_research.binning.config``.

**Bias specs in ``load_config()``:** Prefer a plain dict literal for ``module_name``,
``timeframes``, and ``params`` so you can see and edit values in one place. The small
``build_*_bias_spec`` helpers below remain for tests and scripts that need the composite
shape without duplicating keys; they are optional for day-to-day research edits.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from feature_selection.validation.config import (
    InSamplePermutationConfig,
    OOSCandidateSource,
    OutOfSamplePermutationConfig,
    PermutationTestConfig,
)
from feature_selection.validation.objective_metrics import ObjectiveMetricSpec
from utils.core.enums import (
    Direction,
    DirectionInput,
    Ticker,
    TimeFrame,
    coerce_direction,
)

_FEATURE_RESEARCH_DIR = Path(__file__).resolve().parent
DEFAULT_TIMEFRAME: TimeFrame = TimeFrame.D

RAW_TARGET_COLS: frozenset[str] = frozenset({"log_return", "raw_return"})

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


def build_sma_regime_bias_spec(
    timeframe: TimeFrame,
    *,
    period: int | list[int],
) -> dict[str, Any]:
    """Build a ``bias_spec`` for :class:`~nodes.regime.sma.sma_regime_signal.SmaRegimeSignalNode`.

    Long when ``close > SMA(period)``, short when ``close < SMA(period)`` (see node docstring).
    ``period`` may be a scalar or a list for grid expansion.
    """
    return {
        "module_name": "sma_regime_signal",
        "timeframes": [timeframe],
        "params": {"period": period},
    }


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
class OOSWindowConfig:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    def __post_init__(self) -> None:
        if self.train_end >= self.test_start:
            raise ValueError(
                "OOSWindowConfig: train_end must be before test_start "
                f"(got train_end={self.train_end!s}, test_start={self.test_start!s})."
            )
        if self.test_start >= self.test_end:
            raise ValueError(
                "OOSWindowConfig: test_start must be before test_end "
                f"(got test_start={self.test_start!s}, test_end={self.test_end!s})."
            )


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
class ParamSensitivityConfig:
    stability_threshold: float = 0.5
    plot_3d_mode: str = "surface_slices"
    max_eda_output_combos: int = 25
    smoothing_self_weight: float = 3.0
    metric_floor: float | None = 2.0


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
    bias_spec: dict[str, Any]
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
class VaultSaveConfig:
    """Persistence target for ``python -m feature_research.save_feature_to_vault``.

    The saved feature always uses :attr:`ResearchConfig.eval_bias_spec` (scalar combo under
    ``evaluation_defaults``, or in-sample defaults when eval is unset).

    Set ``direction`` and exactly one of ``ensemble_name`` (create under ``vault/<tf>/``) or
    ``existing_ensemble_dir``. Optional ``tickers`` defaults to :attr:`ResearchConfig.tickers`.
    """

    direction: DirectionInput
    ensemble_name: str | None = None
    existing_ensemble_dir: str | Path | None = None
    tickers: tuple[Ticker, ...] | None = None
    init_vault: bool = False
    vault_root: str | None = None
    dry_run: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "direction",
            coerce_direction(self.direction, field_name="vault_save.direction"),
        )
        has_create = self.ensemble_name is not None and str(self.ensemble_name).strip() != ""
        has_existing = self.existing_ensemble_dir is not None

        if has_create == has_existing:
            raise ValueError(
                "vault_save: set exactly one of ensemble_name (new ensemble) or "
                "existing_ensemble_dir."
            )


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
    validation_window: OOSWindowConfig | None = None
    oos_window: OOSWindowConfig | None = None
    n_jobs: int = 8
    output_root: Path = field(default_factory=lambda: Path("feature_research/shared_results"))
    generate_ticker_tearsheets: bool = False
    tearsheet_target_annual_volatility: float | None = None
    vault_save: VaultSaveConfig | None = None
    sector_allocation_config_path: str | None = None

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
    def bias_spec(self) -> dict[str, Any]:
        return self._phase_defaults.bias_spec

    @property
    def eval_bias_spec(self) -> dict[str, Any]:
        if self.evaluation_defaults is not None:
            return self.evaluation_defaults.for_feature_type(self.feature_type).bias_spec
        return self._phase_defaults.bias_spec

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
        if self.oos_window is not None:
            return self.oos_window.train_start, self.oos_window.train_end
        if self.validation_window is not None:
            return self.validation_window.train_start, self.validation_window.train_end
        return self.start, self.end


def load_config() -> ResearchConfig:
    """Researcher overrides for the main signed-signal feature-research pipeline."""
    tickers = [Ticker.ES]

    start = datetime(2000, 1, 1)
    end = datetime(2025, 9, 18)
    timeframe = DEFAULT_TIMEFRAME
    objective_metric_presets = build_objective_metric_presets(timeframe)

    validation_window = OOSWindowConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2018, 12, 31),
        test_start=datetime(2019, 1, 1),
        test_end=datetime(2022, 12, 31),
    )
    oos_window = OOSWindowConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2022, 12, 31),
        test_start=datetime(2023, 1, 1),
        test_end=datetime(2025, 9, 18),
    )

    permutation = PermutationResearchConfig(
        objective_metric=objective_metric_presets["t_stat"],
        enabled=True,
    )
    feature_type = FeatureType.SIGNED_SIGNAL

    # RSI short (cross above 75, exit cross below 25 or 3 bars) gated by bearish regime:
    # ``sma_below_filter`` → 1.0 when close < SMA(200), else 0.0.
    rsi_short_bearish_spec = build_filter_gate_bias_spec(
        timeframe,
        filter_module="sma_below_filter",
        filter_params={"period": 100},
        signal_module="rsi_signal",
        signal_params={
            "rsi_period": 2,
            "oversold": 35.0,
            "overbought": 65.0,
            "strategy_mode": "short",
            "exit_policy": "threshold_or_bars",
            "exit_bars": 3,
        },
    )

    target_col = "log_return_ewsd"
    signed_reports_dir = (
        _FEATURE_RESEARCH_DIR
        / "in_sample"
        / "results"
        / "signed_signal"
        / "gc_rsi_signal_short_sma_below_200_rsi2_75_25_exit3"
    )

    in_sample_defaults = InSampleDefaultsCatalog(
        continuous=InSamplePhaseDefaultsConfig(
            bias_spec=copy.deepcopy(rsi_short_bearish_spec),
            target_col=target_col,
            strategy=Direction.SHORT,
            reports_dir=signed_reports_dir,
            binning_params_overrides={},
        ),
        signed_signal=InSamplePhaseDefaultsConfig(
            bias_spec=rsi_short_bearish_spec,
            target_col=target_col,
            strategy=Direction.SHORT,
            reports_dir=signed_reports_dir,
            binning_params_overrides={},
        ),
    )

    evaluation_defaults = EvaluationDefaultsCatalog(
        continuous=EvaluationPhaseDefaultsConfig(
            bias_spec=copy.deepcopy(rsi_short_bearish_spec),
        ),
        signed_signal=EvaluationPhaseDefaultsConfig(
            bias_spec=copy.deepcopy(rsi_short_bearish_spec),
        ),
    )

    param_sensitivity = ParamSensitivityConfig()
    vault_save = VaultSaveConfig(
        direction=Direction.SHORT,
        ensemble_name="gc_rsi_signal_short_sma_below_200_rsi2_75_25_exit3_d",
        dry_run=False,
    )

    binning_params = BinningAnalysisConfig(
        strategy=in_sample_defaults.for_feature_type(feature_type).strategy,
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
        validation_window=validation_window,
        oos_window=oos_window,
        n_jobs=8,
        output_root=Path("feature_research/shared_results"),
        generate_ticker_tearsheets=False,
        tearsheet_target_annual_volatility=0.15,
        vault_save=vault_save,
    )
