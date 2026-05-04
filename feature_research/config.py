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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypeVar

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
    PositionMode,
    Ticker,
    TimeFrame,
    coerce_direction,
)
from utils.vault_paths import VaultProfile, resolve_vault_root, resolve_vault_root_for_profile

from feature_research.bias_spec_catalog import first_bias_spec
from nodes.regime.sma.stacked_sma_long_only import stacked_sma_period_combos

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
    """How portfolio-admission loads the current working vault portfolio."""

    tickers: tuple[Ticker, ...] | None = None
    ensemble_dirs: dict[str, str] | None = None
    target_volatility: float = 0.15
    weight_layer_method: str = "ledoit_wolf_min_corr"
    weight_layer_kwargs: dict[str, Any] = field(default_factory=lambda: {"fdm_max": 2.0})
    max_position_pct: float = 3.5
    baseline_mode: str = "equal_weight"
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
    ephemeral_weight_hierarchy_group: str = "momentum"
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
    validation_window: OOSWindowConfig | None = None
    oos_window: OOSWindowConfig | None = None
    n_jobs: int = 8
    output_root: Path = field(default_factory=lambda: Path("feature_research/shared_results"))
    generate_ticker_tearsheets: bool = False
    tearsheet_target_annual_volatility: float | None = None
    portfolio_source: PortfolioSourceConfig | None = None
    portfolio_inclusion: PortfolioInclusionConfig = field(
        default_factory=PortfolioInclusionConfig
    )
    vault_save: VaultSaveConfig | None = None
    sector_allocation_config_path: str | None = None
    #: When True, :mod:`portfolio_research.run_feature_vault_correlation` may export
    #: research-vs-vault correlation tables (also requires
    #: ``PortfolioResearchConfig.feature_vault_correlation.enabled``). See
    #: ``docs/library/Ensemble/portfolio.md`` and vault correlation docs.
    portfolio_vault_correlation: bool = False

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
        if self.oos_window is not None:
            return self.oos_window.train_start, self.oos_window.train_end
        if self.validation_window is not None:
            return self.validation_window.train_start, self.validation_window.train_end
        return self.start, self.end


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

    repo_root = _FEATURE_RESEARCH_DIR.parent

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


def load_config() -> ResearchConfig:
    """Researcher overrides for the main signed-signal feature-research pipeline."""
    tickers = [Ticker.ES]

    start = datetime(2000, 1, 1)
    end = datetime(2026, 1, 1)
    timeframe = DEFAULT_TIMEFRAME
    objective_metric_presets = build_objective_metric_presets(timeframe)

    validation_window = OOSWindowConfig(
        train_start=datetime(2000, 1, 1),
        train_end=datetime(2018, 12, 31),
        test_start=datetime(2019, 1, 1),
        # Temporarily: full post-train tail through dataset end (overlaps calendar with OOS test;
        # OOS phase still fits through ``oos_window.train_end`` — see ``_effective_oos_window``).
        test_end=end,
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

    # In-sample: four-period stacked SMA
    # (``period_1`` < ``period_2`` < ``period_3`` < ``period_4`` — fast → slow) —
    # :class:`~nodes.regime.sma.stacked_sma_long_only.StackedSmaLongOnlyNode`, ``LONG_SHORT``.
    # **12** evenly spaced anchors on **[8, 256]**, each snapped to an **even** integer →
    # ``C(12,4) == 495`` strictly increasing quadruples (~500–600 band; the next step is
    # ``_stacked_ma_n_cand = 13`` → ``C(13,4) == 715``).
    _stacked_ma_n_cand = 13
    _stacked_ma_lo, _stacked_ma_hi = 8, 256
    _stacked_ma_even_candidates: tuple[int, ...] = tuple(
        sorted(
            {
                int(round(_stacked_ma_lo + (_stacked_ma_hi - _stacked_ma_lo) * k / (_stacked_ma_n_cand - 1)))
                // 2
                * 2
                for k in range(_stacked_ma_n_cand)
            }
        )
    )
    _stacked_4p_quads = stacked_sma_period_combos(4, _stacked_ma_even_candidates)
    stacked_sma_4p_is_catalog: list[dict[str, Any]] = [
        {
            "module_name": "stacked_sma_long_only",
            "timeframes": [timeframe],
            "params": {
                "period_1": int(p1),
                "period_2": int(p2),
                "period_3": int(p3),
                "period_4": int(p4),
                "mode": PositionMode.LONG_SHORT,
            },
        }
        for p1, p2, p3, p4 in _stacked_4p_quads
    ]

    # Validation / OOS / ``save_feature_to_vault`` / inclusion: median quad from the same grid.
    _stacked_4p_v1, _stacked_4p_v2, _stacked_4p_v3, _stacked_4p_v4 = _stacked_4p_quads[len(_stacked_4p_quads) // 2]
    stacked_sma_val_eval_bias_spec: dict[str, Any] = {
        "module_name": "stacked_sma_long_only",
        "timeframes": [timeframe],
        "params": {
            "period_1": int(8),
            "period_2": int(194),
            "period_3": int(214),
            "period_4": int(234),
            "mode": PositionMode.LONG_SHORT,
        },
    }

    target_col = "log_return_ewsd"
    signed_reports_dir = (
        _FEATURE_RESEARCH_DIR
        / "in_sample"
        / "results"
        / "signed_signal"
        / "stacked_sma_long_only_4ma_even_8_256_12cand_495quads_long_short"
    )

    in_sample_defaults = InSampleDefaultsCatalog(
        continuous=InSamplePhaseDefaultsConfig(
            bias_spec=copy.deepcopy(stacked_sma_4p_is_catalog),
            target_col=target_col,
            strategy=Direction.LONG,
            reports_dir=signed_reports_dir,
            binning_params_overrides={},
        ),
        signed_signal=InSamplePhaseDefaultsConfig(
            bias_spec=copy.deepcopy(stacked_sma_4p_is_catalog),
            target_col=target_col,
            strategy=Direction.LONG,
            reports_dir=signed_reports_dir,
            binning_params_overrides={},
        ),
    )

    evaluation_defaults = EvaluationDefaultsCatalog(
        continuous=EvaluationPhaseDefaultsConfig(
            bias_spec=copy.deepcopy(stacked_sma_val_eval_bias_spec),
        ),
        signed_signal=EvaluationPhaseDefaultsConfig(
            bias_spec=copy.deepcopy(stacked_sma_val_eval_bias_spec),
        ),
    )

    param_sensitivity = ParamSensitivityConfig()
    vault_save = VaultSaveConfig(
        direction=Direction.LONG,
        ensemble_name="gc_fge_atr10_lb126_r02_ch20_s350_low_d",
        weight_hierarchy_group="momentum",
        dry_run=False,
    )
    portfolio_source = PortfolioSourceConfig(
        tickers=tuple(tickers),
    )
    portfolio_inclusion = PortfolioInclusionConfig(
        ephemeral_weight_hierarchy_group="momentum",
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
        portfolio_source=portfolio_source,
        portfolio_inclusion=portfolio_inclusion,
        vault_save=vault_save,
        portfolio_vault_correlation=True,
    )
