"""Walkforward runtime config schema and validation.

Runtime types and validation only. Researcher-editable values live in
``feature_research.config.load_config()`` and are passed in via ``build_walkforward()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TypeVar

from feature_research.walkforward.metrics import SUPPORTED_OBJECTIVE_METRICS


class WalkforwardSelectionMethod(str, Enum):
    TOP_K = "top_k"
    ENHANCED = "enhanced"


class WeightLayerAlgorithm(str, Enum):
    INVERSE_CORRELATION = "inverse_correlation"
    EQUAL_FLAT = "equal_flat"
    EQUAL_GROUPED = "equal_grouped"
    INV_DOWNSIDE_VOL_GROUPED = "inv_downside_vol_grouped"
    DOWNSIDE_HRP_GROUPED = "downside_hrp_grouped"
    DOWNSIDE_HRP_FLAT = "downside_hrp_flat"


class MemberPredictionMode(str, Enum):
    BINARY = "binary"
    SHARPE_WEIGHTED = "sharpe_weighted"


EnumT = TypeVar("EnumT", bound=Enum)


def _coerce_enum_or_raise(value: object, enum_cls: type[EnumT], field_name: str) -> EnumT:
    valid_values = tuple(item.value for item in enum_cls)
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must be one of {valid_values}, got '{value}'"
            ) from exc
    raise ValueError(
        f"{field_name} must be one of {valid_values}, got '{value}'"
    )


@dataclass(frozen=True)
class WalkforwardResearchConfig:
    """Walkforward research window and selection algorithm.

    Selection is controlled only by selection_method (no boolean overrides):
      - TOP_K: select top_k params by smoothed objective.
      - ENHANCED: three-objective (smoothed obj + trade_freq + diversity); outputs top_k.
    """

    train_start: datetime
    train_end: datetime
    enabled: bool = False
    test_step: int = 730
    num_steps: int = 4
    top_k: int = 5
    objective_metric_name: str = "t_stat"  # Overridden by build_walkforward() from global config
    min_fold_samples: int = 10
    output_root: Path = Path("feature_research/shared_results")
    trade_freq_min: float = 0.01
    # --- Selection algorithm (single source of truth) ---
    selection_method: WalkforwardSelectionMethod | str = WalkforwardSelectionMethod.TOP_K
    # --- Configurable weight layer method ---
    weight_layer_algorithm: WeightLayerAlgorithm | str = WeightLayerAlgorithm.INVERSE_CORRELATION
    weight_layer_config: object = field(default=None)  # WeightLayerConfig | None
    # --- Output: per-fold tearsheets are slow; set False to skip ---
    output_per_fold_tearsheets: bool = False
    # --- Member forecast strength mapping in walkforward stage-2 fast path ---
    member_prediction_mode: MemberPredictionMode | str = MemberPredictionMode.BINARY
    # --- Parallelism: number of jobs for scoring param combos within each fold; 1 = sequential ---
    n_jobs: int = 1  # -1 = use all CPUs (resolved at runtime in runner)
    # --- Smoothing: weight for the center param relative to each 1-step neighbor ---
    # 1.0 = equal weight (most aggressive smoothing; boundary params get diluted).
    # Higher values (e.g. 2.0–3.0) reduce neighbor dilution for boundary params.
    # Must match the value used in EDA/param_sensitivity so researcher sees the same landscape.
    smoothing_self_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.smoothing_self_weight <= 0:
            raise ValueError("smoothing_self_weight must be > 0")
        if self.train_end <= self.train_start:
            raise ValueError("train_end must be greater than train_start")
        if self.test_step < 1:
            raise ValueError("test_step must be >= 1")
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.objective_metric_name not in SUPPORTED_OBJECTIVE_METRICS:
            raise ValueError(
                "objective_metric_name must be one of: "
                f"{SUPPORTED_OBJECTIVE_METRICS}"
            )
        if self.min_fold_samples < 10:
            raise ValueError("min_fold_samples must be >= 10")
        if not isinstance(self.output_root, Path):
            raise ValueError("output_root must be a Path")
        normalized_output_root = str(self.output_root).strip()
        if normalized_output_root in {"", "."}:
            raise ValueError("output_root must be a non-empty Path")
        if not (0.0 <= self.trade_freq_min <= 1.0):
            raise ValueError("trade_freq_min must be in [0, 1]")
        normalized_selection_method = _coerce_enum_or_raise(
            self.selection_method,
            WalkforwardSelectionMethod,
            "selection_method",
        )
        object.__setattr__(self, "selection_method", normalized_selection_method)

        normalized_weight_layer_algorithm = _coerce_enum_or_raise(
            self.weight_layer_algorithm,
            WeightLayerAlgorithm,
            "weight_layer_algorithm",
        )
        object.__setattr__(
            self,
            "weight_layer_algorithm",
            normalized_weight_layer_algorithm,
        )

        normalized_member_prediction_mode = _coerce_enum_or_raise(
            self.member_prediction_mode,
            MemberPredictionMode,
            "member_prediction_mode",
        )
        object.__setattr__(self, "member_prediction_mode", normalized_member_prediction_mode)

    def _effective_selection_method(self) -> str:
        """Return the active selection method string (top_k or enhanced)."""
        return _coerce_enum_or_raise(
            self.selection_method,
            WalkforwardSelectionMethod,
            "selection_method",
        ).value
