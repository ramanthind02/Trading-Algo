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
    STABLE_REGION = "stable_region"


class WeightLayerAlgorithm(str, Enum):
    INVERSE_CORRELATION = "inverse_correlation"
    EQUAL_FLAT = "equal_flat"
    EQUAL_GROUPED = "equal_grouped"
    INV_DOWNSIDE_VOL_GROUPED = "inv_downside_vol_grouped"
    DOWNSIDE_HRP_GROUPED = "downside_hrp_grouped"
    DOWNSIDE_HRP_FLAT = "downside_hrp_flat"


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
    train_start: datetime
    train_end: datetime
    enabled: bool = False
    test_step: int = 365
    num_steps: int = 8
    top_k: int = 3
    objective_metric_name: str = "sortino"
    min_fold_samples: int = 10
    output_root: Path = Path("feature_research/shared_results")
    use_enhanced_selection: bool = False  # deprecated; prefer selection_method="enhanced"
    trade_freq_min: float = 0.05
    # --- Configurable ensemble selection algorithm ---
    selection_method: WalkforwardSelectionMethod | str = WalkforwardSelectionMethod.TOP_K
    stable_region: object = field(default=None)  # StableRegionConfig | None
    # --- Configurable weight layer method ---
    weight_layer_algorithm: WeightLayerAlgorithm | str = WeightLayerAlgorithm.INVERSE_CORRELATION
    weight_layer_config: object = field(default=None)  # WeightLayerConfig | None

    def __post_init__(self) -> None:
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

    def _effective_selection_method(self) -> str:
        """Resolve the active selection method, honouring the legacy flag."""
        selection_method = _coerce_enum_or_raise(
            self.selection_method,
            WalkforwardSelectionMethod,
            "selection_method",
        )
        if (
            self.use_enhanced_selection
            and selection_method == WalkforwardSelectionMethod.TOP_K
        ):
            return WalkforwardSelectionMethod.ENHANCED.value
        return selection_method.value
