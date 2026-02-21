from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from feature_research.walkforward.metrics import SUPPORTED_OBJECTIVE_METRICS

_VALID_SELECTION_METHODS = ("top_k", "enhanced", "stable_region")


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
    selection_method: str = "top_k"  # "top_k" | "enhanced" | "stable_region"
    stable_region: object = field(default=None)  # StableRegionConfig | None
    # --- Configurable weight layer method ---
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
        if self.selection_method not in _VALID_SELECTION_METHODS:
            raise ValueError(
                f"selection_method must be one of {_VALID_SELECTION_METHODS}, "
                f"got '{self.selection_method}'"
            )

    def _effective_selection_method(self) -> str:
        """Resolve the active selection method, honouring the legacy flag."""
        if self.use_enhanced_selection and self.selection_method == "top_k":
            return "enhanced"
        return self.selection_method
