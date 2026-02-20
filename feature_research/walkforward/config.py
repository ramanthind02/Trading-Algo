from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from feature_research.walkforward.metrics import SUPPORTED_OBJECTIVE_METRICS


@dataclass(frozen=True)
class WalkforwardResearchConfig:
    train_start: datetime
    train_end: datetime
    enabled: bool = False
    test_step: int = 252
    num_steps: int = 8
    top_k: int = 3
    objective_metric_name: str = "sortino"
    min_fold_samples: int = 10
    output_root: Path = Path("feature_research/shared_results")
    use_enhanced_selection: bool = False
    trade_freq_min: float = 0.05
    n_robustness_blocks: int = 5
    diversity_weight: float = 0.40
    weight_stability: float = 0.50
    weight_robustness: float = 0.50

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
        if self.n_robustness_blocks < 3:
            raise ValueError("n_robustness_blocks must be >= 3")
        if not (0.0 <= self.diversity_weight <= 1.0):
            raise ValueError("diversity_weight must be in [0, 1]")
        if self.weight_stability < 0.0 or self.weight_robustness < 0.0:
            raise ValueError("weight_stability and weight_robustness must be >= 0")
