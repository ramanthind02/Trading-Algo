"""Shared feature_research configuration. Edit here once; all phases reuse it."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from feature_research.walkforward.config import (
    WalkforwardResearchConfig,
    WalkforwardSelectionMethod,
    WeightLayerAlgorithm,
)
from feature_selection.validation.config import PermutationModeStage2
from feature_selection.validation.objective_metrics import ObjectiveMetricSpec
from utils.core.enums import Ticker, TimeFrame

_FEATURE_RESEARCH_DIR = Path(__file__).resolve().parent

RAW_TARGET_COLS: frozenset[str] = frozenset({"log_return", "raw_return"})


@dataclass(frozen=True)
class PermutationSuiteConfig:
    """Settings for the shared permutation test suite."""

    enabled: bool = False
    nreps: int = 100
    alpha: float = 0.10
    metric_threshold: float = 0.0
    top_k: int = 3
    min_folds_stable: int = 1
    random_seed: int | None = 42
    permutation_mode_stage2: PermutationModeStage2 = "candle_shuffle"
    fold_years: int = 1
    objective_metric: ObjectiveMetricSpec = field(
        default_factory=lambda: ObjectiveMetricSpec(builtin="sharpe")
    )


@dataclass(frozen=True)
class BaseResearchConfig:
    """Shared researcher-editable settings. Phases add reports_dir and phase-specific fields."""

    tickers: list[Ticker]
    start: datetime
    end: datetime
    use_cache: bool
    populate_cache: bool
    permutation_suite: PermutationSuiteConfig
    walkforward_test_step: int = 365
    walkforward_num_steps: int = 8
    walkforward_output_root: Path = Path("feature_research/shared_results")
    walkforward_selection_method: WalkforwardSelectionMethod = WalkforwardSelectionMethod.TOP_K
    weight_layer_algorithm: WeightLayerAlgorithm = WeightLayerAlgorithm.INVERSE_CORRELATION

    def build_walkforward(
        self,
        *,
        test_step: int | None = None,
        num_steps: int | None = None,
        enabled: bool = False,
    ) -> WalkforwardResearchConfig:
        step = test_step if test_step is not None else self.walkforward_test_step
        steps = num_steps if num_steps is not None else self.walkforward_num_steps
        train_end = self.end - timedelta(days=step * steps)
        if train_end <= self.start:
            train_end = self.end - timedelta(days=step)
        return WalkforwardResearchConfig(
            train_start=self.start,
            train_end=train_end,
            enabled=enabled,
            test_step=step,
            num_steps=steps,
            selection_method=self.walkforward_selection_method,
            weight_layer_algorithm=self.weight_layer_algorithm,
            output_root=self.walkforward_output_root,
        )


def load_config() -> BaseResearchConfig:
    """Single source of truth for tickers, date range, cache, and walkforward defaults.

    Edit here; in_sample and walkforward phases import and extend this.
    """
    # ==========================================================================
    # EDIT BELOW
    # ==========================================================================
    tickers = [
        Ticker.ES,
        Ticker.NQ,
        Ticker.YM,
        Ticker.RTY,
    ]
    start = datetime(2000, 1, 1)
    end = datetime(2024, 12, 31)
    use_cache = True
    populate_cache = True
    permutation_suite = PermutationSuiteConfig(enabled=False)
    walkforward_test_step = 365
    walkforward_num_steps = 8
    walkforward_selection_method = WalkforwardSelectionMethod.TOP_K
    weight_layer_algorithm = WeightLayerAlgorithm.INVERSE_CORRELATION
    # ==========================================================================
    # EDIT ABOVE
    # ==========================================================================

    return BaseResearchConfig(
        tickers=tickers,
        start=start,
        end=end,
        use_cache=use_cache,
        populate_cache=populate_cache,
        permutation_suite=permutation_suite,
        walkforward_test_step=walkforward_test_step,
        walkforward_num_steps=walkforward_num_steps,
        walkforward_output_root=Path("feature_research/shared_results"),
        walkforward_selection_method=walkforward_selection_method,
        weight_layer_algorithm=weight_layer_algorithm,
    )
