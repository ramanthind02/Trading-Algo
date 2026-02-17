"""Report dataclasses for permutation testing (T013-T017)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class VectorShuffleReport:
    """Results from T013: Vector Shuffle permutation test."""

    param_combo: str
    original_metric: float
    null_distribution: np.ndarray = field(compare=False, hash=False)
    critical_value: float
    p_value: float
    passed: bool
    alpha: float
    nreps: int


@dataclass(frozen=True)
class PipelinePermutationReport:
    """Results from T014: Pipeline Permutation test."""

    param_combo: str
    feature_type: Literal['continuous', 'rule_based']
    permutation_mode: str
    original_metric: float
    null_distribution: np.ndarray = field(compare=False, hash=False)
    critical_value: float
    p_value: float
    passed: bool
    alpha: float
    nreps: int
    no_trade_permutations: int


@dataclass(frozen=True)
class FoldResult:
    """Result for a single walk-forward fold (T015)."""

    fold_id: str
    fold_period: Tuple[str, str]
    top_k_params: List[str]
    smoothed_objectives: Dict[str, float]
    passed_permutation_overlay: List[bool]


@dataclass(frozen=True)
class WalkforwardStabilityReport:
    """T015: Walk-forward stability analysis report."""

    feature_name: str
    feature_type: Literal['continuous', 'rule_based']
    fold_results: List[FoldResult]
    consistency_metrics: Dict[str, float]
    is_stable: bool
    stability_verdict: str
    top_k: int


@dataclass(frozen=True)
class FunnelStatistics:
    """T016: Funnel statistics across all validation stages."""

    total_params: int
    stage1_pass: int
    stage2_pass: int
    stable_params: int
    ensemble_candidates: int
    computational_savings_pct: float


@dataclass(frozen=True)
class PermutationTestSuite:
    """T013-T017: Complete permutation test suite for a single feature."""

    feature_name: str
    feature_type: Literal['continuous', 'rule_based']
    stage1_reports: Dict[str, VectorShuffleReport]
    stage2_reports: Dict[str, PipelinePermutationReport]
    stage3_report: WalkforwardStabilityReport
    funnel_stats: FunnelStatistics
    ensemble_candidates: List[str]
    summary: str


@dataclass(frozen=True)
class ReportBundle:
    """T017: File bundle for a completed test suite."""

    suite_json: Path
    suite_markdown: Path
    suite_html: Optional[Path]
    stage1_plots: Dict[str, Path]
    stage2_plots: Dict[str, Path]
    stage3_plot: Path
    funnel_plot: Path
    timestamp: datetime
