"""Shared contracts for compatibility-preserving feature_research phases."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Mapping, TypeAlias

import pandas as pd

from lib.core.enums import TimeFrame

ComboKey: TypeAlias = tuple[tuple[str, object], ...]
ComboSignalTargetMap: TypeAlias = dict[ComboKey, pd.DataFrame]
PhaseArtifactMap: TypeAlias = dict[str, Path]
ParameterGrid: TypeAlias = list[dict[str, object]]


class FeatureResearchPhase(str, Enum):
    """Top-level phases in the target feature_research package layout."""

    EXPLORATION = "exploration"
    VALIDATION = "validation"
    PORTFOLIO_ADDITION = "portfolio_addition"


@dataclass(frozen=True)
class PortfolioAdditionBundle:
    """Walkforward outputs needed when evaluating a feature for portfolio addition."""

    combo_signal_target: ComboSignalTargetMap
    selection_summary_df: pd.DataFrame
    eval_tf: TimeFrame
    research_eval_bias_spec: Mapping[str, object]
    target_col: str
    extended_start: datetime
    extended_end: datetime
    module_name: str


# Backward-compatible alias while callers migrate away from OOS-centric naming.
OosCorrelationBundle = PortfolioAdditionBundle

__all__ = [
    "ComboKey",
    "ComboSignalTargetMap",
    "FeatureResearchPhase",
    "OosCorrelationBundle",
    "ParameterGrid",
    "PhaseArtifactMap",
    "PortfolioAdditionBundle",
]
