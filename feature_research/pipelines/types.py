"""Shared types for feature_research pipeline phases."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from utils.core.enums import TimeFrame

ComboSignalTargetMap = Mapping[tuple[tuple[str, object], ...], pd.DataFrame]


@dataclass(frozen=True)
class OosCorrelationBundle:
    """Inputs for portfolio_research feature–vault correlation (OOS extended window)."""

    combo_signal_target: ComboSignalTargetMap
    selection_summary_df: pd.DataFrame
    eval_tf: TimeFrame
    research_eval_bias_spec: dict[str, Any]
    target_col: str
    extended_start: datetime
    extended_end: datetime
    module_name: str
