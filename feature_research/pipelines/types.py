"""Typed payloads produced by feature_research pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

import pandas as pd

from utils.core.enums import TimeFrame

ComboSignalTargetMap = dict[tuple[tuple[str, object], ...], pd.DataFrame]


@dataclass(frozen=True)
class OosCorrelationBundle:
    """OOS walkforward outputs needed for feature–vault correlation exports."""

    combo_signal_target: ComboSignalTargetMap
    selection_summary_df: pd.DataFrame
    eval_tf: TimeFrame
    research_eval_bias_spec: Mapping[str, object]
    target_col: str
    extended_start: datetime
    extended_end: datetime
    module_name: str
