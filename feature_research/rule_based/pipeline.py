from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from feature_research.rule_based.config import RuleBasedResearchConfig


def run_rule_based_eda_pipeline(
    config: "RuleBasedResearchConfig",
    output_dir: Path,
) -> dict[str, Path]:
    raise NotImplementedError
