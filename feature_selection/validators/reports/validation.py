"""Top-level validation report."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
import json

from .eda import EDAReport
from .permutation import PermutationReport
from .stability import StabilityReport


@dataclass(frozen=True)
class ValidationReport:
    """
    Accumulated report from all validation stages.

    Progressive accumulation:
    - Start with EDAReport
    - Add Stage1Report (vector shuffle)
    - Add Stage2Report (pipeline permutation)
    - Add Stage3Report (walkforward stability)
    - Optionally add ParameterReport (sensitivity analysis)
    """

    feature_name: str
    feature_type: Literal['continuous', 'rule_based']
    timestamp: datetime

    # Stage reports (optional, added progressively)
    eda_report: EDAReport | None = None
    stage1_report: PermutationReport | None = None
    stage2_report: PermutationReport | None = None
    stage3_report: StabilityReport | None = None
    parameter_report: Any | None = None  # ParameterReport (to be implemented)

    # Overall validation status
    validation_status: Literal['passed', 'failed', 'incomplete'] = 'incomplete'
    failure_stage: str | None = None  # e.g., "Stage 1: Vector Shuffle"

    # Researcher notes (filled manually after review)
    researcher_notes: str = ""
    ensemble_decision: list[dict[str, Any]] | None = None  # Selected param combos

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'feature_name': self.feature_name,
            'feature_type': self.feature_type,
            'timestamp': self.timestamp.isoformat(),
            'validation_status': self.validation_status,
            'failure_stage': self.failure_stage,
            'researcher_notes': self.researcher_notes,
            'ensemble_decision': self.ensemble_decision,
            # Note: EDA/stage reports would need their own to_dict methods
            # For now, include basic info
            'has_eda_report': self.eda_report is not None,
            'has_stage1_report': self.stage1_report is not None,
            'has_stage2_report': self.stage2_report is not None,
            'has_stage3_report': self.stage3_report is not None,
        }

    def to_json(self) -> str:
        """Generate JSON report."""
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Validation Report: {self.feature_name}",
            f"",
            f"**Feature Type:** {self.feature_type}",
            f"**Timestamp:** {self.timestamp.isoformat()}",
            f"**Status:** {self.validation_status}",
            f"",
        ]

        if self.failure_stage:
            lines.append(f"**Failure Stage:** {self.failure_stage}")
            lines.append("")

        if self.eda_report:
            lines.append("## EDA Report")
            lines.append("")
            lines.append(f"- Feature mean: {self.eda_report.feature_stats.mean:.4f}")
            lines.append(f"- Feature std: {self.eda_report.feature_stats.std:.4f}")
            lines.append(f"- Target correlation (Pearson): {self.eda_report.correlations['pearson']:.4f}")
            lines.append(f"- Stationary (ADF): {self.eda_report.adf_test.is_stationary}")
            lines.append("")

        if self.stage1_report:
            lines.append("## Stage 1: Vector Shuffle")
            lines.append("")
            lines.append(f"- Observed Sharpe: {self.stage1_report.observed_sharpe:.4f}")
            lines.append(f"- P-value: {self.stage1_report.p_value:.4f}")
            lines.append(f"- Passed: {self.stage1_report.passed}")
            lines.append("")

        if self.researcher_notes:
            lines.append("## Researcher Notes")
            lines.append("")
            lines.append(self.researcher_notes)
            lines.append("")

        return "\n".join(lines)
