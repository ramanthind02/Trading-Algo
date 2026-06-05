"""Docs-driven workflow metadata for the feature_research workspace UI."""
from __future__ import annotations

from research.feature.shared import FeatureResearchPhase
from research.feature.ui.contracts import UiWorkflowStep


def workflow_steps() -> tuple[UiWorkflowStep, ...]:
    """Canonical research workflow aligned to `docs/SaaS/robustness_tests`."""

    return (
        UiWorkflowStep(
            phase=FeatureResearchPhase.EXPLORATION,
            title="1. Exploration",
            summary=(
                "Run the parameter sweep, inspect robustness, and judge whether the selected "
                "search space shows a stable in-sample edge."
            ),
            gate="Soft gate: DSR >= 0.50 before moving on to parameter selection.",
            docs_path="docs/SaaS/robustness_tests/in_sample.md",
        ),
        UiWorkflowStep(
            phase=FeatureResearchPhase.VALIDATION,
            title="2. Validation",
            summary=(
                "Lock one parameter choice and test whether the strategy behaves consistently on "
                "the validation slice without refitting."
            ),
            gate="Primary warning gate: clear CUSUM break, acceptable degradation, preserved ranking.",
            docs_path="docs/SaaS/robustness_tests/validation.md",
        ),
        UiWorkflowStep(
            phase=FeatureResearchPhase.PORTFOLIO_ADDITION,
            title="3. Portfolio Addition",
            summary=(
                "Check whether the validated strategy still improves the existing portfolio before "
                "admission, using IS + validation data only."
            ),
            gate="Primary gate: portfolio delta Sharpe >= 0.02 before committing the strategy.",
            docs_path="docs/SaaS/robustness_tests/portfolio_addition.md",
        ),
    )
