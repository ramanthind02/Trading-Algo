"""Thin orchestration for the exploration phase (EDA, robustness, permutation)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from feature_research.config import ResearchConfig
from feature_research.exploration.filter_gate_catalog import (
    resolve_exploration_winner_params,
)
from feature_research.exploration.filter_gate_followup import run_filter_gate_exploration_for_winner
from feature_research.exploration.pass1_vol_regime import (
    exploration_includes_filter_gate,
    run_pass1_vol_regime_binning,
)
from feature_research.pipelines.param_perturbation import (
    PerturbationPipelineResult,
    exploration_perturbation_enabled,
    run_min_step_perturbation_pipeline,
)
from feature_research.pipelines.in_sample import run_eda_pipeline as _run_eda_pipeline
from feature_research.pipelines.permutation import (
    run_permutation_pipeline as _run_permutation_pipeline,
    write_permutation_summary as _write_permutation_summary,
)
from feature_research.pipelines.robustness import (
    run_robustness_pipeline as _run_robustness_pipeline,
    write_robustness_summary as _write_robustness_summary,
)
from feature_research.shared import FeatureResearchPhase, PhaseArtifactMap
from feature_research.shared.phase_api import normalize_phase_artifact_map
from feature_research.ui.artifact_catalog import ensure_phase_visualization_reports

if TYPE_CHECKING:
    from feature_selection.validation.reports import PermutationTestSuite
    from feature_research.pipelines.robustness import InSampleRobustnessReport


@dataclass(frozen=True)
class ExplorationPhaseResult:
    """Artifacts produced by a full exploration run."""

    eda_results: PhaseArtifactMap
    pass1_binning: dict[str, int | str] | None = None
    robustness_report: "InSampleRobustnessReport | None" = None
    robustness_artifacts: PhaseArtifactMap | None = None
    perturbation_result: PerturbationPipelineResult | None = None
    permutation_suite: "PermutationTestSuite | None" = None
    permutation_summary_csv: Path | None = None
    permutation_summary_md: Path | None = None
    matplotlib_plots: tuple[Path, ...] = ()


def exploration_permutation_enabled(config: ResearchConfig) -> bool:
    """Whether vector-shuffle permutation should run during exploration."""

    return config.permutation.enabled and config.permutation.run_vector_shuffle


def execute_exploration_phase(
    config: ResearchConfig,
    output_dir: Path,
) -> ExplorationPhaseResult:
    """Run exploration in canonical order: sweep/EDA, robustness, perturbation, permutation."""

    output_dir.mkdir(parents=True, exist_ok=True)

    pass1_binning: dict[str, int | str] | None = None
    if exploration_includes_filter_gate(config):
        pass1_binning = run_pass1_vol_regime_binning(config)

    eda_results = normalize_phase_artifact_map(_run_eda_pipeline(config, output_dir))

    robustness_report: InSampleRobustnessReport | None = None
    robustness_artifacts: PhaseArtifactMap | None = None
    if config.robustness.enabled:
        report, _parameter_grid = _run_robustness_pipeline(config, output_dir)
        robustness_report = report
        robustness_artifacts = normalize_phase_artifact_map(
            _write_robustness_summary(report, output_dir)
        )

    if (
        config.exploration_filter_gates.enabled
        and config.exploration_filter_gates.scope == "winning_signal_only"
    ):
        try:
            winner_params = resolve_exploration_winner_params(
                config,
                robustness_report=robustness_report,
            )
        except ValueError as exc:
            print(f"Filter-gate follow-up skipped: {exc}")
        else:
            run_filter_gate_exploration_for_winner(config, winner_params)

    perturbation_result: PerturbationPipelineResult | None = None
    if (
        exploration_perturbation_enabled(config)
        and robustness_report is not None
    ):
        chosen_params = dict(robustness_report.best_combination.params)
        if chosen_params:
            print(
                "\nMin-step parameter perturbation: evaluating "
                f"{len(config.param_sensitivity.perturbation_specs)} parameter axes..."
            )
            try:
                perturbation_result = run_min_step_perturbation_pipeline(
                    config,
                    output_dir,
                    chosen_params=chosen_params,
                )
            except Exception as exc:
                print(f"Perturbation skipped: {exc}")
            else:
                print(
                    "Perturbation artifacts: "
                    f"{perturbation_result.report_json} "
                    f"(stability={perturbation_result.result.stability_ratio:.4f}, "
                    f"{'PASS' if perturbation_result.result.passed else 'FAIL'})"
                )

    permutation_suite: PermutationTestSuite | None = None
    permutation_summary_csv: Path | None = None
    permutation_summary_md: Path | None = None
    if exploration_permutation_enabled(config):
        selected_combo_name = (
            None
            if robustness_report is None
            else robustness_report.best_param_combo
        )
        suite, param_grid = _run_permutation_pipeline(
            config,
            output_dir,
            selected_combo_name=selected_combo_name,
        )
        permutation_suite = suite
        csv_path, md_path = _write_permutation_summary(
            suite,
            output_dir,
            objective_metric=config.permutation.objective_metric,
            param_grid=param_grid,
            full_grid_permutation=(
                None
                if robustness_report is None
                else robustness_report.full_grid_permutation
            ),
            selection_metric_label=(
                None if robustness_report is None else robustness_report.selection_metric
            ),
        )
        permutation_summary_csv = csv_path
        permutation_summary_md = md_path

    matplotlib_plots = tuple(
        ensure_phase_visualization_reports(
            config,
            FeatureResearchPhase.EXPLORATION,
        )
    )
    if matplotlib_plots:
        print(
            f"\nMatplotlib reports: {len(matplotlib_plots)} plot(s) under "
            f"{matplotlib_plots[0].parent.resolve()}"
        )

    return ExplorationPhaseResult(
        eda_results=eda_results,
        pass1_binning=pass1_binning,
        robustness_report=robustness_report,
        robustness_artifacts=robustness_artifacts,
        permutation_suite=permutation_suite,
        permutation_summary_csv=permutation_summary_csv,
        permutation_summary_md=permutation_summary_md,
        perturbation_result=perturbation_result,
        matplotlib_plots=matplotlib_plots,
    )
