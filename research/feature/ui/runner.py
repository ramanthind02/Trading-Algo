"""Thin orchestration for the lightweight feature_research UI runner."""
from __future__ import annotations

from research.feature.config import ResearchConfig
from research.feature.exploration import execute_exploration_phase
from research.feature.shared import FeatureResearchPhase
from research.feature.shared.visualization_paths import (
    canonical_in_sample_visualization_dir,
    walkforward_visualization_csv_dir,
)
from research.feature.ui.artifact_catalog import ensure_phase_visualization_reports
from research.feature.ui.workspace_manifest import (
    write_exploration_manifests,
    write_validation_manifest,
)
from research.feature.ui.contracts import FeatureResearchUiRequest, PhaseRunPlan
from research.feature.ui.planner import apply_ui_request
from research.feature.validation import run_validation_pipeline


def render_phase_plan(plan: PhaseRunPlan) -> str:
    """Format a run plan for terminal preview output."""

    field_lines = [f"- {field.label}: {field.value}" for field in plan.fields]
    step_lines = [f"- {step}" for step in plan.will_run]
    note_lines = [f"- {note}" for note in plan.notes]
    sections = [
        plan.title,
        "",
        plan.description,
        "",
        "Run summary:",
        *field_lines,
        "",
        "What will run:",
        *step_lines,
        "",
        "Execution command:",
        plan.command,
        "",
        "Notes:",
        *note_lines,
    ]
    return "\n".join(sections)


def execute_phase_request(
    config: ResearchConfig,
    request: FeatureResearchUiRequest,
) -> int:
    """Execute a requested phase using the current research pipeline entrypoints."""

    configured = apply_ui_request(config, request)
    match request.phase:
        case FeatureResearchPhase.EXPLORATION:
            exit_code = _execute_exploration(configured)
        case FeatureResearchPhase.VALIDATION:
            exit_code = _execute_validation(configured)
        case FeatureResearchPhase.PORTFOLIO_ADDITION:
            exit_code = _execute_portfolio_addition(configured)
    if exit_code == 0:
        _refresh_visualization_reports(configured, request.phase)
    return exit_code


def _execute_exploration(config: ResearchConfig) -> int:
    exit_code = 0
    result = None
    try:
        result = execute_exploration_phase(config, config.reports_dir)
    except Exception:
        exit_code = 1
        raise
    finally:
        write_exploration_manifests(config)
        generated = ensure_phase_visualization_reports(
            config,
            FeatureResearchPhase.EXPLORATION,
        )
        if generated:
            print(
                f"Generated {len(generated)} Matplotlib plot(s) for exploration workspace."
            )

    assert result is not None
    print(
        f"Exploration complete. {len(result.eda_results)} combo(s) written to "
        f"{config.reports_dir}."
    )
    if result.pass1_binning is not None:
        decile_chart = canonical_in_sample_visualization_dir() / "matplotlib" / "atr_pct_decile_chart.png"
        print(f"Pass 1 ATR% decile chart written to {decile_chart}.")
    if result.robustness_artifacts is not None:
        print(
            "Robustness summary written to "
            f"{result.robustness_artifacts['summary_csv']} and "
            f"{result.robustness_artifacts['markdown']}."
        )
    if result.permutation_summary_csv is not None:
        print(
            "Permutation summary written to "
            f"{result.permutation_summary_csv} and {result.permutation_summary_md}."
        )
    viz_dir = canonical_in_sample_visualization_dir()
    print(f"Exploration visualization CSVs live under {viz_dir}.")
    filter_summary = viz_dir / "filter_exploration_summary.csv"
    if filter_summary.is_file():
        print(f"Filter A/B/C summary: {filter_summary}")
        filter_chart = viz_dir / "matplotlib" / "filter_gate_comparison.png"
        if filter_chart.is_file():
            print(f"Filter A/B/C chart: {filter_chart}")
    return exit_code


def _execute_validation(config: ResearchConfig) -> int:
    report = run_validation_pipeline(config, output_dir=None)
    print(
        f"Validation complete. {len(report.folds_df)} fold(s). "
        f"Artifacts written to {config.output_root}."
    )
    from research.feature.shared.visualization_paths import walkforward_visualization_csv_dir

    validation_viz_dir = walkforward_visualization_csv_dir(config.output_root, "validation")
    robustness_json = validation_viz_dir / "validation_robustness_report.json"
    if config.validation_robustness.enabled and robustness_json.exists():
        print(
            "Validation robustness report written to "
            f"{robustness_json} and plots under {validation_viz_dir / 'matplotlib'}."
        )
    gate_json = validation_viz_dir / "portfolio_addition_report.json"
    if config.portfolio_addition_gate.enabled and gate_json.exists():
        print(
            "Portfolio addition gate report written to "
            f"{gate_json} and plots under {validation_viz_dir / 'matplotlib'}."
        )
        gate_ts = validation_viz_dir / "portfolio_gate_tearsheets"
        if config.portfolio_addition_gate.emit_tearsheets and gate_ts.is_dir():
            print(f"Portfolio gate QuantStats tearsheets (12 expected): {gate_ts}")
        elif config.portfolio_addition_gate.emit_sleeve_tearsheets:
            sleeve_dirs = sorted(gate_ts.glob("sleeve_*")) if gate_ts.is_dir() else []
            if sleeve_dirs:
                print(
                    "Sleeve portfolio tearsheets: "
                    + ", ".join(str(path.name) for path in sleeve_dirs)
                    + f" under {gate_ts}."
                )
    write_validation_manifest(config)
    return 0


def _execute_portfolio_addition(config: ResearchConfig) -> int:
    validation_viz_dir = walkforward_visualization_csv_dir(config.output_root, "validation")
    gate_json = validation_viz_dir / "portfolio_addition_report.json"
    if gate_json.exists():
        print(
            "Portfolio addition gate results are available from the latest Validation run at "
            f"{gate_json}."
        )
    else:
        print(
            "Portfolio addition gate runs automatically with Validation. "
            "Run Validation to generate portfolio addition gate artifacts."
        )
    return 0


def _refresh_visualization_reports(
    config: ResearchConfig,
    phase: FeatureResearchPhase,
) -> None:
    if phase is FeatureResearchPhase.EXPLORATION:
        return
    generated = ensure_phase_visualization_reports(config, phase)
    if not generated:
        return
    print(f"Generated {len(generated)} Matplotlib plot(s) for the workspace.")
