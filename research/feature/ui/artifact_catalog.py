"""Discover and categorize feature_research workspace plots and reports."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re

from research.feature.config import ResearchConfig
from research.feature.exploration.pass1_vol_regime import exploration_includes_filter_gate
from research.feature.shared import FeatureResearchPhase
from research.feature.shared.visualization_paths import (
    canonical_in_sample_visualization_dir,
    validation_walkforward_output_dir,
    walkforward_visualization_csv_dir,
)
from research.feature.ui.workspace_manifest import (
    MANIFEST_FILENAME,
    exploration_viz_matches_config,
    read_workspace_manifest,
    validation_artifacts_current,
)
from research.feature.visualization.matplotlib_reports import generate_detected_matplotlib_plots
from lib.core.repo_bootstrap import require_repo_root, resolve_repo_path
from tools.research_workspace.constants import (
    PANEL_KIND_PRIMARY,
    PANEL_KIND_RAW,
    RAW_DATA_SECTION_ID,
)

__all__ = [
    "PANEL_KIND_PRIMARY",
    "PANEL_KIND_RAW",
    "RAW_DATA_SECTION_ID",
    "WorkspaceArtifactCategory",
    "build_phase_report_sections",
    "categorize_artifact_path",
    "discover_artifact_records",
    "ensure_phase_visualization_reports",
    "group_artifact_records",
]
from tools.research_workspace.discovery import discover_artifact_records as _discover_records
from tools.research_workspace.discovery import group_artifact_records as _group_records
from tools.research_workspace.models import CategoryGroup, PanelPolicy
from tools.research_workspace.sections import build_phase_report_sections as _build_sections
from research.feature.inclusion_gates import gate_tearsheet_panel_title
from tools.research_workspace.panels import default_friendly_panel_title

_FEATURE_REPO_ROOT = require_repo_root(Path(__file__))


def _repo_path(path: Path) -> Path:
    return resolve_repo_path(path, repo_root=_FEATURE_REPO_ROOT)
_SUMMARY_JSON_NAMES = frozenset(
    {
        "robustness_report.json",
        "validation_robustness_report.json",
        "portfolio_addition_report.json",
        "perturbation_report.json",
        "report.json",
    }
)


class WorkspaceArtifactCategory(str, Enum):
    """Doc-aligned artifact groups shown in the research workspace."""

    QUANTSTATS_TEARSHEET = "quantstats_tearsheet"
    MATPLOTLIB_REPORT = "matplotlib_report"
    PARAMETER_SENSITIVITY = "parameter_sensitivity"
    PERMUTATION = "permutation"
    VISUALIZATION_DATA = "visualization_data"
    ROBUSTNESS = "robustness"
    WALKFORWARD = "walkforward"
    OTHER = "other"


@dataclass(frozen=True)
class WorkspaceArtifactGroup:
    """One categorized bucket of workspace artifacts."""

    category: WorkspaceArtifactCategory
    label: str
    description: str
    priority: int


_CATEGORY_GROUPS: tuple[WorkspaceArtifactGroup, ...] = (
    WorkspaceArtifactGroup(
        WorkspaceArtifactCategory.QUANTSTATS_TEARSHEET,
        "QuantStats Tearsheets",
        "Twelve portfolio-gate QuantStats tearsheets (candidate, sleeve, full book).",
        0,
    ),
    WorkspaceArtifactGroup(
        WorkspaceArtifactCategory.MATPLOTLIB_REPORT,
        "Matplotlib Reports",
        "PNG charts generated from visualization CSV exports.",
        1,
    ),
    WorkspaceArtifactGroup(
        WorkspaceArtifactCategory.PARAMETER_SENSITIVITY,
        "Parameter Sensitivity",
        "Interactive parameter surface and sensitivity tables.",
        2,
    ),
    WorkspaceArtifactGroup(
        WorkspaceArtifactCategory.PERMUTATION,
        "Permutation Testing",
        "Vector-shuffle permutation summaries and diagnostics.",
        3,
    ),
    WorkspaceArtifactGroup(
        WorkspaceArtifactCategory.VISUALIZATION_DATA,
        "Visualization Data",
        "CSV tables that feed Matplotlib and exploratory review.",
        4,
    ),
    WorkspaceArtifactGroup(
        WorkspaceArtifactCategory.ROBUSTNESS,
        "Robustness Summaries",
        "In-sample robustness JSON, CSV, and markdown outputs.",
        5,
    ),
    WorkspaceArtifactGroup(
        WorkspaceArtifactCategory.WALKFORWARD,
        "Walk-forward Reports",
        "Validation and portfolio-addition fold artifacts.",
        6,
    ),
    WorkspaceArtifactGroup(
        WorkspaceArtifactCategory.OTHER,
        "Other Artifacts",
        "Additional files discovered under the phase output roots.",
        7,
    ),
)

_GROUP_BY_CATEGORY = {group.category: group for group in _CATEGORY_GROUPS}

_PARAM_SENSITIVITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^param_sensitivity(?:_|\.|$)", re.IGNORECASE),
    re.compile(r"^param_sensitivity_by_ticker", re.IGNORECASE),
    re.compile(r"^param_combo_long", re.IGNORECASE),
    re.compile(r"^perturbation", re.IGNORECASE),
)

_PERMUTATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^permutation_vector_shuffle", re.IGNORECASE),
    re.compile(r"^permutation_summary", re.IGNORECASE),
    re.compile(r"^permutation_.*\.png$", re.IGNORECASE),
)

_VISUALIZATION_DATA_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^equity_curve", re.IGNORECASE),
    re.compile(r"^filter_exploration_equity_curve", re.IGNORECASE),
    re.compile(r"^vault_correlation", re.IGNORECASE),
    re.compile(r"^robustness_rolling_cusum", re.IGNORECASE),
)

_ROBUSTNESS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^robustness_", re.IGNORECASE),
    re.compile(r"^validation_robustness", re.IGNORECASE),
    re.compile(r"^validation_rank_correlation", re.IGNORECASE),
    re.compile(r"^validation_z_cusum_sr", re.IGNORECASE),
    re.compile(r"^validation_equity_bands", re.IGNORECASE),
    re.compile(r"^portfolio_addition", re.IGNORECASE),
)

_WALKFORWARD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^report\.json$", re.IGNORECASE),
    re.compile(r"^fold_scores", re.IGNORECASE),
    re.compile(r"^selected_params", re.IGNORECASE),
    re.compile(r"^selection_summary", re.IGNORECASE),
    re.compile(r"^walkforward_", re.IGNORECASE),
)

def _feature_friendly_panel_title(path: Path) -> str:
    gate_title = gate_tearsheet_panel_title(path)
    if gate_title is not None:
        return gate_title
    return default_friendly_panel_title(path)


def _is_primary_panel(path: Path, kind: str) -> bool:
    normalized = path.as_posix().lower()
    if kind == "json" and path.name.lower() == "perturbation_report.json":
        return True
    if kind == "html" and "tearsheet" in path.name.lower():
        return True
    if kind == "image" and path.name.lower() == "atr_pct_decile_chart.png":
        return True
    if kind == "image" and path.name.lower() == "filter_gate_comparison.png":
        return True
    if kind == "image" and path.name.lower() == "filter_exploration_equity_curve.png":
        return True
    if kind == "image" and path.name.lower() == "equity_curve.png":
        return True
    if kind == "image" and path.name.lower() == "cusum_stability.png":
        return True
    if kind == "image" and path.name.lower().startswith("validation_"):
        return True
    if kind == "image" and path.name.lower().startswith("portfolio_addition_"):
        return True
    return False


_FEATURE_PANEL_POLICY = PanelPolicy(
    summary_json_names=_SUMMARY_JSON_NAMES,
    always_visible_categories=frozenset(
        {
            WorkspaceArtifactCategory.PARAMETER_SENSITIVITY.value,
            WorkspaceArtifactCategory.ROBUSTNESS.value,
            WorkspaceArtifactCategory.PERMUTATION.value,
        }
    ),
    is_featured_panel=_is_primary_panel,
    friendly_panel_title=_feature_friendly_panel_title,
)


def _to_category_groups(
    groups: tuple[WorkspaceArtifactGroup, ...],
) -> tuple[CategoryGroup, ...]:
    return tuple(
        CategoryGroup(
            category=group.category.value,
            label=group.label,
            description=group.description,
            priority=group.priority,
        )
        for group in groups
    )


def category_groups_for_phase(phase: FeatureResearchPhase) -> tuple[WorkspaceArtifactGroup, ...]:
    """Return ordered category metadata for one pipeline phase."""

    if phase is FeatureResearchPhase.EXPLORATION:
        excluded = {WorkspaceArtifactCategory.QUANTSTATS_TEARSHEET, WorkspaceArtifactCategory.WALKFORWARD}
        return tuple(group for group in _CATEGORY_GROUPS if group.category not in excluded)
    if phase is FeatureResearchPhase.VALIDATION:
        excluded = {WorkspaceArtifactCategory.PERMUTATION}
        return tuple(group for group in _CATEGORY_GROUPS if group.category not in excluded)
    if phase is FeatureResearchPhase.PORTFOLIO_ADDITION:
        excluded = {
            WorkspaceArtifactCategory.PERMUTATION,
            WorkspaceArtifactCategory.WALKFORWARD,
        }
        return tuple(group for group in _CATEGORY_GROUPS if group.category not in excluded)
    excluded = {WorkspaceArtifactCategory.PERMUTATION, WorkspaceArtifactCategory.ROBUSTNESS}
    return tuple(group for group in _CATEGORY_GROUPS if group.category not in excluded)


def phase_discovery_roots(config: ResearchConfig, phase: FeatureResearchPhase) -> tuple[Path, ...]:
    """Return existing directories to scan for one phase."""

    if phase is FeatureResearchPhase.EXPLORATION:
        from research.feature.binning.config import load_binning_research_config

        candidates: list[Path] = [_repo_path(config.reports_dir)]
        if exploration_includes_filter_gate(config):
            binning_phase_dir = (
                load_binning_research_config().reports_dir / "binning_phase"
            )
            candidates.append(binning_phase_dir)
        viz_dir = canonical_in_sample_visualization_dir()
        viz_manifest = read_workspace_manifest(viz_dir / MANIFEST_FILENAME)
        if exploration_viz_matches_config(viz_manifest, config) or (
            viz_manifest is None
            and (_repo_path(config.reports_dir) / "robustness_report.json").is_file()
        ):
            candidates.extend((viz_dir, viz_dir / "matplotlib"))
    else:
        module_name = str(config.eval_bias_spec.get("module_name", "")).strip()
        phase_subdir = "validation" if phase is FeatureResearchPhase.VALIDATION else "oos"
        output_dir = validation_walkforward_output_dir(config.output_root, module_name)
        validation_viz_dir = walkforward_visualization_csv_dir(config.output_root, "validation")
        visualization_dir = walkforward_visualization_csv_dir(config.output_root, phase_subdir)
        inclusion_tearsheet_dirs = tuple(
            path
            for path in _repo_path(Path(config.output_root)).glob("inclusion_tearsheets_*")
            if path.is_dir()
        )
        candidates = [_repo_path(output_dir), _repo_path(output_dir) / "tearsheets"]
        if validation_artifacts_current(config):
            candidates.extend(
                (
                    _repo_path(validation_viz_dir),
                    _repo_path(visualization_dir),
                    _repo_path(visualization_dir) / "matplotlib",
                    *inclusion_tearsheet_dirs,
                )
            )
            if phase is FeatureResearchPhase.PORTFOLIO_ADDITION:
                gate_dir = _repo_path(validation_viz_dir) / "portfolio_gate_tearsheets"
                if gate_dir.is_dir():
                    candidates.extend((_repo_path(validation_viz_dir) / "matplotlib", gate_dir))
        elif phase is FeatureResearchPhase.PORTFOLIO_ADDITION:
            pass
    return tuple(path for path in candidates if path.exists())


def categorize_artifact_path(path: Path) -> WorkspaceArtifactCategory:
    """Map one artifact path to a doc-aligned workspace category."""

    name = path.name
    normalized = path.as_posix().lower()
    suffix = path.suffix.lower()

    if suffix == ".html" and ("tearsheet" in normalized or "/tearsheets/" in normalized):
        return WorkspaceArtifactCategory.QUANTSTATS_TEARSHEET
    if suffix == ".png" and "/matplotlib/" in normalized:
        return WorkspaceArtifactCategory.MATPLOTLIB_REPORT
    if any(pattern.search(name) for pattern in _PERMUTATION_PATTERNS):
        return WorkspaceArtifactCategory.PERMUTATION
    if any(pattern.search(name) for pattern in _PARAM_SENSITIVITY_PATTERNS):
        return WorkspaceArtifactCategory.PARAMETER_SENSITIVITY
    if any(pattern.search(name) for pattern in _ROBUSTNESS_PATTERNS):
        return WorkspaceArtifactCategory.ROBUSTNESS
    if any(pattern.search(name) for pattern in _WALKFORWARD_PATTERNS):
        return WorkspaceArtifactCategory.WALKFORWARD
    if any(pattern.search(name) for pattern in _VISUALIZATION_DATA_PATTERNS):
        return WorkspaceArtifactCategory.VISUALIZATION_DATA
    if suffix == ".png":
        return WorkspaceArtifactCategory.MATPLOTLIB_REPORT
    return WorkspaceArtifactCategory.OTHER


def discover_artifact_records(
    roots: tuple[Path, ...],
    *,
    phase: FeatureResearchPhase,
    relative_to: Path,
) -> list[dict[str, object]]:
    """Discover supported artifacts under ``roots`` and attach category metadata."""

    return _discover_records(
        roots,
        phase=phase.value,
        relative_to=relative_to,
        categorize=lambda path: categorize_artifact_path(path).value,
        category_label=lambda category: _GROUP_BY_CATEGORY[
            WorkspaceArtifactCategory(category)
        ].label,
    )


def group_artifact_records(
    artifacts: list[dict[str, object]],
    phase: FeatureResearchPhase,
) -> list[dict[str, object]]:
    """Bucket flat artifact records into ordered doc-aligned groups."""

    return _group_records(artifacts, _to_category_groups(category_groups_for_phase(phase)))


def ensure_phase_visualization_reports(
    config: ResearchConfig,
    phase: FeatureResearchPhase,
) -> list[Path]:
    """Generate Matplotlib plots when visualization CSV inputs exist."""

    if phase is FeatureResearchPhase.EXPLORATION:
        input_dir = canonical_in_sample_visualization_dir()
    elif phase is FeatureResearchPhase.PORTFOLIO_ADDITION:
        input_dir = _repo_path(walkforward_visualization_csv_dir(config.output_root, "validation"))
    else:
        phase_subdir = "validation" if phase is FeatureResearchPhase.VALIDATION else "oos"
        input_dir = _repo_path(walkforward_visualization_csv_dir(config.output_root, phase_subdir))

    if not input_dir.exists():
        return []

    output_dir = input_dir / "matplotlib"
    generated = (
        generate_detected_matplotlib_plots(
            input_dir=input_dir,
            output_dir=output_dir,
        )
        if any(input_dir.rglob("*.csv"))
        else []
    )
    validation_report_path = input_dir / "validation_robustness_report.json"
    if (
        phase is FeatureResearchPhase.VALIDATION
        and validation_report_path.exists()
        and config.validation_robustness.enabled
    ):
        from research.feature.validation.robustness_runner import refresh_validation_robustness_plots

        generated.extend(refresh_validation_robustness_plots(input_dir, output_dir=output_dir))
    portfolio_report_path = input_dir / "portfolio_addition_report.json"
    if (
        phase in {FeatureResearchPhase.VALIDATION, FeatureResearchPhase.PORTFOLIO_ADDITION}
        and portfolio_report_path.exists()
        and config.portfolio_addition_gate.enabled
    ):
        from research.feature.portfolio_addition.gate_runner import refresh_portfolio_addition_plots

        generated.extend(refresh_portfolio_addition_plots(input_dir, output_dir=output_dir))
    if phase is FeatureResearchPhase.EXPLORATION:
        from research.feature.binning.pipeline import generate_atr_pct_decile_chart

        decile_chart = generate_atr_pct_decile_chart()
        if decile_chart is not None:
            generated.append(decile_chart)
    return generated


def preferred_preview_artifact(
    grouped_artifacts: list[dict[str, object]],
) -> dict[str, object] | None:
    """Pick the first embeddable artifact from the highest-priority group."""

    embeddable_kinds = {"html", "image"}
    for group in grouped_artifacts:
        match = next(
            (artifact for artifact in group["artifacts"] if artifact["kind"] in embeddable_kinds),
            None,
        )
        if match is not None:
            return match
    return None


def build_phase_report_sections(
    grouped_artifacts: list[dict[str, object]],
    phase: FeatureResearchPhase,
) -> tuple[list[dict[str, object]], str | None]:
    """Build ordered report sections with primary panels and a dedicated raw-data section."""

    return _build_sections(
        grouped_artifacts,
        policy=_FEATURE_PANEL_POLICY,
        default_section_priority=_phase_section_priority(phase),
    )


def _phase_section_priority(phase: FeatureResearchPhase) -> tuple[str, ...]:
    if phase is FeatureResearchPhase.EXPLORATION:
        return (
            WorkspaceArtifactCategory.MATPLOTLIB_REPORT.value,
            WorkspaceArtifactCategory.PARAMETER_SENSITIVITY.value,
            WorkspaceArtifactCategory.ROBUSTNESS.value,
            WorkspaceArtifactCategory.PERMUTATION.value,
            WorkspaceArtifactCategory.VISUALIZATION_DATA.value,
        )
    if phase is FeatureResearchPhase.VALIDATION:
        return (
            WorkspaceArtifactCategory.QUANTSTATS_TEARSHEET.value,
            WorkspaceArtifactCategory.MATPLOTLIB_REPORT.value,
            WorkspaceArtifactCategory.ROBUSTNESS.value,
            WorkspaceArtifactCategory.WALKFORWARD.value,
            WorkspaceArtifactCategory.VISUALIZATION_DATA.value,
        )
    if phase is FeatureResearchPhase.PORTFOLIO_ADDITION:
        return (
            WorkspaceArtifactCategory.QUANTSTATS_TEARSHEET.value,
            WorkspaceArtifactCategory.MATPLOTLIB_REPORT.value,
            WorkspaceArtifactCategory.ROBUSTNESS.value,
            WorkspaceArtifactCategory.VISUALIZATION_DATA.value,
        )
    return (
        WorkspaceArtifactCategory.QUANTSTATS_TEARSHEET.value,
        WorkspaceArtifactCategory.MATPLOTLIB_REPORT.value,
        WorkspaceArtifactCategory.WALKFORWARD.value,
        WorkspaceArtifactCategory.VISUALIZATION_DATA.value,
    )
